// track_centerline_refactored_mst_width.cpp
// ============================================================================
// 트랙 센터라인 + 폭 계산(폐루프) + 최소 곡률 레이싱라인 (논문 흐름에 맞춘 가독화)
// ---------------------------------------------------------------------------
// Pipeline (paper-aligned):
//  1) 입력 링(inner/outer) -> Delaunay -> CDT(제약 강제/복구)
//  2) 트랙 영역 클리핑(+품질 필터)
//  3) 내/외부 라벨 경계 엣지의 중점 추출
//  4) MST 지름 경로 기반 순서화
//  5) 자연 3차 스플라인(TDMA) + 경계 패딩 -> 균일 arc-length 재샘플
//  6) 각 샘플점에서 법선으로 inner/outer까지 거리 -> 코리도 폭 w_L, w_R
//  7) 최소 곡률(∑κ^2 + λ||D1α||^2) 최소화,  lo ≤ α ≤ hi  (α: 법선 오프셋)
//     - GN(선형화) - projected step - outer relinearization
//  8) 결과 저장 (centerline.csv, *_with_geom.csv, *_raceline*.csv)
// ============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

using std::array;
using std::cerr;
using std::cout;
using std::endl;
using std::pair;
using std::string;
using std::vector;

//================================= Config =================================
namespace cfg {
struct Config {
  // 전처리/샘플링
  double dedup_eps = 1e-12;
  bool   add_micro_jitter = true;
  double jitter_eps = 1e-9;
  int    samples = 300;   // 센터라인 재샘플 수
  int    knn_k   = 8;      // MST k-NN

  // CDT 강제삽입/복구 가드
  int  max_flips_per_segment = 20000;
  int  max_global_flips      = 500000;
  int  max_segment_splits    = 8;
  int  max_cdt_rebuilds      = 12;
  bool verbose = true;

  // 삼각형 품질 필터
  bool   enable_quality_filter = true;
  double min_triangle_area      = 1e-10; // 면적 하한
  double min_triangle_angle_deg = 5.0;   // 최소 내각(deg)
  double max_edge_length_scale  = 5.0;   // 중앙값 대비 엣지 길이 상한 배수

  // 클리핑 실패 시 전체 사용 허용
  bool allow_fallback_clip = true;

  // 차량 폭/마진 (코리도 가드)
  double veh_width_m      = 1.0;
  double safety_margin_m  = 0.05;

  // 출력 옵션: 폐루프 시 마지막 샘플 = 첫 샘플 복제 여부
  bool emit_closed_duplicate = false;

  // --- 논문식 raceline 최적화 파라미터 ---
  double lambda_smooth   = 1e-3;  // 스무딩 가중치 λ (||D1 α||^2)
  int    max_outer_iters = 8;     // GN 재선형화 횟수
  int    max_inner_iters = 150;   // 내부 projected step
  double step_init       = 0.6;   // 초기 스텝 (Armijo)
  double step_min        = 1e-6;  // 최소 스텝
  double armijo_c        = 1e-5;  // Armijo 조건 계수
};
inline Config& get(){ static Config C; return C; }
} // namespace cfg

//============================== Geometry ===================================
namespace geom {
struct Vec2 { double x=0.0, y=0.0; };

inline Vec2 operator+(const Vec2& a,const Vec2& b){ return {a.x+b.x, a.y+b.y}; }
inline Vec2 operator-(const Vec2& a,const Vec2& b){ return {a.x-b.x, a.y-b.y}; }
inline Vec2 operator*(const Vec2& a,double s){ return {a.x*s, a.y*s}; }

inline double dot(const Vec2& a,const Vec2& b){ return a.x*b.x + a.y*b.y; }
inline double norm2(const Vec2& a){ return dot(a,a); }
inline double norm (const Vec2& a){ return std::sqrt(norm2(a)); }

inline Vec2 normalize(const Vec2& v, double eps=1e-12){
  double n = norm(v);
  if(n<eps) return {0,0};
  return {v.x/n, v.y/n};
}

inline bool almostEq(const Vec2& a,const Vec2& b,double e=1e-12){
  return (std::fabs(a.x-b.x)<=e && std::fabs(a.y-b.y)<=e);
}

// robust orient/incircle
inline double orient2d_filt(const Vec2& a,const Vec2& b,const Vec2& c){
  double det = (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
  double absa = std::fabs(b.x-a.x)+std::fabs(b.y-a.y);
  double absb = std::fabs(c.x-a.x)+std::fabs(c.y-a.y);
  double err = (absa*absb)*std::numeric_limits<double>::epsilon()*4.0;
  if (std::fabs(det)>err) return det;
  long double adx=(long double)b.x-(long double)a.x;
  long double ady=(long double)b.y-(long double)a.y;
  long double bdx=(long double)c.x-(long double)a.x;
  long double bdy=(long double)c.y-(long double)a.y;
  long double detl = adx*bdy - ady*bdx;
  return (double)detl;
}
inline int orientSign(const Vec2& a,const Vec2& b,const Vec2& c){
  double v=orient2d_filt(a,b,c); return (v>0)-(v<0);
}
inline double incircle_filt(const Vec2& a,const Vec2& b,const Vec2& c,const Vec2& d){
  double adx=a.x-d.x, ady=a.y-d.y;
  double bdx=b.x-d.x, bdy=b.y-d.y;
  double cdx=c.x-d.x, cdy=c.y-d.y;
  double ad=adx*adx+ady*ady, bd=bdx*bdx+bdy*bdy, cd=cdx*cdx+cdy*cdy;
  double det=adx*(bdy*cd-bd*cdy) - ady*(bdx*cd-bd*cdx) + ad*(bdx*cdy-bdy*cdx);
  double mags=(std::fabs(adx)+std::fabs(ady))*(std::fabs(bdx)+std::fabs(bdy))*(std::fabs(cdx)+std::fabs(cdy));
  double err = mags*std::numeric_limits<double>::epsilon()*16.0;
  if (std::fabs(det)>err) return det;
  long double AX=a.x, AY=a.y, BX=b.x, BY=b.y, CX=c.x, CY=c.y, DX=d.x, DY=d.y;
  long double adxl=AX-DX, adyl=AY-DY, bdxl=BX-DX, bdyl=BY-DY, cdxl=CX-DX, cdyl=CY-DY;
  long double adl=adxl*adxl+adyl*adyl, bdl=bdxl*bdxl+bdyl*bdyl, cdl=cdxl*cdxl+cdyl*cdxl;
  long double detl=adxl*(bdyl*cdl-bdl*cdyl) - adyl*(bdxl*cdl-bdl*cdxl) + adl*(bdxl*cdyl-bdyl*cdxl);
  return (double)detl;
}

inline bool ccw(const Vec2&A,const Vec2&B,const Vec2&C){ return orient2d_filt(A,B,C)>0; }

inline bool segIntersectProper(const Vec2& a,const Vec2& b,const Vec2& c,const Vec2& d){
  int s1=orientSign(a,b,c), s2=orientSign(a,b,d), s3=orientSign(c,d,a), s4=orientSign(c,d,b);
  if(s1==0 && s2==0 && s3==0 && s4==0){
    auto mm=[](double u,double v){ if(u>v) std::swap(u,v); return std::make_pair(u,v); };
    auto [ax,bx]=mm(a.x,b.x); auto [cx,dx]=mm(c.x,d.x);
    auto [ay,by]=mm(a.y,b.y); auto [cy,dy]=mm(c.y,d.y);
    bool strict = (bx>cx && dx>ax && by>cy && dy>ay);
    return strict;
  }
  return (s1*s2<0 && s3*s4<0);
}
inline bool pointInPoly(const vector<Vec2>& poly, const Vec2& p){
  bool inside=false; int n=(int)poly.size();
  for(int i=0,j=n-1;i<n;j=i++){
    const Vec2&a=poly[j], &b=poly[i];
    bool cond = ((a.y>p.y)!=(b.y>p.y)) && (p.x < (b.x-a.x)*(p.y-a.y)/(b.y-a.y+1e-30) + a.x);
    if(cond) inside=!inside;
  }
  return inside;
}
inline double triArea2(const Vec2& A,const Vec2& B,const Vec2& C){
  return std::fabs(orient2d_filt(A,B,C));
}
inline double angleAt(const Vec2& A,const Vec2& B,const Vec2& C){
  Vec2 u = A-B, v = C-B;
  double nu = norm(u), nv = norm(v);
  if(nu*nv < 1e-30) return 0.0;
  double c = std::clamp(dot(u,v)/(nu*nv), -1.0, 1.0);
  return std::acos(c);
}
} // namespace geom

//============================== Delaunay / CDT =============================
namespace delaunay {
using geom::Vec2;

struct Tri { int a,b,c; }; // CCW

// Bowyer–Watson (증분형)
static vector<Tri> bowyerWatson(const vector<Vec2>& pts){
  auto& C = cfg::get();
  vector<Vec2> P=pts;
  if(C.add_micro_jitter){
    std::mt19937_64 rng(1234567);
    std::uniform_real_distribution<double> U(-C.jitter_eps, C.jitter_eps);
    for(auto& p:P){ p.x+=U(rng); p.y+=U(rng); }
  }
  // super triangle
  geom::Vec2 lo{+1e300,+1e300}, hi{-1e300,-1e300};
  for(const auto& p:P){ lo.x=std::min(lo.x,p.x); lo.y=std::min(lo.y,p.y); hi.x=std::max(hi.x,p.x); hi.y=std::max(hi.y,p.y); }
  geom::Vec2 c = (lo+hi)*0.5; double d=std::max(hi.x-lo.x, hi.y-lo.y)*1000.0 + 1.0;
  int n0=(int)P.size();
  P.push_back({c.x-2*d, c.y-d});
  P.push_back({c.x+2*d, c.y-d});
  P.push_back({c.x,     c.y+2*d});
  int s1=n0, s2=n0+1, s3=n0+2;

  vector<Tri> T; T.push_back({s1,s2,s3});

  // incremental insert
  for(int ip=0; ip<n0; ++ip){
    const Vec2& p=P[ip];

    // bad triangles
    vector<int> bad; bad.reserve(T.size()/3);
    for(int t=0;t<(int)T.size();++t){
      auto& tr=T[t]; if(!geom::ccw(P[tr.a],P[tr.b],P[tr.c])) std::swap(tr.b,tr.c);
      if(geom::incircle_filt(P[tr.a],P[tr.b],P[tr.c], p)>0) bad.push_back(t);
    }

    // boundary polygon (cancel duplicates)
    struct E{int u,v;};
    vector<E> poly;
    auto addE=[&](int u,int v){
      for(auto it=poly.begin(); it!=poly.end(); ++it){
        if(it->u==v && it->v==u){ poly.erase(it); return; }
      }
      poly.push_back({u,v});
    };

    vector<char> del(T.size(),0);
    for(int id:bad){
      del[id]=1; auto tr=T[id];
      addE(tr.a,tr.b); addE(tr.b,tr.c); addE(tr.c,tr.a);
    }

    vector<Tri> keep; keep.reserve(T.size());
    for(int i=0;i<(int)T.size();++i) if(!del[i]) keep.push_back(T[i]);
    T.swap(keep);

    // retriangulate
    for(const auto& e: poly){
      Tri nt{e.u, e.v, ip};
      if(!geom::ccw(P[nt.a],P[nt.b],P[nt.c])) std::swap(nt.b,nt.c);
      T.push_back(nt);
    }
  }

  // remove super triangles
  vector<Tri> out; out.reserve(T.size());
  for(const auto& tr:T){
    if(tr.a>=n0 || tr.b>=n0 || tr.c>=n0) continue;
    out.push_back(tr);
  }
  return out;
}

// --- Edge map / flip helpers ---
struct EdgeKey{ int u,v; EdgeKey(){} EdgeKey(int a,int b){ u=std::min(a,b); v=std::max(a,b);} bool operator==(const EdgeKey&o)const{ return u==o.u && v==o.v; } };
struct EdgeKeyHash{ size_t operator()(const EdgeKey&k) const { return ( (uint64_t)k.u<<32 ) ^ (uint64_t)k.v; } };
struct EdgeRef { int tri; int a,b; };

inline void buildEdgeMap(const vector<Tri>& T, std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash>& M){
  M.clear(); M.reserve(T.size()*2);
  for(int t=0;t<(int)T.size();++t){
    const Tri& tr=T[t];
    int A[3]={tr.a,tr.b,tr.c};
    for(int i=0;i<3;i++){
      int u=A[i], v=A[(i+1)%3];
      M[EdgeKey(u,v)].push_back({t,u,v});
    }
  }
}
inline bool hasEdge(const std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash>& M, int a,int b){
  auto it = M.find(EdgeKey(a,b));
  return (it!=M.end() && !it->second.empty());
}
inline bool findEdgeTris(const std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash>& M, int a,int b, int& t1,int& t2){
  auto it = M.find(EdgeKey(a,b));
  if(it==M.end()) return false;
  const auto& vec = it->second;
  int found=0; t1=-1; t2=-1;
  for(const auto& er: vec){
    if( (er.a==a && er.b==b) || ( er.a==b && er.b==a ) ){
      if(found==0){ t1=er.tri; found=1; }
      else if(er.tri!=t1){ t2=er.tri; found=2; break; }
    }
  }
  if(found<2){
    for(const auto& er: vec){
      if(er.tri!=t1){
        if(found==0){ t1=er.tri; found=1; }
        else { t2=er.tri; found=2; break; }
      }
    }
  }
  return (found==2);
}
inline bool flipDiagonal(vector<Tri>& T, const vector<Vec2>& P, int t1, int t2, int u,int v){
  int a1=T[t1].a, b1=T[t1].b, c1=T[t1].c; int c=-1;
  if(a1!=u && a1!=v) c=a1; if(b1!=u && b1!=v) c=b1; if(c1!=u && c1!=v) c=c1;
  int a2=T[t2].a, b2=T[t2].b, c2=T[t2].c; int d=-1;
  if(a2!=u && a2!=v) d=a2; if(b2!=u && b2!=v) d=b2; if(c2!=u && c2!=v) d=c2;
  if(c==-1 || d==-1) return false;

  if(geom::orient2d_filt(P[u],P[v],P[c])<=0) return false;
  if(geom::orient2d_filt(P[v],P[u],P[d])<=0) return false;

  Tri Tleft  = {c, d, v};
  if(!geom::ccw(P[Tleft.a],P[Tleft.b],P[Tleft.c])) std::swap(Tleft.b, Tleft.c);
  Tri Tright = {d, c, u};
  if(!geom::ccw(P[Tright.a],P[Tright.b],P[Tright.c])) std::swap(Tright.b,Tright.c);

  T[t1]=Tleft; T[t2]=Tright; return true;
}
inline bool intersectParamT(const Vec2&A,const Vec2&B,const Vec2&C,const Vec2&D,double& t){
  double x1=A.x,y1=A.y,x2=B.x,y2=B.y,x3=C.x,y3=C.y,x4=D.x,y4=D.y;
  double den=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4);
  if(std::fabs(den)<1e-20) return false;
  double tnum=(x1-x3)*(y3-y4)-(y1-y3)*(x3-x4);
  double unum=(x1-x3)*(y1-y2)-(y1-y3)*(x1-x2);
  t=tnum/den; double u=unum/den;
  return (t>0.0 && t<1.0 && u>0.0 && u<1.0);
}

// 제약 엣지 삽입 (플립 반복)
inline bool insertConstraintEdge(vector<Tri>& T, const vector<Vec2>& P,
                                 int a,int b,
                                 const std::unordered_set<EdgeKey,EdgeKeyHash>& forced_set,
                                 int& globalFlipBudget)
{
  if(a==b) return true;
  std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash> M; buildEdgeMap(T,M);
  if(hasEdge(M,a,b)) return true;

  const Vec2& A=P[a]; const Vec2& B=P[b];
  int flips=0; auto& C = cfg::get();

  while(!hasEdge(M,a,b)){
    if(globalFlipBudget<=0 || flips>=C.max_flips_per_segment) return false;

    struct Hit{ int u,v; int t1,t2; double t; };
    vector<Hit> hits; hits.reserve(64);

    for(const auto& kv : M){
      int u=kv.first.u, v=kv.first.v;
      if(u==a||v==a||u==b||v==b) continue;
      if(forced_set.count(kv.first)) continue;

      int t1=-1,t2=-1;
      if(!findEdgeTris(M,u,v,t1,t2)) continue; // 경계 엣지

      if(geom::segIntersectProper(A,B, P[u],P[v])){
        double tp; if(intersectParamT(A,B,P[u],P[v],tp)){
          hits.push_back({u,v,t1,t2,tp});
        }
      }
    }
    if(hits.empty()) return false;
    std::sort(hits.begin(), hits.end(), [](const Hit&x,const Hit&y){ return x.t<y.t; });

    bool did=false;
    for(const auto& h: hits){
      if(globalFlipBudget<=0) return false;
      if(flipDiagonal(T,P,h.t1,h.t2, h.u,h.v)){
        did=true; flips++; globalFlipBudget--; break;
      }
    }
    if(!did) return false;
    buildEdgeMap(T,M);
  }
  return true;
}

// 비제약 엣지 델로네화
inline void legalizeCDT(vector<Tri>& T, const vector<Vec2>& P,
                        const std::unordered_set<EdgeKey,EdgeKeyHash>& forced_set,
                        int max_passes=3)
{
  for(int pass=0; pass<max_passes; ++pass){
    bool changed=false;
    std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash> M; buildEdgeMap(T,M);
    for(const auto& kv : M){
      if(forced_set.count(kv.first)) continue;
      int a=kv.first.u, b=kv.first.v;
      int t1=-1, t2=-1;
      if(!findEdgeTris(M,a,b,t1,t2)) continue;

      int c=-1,d=-1;
      { auto tri=T[t1]; int vv[3]={tri.a,tri.b,tri.c};
        for(int k=0;k<3;k++) if(vv[k]!=a && vv[k]!=b){ c=vv[k]; break; } }
      { auto tri=T[t2]; int vv[3]={tri.a,tri.b,tri.c};
        for(int k=0;k<3;k++) if(vv[k]!=a && vv[k]!=b){ d=vv[k]; break; } }
      if(c==-1||d==-1) continue;

      if(geom::orient2d_filt(P[a],P[b],P[c])<=0) continue;
      if(geom::orient2d_filt(P[b],P[a],P[d])<=0) continue;

      bool bad = (geom::incircle_filt(P[a],P[b],P[c],P[d])>0.0);
      if(!bad) continue;

      EdgeKey newk(c,d);
      if(forced_set.count(newk)) continue;

      if(flipDiagonal(T,P,t1,t2,a,b)) changed=true;
    }
    if(!changed) break;
  }
}
} // namespace delaunay

//=========================== Clip & Quality Filter =========================
namespace clip {
using geom::Vec2;

inline vector<pair<Vec2, Vec2>> ringEdges(const vector<Vec2>& R){
  vector<pair<Vec2, Vec2>> E; int n=(int)R.size(); E.reserve(n);
  for(int i=0;i<n;++i){ int j=(i+1)%n; E.push_back({R[i], R[j]}); }
  return E;
}

inline bool triQualityOK(const Vec2& A,const Vec2& B,const Vec2& C,double medEdge){
  auto& Cfg = cfg::get();
  if(!Cfg.enable_quality_filter) return true;

  // (1) 최소 면적
  double area2 = geom::triArea2(A,B,C);
  if(area2*0.5 < Cfg.min_triangle_area) return false;

  // (2) 최소 내각
  double angA = geom::angleAt(B,A,C);
  double angB = geom::angleAt(A,B,C);
  double angC = geom::angleAt(A,C,B);
  double minAngDeg = std::min({angA,angB,angC}) * 180.0 / M_PI;
  if(minAngDeg < Cfg.min_triangle_angle_deg) return false;

  // (3) 최대 엣지 길이(중앙값 배수)
  double e1 = geom::norm(B-A), e2 = geom::norm(C-B), e3 = geom::norm(A-C);
  double edges[3] = {e1,e2,e3}; std::sort(edges, edges+3);
  double maxEdge = edges[2];
  if(medEdge > 1e-12 && maxEdge > Cfg.max_edge_length_scale * medEdge) return false;

  return true;
}

inline bool triangleKeep(const Vec2&A,const Vec2&B,const Vec2&C,
                         const vector<Vec2>& inner,const vector<Vec2>& outer,
                         const vector<pair<Vec2,Vec2>>& innerE,
                         const vector<pair<Vec2,Vec2>>& outerE,
                         double medEdgeForQuality)
{
  // (a) 중심점이 트랙 영역(outer 내부 & inner 외부)
  Vec2 cent=(A+B+C)*(1.0/3.0);
  if(!geom::pointInPoly(outer, cent)) return false;
  if(geom::pointInPoly(inner, cent))  return false;

  // (b) 변이 경계와 부적절 교차 금지
  auto crosses=[&](const Vec2& u,const Vec2& v)->bool{
    for(const auto&e: innerE){
      if(geom::almostEq(u,e.first)||geom::almostEq(u,e.second)||geom::almostEq(v,e.first)||geom::almostEq(v,e.second)) continue;
      if(geom::segIntersectProper(u,v,e.first,e.second)) return true;
    }
    for(const auto&e: outerE){
      if(geom::almostEq(u,e.first)||geom::almostEq(u,e.second)||geom::almostEq(v,e.first)||geom::almostEq(v,e.second)) continue;
      if(geom::segIntersectProper(u,v,e.first,e.second)) return true;
    }
    return false;
  };
  if(crosses(A,B)||crosses(B,C)||crosses(C,A)) return false;

  // (c) 품질 필터
  if(!triQualityOK(A,B,C, medEdgeForQuality)) return false;

  return true;
}
} // namespace clip

//============================== Centerline =================================
namespace centerline {
using geom::Vec2;
using delaunay::Tri;

// 라벨 경계(내부-외부) 엣지 중점 추출
inline vector<Vec2> midpointsInnerOuter_byLabels(const vector<Vec2>& all,const vector<Tri>& T,
                                                 const vector<int>& labels /*0 inner,1 outer*/)
{
  std::map<std::pair<int,int>, char> S;
  for(const auto& tr:T){
    int v[3]={tr.a,tr.b,tr.c};
    for(int k=0;k<3;k++){
      int i=v[k], j=v[(k+1)%3];
      if(labels[i]<0 || labels[j]<0) continue;
      if(labels[i]!=labels[j]){
        int a=std::min(i,j), b=std::max(i,j);
        S[{a,b}]=1;
      }
    }
  }
  vector<Vec2> mids; mids.reserve(S.size());
  for(const auto& kv:S){
    int i=kv.first.first, j=kv.first.second;
    mids.push_back( (all[i]+all[j]) * 0.5 );
  }
  return mids;
}

// MST(KNN) -> 지름 경로(두 번의 BFS) -> 경로 순서 + 잔여점 연결
inline vector<Vec2> orderByMST(const vector<Vec2>& pts){
  auto& C = cfg::get();
  int n=(int)pts.size();
  if(n<=2) return pts;

  // 1) k-NN 그래프 구성(대칭)
  int K=std::min(C.knn_k, n-1);
  vector<vector<pair<int,double>>> adj(n);
  for(int i=0;i<n;i++){
    vector<pair<double,int>> cand; cand.reserve(n-1);
    for(int j=0;j<n;j++) if(i!=j){
      double d2 = (pts[i].x-pts[j].x)*(pts[i].x-pts[j].x) + (pts[i].y-pts[j].y)*(pts[i].y-pts[j].y);
      cand.push_back({d2, j});
    }
    if((int)cand.size()>K){
      std::nth_element(cand.begin(), cand.begin()+K, cand.end(),
        [](const auto& A,const auto& B){ return A.first<B.first; });
      cand.resize(K);
    }
    for(auto& c:cand){
      double w=std::sqrt(std::max(0.0,c.first));
      adj[i].push_back({c.second,w});
      adj[c.second].push_back({i,w});
    }
  }

  // 2) Prim으로 MST
  vector<double> key(n,1e300); vector<int> par(n,-1); vector<char> in(n,0); key[0]=0;
  for(int it=0; it<n; ++it){
    int u=-1; double best=1e301;
    for(int i=0;i<n;i++) if(!in[i] && key[i]<best){ best=key[i]; u=i; }
    if(u==-1) break; in[u]=1;
    for(auto [v,w]:adj[u]) if(!in[v] && w<key[v]){ key[v]=w; par[v]=u; }
  }

  // 3) MST를 무방향 트리로
  vector<vector<int>> tree(n);
  for(int v=0; v<n; v++) if(par[v]>=0){ tree[v].push_back(par[v]); tree[par[v]].push_back(v); }

  // 4) 두 번의 BFS로 지름 경로 근사
  auto bfs=[&](int s){
    vector<double>d(n,1e300); vector<int>p(n,-1); std::queue<int>q; q.push(s); d[s]=0;
    while(!q.empty()){
      int u=q.front(); q.pop();
      for(int v:tree[u]) if(d[v]>1e299){
        double w = std::sqrt( (pts[u].x-pts[v].x)*(pts[u].x-pts[v].x) + (pts[u].y-pts[v].y)*(pts[u].y-pts[v].y) );
        d[v]=d[u]+w; p[v]=u; q.push(v);
      }
    }
    int far=s; for(int i=0;i<n;i++) if(d[i]>d[far]) far=i;
    return std::tuple<int,vector<int>,vector<double>>(far,p,d);
  };
  auto [s1,p1,d1]=bfs(0);
  auto [s2,p2,d2]=bfs(s1);

  // 5) 지름 경로 추출 + 잔여점 뒤에 이어붙이기
  vector<int> path; for(int v=s2; v!=-1; v=p2[v]) path.push_back(v);
  vector<char> used(n,0); vector<Vec2> out; out.reserve(n);
  for(int id:path){ out.push_back(pts[id]); used[id]=1; }
  for(int i=0;i<n;i++) if(!used[i]) out.push_back(pts[i]);
  return out;
}

// --------- 1D Natural Cubic Spline (TDMA) ----------
struct Spline1D {
  // 구간 i: f(t) = a[i] + b[i]*(t-s[i]) + c[i]*(t-s[i])^2 + d[i]*(t-s[i])^3
  vector<double> s, a, b, c, d;

  static void triSolve(vector<double>& dl, vector<double>& dm, vector<double>& du, vector<double>& rhs){
    int n=(int)dm.size();
    for(int i=1;i<n;++i){
      double w = dl[i-1] / dm[i-1];
      dm[i]   -= w * du[i-1];
      rhs[i]  -= w * rhs[i-1];
    }
    rhs[n-1] /= dm[n-1];
    for(int i=n-2;i>=0;--i){
      rhs[i] = (rhs[i] - du[i]*rhs[i+1]) / dm[i];
    }
  }

  void fit(const vector<double>& _s, const vector<double>& y){
    int n = (int)_s.size();
    s = _s; a = y;
    b.assign(n, 0.0); c.assign(n, 0.0); d.assign(n, 0.0);
    if(n<3){ if(n==2) b[0] = (a[1]-a[0]) / std::max(1e-30, s[1]-s[0]); return; }

    vector<double> h(n-1);
    for(int i=0;i<n-1;++i) h[i] = std::max(1e-30, s[i+1]-s[i]);

    // 내부 1..n-2
    vector<double> dl(n-2), dm(n-2), du(n-2), rhs(n-2);
    for(int i=1;i<=n-2;++i){
      double hi_1 = h[i-1], hi = h[i];
      dl[i-1] = hi_1;
      dm[i-1] = 2.0*(hi_1 + hi);
      du[i-1] = hi;
      rhs[i-1] = 3.0 * ( (a[i+1]-a[i]) / hi - (a[i]-a[i-1]) / hi_1 );
    }
    if(n-2 > 0) triSolve(dl, dm, du, rhs);

    // 자연 경계: c0 = c_{n-1} = 0
    for(int i=1;i<=n-2;++i) c[i]=rhs[i-1];
    c[0]=0.0; c[n-1]=0.0;

    // b, d
    for(int i=0;i<n-1;++i){
      b[i] = (a[i+1]-a[i])/h[i] - (2.0*c[i]+c[i+1]) * h[i]/3.0;
      d[i] = (c[i+1]-c[i]) / (3.0*h[i]);
    }
  }

  double eval(double si) const {
    int n=(int)s.size(); if(n==0) return 0.0; if(n==1) return a[0];

    int lo=0, hi=n-1;
    if(si<=s.front()) lo=0;
    else if(si>=s.back()) lo=n-2;
    else{
      while(hi-lo>1){ int mid=(lo+hi)>>1; if(s[mid]<=si) lo=mid; else hi=mid; }
    }
    double t = si - s[lo];
    return a[lo] + b[lo]*t + c[lo]*t*t + d[lo]*t*t*t;
  }

  void eval_with_deriv(double si, double& f, double& fp, double& fpp) const {
    int n=(int)s.size(); if(n==0){ f=fp=fpp=0; return; } if(n==1){ f=a[0]; fp=fpp=0; return; }

    int lo=0, hi=n-1;
    if(si<=s.front()) lo=0;
    else if(si>=s.back()) lo=n-2;
    else{ while(hi-lo>1){ int mid=(lo+hi)>>1; if(s[mid]<=si) lo=mid; else hi=mid; } }

    double t = si - s[lo];
    f   = a[lo] + b[lo]*t + c[lo]*t*t + d[lo]*t*t*t;
    fp  = b[lo] + 2.0*c[lo]*t + 3.0*d[lo]*t*t;
    fpp = 2.0*c[lo] + 6.0*d[lo]*t;
  }
};

// --------- 자연 3차 스플라인 + 균일 재샘플 (폐곡 안정화 위해 패딩) ---------
inline vector<Vec2> splineUniformClosed_EXPORT_CONTEXT(
    const vector<Vec2>& ordered,
    int samples,
    int paddingK,
    bool close_loop,
    Spline1D& spx_out, Spline1D& spy_out,
    double& s0_out, double& L_out)
{
  int N=(int)ordered.size();
  if(N<3) return ordered;

  // 1) 앞/뒤 패딩
  vector<Vec2> P;
  P.reserve(N + 2*paddingK);
  for(int i=0;i<paddingK;++i) P.push_back(ordered[N - paddingK + i]); // tail -> head
  for(const auto& q:ordered) P.push_back(q);
  for(int i=0;i<paddingK;++i) P.push_back(ordered[i]);                 // head -> tail

  // 2) 누적 호장길이 s
  int M=(int)P.size();
  vector<double> s(M,0.0), xs(M), ys(M);
  for(int i=1;i<M;++i){
    double dx=P[i].x-P[i-1].x, dy=P[i].y-P[i-1].y;
    s[i] = s[i-1] + std::sqrt(dx*dx + dy*dy);
  }
  for(int i=0;i<M;++i){ xs[i]=P[i].x; ys[i]=P[i].y; }

  // 3) 1D 자연 3차 스플라인 두 개(x(s), y(s))
  Spline1D spx, spy; spx.fit(s, xs); spy.fit(s, ys);

  // 4) 원래 데이터 영역만 균일 재샘플 ([K, M-K-1])
  double s0 = s[paddingK], s1 = s[M - paddingK - 1];
  double L  = std::max(1e-30, s1 - s0);

  vector<Vec2> out; out.reserve(samples + (close_loop?1:0));
  for(int k=0;k<samples;++k){
    double si = s0 + L * (double(k) / double(samples));
    out.push_back({ spx.eval(si), spy.eval(si) });
  }
  if(close_loop) out.push_back(out.front());

  spx_out = std::move(spx);
  spy_out = std::move(spy);
  s0_out  = s0;
  L_out   = L;
  return out;
}
} // namespace centerline

// ---- forward declarations (Ray–Segment utils) ----
static double rayToRingDistance(const geom::Vec2& P,
                                const geom::Vec2& dir,
                                const std::vector<std::pair<geom::Vec2, geom::Vec2>>& ringEdges);

// ========================= MIN-CURV EXTENSION (raceline) =========================
// 논문식: 경로 = Center + n * α,   목적: ∑ κ^2 + λ||D1 α||^2,   제약: lo ≤ α ≤ hi
// 재선형화(GN)에서 A1, A2, N0, W를 "고정"한 뒤 선형 근사에서 projected step 반복 → 경로 갱신
namespace raceline_min_curv {
using geom::Vec2;

// 주기 인덱싱
inline int wrap(int i, int n){ i%=n; if(i<0) i+=n; return i; }

// 중앙차분 1차/2차 (주기)
struct DiffOps {
  int N; double h, inv2h, invh2;
  DiffOps(int N_, double h_) : N(N_), h(h_) { inv2h = 1.0/(2.0*h); invh2 = 1.0/(h*h); }

  // (D1 * a)_i = (a_{i+1} - a_{i-1})/(2h)
  void D1(const std::vector<double>& a, std::vector<double>& out) const {
    out.resize(N);
    for(int i=0;i<N;++i){
      int ip=wrap(i+1,N), im=wrap(i-1,N);
      out[i] = (a[ip] - a[im]) * inv2h;
    }
  }
  // (D2 * a)_i = (a_{i+1} - 2a_i + a_{i-1})/(h^2)
  void D2(const std::vector<double>& a, std::vector<double>& out) const {
    out.resize(N);
    for(int i=0;i<N;++i){
      int ip=wrap(i+1,N), im=wrap(i-1,N);
      out[i] = (a[ip] - 2.0*a[i] + a[im]) * invh2;
    }
  }
  // D1^T v = (v_{i-1} - v_{i+1})/(2h)
  void D1T(const std::vector<double>& v, std::vector<double>& out) const {
    out.resize(N);
    for(int i=0;i<N;++i){
      int im=wrap(i-1,N), ip=wrap(i+1,N);
      out[i] = (v[im] - v[ip]) * inv2h;
    }
  }
  // D2^T = D2 (대칭)
  void D2T(const std::vector<double>& v, std::vector<double>& out) const {
    D2(v,out);
  }
};

// 헤딩/곡률 (중앙차분)
static void heading_curv_from_points(const std::vector<Vec2>& P, double h,
                                     std::vector<double>& heading, std::vector<double>& kappa)
{
  int N=(int)P.size();
  heading.resize(N); kappa.resize(N);
  for(int i=0;i<N;++i){
    int ip=wrap(i+1,N), im=wrap(i-1,N);
    double xp = (P[ip].x - P[im].x) / (2.0*h);
    double yp = (P[ip].y - P[im].y) / (2.0*h);
    double xpp = (P[ip].x - 2.0*P[i].x + P[im].x) / (h*h);
    double ypp = (P[ip].y - 2.0*P[i].y + P[im].y) / (h*h);
    heading[i] = std::atan2(yp, xp);
    double denom = std::pow(std::max(1e-12, xp*xp + yp*yp), 1.5);
    kappa[i] = (xp*ypp - yp*xpp) / denom;
  }
}

// 현재 경로에서 좌측 법선( -y', x' ) 계산
static std::vector<Vec2> normals_from_points(const std::vector<Vec2>& P){
  int N=(int)P.size();
  std::vector<Vec2> n(N);
  for(int i=0;i<N;++i){
    int ip=wrap(i+1,N), im=wrap(i-1,N);
    double dx = (P[ip].x - P[im].x) * 0.5; // ds로 나누지 않아도 방향은 동일
    double dy = (P[ip].y - P[im].y) * 0.5;
    Vec2 t{dx,dy}; if(geom::norm(t)<1e-15) t={1,0};
    Vec2 nv{-t.y, t.x};
    n[i] = geom::normalize(nv, 1e-15);
  }
  return n;
}

// 코리도 폭 → α 박스제약 계산 (좌/우 모두 레이캐스트 후 차량폭/마진 가드 적용)
static void compute_alpha_bounds(const std::vector<Vec2>& P,
                                 const std::vector<Vec2>& n,
                                 const std::vector<std::pair<Vec2,Vec2>>& innerE,
                                 const std::vector<std::pair<Vec2,Vec2>>& outerE,
                                 double veh_width, double safety_margin,
                                 std::vector<double>& lo, std::vector<double>& hi)
{
  int N=(int)P.size(); lo.assign(N,0.0); hi.assign(N,0.0);
  for(int i=0;i<N;++i){
    Vec2 nv = n[i], P0=P[i], nneg{-nv.x,-nv.y};
    double dpos_in  = ::rayToRingDistance(P0, nv,   innerE);
    double dpos_out = ::rayToRingDistance(P0, nv,   outerE);
    double dneg_in  = ::rayToRingDistance(P0, nneg, innerE);
    double dneg_out = ::rayToRingDistance(P0, nneg, outerE);
    double dpos = std::min(dpos_in, dpos_out);   // +n 쪽 여유
    double dneg = std::min(dneg_in, dneg_out);   // -n 쪽 여유
    double guard = veh_width*0.5 + safety_margin;
    hi[i] = std::max(0.0, dpos - guard);  // 좌측(+) 최대 오프셋
    lo[i] = -std::max(0.0, dneg - guard); // 우측(-) 최소 오프셋
    if(!std::isfinite(hi[i])) hi[i]=0.0;
    if(!std::isfinite(lo[i])) lo[i]=0.0;
  }
}

// ── GN 선형화 계수 구조체 ───────────────────────────────────────────────
struct LinGeom {
  std::vector<double> A1, A2, N0, W;
};

// (outer 루프 한 번당) Pbase, n, h에서 A1/A2/N0/W를 "고정" 계산
static LinGeom precompute_lin_geom(const std::vector<Vec2>& Pbase,
                                   const std::vector<Vec2>& n,
                                   double h)
{
  int N=(int)Pbase.size();
  std::vector<double> xp(N), yp(N), xpp(N), ypp(N);
  for(int i=0;i<N;++i){
    int ip=wrap(i+1,N), im=wrap(i-1,N);
    xp[i]  = (Pbase[ip].x - Pbase[im].x) / (2.0*h);
    yp[i]  = (Pbase[ip].y - Pbase[im].y) / (2.0*h);
    xpp[i] = (Pbase[ip].x - 2.0*Pbase[i].x + Pbase[im].x) / (h*h);
    ypp[i] = (Pbase[ip].y - 2.0*Pbase[i].y + Pbase[im].y) / (h*h);
  }

  LinGeom G;
  G.A1.resize(N); G.A2.resize(N); G.N0.resize(N); G.W.resize(N);
  for(int i=0;i<N;++i){
    G.A1[i] = n[i].x * ypp[i] - n[i].y * xpp[i];
    G.A2[i] = xp[i] * n[i].y - yp[i] * n[i].x;
    G.N0[i] = xp[i]*ypp[i] - yp[i]*xpp[i];
    double denom = std::pow(std::max(1e-12, xp[i]*xp[i] + yp[i]*yp[i]), 1.5);
    G.W[i] = 1.0 / denom;
  }
  return G;
}

// --- "고정된" 선형화 계수를 쓰는 비용/그래디언트 ---
// κ ≈ W * (N0 + A1 * (D1 α) + A2 * (D2 α))
// J = ||κ||^2 + λ ||D1 α||^2
struct CostGrad {
  double J;
  std::vector<double> grad;
};

static CostGrad eval_cost_grad_frozen(const std::vector<double>& A1,
                                      const std::vector<double>& A2,
                                      const std::vector<double>& N0,
                                      const std::vector<double>& W,
                                      double h, double lambda_smooth,
                                      const std::vector<double>& alpha)
{
  int N=(int)alpha.size();
  DiffOps D(N,h);

  // D1 α, D2 α
  std::vector<double> a1, a2;
  D.D1(alpha, a1);
  D.D2(alpha, a2);

  // κ 근사
  std::vector<double> z(N);
  for(int i=0;i<N;++i) z[i] = W[i]*( N0[i] + A1[i]*a1[i] + A2[i]*a2[i] );

  // J = ||z||^2 + λ||D1 α||^2   (필요시 적분 스케일링으로 *h 고려 가능)
  double J=0.0; for(double v: z) J += v*v;
  double Jsm=0.0; for(double v: a1) Jsm += v*v;
  J += lambda_smooth * Jsm;

  // grad = 2*( D1^T( A1 ∘ (W ∘ z) ) + D2^T( A2 ∘ (W ∘ z) ) ) + 2λ D1^T(D1 α)
  std::vector<double> Wz(N), q1(N), q2(N), g1, g2, gsm, D1a;
  for(int i=0;i<N;++i){ Wz[i]=W[i]*z[i]; q1[i]=A1[i]*Wz[i]; q2[i]=A2[i]*Wz[i]; }
  D.D1T(q1, g1);
  D.D2T(q2, g2);
  D.D1(alpha, D1a);
  D.D1T(D1a, gsm);

  std::vector<double> grad(N);
  for(int i=0;i<N;++i) grad[i] = 2.0*(g1[i] + g2[i]) + 2.0*lambda_smooth*gsm[i];

  return {J, std::move(grad)};
}

struct Result {
  std::vector<Vec2>   raceline;     // 최종 좌표 (Center + n * 누적오프셋)
  std::vector<double> heading;
  std::vector<double> curvature;
  std::vector<double> alpha_total;  // 모든 GN 스테이지 누적 α
  std::vector<double> alpha_last;   // 마지막 GN 스테이지의 α (디버그/보고용)
};

static Result compute_min_curvature_raceline(const std::vector<Vec2>& center,
                                             const std::vector<std::pair<Vec2,Vec2>>& innerE,
                                             const std::vector<std::pair<Vec2,Vec2>>& outerE,
                                             double veh_width,
                                             double L) // 총 길이
{
  auto& C = cfg::get();
  int N=(int)center.size();
  double h = L / double(N);

  // Step 0) 시작 경로/법선/박스제약 (논문: Frenet폭 기반 코리도)
  std::vector<Vec2>  Pbase = center;
  std::vector<Vec2>  n     = normals_from_points(Pbase);
  std::vector<double> lo, hi;
  compute_alpha_bounds(Pbase, n, innerE, outerE, veh_width, C.safety_margin_m, lo, hi);

  // 디버깅: 코리도 폭 통계
  {
    double hi_avg=0, lo_avg=0, hi_max=0, lo_max=0;
    for (int i=0;i<N;++i){
      hi_avg += hi[i]; lo_avg += -lo[i];
      hi_max = std::max(hi_max, hi[i]);
      lo_max = std::max(lo_max, -lo[i]);
    }
    hi_avg/=N; lo_avg/=N;
    cerr << "[corridor] mean+ = " << hi_avg << "  mean- = " << lo_avg
         << "  max+ = " << hi_max << "  max- = " << lo_max << "\n";
  }

  // 누적 오프셋 / 내부 반복용 α
  std::vector<double> alpha(N, 0.0);
  std::vector<double> alpha_accum(N, 0.0);
  std::vector<double> alpha_last_stage(N, 0.0);

  // Step 1) GN 재선형화 루프
  for(int outer=0; outer<C.max_outer_iters; ++outer){
    // 선형화 계수 "고정" 계산
    auto G = precompute_lin_geom(Pbase, n, h);

    double step = C.step_init;
    auto cg = eval_cost_grad_frozen(G.A1, G.A2, G.N0, G.W, h, C.lambda_smooth, alpha);
    double J_prev = cg.J;

    if(C.verbose){
      cerr << "[GN " << outer << "]  J0=" << J_prev
           << "  step=" << step << "  lambda=" << C.lambda_smooth << "\n";
    }

    // Step 2) 선형화 고정 상태에서 projected step (Armijo)
    for(int it=0; it<C.max_inner_iters; ++it){
      bool accepted=false;
      int bt=0;
      while(bt<20){
        std::vector<double> a_new(N);
        for(int i=0;i<N;++i){
          double ai = alpha[i] - step * cg.grad[i];
          // 박스 투영
          a_new[i] = std::min(hi[i], std::max(lo[i], ai));
        }

        auto cg_new = eval_cost_grad_frozen(G.A1, G.A2, G.N0, G.W, h, C.lambda_smooth, a_new);

        // Armijo:  f(α_new) ≤ f(α) + c * g^T (α_new - α)
        double dec = 0.0;
        for(int i=0;i<N;++i) dec += cg.grad[i] * (a_new[i] - alpha[i]);

        if(cg_new.J <= cg.J + C.armijo_c * dec){
          alpha.swap(a_new);
          cg  = std::move(cg_new);
          accepted=true;
          break;
        }
        step *= 0.5; bt++;
        if(step < C.step_min) break;
      }
      if(!accepted) break;
      if(std::fabs(J_prev - cg.J) < 1e-10) break;
      J_prev = cg.J;
    }

    // 마지막 스테이지 α 기록(리셋 전에)
    alpha_last_stage = alpha;

    // Step 3) 경로 업데이트 & 누적량 기록, 법선/제약 재계산 (relinearize)
    for(int i=0;i<N;++i){
      Pbase[i].x += n[i].x*alpha[i];
      Pbase[i].y += n[i].y*alpha[i];
      alpha_accum[i] += alpha[i];
    }
    n = normals_from_points(Pbase);
    compute_alpha_bounds(Pbase, n, innerE, outerE, veh_width, C.safety_margin_m, lo, hi);

    // 다음 선형화에서 기준 α는 0부터 다시 시작 (관행)
    std::fill(alpha.begin(), alpha.end(), 0.0);
  }

  // 최종 geom
  std::vector<double> heading, kappa;
  heading_curv_from_points(Pbase, h, heading, kappa);

  if(cfg::get().verbose){
    double ksum=0; for(double v:kappa) ksum+=v*v;
    cerr<<"[done] ∑κ^2 = "<<ksum<<"  (with λ="<<cfg::get().lambda_smooth<<")\n";
  }

  return {
    std::move(Pbase),
    std::move(heading),
    std::move(kappa),
    std::move(alpha_accum),     // alpha_total
    std::move(alpha_last_stage) // alpha_last
  };
}

} // namespace raceline_min_curv
// ======================= END: MIN-CURV EXTENSION (raceline) =======================

//=============================== IO Utils ==================================
namespace io {
using geom::Vec2;

inline vector<Vec2> loadCSV_XY(const string& path){
  vector<Vec2> pts; std::ifstream fin(path);
  if(!fin){ cerr<<"[ERR] cannot open: "<<path<<"\n"; return pts; }
  string line;
  while(std::getline(fin,line)){
    if(line.empty()) continue;
    for(char&ch:line) if(ch==';'||ch=='\t') ch=' ';
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream iss(line); double x,y;
    if(iss>>x>>y) pts.push_back({x,y});
  }
  return pts;
}
inline bool saveCSV_pointsXY(const string& path,const vector<Vec2>& pts){
  std::ofstream fo(path); if(!fo){ cerr<<"[ERR] write failed: "<<path<<"\n"; return false; }
  fo.setf(std::ios::fixed); fo.precision(9);
  for(auto&p:pts) fo<<p.x<<","<<p.y<<"\n";
  return true;
}
inline bool saveCSV_pointsLabeled(const string& path,const vector<Vec2>& pts,const vector<int>& label){
  std::ofstream fo(path); if(!fo){ cerr<<"[ERR] write failed: "<<path<<"\n"; return false; }
  fo<<"id,x,y,label\n"; fo.setf(std::ios::fixed); fo.precision(9);
  for(size_t i=0;i<pts.size();++i) fo<<i<<","<<pts[i].x<<","<<pts[i].y<<","<<label[i]<<"\n";
  return true;
}
inline bool saveCSV_edgesIdx(const string& path,const vector<pair<int,int>>& E){
  std::ofstream fo(path); if(!fo){ cerr<<"[ERR] write failed: "<<path<<"\n"; return false; }
  for(auto&e:E) fo<<e.first<<","<<e.second<<"\n";
  return true;
}
inline bool saveCSV_trisIdx(const string& path,const vector<delaunay::Tri>& T){
  std::ofstream fo(path); if(!fo){ cerr<<"[ERR] write failed: "<<path<<"\n"; return false; }
  for(auto&t:T) fo<<t.a<<","<<t.b<<","<<t.c<<"\n";
  return true;
}
inline string dropExt(const string& s){
  size_t p=s.find_last_of('.'); if(p==string::npos) return s; return s.substr(0,p);
}
} // namespace io

//============================= CDT with Recovery ============================
struct Constraint { int a,b; int splits=0; };

struct CDTResult{
  vector<geom::Vec2> all;
  vector<int>  label; // 0 inner,1 outer
  vector<delaunay::Tri>  tris;
  vector<pair<int,int>> forced_edges;
  bool all_forced_ok=false;
};

static CDTResult buildCDT_withRecovery(vector<geom::Vec2> inner, vector<geom::Vec2> outer){
  auto& C = cfg::get();

  CDTResult R;
  // 포인트 병합 + 라벨
  R.all = inner;
  R.all.insert(R.all.end(), outer.begin(), outer.end());
  R.label.assign(R.all.size(), 0);
  for(size_t i=0;i<R.all.size();++i) R.label[i] = (i<inner.size()?0:1);

  auto rebuildDT = [&](vector<delaunay::Tri>& T){ T = delaunay::bowyerWatson(R.all); };

  vector<delaunay::Tri> T; rebuildDT(T);

  // 초기 제약: 각 링의 모든 변
  vector<Constraint> cons;
  int nIn=(int)inner.size(), nOut=(int)outer.size();
  auto pushRing=[&](int base,int n){
    for(int i=0;i<n;i++){ int j=(i+1)%n; cons.push_back({base+i, base+j, 0}); }
  };
  pushRing(0, nIn);
  pushRing(nIn, nOut);

  auto rebuildForcedSet=[&](std::unordered_set<delaunay::EdgeKey,delaunay::EdgeKeyHash>& F){
    F.clear(); F.reserve(cons.size()*2);
    for(auto& c: cons) F.insert( delaunay::EdgeKey(c.a,c.b) );
  };

  int  globalFlipBudget = C.max_global_flips;
  bool ok=false;

  for(int rebuilds=0; rebuilds<=C.max_cdt_rebuilds; ++rebuilds){
    std::unordered_set<delaunay::EdgeKey,delaunay::EdgeKeyHash> forced; rebuildForcedSet(forced);

    ok=true;
    for(size_t k=0;k<cons.size();++k){
      auto& seg=cons[k];
      if(globalFlipBudget<=0){ ok=false; break; }

      if(delaunay::insertConstraintEdge(T, R.all, seg.a, seg.b, forced, globalFlipBudget))
        continue;

      // 실패 → 슈타이너 분할
      if(seg.splits >= C.max_segment_splits){ ok=false; break; }
      geom::Vec2 A=R.all[seg.a], B=R.all[seg.b];
      geom::Vec2 M = (A+B)*0.5;
      int newIdx = (int)R.all.size();
      R.all.push_back(M);
      R.label.push_back(R.label[seg.a]);

      Constraint left { seg.a, newIdx,  seg.splits+1 };
      Constraint right{ newIdx, seg.b,  seg.splits+1 };
      cons.erase(cons.begin()+k);
      cons.insert(cons.begin()+k, right);
      cons.insert(cons.begin()+k, left);

      rebuildDT(T);
      rebuildForcedSet(forced);

      ok=false; // 다시 루프
      break;
    }
    if(ok){
      delaunay::legalizeCDT(T, R.all, /*forced*/ std::unordered_set<delaunay::EdgeKey,delaunay::EdgeKeyHash>(), 2);
      break;
    }
    if(C.verbose) cerr<<"[CDT] rebuild "<<(rebuilds+1)<<" due to split; total pts="<<R.all.size()<<"\n";
    if(rebuilds==C.max_cdt_rebuilds) break;
  }

  R.tris = std::move(T);
  R.all_forced_ok = ok;

  R.forced_edges.clear();
  R.forced_edges.reserve(cons.size());
  for(const auto& c: cons) R.forced_edges.push_back({c.a,c.b});

  return R;
}

//============================ Ray–Segment Utils ============================
// 레이 A + t*d 와 세그먼트 S0 + u*(S1-S0)의 교차 (t>=0, u∈[0,1])
static bool rayIntersectSegment(const geom::Vec2& A, const geom::Vec2& d,
                                const geom::Vec2& S0, const geom::Vec2& S1,
                                double& t_out, double eps=1e-15)
{
  double vx = S1.x - S0.x, vy = S1.y - S0.y;
  double den = d.x * (-vy) + d.y * (vx); // det([d, S0->S1])
  if (std::fabs(den) < eps) return false; // 평행

  // A + t d = S0 + u (S1-S0)
  double ax = S0.x - A.x, ay = S0.y - A.y;
  double inv = 1.0 / den;
  double t = ( ax * (-vy) + ay * (vx) ) * inv;
  double u = ( d.x * ay - d.y * ax ) * inv;

  if (t >= 0.0 && u >= -1e-12 && u <= 1.0 + 1e-12) { t_out = t; return true; }
  return false;
}

// P에서 법선 dir로 링 세그먼트와의 최단 양의 교차거리
static double rayToRingDistance(const geom::Vec2& P, const geom::Vec2& dir,
                                const vector<pair<geom::Vec2, geom::Vec2>>& ringEdges)
{
  double best = std::numeric_limits<double>::infinity();
  for (const auto& e : ringEdges) {
    double t;
    if (rayIntersectSegment(P, dir, e.first, e.second, t)) {
      if (t > 0.0 && t < best) best = t;
    }
  }
  return best;
}

// 한 점 P와 법선 n에 대해 inner/outer까지의 거리(d_inner, d_outer)
static void distancesToRings(const geom::Vec2& P, const geom::Vec2& n,
                             const vector<pair<geom::Vec2, geom::Vec2>>& innerE,
                             const vector<pair<geom::Vec2, geom::Vec2>>& outerE,
                             double& d_inner, double& d_outer)
{
  geom::Vec2 npos = n;
  geom::Vec2 nneg = geom::Vec2{-n.x, -n.y};

  double di1 = rayToRingDistance(P, npos, innerE);
  double di2 = rayToRingDistance(P, nneg, innerE);
  d_inner = std::min(di1, di2);

  double do1 = rayToRingDistance(P, npos, outerE);
  double do2 = rayToRingDistance(P, nneg, outerE);
  d_outer = std::min(do1, do2);
}

//==================================== MAIN =================================
int main(int argc, char** argv){
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  if(argc < 4){
    cerr<<"Usage: "<<argv[0]<<" inner.csv outer.csv centerline.csv\n";
    return 1;
  }
  std::string innerPath = argv[1], outerPath = argv[2], outPath = argv[3];
  const std::string base = io::dropExt(outPath); // 단계별 CSV 자동 prefix

  auto& C = cfg::get();

  // [1] 입력 로드
  std::vector<geom::Vec2> inner = io::loadCSV_XY(innerPath);
  std::vector<geom::Vec2> outer = io::loadCSV_XY(outerPath);
  if(inner.size() < 3 || outer.size() < 3){
    cerr<<"[ERR] need >= 3 points per ring\n"; return 2;
  }

  // [2] CDT(+슈타이너 복구)
  auto cdt = buildCDT_withRecovery(inner, outer);
  if(C.verbose){
    cerr<<"[CDT] forced insert "<<(cdt.all_forced_ok ? "OK" : "RECOVERED with Steiner")
        <<", total points="<<cdt.all.size()<<", faces="<<cdt.tris.size()<<"\n";
  }

  // 디버그 덤프 1
  io::saveCSV_pointsLabeled(base + "_all_points.csv",     cdt.all, cdt.label);
  io::saveCSV_edgesIdx     (base + "_forced_edges_idx.csv", cdt.forced_edges);
  io::saveCSV_trisIdx      (base + "_tri_raw_idx.csv",      cdt.tris);

  // [3] 품질 필터 기준(엣지 길이 중앙값)
  std::vector<double> edgeLens; edgeLens.reserve(cdt.tris.size()*3);
  for(const auto& t: cdt.tris){
    const auto& A=cdt.all[t.a]; const auto& B=cdt.all[t.b]; const auto& Cc=cdt.all[t.c];
    edgeLens.push_back(geom::norm(B-A));
    edgeLens.push_back(geom::norm(Cc-B));
    edgeLens.push_back(geom::norm(A-Cc));
  }
  double medEdge=0.0;
  if(!edgeLens.empty()){
    size_t m = edgeLens.size()/2;
    std::nth_element(edgeLens.begin(), edgeLens.begin()+m, edgeLens.end());
    medEdge = edgeLens[m];
  }

  // [4] 트랙 영역 클리핑 (+ 품질 필터)
  std::vector<delaunay::Tri> faces_kept, faces_drop;
  {
    auto innerE = clip::ringEdges(inner);
    auto outerE = clip::ringEdges(outer);
    for(const auto& t: cdt.tris){
      const auto&A=cdt.all[t.a]; const auto&B=cdt.all[t.b]; const auto&C3=cdt.all[t.c];
      if(clip::triangleKeep(A,B,C3, inner, outer, innerE, outerE, medEdge))
        faces_kept.push_back(t);
      else
        faces_drop.push_back(t);
    }
  }
  if(faces_kept.empty()){
    if(!C.allow_fallback_clip){ cerr<<"[ERR] no faces after clipping\n"; return 3; }
    faces_kept = cdt.tris;
  }

  // 디버그 덤프 2
  io::saveCSV_trisIdx(base + "_faces_kept_idx.csv", faces_kept);
  io::saveCSV_trisIdx(base + "_faces_drop_idx.csv", faces_drop);

  // [5] 경계 엣지 중점 -> MST 지름 경로 정렬
  auto mids = centerline::midpointsInnerOuter_byLabels(cdt.all, faces_kept, cdt.label);
  if(mids.size() < 4){
    cerr<<"[ERR] not enough midpoints ("<<mids.size()<<")\n";
    io::saveCSV_pointsXY(base + "_mids_raw.csv", mids);
    return 4;
  }
  auto ordered = centerline::orderByMST(mids);
  io::saveCSV_pointsXY(base + "_mids_raw.csv",     mids);
  io::saveCSV_pointsXY(base + "_mids_ordered.csv", ordered);

  // [6] 자연 3차 스플라인 + 균일 재샘플 + 헤딩/곡률
  centerline::Spline1D spx, spy;
  double s0=0.0, L=0.0;
  auto center = centerline::splineUniformClosed_EXPORT_CONTEXT(
      ordered, C.samples, /*paddingK=*/3, /*close_loop=*/C.emit_closed_duplicate,
      spx, spy, s0, L
  );

  // [7] 센터라인 CSV (좌표만)
  {
    std::ofstream fo(outPath);
    if(!fo){ cerr<<"[ERR] save centerline "<<outPath<<"\n"; return 5; }
    fo.setf(std::ios::fixed); fo.precision(9);
    for(const auto& p: center) fo<<p.x<<","<<p.y<<"\n";
  }

  // [8] 기하량 + 폭(width) CSV (w_L,w_R을 간접적으로 제공: d_inner,d_outer, width=d_in+d_out)
  auto innerE = clip::ringEdges(inner);
  auto outerE = clip::ringEdges(outer);

  {
    std::ofstream fo2(base + "_with_geom.csv");
    if(!fo2){ cerr<<"[ERR] save centerline_with_geom\n"; return 6; }
    fo2.setf(std::ios::fixed); fo2.precision(9);
    fo2 << "s,x,y,heading_rad,curvature,dist_to_inner,dist_to_outer,width\n";

    for(int k=0;k<C.samples;++k){
      double si = s0 + L * (double(k) / double(C.samples));  // s-축 균일
      double x, xp, xpp, y, yp, ypp;
      spx.eval_with_deriv(si, x, xp, xpp);
      spy.eval_with_deriv(si, y, yp, ypp);

      // heading & curvature
      double heading   = std::atan2(yp, xp);
      double speed2    = xp*xp + yp*yp;
      double denom     = std::pow(std::max(1e-12, speed2), 1.5);
      double curvature = (xp*ypp - yp*xpp) / denom;

      // 법선 (좌측: -y', x')
      geom::Vec2 nvec = geom::normalize( geom::Vec2{ -yp, xp }, 1e-12 );

      // 경계까지 거리(양 방향 레이 중 짧은 것)
      double d_in=std::numeric_limits<double>::infinity();
      double d_out=std::numeric_limits<double>::infinity();
      if (nvec.x!=0 || nvec.y!=0) {
        geom::Vec2 P { x, y };
        distancesToRings(P, nvec, innerE, outerE, d_in, d_out);
      } else {
        d_in = d_out = std::numeric_limits<double>::quiet_NaN();
      }

      double width = d_in + d_out;
      double si_rel = si - s0;  // 0..L

      fo2 << si_rel << "," << x << "," << y << "," << heading << "," << curvature
          << "," << d_in << "," << d_out << "," << width << "\n";
    }
  }

  if(C.verbose){
    cerr<<"[OK] centerline saved: "<<outPath<<" (N="<<center.size()<<")\n";
    cerr<<"[OK] geom+width saved: "<<base<<"_with_geom.csv\n";
    cerr<<"[OK] dumps: "<<base<<"_* .csv\n";
  }

  // [9] 최소 곡률 raceline (논문식 흐름으로 재구성)
  {
    auto res = raceline_min_curv::compute_min_curvature_raceline(
        center, innerE, outerE, C.veh_width_m, /*L=*/L
    );

    // raceline 좌표
    {
      std::ofstream fo(base + "_raceline.csv");
      if(!fo){ cerr<<"[ERR] save raceline\n"; return 7; }
      fo.setf(std::ios::fixed); fo.precision(9);
      for(const auto& p: res.raceline) fo<<p.x<<","<<p.y<<"\n";
    }
    // raceline with geom
    {
      std::ofstream fo(base + "_raceline_with_geom.csv");
      if(!fo){ cerr<<"[ERR] save raceline_with_geom\n"; return 8; }
      fo.setf(std::ios::fixed); fo.precision(9);
      fo<<"s,x,y,heading_rad,curvature,alpha_last\n";
      for(int k=0;k<C.samples;++k){
        double si = s0 + L * (double(k) / double(C.samples));
        double si_rel = si - s0;
        fo<<si_rel<<","<<res.raceline[k].x<<","<<res.raceline[k].y<<","
          <<res.heading[k]<<","<<res.curvature[k]<<","<<res.alpha_last[k]<<"\n";
      }
    }
    if(C.verbose){
      cerr<<"[OK] raceline saved: "<<base<<"_raceline.csv\n";
      cerr<<"[OK] raceline+geom saved: "<<base<<"_raceline_with_geom.csv\n";
    }
  }
  return 0;
}
