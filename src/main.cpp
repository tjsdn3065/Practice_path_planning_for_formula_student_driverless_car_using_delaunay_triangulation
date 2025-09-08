#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <algorithm>
#include <queue>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <numeric>
#include <functional>
#include <cassert>

using std::vector;
using std::array;
using std::pair;
using std::make_pair;
using std::string;
using std::cout;
using std::cerr;
using std::endl;

//================== Config ==================
struct Config {
    // preprocess
    double dedup_eps = 1e-12;
    bool   add_micro_jitter = true;
    double jitter_eps = 1e-9;

    // centerline
    int samples = 1000;
    int knn_k   = 8;

    // constrained insertion
    int  max_flips_per_segment = 20000;   // flip guard
    int  max_global_flips      = 500000;  // flip guard
    int  max_segment_splits    = 8;       // Steiner splits per original segment
    int  max_cdt_rebuilds      = 12;      // global rebuild guard
    bool verbose = true;

    // fallback
    bool allow_fallback_clip = true;
} CFG;

//================== Geometry =================
struct Vec2 { double x=0, y=0; };
static inline Vec2 operator+(const Vec2&a,const Vec2&b){ return {a.x+b.x,a.y+b.y}; }
static inline Vec2 operator-(const Vec2&a,const Vec2&b){ return {a.x-b.x,a.y-b.y}; }
static inline Vec2 operator*(const Vec2&a,double s){ return {a.x*s, a.y*s}; }
static inline double dot(const Vec2&a,const Vec2&b){ return a.x*b.x + a.y*b.y; }
static inline double norm2(const Vec2&a){ return dot(a,a); }
static inline double norm (const Vec2&a){ return std::sqrt(norm2(a)); }
static inline bool almostEq(const Vec2&a,const Vec2&b,double e=1e-12){ return (std::fabs(a.x-b.x)<=e && std::fabs(a.y-b.y)<=e); }

static inline double orient2d_filt(const Vec2& a, const Vec2& b, const Vec2& c){
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
static inline int orientSign(const Vec2& a,const Vec2& b,const Vec2& c){
    double v=orient2d_filt(a,b,c); return (v>0)-(v<0);
}
static inline double incircle_filt(const Vec2& a,const Vec2& b,const Vec2& c,const Vec2& d){
    double adx=a.x-d.x, ady=a.y-d.y;
    double bdx=b.x-d.x, bdy=b.y-d.y;
    double cdx=c.x-d.x, cdy=c.y-d.y;
    double ad=adx*adx+ady*ady;
    double bd=bdx*bdx+bdy*bdy;
    double cd=cdx*cdx+cdy*cdy;
    double det=adx*(bdy*cd-bd*cdy) - ady*(bdx*cd-bd*cdx) + ad*(bdx*cdy-bdy*cdx);
    double mags=(std::fabs(adx)+std::fabs(ady))*(std::fabs(bdx)+std::fabs(bdy))*(std::fabs(cdx)+std::fabs(cdy));
    double err = mags*std::numeric_limits<double>::epsilon()*16.0;
    if (std::fabs(det)>err) return det;
    long double AX=a.x, AY=a.y, BX=b.x, BY=b.y, CX=c.x, CY=c.y, DX=d.x, DY=d.y;
    long double adxl=AX-DX, adyl=AY-DY, bdxl=BX-DX, bdyl=BY-DY, cdxl=CX-DX, cdyl=CY-DY;
    long double adl=adxl*adxl+adyl*adyl, bdl=bdxl*bdxl+bdyl*bdyl, cdl=cdxl*cdxl+cdyl*cdyl;
    long double detl=adxl*(bdyl*cdl-bdl*cdyl) - adyl*(bdxl*cdl-bdl*cdxl) + adl*(bdxl*cdyl-bdyl*cdxl);
    return (double)detl;
}

static bool segIntersectProper(const Vec2& a,const Vec2& b,const Vec2& c,const Vec2& d){
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

static bool pointInPoly(const vector<Vec2>& poly, const Vec2& p){
    bool inside=false; int n=(int)poly.size();
    for(int i=0,j=n-1;i<n;j=i++){
        const Vec2&a=poly[j], &b=poly[i];
        bool cond = ((a.y>p.y)!=(b.y>p.y)) && (p.x < (b.x-a.x)*(p.y-a.y)/(b.y-a.y+1e-30) + a.x);
        if(cond) inside=!inside;
    }
    return inside;
}

//================== Bowyer–Watson Incremental DT =================
struct Tri { int a,b,c; }; // CCW
static bool ccw(const Vec2&A,const Vec2&B,const Vec2&C){ return orient2d_filt(A,B,C)>0; }
static bool inCircum(const Vec2&A,const Vec2&B,const Vec2&C,const Vec2&P){ return incircle_filt(A,B,C,P)>0; }

static vector<Tri> bowyerWatson(const vector<Vec2>& pts){
    vector<Vec2> P=pts;
    if(CFG.add_micro_jitter){
        std::mt19937_64 rng(1234567);
        std::uniform_real_distribution<double> U(-CFG.jitter_eps, CFG.jitter_eps);
        for(auto& p:P){ p.x+=U(rng); p.y+=U(rng); }
    }
    // super triangle
    Vec2 lo{+1e300,+1e300}, hi{-1e300,-1e300};
    for(const auto& p:P){ lo.x=std::min(lo.x,p.x); lo.y=std::min(lo.y,p.y); hi.x=std::max(hi.x,p.x); hi.y=std::max(hi.y,p.y); }
    Vec2 c = (lo+hi)*0.5; double d=std::max(hi.x-lo.x, hi.y-lo.y)*1000.0 + 1.0;
    int n0=(int)P.size();
    P.push_back({c.x-2*d, c.y-d});
    P.push_back({c.x+2*d, c.y-d});
    P.push_back({c.x,     c.y+2*d});
    int i1=n0, i2=n0+1, i3=n0+2;

    vector<Tri> T; T.push_back({i1,i2,i3});

    // incremental insert
    for(int ip=0; ip<n0; ++ip){
        const Vec2& p=P[ip];
        // bad triangles
        vector<int> bad; bad.reserve(T.size()/3);
        for(int t=0;t<(int)T.size();++t){
            Tri &tr=T[t]; if(!ccw(P[tr.a],P[tr.b],P[tr.c])) std::swap(tr.b,tr.c);
            if(inCircum(P[tr.a],P[tr.b],P[tr.c], p)) bad.push_back(t);
        }
        // boundary polygon
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
            del[id]=1; Tri tr=T[id];
            addE(tr.a,tr.b); addE(tr.b,tr.c); addE(tr.c,tr.a);
        }
        vector<Tri> keep; keep.reserve(T.size());
        for(int i=0;i<(int)T.size();++i) if(!del[i]) keep.push_back(T[i]);
        T.swap(keep);
        // retriangulate
        for(const auto& e: poly){
            Tri nt{e.u, e.v, ip};
            if(!ccw(P[nt.a],P[nt.b],P[nt.c])) std::swap(nt.b,nt.c);
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

//================== Edge Map & Flips =================
struct EdgeKey{
    int u,v; EdgeKey(){} EdgeKey(int a,int b){ u=std::min(a,b); v=std::max(a,b); }
    bool operator==(const EdgeKey&o)const{ return u==o.u && v==o.v; }
};
struct EdgeKeyHash{
    size_t operator()(const EdgeKey&k) const { return ( (uint64_t)k.u<<32 ) ^ (uint64_t)k.v; }
};
struct EdgeRef { int tri; int a,b; };

static void buildEdgeMap(const vector<Tri>& T, std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash>& M){
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
static bool hasEdge(const std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash>& M, int a,int b){
    auto it = M.find(EdgeKey(a,b));
    return (it!=M.end() && !it->second.empty());
}
static bool findEdgeTris(const std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash>& M, int a,int b, int& t1,int& t2){
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
static bool flipDiagonal(vector<Tri>& T, const vector<Vec2>& P, int t1, int t2, int u,int v){
    int a1=T[t1].a, b1=T[t1].b, c1=T[t1].c;
    int c=-1;
    if(a1!=u && a1!=v) c=a1;
    if(b1!=u && b1!=v) c=b1;
    if(c1!=u && c1!=v) c=c1;
    int a2=T[t2].a, b2=T[t2].b, c2=T[t2].c;
    int d=-1;
    if(a2!=u && a2!=v) d=a2;
    if(b2!=u && b2!=v) d=b2;
    if(c2!=u && c2!=v) d=c2;
    if(c==-1 || d==-1) return false;

    if(orient2d_filt(P[u],P[v],P[c])<=0) return false;
    if(orient2d_filt(P[v],P[u],P[d])<=0) return false;

    Tri Tleft  = {c, d, v};
    if(!ccw(P[Tleft.a],P[Tleft.b],P[Tleft.c])) std::swap(Tleft.b, Tleft.c);
    Tri Tright = {d, c, u};
    if(!ccw(P[Tright.a],P[Tright.b],P[Tright.c])) std::swap(Tright.b,Tright.c);

    T[t1]=Tleft;
    T[t2]=Tright;
    return true;
}
static bool intersectParamT(const Vec2&A,const Vec2&B,const Vec2&C,const Vec2&D,double& t){
    double x1=A.x,y1=A.y,x2=B.x,y2=B.y,x3=C.x,y3=C.y,x4=D.x,y4=D.y;
    double den=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4);
    if(std::fabs(den)<1e-20) return false;
    double tnum=(x1-x3)*(y3-y4)-(y1-y3)*(x3-x4);
    double unum=(x1-x3)*(y1-y2)-(y1-y3)*(x1-x2);
    t=tnum/den; double u=unum/den;
    return (t>0.0 && t<1.0 && u>0.0 && u<1.0);
}

// Constrained edge via flips
static bool insertConstraintEdge(vector<Tri>& T, const vector<Vec2>& P,
                                 int a,int b,
                                 const std::unordered_set<EdgeKey,EdgeKeyHash>& forced_set,
                                 int& globalFlipBudget)
{
    if(a==b) return true;

    std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash> M;
    buildEdgeMap(T,M);
    if(hasEdge(M,a,b)) return true;

    const Vec2& A=P[a]; const Vec2& B=P[b];

    int flips=0;
    while(!hasEdge(M,a,b)){
        if(globalFlipBudget<=0 || flips>=CFG.max_flips_per_segment){
            return false;
        }
        // collect intersecting edges (skip forced/incident)
        struct Hit{ int u,v; int t1,t2; double t; };
        vector<Hit> hits; hits.reserve(64);

        for(const auto& kv : M){
            int u=kv.first.u, v=kv.first.v;
            if(u==a||v==a||u==b||v==b) continue;
            if(forced_set.count(kv.first)) continue;

            int t1=-1,t2=-1;
            if(!findEdgeTris(M,u,v,t1,t2)) continue; // boundary edge => cannot flip

            if(segIntersectProper(A,B, P[u],P[v])){
                double tp; if(intersectParamT(A,B,P[u],P[v],tp)){
                    hits.push_back({u,v,t1,t2,tp});
                }
            }
        }
        if(hits.empty()){
            // cannot progress by flips
            return false;
        }
        std::sort(hits.begin(), hits.end(), [](const Hit&x,const Hit&y){ return x.t<y.t; });

        bool did=false;
        for(const auto& h: hits){
            if(globalFlipBudget<=0) return false;
            if(flipDiagonal(T,P,h.t1,h.t2, h.u,h.v)){
                did=true; flips++; globalFlipBudget--;
                break;
            }
        }
        if(!did) return false;
        buildEdgeMap(T,M);
    }
    return true;
}

static void legalizeCDT(vector<Tri>& T, const vector<Vec2>& P,
                        const std::unordered_set<EdgeKey,EdgeKeyHash>& forced_set,
                        int max_passes=3)
{
    for(int pass=0; pass<max_passes; ++pass){
        bool changed=false;
        std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash> M;
        buildEdgeMap(T,M);
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

            if(orient2d_filt(P[a],P[b],P[c])<=0) continue;
            if(orient2d_filt(P[b],P[a],P[d])<=0) continue;
            bool bad = (incircle_filt(P[a],P[b],P[c],P[d])>0.0);
            if(!bad) continue;

            EdgeKey newk(c,d);
            if(forced_set.count(newk)) continue;

            if(flipDiagonal(T,P,t1,t2,a,b)){ changed=true; }
        }
        if(!changed) break;
    }
}

//================== Clip triangles to track region =================
static vector<pair<Vec2,Vec2>> ringEdges(const vector<Vec2>& R){
    vector<pair<Vec2,Vec2>> E; int n=(int)R.size();
    for(int i=0;i<n;i++){ int j=(i+1)%n; E.push_back({R[i],R[j]}); }
    return E;
}
static bool triangleKeep(const Vec2&A,const Vec2&B,const Vec2&C,
                         const vector<Vec2>& inner,const vector<Vec2>& outer,
                         const vector<pair<Vec2,Vec2>>& innerE,
                         const vector<pair<Vec2,Vec2>>& outerE)
{
    Vec2 cent=(A+B+C)*(1.0/3.0);
    if(!pointInPoly(outer, cent)) return false;
    if(pointInPoly(inner, cent))  return false;

    auto crosses=[&](const Vec2& u,const Vec2& v)->bool{
        for(const auto&e: innerE){
            if(almostEq(u,e.first)||almostEq(u,e.second)||almostEq(v,e.first)||almostEq(v,e.second)) continue;
            if(segIntersectProper(u,v,e.first,e.second)) return true;
        }
        for(const auto&e: outerE){
            if(almostEq(u,e.first)||almostEq(u,e.second)||almostEq(v,e.first)||almostEq(v,e.second)) continue;
            if(segIntersectProper(u,v,e.first,e.second)) return true;
        }
        return false;
    };
    if(crosses(A,B)||crosses(B,C)||crosses(C,A)) return false;
    return true;
}

//================== Centerline =================
static vector<Vec2> midpointsInnerOuter_byLabels(const vector<Vec2>& all,const vector<Tri>& T,
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
static vector<Vec2> orderByMST(const vector<Vec2>& pts){
    int n=(int)pts.size(); if(n<=2) return pts;
    int K=std::min(CFG.knn_k, n-1);
    vector<vector<pair<int,double>>> adj(n);
    for(int i=0;i<n;i++){
        vector<pair<double,int>> cand; cand.reserve(n-1);
        for(int j=0;j<n;j++) if(i!=j) cand.push_back({ (pts[i].x-pts[j].x)*(pts[i].x-pts[j].x) + (pts[i].y-pts[j].y)*(pts[i].y-pts[j].y), j });
        if((int)cand.size()>K){
            std::nth_element(cand.begin(), cand.begin()+K, cand.end(),
                [](const auto&A,const auto&B){ return A.first<B.first; });
            cand.resize(K);
        }
        for(auto&c:cand){ double w=std::sqrt(std::max(0.0,c.first));
            adj[i].push_back({c.second,w}); adj[c.second].push_back({i,w}); }
    }
    // Prim -> MST
    vector<double> key(n,1e300); vector<int> par(n,-1); vector<char> in(n,0); key[0]=0;
    for(int it=0; it<n; ++it){
        int u=-1; double best=1e301;
        for(int i=0;i<n;i++) if(!in[i] && key[i]<best){ best=key[i]; u=i; }
        if(u==-1) break; in[u]=1;
        for(auto [v,w]:adj[u]) if(!in[v] && w<key[v]){ key[v]=w; par[v]=u; }
    }
    vector<vector<int>> tree(n);
    for(int v=0;v<n;v++) if(par[v]>=0){ tree[v].push_back(par[v]); tree[par[v]].push_back(v); }

    auto bfs=[&](int s){
        vector<double>d(n,1e300); vector<int>p(n,-1); std::queue<int>q; q.push(s); d[s]=0;
        while(!q.empty()){ int u=q.front(); q.pop();
            for(int v:tree[u]) if(d[v]>1e299){ d[v]=d[u]+std::sqrt((pts[u].x-pts[v].x)*(pts[u].x-pts[v].x)+(pts[u].y-pts[v].y)*(pts[u].y-pts[v].y));
                p[v]=u; q.push(v); } }
        int far=s; for(int i=0;i<n;i++) if(d[i]>d[far]) far=i; return std::tuple<int,vector<int>,vector<double>>(far,p,d);
    };
    auto [s1,p1,d1]=bfs(0);
    auto [s2,p2,d2]=bfs(s1);
    vector<int> path; for(int v=s2; v!=-1; v=p2[v]) path.push_back(v);
    vector<char> used(n,0); vector<Vec2> out; out.reserve(n);
    for(int id:path){ out.push_back(pts[id]); used[id]=1; }
    for(int i=0;i<n;i++) if(!used[i]) out.push_back(pts[i]);
    return out;
}
static vector<Vec2> catmullClosed(const vector<Vec2>& pts,int samples){
    int n=(int)pts.size(); if(n==0) return {}; if(n==1) return pts;
    if(n==2){ vector<Vec2> out; out.reserve(samples);
        for(int i=0;i<samples;i++){ double t=double(i)/std::max(1,samples-1); out.push_back(pts[0]*(1.0-t)+pts[1]*t); } return out; }
    auto P=[&](int i){ int k=(i%n+n)%n; return pts[k]; };
    vector<Vec2> out; int segs=n, m=std::max(2, samples/segs);
    for(int s=0;s<segs;s++){
        Vec2 p0=P(s-1), p1=P(s), p2=P(s+1), p3=P(s+2);
        auto tj=[&](double ti,const Vec2&a,const Vec2&b){ double a2=0.5; double d=std::pow(std::sqrt(norm2(b-a)), a2); return ti+d; };
        double t0=0, t1=tj(t0,p0,p1), t2=tj(t1,p1,p2), t3=tj(t2,p2,p3);
        for(int i=0;i<m;i++){
            double t=t1+(t2-t1)*(double(i)/double(m));
            Vec2 A1 = (p0*((t1 - t)/(t1 - t0 + 1e-30)) + p1*((t - t0)/(t1 - t0 + 1e-30)));
            Vec2 A2 = (p1*((t2 - t)/(t2 - t1 + 1e-30)) + p2*((t - t1)/(t2 - t1 + 1e-30)));
            Vec2 A3 = (p2*((t3 - t)/(t3 - t2 + 1e-30)) + p3*((t - t2)/(t3 - t2 + 1e-30)));
            Vec2 B1 = (A1*((t2 - t)/(t2 - t0 + 1e-30)) + A2*((t - t0)/(t2 - t0 + 1e-30)));
            Vec2 B2 = (A2*((t3 - t)/(t3 - t1 + 1e-30)) + A3*((t - t1)/(t3 - t1 + 1e-30)));
            Vec2 C  = (B1*((t2 - t)/(t2 - t1 + 1e-30)) + B2*((t - t1)/(t2 - t1 + 1e-30)));
            out.push_back(C);
        }
    }
    vector<Vec2> clean; clean.reserve(out.size());
    for(auto&p:out){ if(clean.empty() || norm(p-clean.back())>1e-6) clean.push_back(p); }
    if((int)clean.size()>samples) clean.resize(samples);
    while((int)clean.size()<samples) clean.push_back(clean.back());
    return clean;
}

static vector<Vec2> resampleUniformClosed(const vector<Vec2>& pts, int samples) {
    if (pts.size() < 2) return pts;

    // 닫힌 형태 보장용: 마지막이 첫점과 다르면 닫아둠
    vector<Vec2> P = pts;
    if (std::fabs(P.front().x - P.back().x) > 1e-9 ||
        std::fabs(P.front().y - P.back().y) > 1e-9) {
        P.push_back(P.front());
    }

    int n = (int)P.size();
    // 누적 거리
    vector<double> s(n, 0.0);
    for (int i=1; i<n; ++i) s[i] = s[i-1] + std::sqrt((P[i].x-P[i-1].x)*(P[i].x-P[i-1].x)
                                                    + (P[i].y-P[i-1].y)*(P[i].y-P[i-1].y));
    double L = s.back();
    if (L <= 1e-12) return vector<Vec2>(samples, P.front());

    // 균일 간격 위치로 보간
    vector<Vec2> out; out.reserve(samples+1);
    for (int k=0; k<samples; ++k) {
        double t = (L * k) / double(samples);
        auto it = std::upper_bound(s.begin(), s.end(), t);
        int i = std::max(1, int(it - s.begin())) - 1; // [i, i+1]
        double seg = s[i+1] - s[i];
        double u = (seg > 1e-20) ? (t - s[i]) / seg : 0.0;
        Vec2 q{ P[i].x*(1.0-u) + P[i+1].x*u, P[i].y*(1.0-u) + P[i+1].y*u };
        out.push_back(q);
    }
    // 닫힘 보장(시각화용)
    if (std::fabs(out.front().x - out.back().x) > 1e-9 ||
        std::fabs(out.front().y - out.back().y) > 1e-9) {
        out.push_back(out.front());
    }
    return out;
}

//================== IO =================
static vector<Vec2> loadCSV(const string& path){
    vector<Vec2> pts; std::ifstream fin(path);
    if(!fin){ cerr<<"[ERR] open "<<path<<"\n"; return pts; }
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
static bool saveCSV_pointsXY(const string& path,const vector<Vec2>& pts){
    std::ofstream fo(path); if(!fo){ cerr<<"[ERR] write "<<path<<"\n"; return false; }
    fo.setf(std::ios::fixed); fo.precision(9);
    for(auto&p:pts) fo<<p.x<<","<<p.y<<"\n"; return true;
}
static bool saveCSV_pointsLabeled(const string& path,const vector<Vec2>& pts, const vector<int>& label){
    std::ofstream fo(path); if(!fo){ cerr<<"[ERR] write "<<path<<"\n"; return false; }
    fo<<"id,x,y,label\n"; fo.setf(std::ios::fixed); fo.precision(9);
    for(size_t i=0;i<pts.size();++i) fo<<i<<","<<pts[i].x<<","<<pts[i].y<<","<<label[i]<<"\n"; return true;
}
static bool saveCSV_edgesIdx(const string& path,const vector<pair<int,int>>& E){
    std::ofstream fo(path); if(!fo){ cerr<<"[ERR] write "<<path<<"\n"; return false; }
    for(auto&e:E) fo<<e.first<<","<<e.second<<"\n"; return true;
}
static bool saveCSV_trisIdx(const string& path,const vector<Tri>& T){
    std::ofstream fo(path); if(!fo){ cerr<<"[ERR] write "<<path<<"\n"; return false; }
    for(auto&t:T) fo<<t.a<<","<<t.b<<","<<t.c<<"\n"; return true;
}
static string dropExt(const string& s){
    size_t p=s.find_last_of('.'); if(p==string::npos) return s; return s.substr(0,p);
}
struct Constraint { int a,b; int splits=0; }; // dynamic (indices refer to all[])

//================== Build CDT with Steiner fallback =================
struct CDTResult{
    vector<Vec2> all;
    vector<int>  label; // 0 inner,1 outer
    vector<Tri>  tris;
    vector<pair<int,int>> forced_edges; // 최종 강제 엣지 집합(슈타이너 포함)
    bool all_forced_ok=false;
};

static CDTResult buildCDT_withRecovery(vector<Vec2> inner, vector<Vec2> outer){
    CDTResult R;
    // merge and label
    R.all = inner; R.all.insert(R.all.end(), outer.begin(), outer.end());
    R.label.assign(R.all.size(), 0);
    for(size_t i=0;i<R.all.size();++i) R.label[i] = (i<inner.size()?0:1);

    auto rebuildDT = [&](vector<Tri>& T){
        T = bowyerWatson(R.all);
    };

    // initial DT
    vector<Tri> T; rebuildDT(T);

    // initial constraint list (ring edges)
    vector<Constraint> cons;
    int nIn=(int)inner.size(), nOut=(int)outer.size();
    auto pushRing=[&](int base,int n){
        for(int i=0;i<n;i++){ int j=(i+1)%n; cons.push_back({base+i, base+j, 0}); }
    };
    pushRing(0, nIn);
    pushRing(nIn, nOut);

    // forced set (dynamic)
    auto rebuildForcedSet=[&](std::unordered_set<EdgeKey,EdgeKeyHash>& F){
        F.clear(); F.reserve(cons.size()*2);
        for(auto& c: cons) F.insert( EdgeKey(c.a,c.b) );
    };

    // global flip budget
    int globalFlipBudget = CFG.max_global_flips;
    bool ok=false;
    for(int rebuilds=0; rebuilds<=CFG.max_cdt_rebuilds; ++rebuilds){
        std::unordered_set<EdgeKey,EdgeKeyHash> forced;
        rebuildForcedSet(forced);

        ok=true;
        for(size_t k=0;k<cons.size();++k){
            auto& seg=cons[k];
            if(globalFlipBudget<=0){ ok=false; break; }
            // try insert
            if(insertConstraintEdge(T, R.all, seg.a, seg.b, forced, globalFlipBudget)) continue;

            // failed → split if under limit
            if(seg.splits >= CFG.max_segment_splits){ ok=false; break; }

            // add Steiner midpoint
            Vec2 A=R.all[seg.a], B=R.all[seg.b];
            Vec2 M = (A+B)*0.5;
            int newIdx = (int)R.all.size();
            R.all.push_back(M);
            R.label.push_back(R.label[seg.a]); // same ring label

            // replace [a,b] by [a,new], [new,b]
            Constraint left { seg.a, newIdx,  seg.splits+1 };
            Constraint right{ newIdx, seg.b,  seg.splits+1 };
            // erase seg and insert two
            cons.erase(cons.begin()+k);
            cons.insert(cons.begin()+k, right);
            cons.insert(cons.begin()+k, left);

            // rebuild DT for new point
            rebuildDT(T);
            rebuildForcedSet(forced);

            ok=false; // need another outer loop
            break;
        }
        if(ok) { // optional CDT legalization (non-forced)
            legalizeCDT(T, R.all, /*forced*/ std::unordered_set<EdgeKey,EdgeKeyHash>(), 2);
            break;
        }
        if(CFG.verbose) cerr<<"[CDT] rebuild "<<(rebuilds+1)<<" due to split; total pts="<<R.all.size()<<"\n";
        if(rebuilds==CFG.max_cdt_rebuilds){ break; }
    }

    R.tris = std::move(T);
    R.all_forced_ok = ok;

    // collect final forced edges list (슈타이너 포함)
    R.forced_edges.clear();
    R.forced_edges.reserve(cons.size());
    for(const auto& c: cons) R.forced_edges.push_back({c.a,c.b});

    return R;
}

//================== MAIN =================
int main(int argc,char**argv){
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if(argc<4){
        cerr<<"Usage: "<<argv[0]<<" inner.csv outer.csv centerline.csv [dump_prefix]\n";
        return 1;
    }
    string innerPath=argv[1], outerPath=argv[2], outPath=argv[3];
    string dumpPrefix = (argc>=5? string(argv[4]) : dropExt(outPath));

    vector<Vec2> inner=loadCSV(innerPath), outer=loadCSV(outerPath);
    if(inner.size()<3 || outer.size()<3){ cerr<<"[ERR] need >=3 points per ring\n"; return 2; }

    // 1) CDT with Steiner fallback
    auto cdt = buildCDT_withRecovery(inner, outer);
    if(CFG.verbose){
        cerr<<"[CDT] forced insert "<<(cdt.all_forced_ok?"OK":"RECOVERED with Steiner")
            <<", total points="<<cdt.all.size()<<", faces="<<cdt.tris.size()<<"\n";
    }

    // === Debug dump 1: all points (labeled), forced edges, raw triangles ===
    saveCSV_pointsLabeled(dumpPrefix + "_all_points.csv", cdt.all, cdt.label);
    saveCSV_edgesIdx     (dumpPrefix + "_forced_edges_idx.csv", cdt.forced_edges);
    saveCSV_trisIdx      (dumpPrefix + "_tri_raw_idx.csv",      cdt.tris);

    // 2) Clip to track region (outer-in, inner-out, boundary respectful)
    vector<Tri> faces_kept, faces_drop;
    {
        auto innerE = ringEdges(inner);
        auto outerE = ringEdges(outer);
        for(const auto& t: cdt.tris){
            const Vec2&A=cdt.all[t.a], &B=cdt.all[t.b], &C=cdt.all[t.c];
            if(triangleKeep(A,B,C, inner,outer, innerE,outerE)) faces_kept.push_back(t);
            else faces_drop.push_back(t);
        }
    }
    if(faces_kept.empty()){
        if(!CFG.allow_fallback_clip){
            cerr<<"[ERR] no faces after clipping\n"; return 3;
        }
        faces_kept = cdt.tris; // fallback
    }

    // === Debug dump 2: kept/drop faces ===
    saveCSV_trisIdx(dumpPrefix + "_faces_kept_idx.csv", faces_kept);
    saveCSV_trisIdx(dumpPrefix + "_faces_drop_idx.csv", faces_drop);

    // 3) Internal edge midpoints → order → spline
    auto mids    = midpointsInnerOuter_byLabels(cdt.all, faces_kept, cdt.label);
    if(mids.size()<4){
        cerr<<"[ERR] not enough midpoints ("<<mids.size()<<")\n";
        // 그래도 덤프는 찍어 주자
        saveCSV_pointsXY(dumpPrefix + "_mids_raw.csv", mids);
        return 4;
    }
    auto ordered = orderByMST(mids);
    auto center  = catmullClosed(ordered, CFG.samples);
    center = resampleUniformClosed(center, CFG.samples);

    // === Debug dump 3: midpoints ===
    saveCSV_pointsXY(dumpPrefix + "_mids_raw.csv",     mids);
    saveCSV_pointsXY(dumpPrefix + "_mids_ordered.csv", ordered);

    // 4) 최종 센터라인 저장(원래 3번째 인자)
    std::ofstream fo(outPath);
    if(!fo){ cerr<<"[ERR] save centerline "<<outPath<<"\n"; return 5; }
    fo.setf(std::ios::fixed); fo.precision(9);
    for(auto&p:center) fo<<p.x<<","<<p.y<<"\n";
    fo.close();

    if(CFG.verbose){
        cerr<<"[OK] centerline saved: "<<outPath<<"  (N="<<center.size()<<")\n";
        cerr<<"[OK] dumps: "<<dumpPrefix<<"_* .csv (points/edges/triangles/midpoints)\n";
    }
    return 0;
}
