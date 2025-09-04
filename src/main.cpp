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
#include <random>
#include <numeric>
#include <functional>
#include <cassert>

using std::vector;
using std::array;
using std::pair;
using std::make_pair;
using std::size_t;
using std::cout;
using std::cerr;
using std::endl;
using std::string;

// ---------------- Config ----------------
struct Config {
    // Bowyer–Watson super triangle scale (relative to bbox)
    double superScale = 1000.0;

    // Preprocess
    double dedup_eps = 1e-9;      // merge near-identical points
    bool add_micro_jitter = true; // to avoid perfect collinearity
    double jitter_eps = 1e-9;

    // Edge tests
    double orient_eps = 1e-15; // filtered tolerance

    // Midpoints → sampling
    int centerline_samples = 300; // number of output samples on centerline

    // MST k-NN
    int knn_k = 8;
};
static Config CFG;

// ---------------- Geometry ----------------
struct Vec2 {
    double x=0.0, y=0.0;
    Vec2() {}
    Vec2(double _x,double _y):x(_x),y(_y){}
    Vec2 operator+(const Vec2& o) const { return {x+o.x, y+o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x-o.x, y-o.y}; }
    Vec2 operator*(double s) const { return {x*s, y*s}; }
    Vec2 operator/(double s) const { return {x/s, y/s}; }
};
static inline double dot(const Vec2&a,const Vec2&b){ return a.x*b.x + a.y*b.y; }
static inline double cross(const Vec2&a,const Vec2&b){ return a.x*b.y - a.y*b.x; }
static inline double norm2(const Vec2&a){ return dot(a,a); }
static inline double norm (const Vec2&a){ return std::sqrt(norm2(a)); }
static inline Vec2   normalL(const Vec2& t){ return Vec2{-t.y, t.x}; }

struct BBox { Vec2 lo, hi; };
static BBox bboxOf(const vector<Vec2>& pts) {
    BBox b; b.lo.x = b.lo.y =  std::numeric_limits<double>::max();
    b.hi.x = b.hi.y = -std::numeric_limits<double>::max();
    for (auto &p: pts) {
        b.lo.x = std::min(b.lo.x, p.x);
        b.lo.y = std::min(b.lo.y, p.y);
        b.hi.x = std::max(b.hi.x, p.x);
        b.hi.y = std::max(b.hi.y, p.y);
    }
    return b;
}

// Filtered orient2d with long double fallback
static inline double orient2d_filt(const Vec2& a, const Vec2& b, const Vec2& c){
    double det = (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
    double absa = std::fabs(b.x - a.x) + std::fabs(b.y - a.y);
    double absb = std::fabs(c.x - a.x) + std::fabs(c.y - a.y);
    double err = (absa*absb)*std::numeric_limits<double>::epsilon()*4.0;
    if (std::fabs(det) > err) return det;
    // fallback
    long double adx = (long double)b.x - (long double)a.x;
    long double ady = (long double)b.y - (long double)a.y;
    long double bdx = (long double)c.x - (long double)a.x;
    long double bdy = (long double)c.y - (long double)a.y;
    long double detl = adx*bdy - ady*bdx;
    return (double)detl;
}

// incircle test filtered + long double fallback
static inline double incircle_filt(const Vec2& a,const Vec2& b,const Vec2& c,const Vec2& d){
    // determinant of (ax ay ax^2+ay^2 1; ...)
    double adx = a.x - d.x, ady = a.y - d.y;
    double bdx = b.x - d.x, bdy = b.y - d.y;
    double cdx = c.x - d.x, cdy = c.y - d.y;

    double ad = adx*adx + ady*ady;
    double bd = bdx*bdx + bdy*bdy;
    double cd = cdx*cdx + cdy*cdy;

    double det = adx*(bdy*cd - bd*cdy) - ady*(bdx*cd - bd*cdx) + ad*(bdx*cdy - bdy*cdx);

    double mags = (std::fabs(adx)+std::fabs(ady))*(std::fabs(bdx)+std::fabs(bdy))*(std::fabs(cdx)+std::fabs(cdy));
    double err = mags*std::numeric_limits<double>::epsilon()*16.0;
    if (std::fabs(det) > err) return det;

    // fallback long double
    long double ADX=a.x; long double ADY=a.y;
    long double BDX=b.x; long double BDY=b.y;
    long double CDX=c.x; long double CDY=c.y;
    long double DX = d.x; long double DY = d.y;
    long double adxl = ADX - DX, adyl = ADY - DY;
    long double bdxl = BDX - DX, bdyl = BDY - DY;
    long double cdxl = CDX - DX, cdyl = CDY - DY;
    long double adl = adxl*adxl + adyl*adyl;
    long double bdl = bdxl*bdxl + bdyl*bdyl;
    long double cdl = cdxl*cdxl + cdyl*cdyl;
    long double detl = adxl*(bdyl*cdl - bdl*cdyl) - adyl*(bdxl*cdl - bdl*cdxl) + adl*(bdxl*cdyl - bdyl*cdxl);
    return (double)detl;
}

static inline int orientSign(const Vec2& a,const Vec2& b,const Vec2& c){
    double d = orient2d_filt(a,b,c);
    if (d>0) return 1; if (d<0) return -1; return 0;
}

// Segment intersection (proper or touching at interiors; endpoints-sharing allowed toggled)
static bool segIntersect(const Vec2& a,const Vec2& b,const Vec2& c,const Vec2& d, bool count_touches=true){
    int s1 = orientSign(a,b,c);
    int s2 = orientSign(a,b,d);
    int s3 = orientSign(c,d,a);
    int s4 = orientSign(c,d,b);
    if (s1==0 && s2==0 && s3==0 && s4==0) {
        // collinear: check overlap in 1D
        auto minmax = [](double u,double v){ if(u>v) std::swap(u,v); return std::make_pair(u,v); };
        auto [ax,bx]=minmax(a.x,b.x); auto [cx,dx]=minmax(c.x,d.x);
        auto [ay,by]=minmax(a.y,b.y); auto [cy,dy]=minmax(c.y,d.y);
        bool overlap = !(bx < cx || dx < ax || by < cy || dy < ay);
        return count_touches ? overlap : (bx>cx && dx>ax && by>cy && dy>ay);
    }
    bool proper = (s1*s2<=0 && s3*s4<=0);
    if(!count_touches){
        proper = (s1*s2<0 && s3*s4<0);
    }
    return proper;
}

// point-in-polygon (ray casting)
static bool pointInPolygon(const vector<Vec2>& poly, const Vec2& p){
    bool inside=false;
    int n=poly.size();
    for(int i=0,j=n-1;i<n;j=i++){
        const Vec2& a=poly[j]; const Vec2& b=poly[i];
        bool cond = ((a.y>p.y)!=(b.y>p.y)) && (p.x < (b.x-a.x)*(p.y-a.y)/(b.y-a.y + 1e-30) + a.x);
        if(cond) inside=!inside;
    }
    return inside;
}

// ---------------- Bowyer–Watson Delaunay ----------------
struct Triangle { int a,b,c; }; // CCW

static bool ccw(const Vec2& a,const Vec2& b,const Vec2& c){ return orient2d_filt(a,b,c)>0; }

static bool inCircumcircle(const Vec2& a,const Vec2& b,const Vec2& c,const Vec2& p){
    // Returns true if p is inside circumcircle of (a,b,c) (CCW assumed)
    double s = incircle_filt(a,b,c,p);
    return s > 0.0;
}

static bool almostEqual(const Vec2&a,const Vec2&b,double eps){ return (std::fabs(a.x-b.x)<=eps && std::fabs(a.y-b.y)<=eps); }

static void dedupPoints(vector<Vec2>& pts, vector<int>& map_old2new, double eps){
    vector<Vec2> out; out.reserve(pts.size());
    map_old2new.assign(pts.size(), -1);
    for(size_t i=0;i<pts.size();++i){
        bool found=false; int idx=-1;
        for(int j=0;j<(int)out.size();++j){
            if(almostEqual(pts[i], out[j], eps)){ found=true; idx=j; break; }
        }
        if(!found){ idx=(int)out.size(); out.push_back(pts[i]); }
        map_old2new[i]=idx;
    }
    pts.swap(out);
}

static vector<Triangle> bowyerWatson(const vector<Vec2>& inpts, double superScale){
    // Copy points and add jitter if requested
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> U(-CFG.jitter_eps, CFG.jitter_eps);
    vector<Vec2> pts = inpts;
    if(CFG.add_micro_jitter){
        for(auto& p:pts){ p.x += U(rng); p.y += U(rng); }
    }

    // Deduplicate
    vector<int> o2n;
    dedupPoints(pts, o2n, CFG.dedup_eps);

    // Build super triangle
    BBox b = bboxOf(pts);
    Vec2 c = (b.lo + b.hi)/2.0;
    double dx = b.hi.x - b.lo.x;
    double dy = b.hi.y - b.lo.y;
    double delta = std::max(dx,dy)*superScale + 1.0;
    // Big equilateral-ish super triangle
    Vec2 p1 = {c.x - 2*delta, c.y - delta};
    Vec2 p2 = {c.x + 2*delta, c.y - delta};
    Vec2 p3 = {c.x,          c.y + 2*delta};
    int n0 = (int)pts.size();
    vector<Vec2> P = pts;
    P.push_back(p1); P.push_back(p2); P.push_back(p3);
    int si1=n0, si2=n0+1, si3=n0+2;

    vector<Triangle> tris;
    tris.push_back({si1,si2,si3});

    // Incremental
    for(int ip=0; ip<n0; ++ip){
        const Vec2& p = P[ip];

        // find all triangles whose circumcircle contains p
        vector<int> bad;
        bad.reserve(tris.size()/3);
        for(int t=0;t<(int)tris.size();++t){
            Triangle &T = tris[t];
            // ensure CCW
            if(!ccw(P[T.a],P[T.b],P[T.c])) std::swap(T.b,T.c);
            if(inCircumcircle(P[T.a],P[T.b],P[T.c], p)){
                bad.push_back(t);
            }
        }
        // find boundary polygon (edges of bad triangles not shared twice)
        struct Edge{int u,v;};
        vector<Edge> polygon;
        auto add_edge = [&](int u,int v){
            // directed; boundary if (v,u) not present
            for(auto it=polygon.begin(); it!=polygon.end(); ++it){
                if(it->u==v && it->v==u){ polygon.erase(it); return; }
            }
            polygon.push_back({u,v});
        };
        // mark removed
        vector<bool> removed(tris.size(), false);
        for(int id: bad){
            removed[id]=true;
            Triangle T=tris[id];
            add_edge(T.a,T.b);
            add_edge(T.b,T.c);
            add_edge(T.c,T.a);
        }
        // compact tris
        vector<Triangle> keep; keep.reserve(tris.size());
        for(int i=0;i<(int)tris.size();++i) if(!removed[i]) keep.push_back(tris[i]);
        tris.swap(keep);

        // retriangulate cavity
        for(const auto& e: polygon){
            Triangle nt{e.u, e.v, ip};
            // enforce CCW
            if(!ccw(P[nt.a],P[nt.b],P[nt.c])) std::swap(nt.b,nt.c);
            tris.push_back(nt);
        }
    }

    // remove triangles using super triangle vertices
    vector<Triangle> out;
    out.reserve(tris.size());
    for(const auto& T: tris){
        if(T.a>=n0 || T.b>=n0 || T.c>=n0) continue;
        out.push_back(T);
    }
    return out;
}

// ---------------- Constraints via Polygon Region ----------------
//
// 트랙을 "outer polygon 내부 AND inner polygon 외부"로 정의.
// 또한 삼각형 엣지가 경계 세그먼트와 교차하면 제외(경계 준수).
//
static bool triangleOK(const Vec2& A,const Vec2& B,const Vec2& C,
                       const vector<Vec2>& inner, const vector<Vec2>& outer,
                       const vector<pair<Vec2,Vec2>>& innerE,
                       const vector<pair<Vec2,Vec2>>& outerE)
{
    Vec2 cent = (A+B+C)/3.0;
    bool insideOuter = pointInPolygon(outer, cent);
    bool outsideInner = !pointInPolygon(inner, cent);
    if(!(insideOuter && outsideInner)) return false;

    // boundary crossing check
    auto crossesAny = [&](const Vec2& u,const Vec2& v)->bool{
        for(const auto& e: innerE){
            // shared endpoints allowed
            if (almostEqual(u,e.first,1e-12) || almostEqual(u,e.second,1e-12) ||
                almostEqual(v,e.first,1e-12) || almostEqual(v,e.second,1e-12)) continue;
            if(segIntersect(u,v,e.first,e.second,false)) return true;
        }
        for(const auto& e: outerE){
            if (almostEqual(u,e.first,1e-12) || almostEqual(u,e.second,1e-12) ||
                almostEqual(v,e.first,1e-12) || almostEqual(v,e.second,1e-12)) continue;
            if(segIntersect(u,v,e.first,e.second,false)) return true;
        }
        return false;
    };
    if(crossesAny(A,B)) return false;
    if(crossesAny(B,C)) return false;
    if(crossesAny(C,A)) return false;
    return true;
}

// Build ring edges (closed polygon)
static vector<pair<Vec2,Vec2>> ringEdges(const vector<Vec2>& ring){
    vector<pair<Vec2,Vec2>> E;
    int n = (int)ring.size();
    for(int i=0;i<n;i++){
        int j=(i+1)%n;
        E.push_back({ring[i], ring[j]});
    }
    return E;
}

// Label points as inner(0)/outer(1)/unknown(-1) based on exact match
static vector<int> labelPoints(const vector<Vec2>& pts,
                               const vector<Vec2>& inner,
                               const vector<Vec2>& outer)
{
    vector<int> lab(pts.size(), -1);
    for(size_t i=0;i<pts.size();++i){
        for(const auto& p: inner) if(almostEqual(pts[i],p,1e-8)){ lab[i]=0; break; }
        if(lab[i]==-1){
            for(const auto& p: outer) if(almostEqual(pts[i],p,1e-8)){ lab[i]=1; break; }
        }
    }
    return lab;
}

struct EdgeKey {
    int u,v;
    EdgeKey(){}
    EdgeKey(int a,int b){ u=std::min(a,b); v=std::max(a,b); }
    bool operator<(const EdgeKey& o) const {
        if(u!=o.u) return u<o.u;
        return v<o.v;
    }
};

// ---------------- Centerline: internal edge midpoints ----------------
static vector<Vec2> extractInternalMidpoints(const vector<Vec2>& pts,
                                            const vector<Triangle>& tris,
                                            const vector<int>& label)
{
    std::map<EdgeKey, bool> chosen;
    for(const auto& T: tris){
        int vids[3]={T.a,T.b,T.c};
        for(int e=0;e<3;e++){
            int i=vids[e], j=vids[(e+1)%3];
            int li=label[i], lj=label[j];
            if(li==-1 || lj==-1) continue;
            // internal edge = connects inner↔outer
            if(li!=lj){
                chosen[EdgeKey(i,j)]=true;
            }
        }
    }
    vector<Vec2> mids; mids.reserve(chosen.size());
    for(const auto& kv: chosen){
        int i=kv.first.u, j=kv.first.v;
        Vec2 m = (pts[i]+pts[j])*0.5;
        mids.push_back(m);
    }
    return mids;
}

// ---------------- Order midpoints using MST longest path ----------------
static double dist2(const Vec2&a,const Vec2&b){ return (a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y); }

static vector<Vec2> orderByMSTLongestPath(const vector<Vec2>& pts){
    int n=(int)pts.size();
    if(n<=2) return pts;
    // Build kNN graph
    int K = std::min(CFG.knn_k, n-1);
    vector<vector<pair<int,double>>> adj(n);
    for(int i=0;i<n;i++){
        vector<pair<double,int>> cand;
        cand.reserve(n-1);
        for(int j=0;j<n;j++) if(i!=j){
            cand.push_back({dist2(pts[i],pts[j]), j});
        }
        std::nth_element(cand.begin(), cand.begin()+K, cand.end(),
                         [](const auto& A,const auto& B){ return A.first<B.first; });
        cand.resize(K);
        for(auto &c: cand){
            adj[i].push_back({c.second, std::sqrt(std::max(0.0,c.first))});
            adj[c.second].push_back({i, std::sqrt(std::max(0.0,c.first))});
        }
    }
    // Prim MST
    vector<double> key(n,std::numeric_limits<double>::infinity());
    vector<int> parent(n,-1);
    vector<bool> inMST(n,false);
    key[0]=0;
    for(int it=0; it<n; ++it){
        int u=-1; double best=1e300;
        for(int i=0;i<n;i++) if(!inMST[i] && key[i]<best){ best=key[i]; u=i; }
        if(u==-1) break;
        inMST[u]=true;
        for(auto [v,w]: adj[u]){
            if(!inMST[v] && w<key[v]){
                key[v]=w; parent[v]=u;
            }
        }
    }
    // build MST adjacency
    vector<vector<int>> tree(n);
    for(int v=0;v<n;v++){
        if(parent[v]>=0){ tree[v].push_back(parent[v]); tree[parent[v]].push_back(v); }
    }
    auto bfs_far = [&](int s)->std::tuple<int, vector<int>, vector<double>>{
        vector<double> d(n, 1e300);
        vector<int> par(n,-1);
        std::queue<int> q; q.push(s); d[s]=0;
        while(!q.empty()){
            int u=q.front(); q.pop();
            for(int v: tree[u]){
                if(d[v]>1e299){
                    d[v]=d[u] + std::sqrt(dist2(pts[u],pts[v]));
                    par[v]=u;
                    q.push(v);
                }
            }
        }
        int far=s;
        for(int i=0;i<n;i++) if(d[i]>d[far]) far=i;
        return {far, par, d};
    };
    int s0=0;
    auto t1 = bfs_far(s0);
    int s1 = std::get<0>(t1);
    auto t2 = bfs_far(s1);
    int s2 = std::get<0>(t2);
    vector<int> par2 = std::get<1>(t2);

    vector<int> path;
    for(int v=s2; v!=-1; v=par2[v]) path.push_back(v);
    // order points along path; if some points are not on path, append arbitrarily
    vector<bool> used(n,false);
    vector<Vec2> ordered; ordered.reserve(n);
    for(int idx: path){ ordered.push_back(pts[idx]); used[idx]=true; }
    for(int i=0;i<n;i++) if(!used[i]) ordered.push_back(pts[i]);
    return ordered;
}

// ---------------- Spline (Catmull-Rom centripetal, loop) ----------------
static vector<Vec2> catmullRom(const vector<Vec2>& p, int samples){
    if(p.size()<2){ return p; }
    vector<Vec2> pts = p;
    // 트랙은 폐곡선이므로 loop로 처리
    bool loop=true;
    auto P = [&](int i)->Vec2{
        int n=pts.size();
        if(loop){ int k=((i%n)+n)%n; return pts[k]; }
        else{
            if(i<0) return pts[0];
            if(i>=n) return pts[n-1];
            return pts[i];
        }
    };
    int segs = (int)pts.size();
    vector<Vec2> out; out.reserve(samples);
    for(int s=0;s<segs;s++){
        Vec2 p0 = P(s-1), p1=P(s), p2=P(s+1), p3=P(s+2);
        auto tj = [&](double ti, const Vec2& a, const Vec2& b){
            double alpha=0.5; // centripetal
            double d = std::pow(std::sqrt(norm2(b-a)), alpha);
            return ti + d;
        };
        double t0=0;
        double t1=tj(t0,p0,p1);
        double t2=tj(t1,p1,p2);
        double t3=tj(t2,p2,p3);
        int m = std::max(2, samples/segs);
        for(int i=0;i<m;i++){
            double t = t1 + (t2 - t1)*(double(i)/double(m));
            Vec2 A1 = (p0*( (t1 - t)/(t1 - t0 + 1e-30) ) + p1*( (t - t0)/(t1 - t0 + 1e-30) ));
            Vec2 A2 = (p1*( (t2 - t)/(t2 - t1 + 1e-30) ) + p2*( (t - t1)/(t2 - t1 + 1e-30) ));
            Vec2 A3 = (p2*( (t3 - t)/(t3 - t2 + 1e-30) ) + p3*( (t - t2)/(t3 - t2 + 1e-30) ));
            Vec2 B1 = (A1*( (t2 - t)/(t2 - t0 + 1e-30) ) + A2*( (t - t0)/(t2 - t0 + 1e-30) ));
            Vec2 B2 = (A2*( (t3 - t)/(t3 - t1 + 1e-30) ) + A3*( (t - t1)/(t3 - t1 + 1e-30) ));
            Vec2 C  = (B1*( (t2 - t)/(t2 - t1 + 1e-30) ) + B2*( (t - t1)/(t2 - t1 + 1e-30) ));
            out.push_back(C);
        }
    }
    // remove near-duplicates
    vector<Vec2> cleaned; cleaned.reserve(out.size());
    for(const auto& pnt: out){
        if(cleaned.empty() || norm(pnt - cleaned.back())>1e-6) cleaned.push_back(pnt);
    }
    return cleaned;
}

// ---------------- IO: read CSV with x,y per line ----------------
static vector<Vec2> loadCSV(const string& path){
    vector<Vec2> pts;
    std::ifstream fin(path);
    if(!fin) return pts;
    string line;
    while(std::getline(fin,line)){
        if(line.empty()) continue;
        std::replace(line.begin(), line.end(), ';', ' ');
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        double x,y;
        if( (iss>>x>>y) ){
            pts.push_back({x,y});
        }
    }
    return pts;
}

// If no input given, build a small oval inner/outer
static void buildSample(vector<Vec2>& inner, vector<Vec2>& outer){
    int N=120; double a=30, b=15, w=3;
    for(int i=0;i<N;i++){
        double th = 2.0*M_PI*i/N;
        Vec2 c = { a*std::cos(th), b*std::sin(th) };
        Vec2 t = { -a*std::sin(th),  b*std::cos(th) };
        Vec2 nL= normalL(t); nL = nL/(norm(nL)+1e-30);
        inner.push_back(c - nL*w);
        outer.push_back(c + nL*w);
    }
}

// ---------------- MAIN ----------------
int main(int argc, char** argv){
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // 1) Load inner/outer points
    vector<Vec2> inner, outer;
    if(argc>=3){
        inner = loadCSV(argv[1]);
        outer = loadCSV(argv[2]);
        if(inner.empty() || outer.empty()){
            cerr<<"[Warn] Failed to read CSVs or empty. Falling back to sample.\n";
            inner.clear(); outer.clear();
            buildSample(inner, outer);
        }
    } else {
        cerr<<"[Info] No CSV specified. Using built-in sample oval.\n";
        buildSample(inner, outer);
    }

    // 2) Merge all points for DT
    vector<Vec2> all = inner;
    all.insert(all.end(), outer.begin(), outer.end());

    // 3) Delaunay triangulation (Bowyer-Watson)
    auto tris = bowyerWatson(all, CFG.superScale);

    // 4) Region constraints: keep only triangles in (outer inside ∧ inner outside), no boundary crossing
    auto innerE = ringEdges(inner);
    auto outerE = ringEdges(outer);
    vector<Triangle> kept;
    kept.reserve(tris.size());
    for(const auto& T: tris){
        Vec2 A=all[T.a], B=all[T.b], C=all[T.c];
        if(triangleOK(A,B,C, inner,outer, innerE,outerE)){
            kept.push_back(T);
        }
    }

    // 5) Label inner/outer, extract internal edge midpoints
    vector<int> label = labelPoints(all, inner, outer);
    auto mids = extractInternalMidpoints(all, kept, label);
    if(mids.size()<4){
        cerr<<"[Error] Not enough internal edges to form centerline.\n";
        return 1;
    }

    // 6) Order midpoints (MST longest path) and spline → CENTERLINE
    auto ordered = orderByMSTLongestPath(mids);
    auto center  = catmullRom(ordered, CFG.centerline_samples);

    // 7) Output CENTERLINE CSV (x,y per line)
    for(const auto& p: center){
        cout<<p.x<<","<<p.y<<"\n";
    }
    return 0;
}
