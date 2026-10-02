#include "mercury_cuda.h"
#include "mercury_cuda_memory.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int THREADS=256;
constexpr double K2=2.959122082855911e-4;
constexpr double C_PN=299792458.0*86400.0/1.495978707e11;
// Preserve the original Fortran assignment c = 173.1 (default REAL literal).
constexpr double C_PR=static_cast<double>(173.1f);
struct Config { int n,nbig,nmass,ngflag; double mu,j2,j4,j6; } cfg;
bool pn_enabled=false;
int algorithm=2;
int capacity=0,event_capacity=0;
double *x=nullptr,*v=nullptr,*oldx=nullptr,*oldv=nullptr;
double *wx=nullptr,*wv=nullptr,*ex=nullptr,*ev=nullptr,*acc=nullptr,*acc0=nullptr;
double *table=nullptr,*scale=nullptr,*partial=nullptr,*maximum=nullptr;
double *mass=nullptr,*ngf=nullptr,*rce=nullptr,*rphys=nullptr,*boxes=nullptr;
double *indirect=nullptr,*transfer=nullptr;
int *fault=nullptr,*event_count=nullptr;
MercuryEvent *device_events=nullptr;
std::vector<MercuryEvent> host_events;
std::vector<void*> allocations;
int64_t nforces=0;
double* critical=nullptr;
bool encounter_mode=false;
int *pair_i=nullptr,*pair_j=nullptr,pair_count=0,pair_capacity=0;
void encounter_force(const double*,const double*,double*);

void check(cudaError_t code) {
    if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
template<class T> void allocate(T*& p,size_t count) {
    check(cudaMalloc(reinterpret_cast<void**>(&p),sizeof(T)*count));
    allocations.push_back(p);
}
int error(const std::exception& e) {
    std::fprintf(stderr,"MERCURY CUDA: %s\n",e.what());
    return 1;
}
int blocks(int n) { return (n+THREADS-1)/THREADS; }
void release(void* p) {
    auto it=std::find(allocations.begin(),allocations.end(),p);
    if(it==allocations.end()) return;
    allocations.erase(it); check(cudaFree(p));
}
struct MaxOp { __device__ double operator()(double a,double b) const { return fmax(a,b); } };
struct SumOp { __device__ double operator()(double a,double b) const { return a+b; } };
// Fixed-order tree reduction across one THREADS-wide block; every thread
// receives the result. All threads of the block must call it.
template<class Op> __device__ double block_reduce(double value,Op op) {
    __shared__ double values[THREADS];
    values[threadIdx.x]=value; __syncthreads();
    for(int s=THREADS/2;s;s/=2) { if(threadIdx.x<s) values[threadIdx.x]=op(values[threadIdx.x],values[threadIdx.x+s]); __syncthreads(); }
    double result=values[0]; __syncthreads();
    return result;
}

__global__ void layout(int n,const double* src,double* dst,int components,int to_soa) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=n) return;
    for(int k=0;k<components;k++) {
        if(to_soa) dst[k*n+j]=src[components*j+k];
        else dst[components*j+k]=src[k*n+j];
    }
}
// Newtonian J2/J4/J6: Murray & Dermott (1999), Solar System Dynamics,
// https://doi.org/10.1017/CBO9781139174817; matches Fortran mfo_obl.
// Config stores Jn*Rcentral^n, not bare dimensionless Jn.
__device__ void obl(Config c,const double r[3],double inv,double a[3]) {
    double inv2=inv*inv,u2=r[2]*r[2]*inv2,u4=u2*u2,u6=u4*u2;
    double j2=c.j2*inv2,j4=c.j4*inv2*inv2,j6=c.j6*inv2*inv2*inv2;
    double s=c.mu*inv2*inv;
    double f=j2*(7.5*u2-1.5)+j4*(39.375*u4-26.25*u2+1.875)
        +j6*(187.6875*u6-216.5625*u4+59.0625*u2-2.1875);
    double g=j2*3.0+j4*(17.5*u2-7.5)+j6*(86.625*u4-78.75*u2+13.125);
    a[0]=r[0]*s*f; a[1]=r[1]*s*f; a[2]=r[2]*s*(f-g);
}
__global__ void indirect_force(Config c,const double* pos,const double* m,double* result) {
    double out[3]={0,0,0};
    for(int j=1+threadIdx.x;j<c.nmass;j+=THREADS) {
        if(m[j]==0) continue;
        double r[3]={pos[j],pos[c.n+j],pos[2*c.n+j]};
        double inv=1.0/sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        for(int k=0;k<3;k++) out[k]-=m[j]*inv*inv*inv*r[k];
        if(c.j2!=0||c.j4!=0||c.j6!=0) {
            double a[3]; obl(c,r,inv,a);
            for(int k=0;k<3;k++) out[k]+=m[j]/c.mu*a[k];
        }
    }
    for(int k=0;k<3;k++) { double sum=block_reduce(out[k],SumOp()); if(threadIdx.x==0) result[k]=sum; }
}

__device__ void nongrav(Config c,int j,const double r[3],double r2,double inv,
    const double u[3],double rv,const double* m,const double* ng,double a[3],int* bad) {
    // A1/A2/A3: Marsden, Sekanina & Yeomans (1973), AJ 78, 211-225.
    // https://doi.org/10.1086/111402; original cometary g(r) and cutoff.
    // YAR: Farnocchia et al. (2013), Icarus 224, 1-13, distance exponent 2.
    // https://doi.org/10.1016/j.icarus.2013.02.004
    // a_Y = yar*(1 AU/r)^2*unit(v - (r.v)/r^2*r).
    if(c.ngflag==1||c.ngflag==3) {
        double a1=ng[j],a2=ng[c.n+j],a3=ng[2*c.n+j],yar=ng[4*c.n+j];
        double t[3],norm2=0;
        for(int k=0;k<3;k++) { t[k]=u[k]-(rv/r2)*r[k]; norm2+=t[k]*t[k]; }
        if((a1!=0||a2!=0||a3!=0)&&(r2<88.0||fabs(a1)>1e-7||fabs(a2)>1e-7||fabs(a3)>1e-7)) {
            double q=sqrt(r2)*.3561253561253561;
            double g=.111262*pow(q,-2.15)*pow(1.0+pow(q,5.093),-4.6142);
            double normal[3]={r[1]*u[2]-r[2]*u[1],r[2]*u[0]-r[0]*u[2],r[0]*u[1]-r[1]*u[0]};
            double nn=sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
            if(a3!=0&&nn==0) { atomicExch(bad,1); return; }
            if(a2!=0&&!(norm2>0)) { atomicExch(bad,1); return; }
            for(int k=0;k<3;k++) a[k]+=a1*g*inv*r[k]+(a2==0?0:a2*g/sqrt(norm2)*t[k])
                +(a3==0?0:a3*g/nn*normal[k]);
        }
        if(yar!=0) {
            if(!(norm2>0)) { atomicExch(bad,1); return; }
            double f=yar/(r2*sqrt(norm2));
            for(int k=0;k<3;k++) a[k]+=f*t[k];
        }
    }
    // Radiation pressure/PR background: Burns, Lamy & Soter (1979),
    // Icarus 40, 1-48; https://doi.org/10.1016/0019-1035(79)90050-2.
    // Standard vector force is beta*mu/r^2*((1-vr/c)*r_hat-v/c).
    // Compatibility: match supplied mfo_pr exactly, NOT that standard
    // formula: component-wise Vt, sw=0.3, C_PR default REAL, K2 and
    // massless-only gating. Includes radiation pressure as well as drag.
    if((c.ngflag==2||c.ngflag==3)&&m[j]==0) {
        double radius=sqrt(r2),vr=rv/radius;
        double f=K2*ng[3*c.n+j]/(C_PR*radius*radius);
        for(int k=0;k<3;k++) {
            double vt=u[k]*(1.0-r[k]/radius);
            a[k]+=f*((C_PR-1.3*2.0*vr)*r[k]/radius-1.3*vt);
        }
    }
}

template<bool PN> __global__ void force_kernel(Config c,const double* pos,
    const double* vel,const double* m,const double* ng,const double* ind,
    double* out,int* bad) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=c.n) return;
    if(j==0) { for(int k=0;k<3;k++) out[k*c.n]=0; return; }
    double r[3]={pos[j],pos[c.n+j],pos[2*c.n+j]};
    double r2=r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    if(!(r2>0)) { atomicExch(bad,1); return; }
    double inv=1.0/sqrt(r2),inv3=inv*inv*inv;
    double a[3]={0,0,0};
    // Big bodies feel small-body back-reaction. Small bodies never feel
    // other small bodies, matching mfo_grav (including semi-active inputs).
    for(int i=1;i<(j<c.nbig?c.nmass:c.nbig);i++) {
        if(i==j||m[i]==0) continue;
        double dx=pos[i]-r[0],dy=pos[c.n+i]-r[1],dz=pos[2*c.n+i]-r[2];
        double d2=dx*dx+dy*dy+dz*dz;
        if(!(d2>0)) { atomicExch(bad,1); return; }
        double di=1.0/sqrt(d2),f=m[i]*di*di*di;
        a[0]+=f*dx; a[1]+=f*dy; a[2]+=f*dz;
    }
    for(int k=0;k<3;k++) a[k]+=ind[k]-c.mu*inv3*r[k];
    if(c.j2!=0||c.j4!=0||c.j6!=0) {
        double ao[3]; obl(c,r,inv,ao);
        for(int k=0;k<3;k++) a[k]+=ao[k];
    }
    // Compile-time specialization removes all PN-only work when disabled.
    double u[3]={0,0,0},rv=0;
    if(PN||c.ngflag) {
        for(int k=0;k<3;k++) { u[k]=vel[k*c.n+j]; rv+=r[k]*u[k]; }
    }
    nongrav(c,j,r,r2,inv,u,rv,m,ng,a,bad);
    // 1PN formula: Will (2014), Living Rev. Relativity 17, 4,
    // Eq. (79), eta->0 with G,c restored; central-mass test-body limit.
    // https://doi.org/10.12942/lrr-2014-4
    // Approximation background: Tamayo, Rein, Shi & Hernandez (2020),
    // MNRAS 491, 2885-2901, Appendix B; matches mfo_pn/mfo_grav.
    // https://doi.org/10.1093/mnras/stz2870
    // mu/(c^2*r^3)*((4*mu/r-v^2)*r + 4*(r.v)*v); heliocentric
    // physical velocities, no planetary PN cross terms/spin/higher orders.
    if(PN) {
        double v2=u[0]*u[0]+u[1]*u[1]+u[2]*u[2];
        double f=c.mu*inv3/(C_PN*C_PN),radial=4.0*c.mu*inv-v2;
        for(int k=0;k<3;k++) a[k]+=f*(radial*r[k]+4.0*rv*u[k]);
    }
    for(int k=0;k<3;k++) {
        out[k*c.n+j]=a[k];
        if(!isfinite(a[k])) atomicExch(bad,1);
    }
}
void force(const double* xx,const double* vv,double* aa) {
    if(encounter_mode) { encounter_force(xx,vv,aa); return; }
    indirect_force<<<1,THREADS>>>(cfg,xx,mass,indirect);
    if(pn_enabled) force_kernel<true><<<blocks(cfg.n),THREADS>>>(cfg,xx,vv,mass,ngf,indirect,aa,fault);
    else force_kernel<false><<<blocks(cfg.n),THREADS>>>(cfg,xx,vv,mass,ngf,indirect,aa,fault);
    check(cudaGetLastError());
    ++nforces;
}
__global__ void begin_step(int n,const double* xx,const double* vv,double* ox,double* ov,double* scales) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=n) return;
    double r2=0,v2=0;
    for(int k=0;k<3;k++) { int z=k*n+j; ox[z]=xx[z]; ov[z]=vv[z]; r2+=xx[z]*xx[z]; v2+=vv[z]*vv[z]; }
    scales[j]=1.0/fmax(r2,1e-300); scales[n+j]=1.0/fmax(v2,1e-300);
}
__global__ void midpoint(int n,double h,const double* base_x,const double* base_v,
    const double* use_v,const double* aa,double* dst_x,double* dst_v) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=3*n) return;
    dst_x[j]=base_x[j]+h*use_v[j]; dst_v[j]=base_v[j]+h*aa[j];
}
__global__ void endpoint(int n,int level,double h,const double* xx,const double* vv,
    const double* xe,const double* ve,const double* aa,double* d) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=3*n) return;
    d[(level*6)*n+j]=.5*(xe[j]+xx[j]+h*ve[j]);
    d[(level*6+3)*n+j]=.5*(ve[j]+vv[j]+h*aa[j]);
}
__global__ void extrapolate(int n,int level,int col,double f1,double f2,double* d) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=6*n) return;
    d[col*6*n+j]=f1*d[(col+1)*6*n+j]-f2*d[col*6*n+j];
}
__global__ void estimate_error(int n,const double* d,const double* scales,const int* bad,double* p) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    double e=0;
    if(j>0&&j<n) for(int k=0;k<6;k++) e=fmax(e,d[k*n+j]*d[k*n+j]*scales[(k/3)*n+j]);
    if(*bad) e=INFINITY;
    e=block_reduce(e,MaxOp());
    if(threadIdx.x==0) p[blockIdx.x]=e;
}
__global__ void reduce_max(int n,const double* p,double* result) {
    double e=0;
    for(int j=threadIdx.x;j<n;j+=THREADS) e=fmax(e,p[j]);
    e=block_reduce(e,MaxOp());
    if(threadIdx.x==0) *result=e;
}
__global__ void accept(int n,int levels,const double* d,double* xx,double* vv) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=3*n) return;
    double a=d[j],b=d[3*n+j];
    for(int k=1;k<levels;k++) { a+=d[k*6*n+j]; b+=d[(k*6+3)*n+j]; }
    xx[j]=a; vv[j]=b;
}

#include "mercury_cuda_adaptive.cuh"
#include "mercury_cuda_radau.cuh"
#include "mercury_cuda_symplectic.cuh"

__device__ void minimum(double d0,double d1,double v0,double v1,double h,double& d,double& t) {
    if(v0*h>0||v1*h<0) { d=fmin(d0,d1); t=d0<=d1?-h:0; return; }
    double temp=6.0*(d0-d1),a=temp+3.0*h*(v0+v1),b=temp+2.0*h*(v0+2.0*v1),c=h*v1;
    temp=-.5*(b+copysign(sqrt(fmax(b*b-4.0*a*c,0.0)),b));
    double tau=temp==0?0:c/temp;
    tau=fmax(-1.0,fmin(tau,0.0)); t=tau*h;
    temp=1.0+tau;
    d=fmax(0.0,tau*tau*((3.0+2.0*tau)*d0+temp*h*v0)+temp*temp*((1.0-2.0*tau)*d1+tau*h*v1));
}
// MCE_BOX plus expansion: MCE_STAT pads every body by 1.2*rce, while
// MCE_SNIF pads only Big bodies by rcrit.
__global__ void bounding_boxes(int n,int nbig,double big_pad,double small_pad,double h,
    const double* ox,const double* ov,const double* xx,const double* vv,const double* limits,double* bb) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=n) return;
    double pad=(j<nbig?big_pad:small_pad)*limits[j];
    for(int k=0;k<2;k++) {
        int z=k*n+j; double lo=fmin(ox[z],xx[z]),hi=fmax(ox[z],xx[z]);
        if((ov[z]<0&&vv[z]>0)||(ov[z]>0&&vv[z]<0)) {
            double tmp=(ov[z]*xx[z]-vv[z]*ox[z]-.5*h*ov[z]*vv[z])/(ov[z]-vv[z]);
            lo=fmin(lo,tmp); hi=fmax(hi,tmp);
        }
        bb[2*k*n+j]=lo-pad; bb[(2*k+1)*n+j]=hi+pad;
    }
}
__global__ void pair_events(Config c,double time,double h,const double* ox,const double* ov,
    const double* xx,const double* vv,const double* m,const double* limits,const double* radii,
    const double* bb,int cap,int* count,MercuryEvent* out) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<2||j>=c.n) return;
    for(int i=1;i<j&&i<c.nbig;i++) {
        if(bb[c.n+i]<bb[j]||bb[c.n+j]<bb[i]||bb[3*c.n+i]<bb[2*c.n+j]||bb[3*c.n+j]<bb[2*c.n+i]) continue;
        double d0=0,d1=0,dt0=0,dt1=0;
        for(int k=0;k<3;k++) {
            int a=k*c.n+i,b=k*c.n+j;
            double dx0=ox[a]-ox[b],dx1=xx[a]-xx[b];
            d0+=dx0*dx0; d1+=dx1*dx1;
            dt0+=dx0*(ov[a]-ov[b])*2.0; dt1+=dx1*(vv[a]-vv[b])*2.0;
        }
        double d,t; minimum(d0,d1,dt0,dt1,h,d,t);
        double close=fmax(limits[i],limits[j]),hit=radii[i]+radii[j];
        bool clo=(d<=close*close&&dt0*h<=0&&dt1*h>=0)||d<=hit*hit;
        bool near=d<=4.0*hit*hit;
        if(!clo&&!near) continue;
        int slot=atomicAdd(count,1); if(slot>=cap) continue;
        MercuryEvent e{}; int a=i,b=j;
        if(j<c.nbig&&m[j]>m[i]) { a=j; b=i; }
        e.i=a+1; e.j=b+1; e.kind=(clo?1:0)|(near?2:0)|(d<=hit*hit?4:0);
        e.now=d1<=hit*hit; e.time=time+t; e.distance=sqrt(d);
        double w0=-t/h,w1=1.0+t/h;
        for(int k=0;k<3;k++) {
            e.xi[k]=w0*ox[k*c.n+a]+w1*xx[k*c.n+a];
            e.xj[k]=w0*ox[k*c.n+b]+w1*xx[k*c.n+b];
            e.xi[k+3]=w0*ov[k*c.n+a]+w1*vv[k*c.n+a];
            e.xj[k+3]=w0*ov[k*c.n+b]+w1*vv[k*c.n+b];
        }
        out[slot]=e;
    }
}
__global__ void central_events(Config c,double time,double h,double radius,const double* ox,
    const double* ov,const double* xx,const double* vv,const double* m,int cap,int* count,MercuryEvent* out) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<1||j>=c.n) return;
    double r[3],u[3],r0sq=0,r1sq=0,rv0=0,rv1=0,v2=0;
    for(int k=0;k<3;k++) {
        int z=k*c.n+j; r[k]=ox[z]; u[k]=ov[z];
        r0sq+=r[k]*r[k]; r1sq+=xx[z]*xx[z]; rv0+=r[k]*u[k]; rv1+=xx[z]*vv[z]; v2+=u[k]*u[k];
    }
    if(!(rv0*h<=0&&rv1*h>=0)&&fmin(r0sq,r1sq)>radius*radius) return;
    double cross[3]={r[1]*u[2]-r[2]*u[1],r[2]*u[0]-r[0]*u[2],r[0]*u[1]-r[1]*u[0]};
    double mu=c.mu+m[j],p=(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2])/mu,r0=sqrt(r0sq);
    double e=sqrt(fmax(0.0,1.0+p*(v2/mu-2.0/r0))),q=p/(1.0+e);
    if(q>radius) return;
    double dt=0;
    if(e>0&&e<1) {
        double a=q/(1.0-e);
        double hit=copysign(acos(fmax(-1.0,fmin(1.0,(1.0-radius/a)/e))),-h);
        double init=copysign(acos(fmax(-1.0,fmin(1.0,(1.0-r0/a)/e))),rv0);
        double mhit=hit-e*sin(hit),m0=init-e*sin(init);
        dt=(mhit-m0)/sqrt(mu/(a*a*a));
    } else if(e>1) {
        double a=q/(e-1.0);
        double hit=copysign(acosh(fmax(1.0,(1.0+radius/a)/e)),-h);
        double init=copysign(acosh(fmax(1.0,(1.0+r0/a)/e)),rv0);
        dt=((e*sinh(hit)-hit)-(e*sinh(init)-init))/sqrt(mu/(a*a*a));
    } else if(e==1&&q>0) {
        double hit=copysign(sqrt(fmax(0.0,radius/q-1.0)),-h),init=copysign(sqrt(fmax(0.0,r0/q-1.0)),rv0);
        dt=sqrt(2.0*q*q*q/mu)*(hit+hit*hit*hit/3.0-init-init*init*init/3.0);
    }
    if(q==0&&r1sq!=r0sq) dt=h*fmax(0.0,fmin(1.0,(r0-radius)/(r0-sqrt(r1sq))));
    int slot=atomicAdd(count,1); if(slot>=cap) return;
    MercuryEvent result{}; result.i=1; result.j=j+1; result.kind=8;
    result.time=time-h+dt; result.distance=radius;
    out[slot]=result;
}
void reserve_events(int count) {
    if(count<=event_capacity) return;
    if(device_events) check(cudaFree(device_events));
    device_events=nullptr; event_capacity=0;
    check(cudaMalloc(reinterpret_cast<void**>(&device_events),sizeof(MercuryEvent)*size_t(count)));
    event_capacity=count;
}
void upload_array(const double* source,double* target,int components) {
    check(cudaMemcpy(transfer,source,size_t(components)*cfg.n*sizeof(double),cudaMemcpyHostToDevice));
    layout<<<blocks(cfg.n),THREADS>>>(cfg.n,transfer,target,components,1);
}
void download_array(const double* source,double* target) {
    layout<<<blocks(cfg.n),THREADS>>>(cfg.n,source,transfer,3,0);
    check(cudaMemcpy(target,transfer,size_t(3)*cfg.n*sizeof(double),cudaMemcpyDeviceToHost));
}
#include "mercury_cuda_hybrid.cuh"
} // namespace

extern "C" int mercury_cuda_available() {
    int n=0; return cudaGetDeviceCount(&n)==cudaSuccess&&n>0;
}
extern "C" void mercury_cuda_free() {
    free_other_context();
    pair_capacity=0; encounter_mode=false;
    for(void* p:allocations) cudaFree(p);
    allocations.clear();
    if(device_events) cudaFree(device_events);
    device_events=nullptr; capacity=0; event_capacity=0; host_events.clear();
}
extern "C" int mercury_cuda_configure(int method) {
    if(method!=1&&method!=2&&method!=3&&method!=4&&method!=9&&method!=10) return 1;
    if(method!=algorithm) mercury_cuda_free();
    algorithm=method;
    try { if(method==4) initialize_radau(); }
    catch(const std::exception& e) { return error(e); }
    ra_reset=true; sym_reset=true; return 0;
}
extern "C" void mercury_cuda_reset(int flag) { if(flag!=2) { ra_reset=true; sym_reset=true; } }
extern "C" int mercury_cuda_upload(int n,int nbig,int pn,int ngflag,const double* m,
    const double* xx,const double* vv,const double* ng,const double* jcen,
    const double* limits,const double* radii) {
    try {
        if(n>capacity) {
            mercury_cuda_free();
            size_t free_bytes,total_bytes; check(cudaMemGetInfo(&free_bytes,&total_bytes));
            if(size_t(n)*(algorithm==2?800:1100)>free_bytes*8/10) throw std::runtime_error("Insufficient device workspace memory");
            for(double** p:{&x,&v,&oldx,&oldv,&wx,&wv,&ex,&ev,&acc,&acc0}) allocate(*p,size_t(3)*n);
            allocate(table,size_t(algorithm==3?72:algorithm==4?63:48)*n); allocate(scale,size_t(2)*n);
            allocate(mass,n); allocate(ngf,MercuryForceComponents*n); allocate(rce,n); allocate(rphys,n);
            allocate(boxes,size_t(4)*n); allocate(transfer,size_t(6)*n);
            allocate(partial,blocks(n)); allocate(maximum,1); allocate(indirect,3);
            allocate(fault,1); allocate(event_count,1); reserve_events(4096);
            if(algorithm==1||algorithm==9||algorithm==10) allocate(sym,size_t(18)*n);
            if(algorithm==3||algorithm==10) allocate(critical,n);
            if(algorithm==10) allocate(selected_device,n);
            capacity=n;
        }
        ra_reset=true; sym_reset=true;
        int nmass=nbig; for(int j=nbig;j<n;j++) if(m[j]!=0) nmass=j+1;
        cfg={n,nbig,nmass,ngflag,m[0],jcen[0],jcen[1],jcen[2]}; pn_enabled=pn!=0;
        check(cudaMemcpy(mass,m,size_t(n)*sizeof(double),cudaMemcpyHostToDevice));
        check(cudaMemcpy(rce,limits,size_t(n)*sizeof(double),cudaMemcpyHostToDevice));
        check(cudaMemcpy(rphys,radii,size_t(n)*sizeof(double),cudaMemcpyHostToDevice));
        upload_array(xx,x,3); upload_array(vv,v,3); upload_array(ng,ngf,MercuryForceComponents);
        check(cudaMemset(fault,0,sizeof(int))); check(cudaDeviceSynchronize());
        return 0;
    } catch(const std::exception& e) { return error(e); }
}
extern "C" int mercury_cuda_step(double time,double* h,double* hdid,double tol,int64_t* forces,int64_t* rejected) {
    try {
        int n=cfg.n; int64_t before=nforces; *rejected=0;
        check(cudaMemset(fault,0,sizeof(int)));
        if(algorithm==1||algorithm==9) {
            mvs_step(*h); *hdid=*h; *forces=nforces-before; return 0;
        }
        if(algorithm==4) {
            radau_step(time,h,hdid,tol,rejected);
            *forces=nforces-before; return 0;
        }
        if(algorithm==3) {
            bs2_step(time,h,hdid,tol,rejected);
            *forces=nforces-before; return 0;
        }
        begin_step<<<blocks(n),THREADS>>>(n,x,v,oldx,oldv,scale);
        force(oldx,oldv,acc0);
        for(;;) {
            if(time+*h==time) throw std::runtime_error("BS step cannot advance time");
            for(int level=0;level<8;level++) {
                int order=level+1; double hs=*h/(2.0*order);
                midpoint<<<blocks(3*n),THREADS>>>(n,hs,oldx,oldv,oldv,acc0,wx,wv);
                force(wx,wv,acc);
                midpoint<<<blocks(3*n),THREADS>>>(n,2*hs,oldx,oldv,wv,acc,ex,ev);
                for(int sub=2;sub<=order;sub++) {
                    force(ex,ev,acc);
                    midpoint<<<blocks(3*n),THREADS>>>(n,2*hs,wx,wv,ev,acc,wx,wv);
                    force(wx,wv,acc);
                    midpoint<<<blocks(3*n),THREADS>>>(n,2*hs,ex,ev,wv,acc,ex,ev);
                }
                force(ex,ev,acc);
                endpoint<<<blocks(3*n),THREADS>>>(n,level,hs,wx,wv,ex,ev,acc,table);
                double hn=.25/(order*order);
                for(int col=level-1;col>=0;col--) {
                    double hj=.25/((col+1)*(col+1)),hj1=.25/((col+2)*(col+2));
                    double inv=1.0/(hj-hn);
                    extrapolate<<<blocks(6*n),THREADS>>>(n,level,col,inv*hj1,inv*hn,table);
                }
                if(order>3) {
                    estimate_error<<<blocks(n),THREADS>>>(n,table,scale,fault,partial);
                    reduce_max<<<1,THREADS>>>(blocks(n),partial,maximum);
                    double err; check(cudaMemcpy(&err,maximum,sizeof(double),cudaMemcpyDeviceToHost));
                    if(!std::isfinite(err)) throw std::runtime_error("Nonfinite force or undefined orbit direction");
                    if(err<=tol*tol) {
                        accept<<<blocks(3*n),THREADS>>>(n,order,table,x,v);
                        check(cudaGetLastError()); *hdid=*h;
                        if(order==8) *h*=.55;
                        if(order<7) *h*=1.3;
                        *forces=nforces-before; return 0;
                    }
                }
            }
            *h*=.5; ++*rejected;
        }
    } catch(const std::exception& e) { return error(e); }
}
extern "C" int mercury_cuda_download(double* xx,double* vv,int previous) {
    try { download_array(previous?oldx:x,xx); download_array(previous?oldv:v,vv); return 0; }
    catch(const std::exception& e) { return error(e); }
}
extern "C" int mercury_cuda_events(double time,double h,double radius,const MercuryEvent** result,int* count) {
    try {
        bool mvs=algorithm==1||algorithm==9;
        const double* central_v=v;
        if(algorithm==10) { physical_velocity(v,sym+15*cfg.n); central_v=sym+15*cfg.n; }
        const double *px=mvs?sym:oldx,*pv=mvs?sym+3*cfg.n:oldv,*fv=mvs?sym+6*cfg.n:v;
        if(algorithm!=10) bounding_boxes<<<blocks(cfg.n),THREADS>>>(cfg.n,cfg.nbig,1.2,1.2,h,px,pv,x,fv,rce,boxes);
        for(;;) {
            check(cudaMemset(event_count,0,sizeof(int)));
            if(algorithm!=10) pair_events<<<blocks(cfg.n),THREADS>>>(cfg,time,h,px,pv,x,fv,mass,rce,rphys,boxes,event_capacity,event_count,device_events);
            central_events<<<blocks(cfg.n),THREADS>>>(cfg,time,h,radius,oldx,oldv,x,central_v,mass,event_capacity,event_count,device_events);
            check(cudaMemcpy(count,event_count,sizeof(int),cudaMemcpyDeviceToHost));
            if(*count<=event_capacity) break;
            reserve_events(*count); // repeat screening after growth; nothing is dropped
        }
        host_events.resize(*count);
        if(*count) check(cudaMemcpy(host_events.data(),device_events,sizeof(MercuryEvent)*size_t(*count),cudaMemcpyDeviceToHost));
        std::sort(host_events.begin(),host_events.end(),[](const MercuryEvent& a,const MercuryEvent& b) {
            int ai=std::min(a.i,a.j),aj=std::max(a.i,a.j),bi=std::min(b.i,b.j),bj=std::max(b.i,b.j);
            return ai<bi||(ai==bi&&aj<bj);
        });
        *result=host_events.data(); return 0;
    } catch(const std::exception& e) { return error(e); }
}
extern "C" int mercury_cuda_force(double* aa) {
    try {
        check(cudaMemset(fault,0,sizeof(int))); force(x,v,acc);
        int bad; check(cudaMemcpy(&bad,fault,sizeof(int),cudaMemcpyDeviceToHost));
        if(bad) throw std::runtime_error("Nonfinite force or undefined orbit direction");
        download_array(acc,aa); return 0;
    } catch(const std::exception& e) { return error(e); }
}

extern "C" int mercury_cuda_export(double h,int physical,double* xx,double* vv) {
    try {
        if(physical&&algorithm==10) {
            physical_velocity(v,sym+15*cfg.n);
            download_array(x,xx); download_array(sym+15*cfg.n,vv);
        } else if(physical&&algorithm==1) {
            copy_vector(sym+12*cfg.n,x); copy_vector(sym+15*cfg.n,v);
            corrector(sym+12*cfg.n,sym+15*cfg.n,h,false);
            download_array(sym+12*cfg.n,xx); download_array(sym+15*cfg.n,vv);
            int bad; check(cudaMemcpy(&bad,fault,sizeof(int),cudaMemcpyDeviceToHost));
            if(bad) throw std::runtime_error("Output corrector failed");
        } else { download_array(x,xx); download_array(v,vv); }
        return 0;
    } catch(const std::exception& e) { return error(e); }
}

extern "C" int mercury_cuda_hybrid_begin(double h,const double* crit,int flag,int cap,
    int* ce,int* count,int* pi,int* pj,double* xx,double* vv,int64_t* forces) {
    try {
        int64_t before=nforces; int n=cfg.n;
        check(cudaMemset(fault,0,sizeof(int)));
        if(flag!=2||sym_reset) {
            check(cudaMemcpy(critical,crit,n*sizeof(double),cudaMemcpyHostToDevice));
            regular_hybrid_force();
        }
        copy_vector(oldx,x); physical_velocity(v,oldv);
        kick<<<blocks(3*n),THREADS>>>(n,h*.5,sym+9*n,v);
        solar_drift(h*.5); copy_vector(sym,x); copy_vector(sym+3*n,v);
        kepler_drift<<<blocks(n),THREADS>>>(cfg,h,mass,x,v,wx,wv,0);
        bounding_boxes<<<blocks(n),THREADS>>>(n,cfg.nbig,1.0,0.0,h,sym,sym+3*n,x,v,critical,boxes);
        for(;;) {
            check(cudaMemset(event_count,0,sizeof(int)));
            sniff<<<blocks(n),THREADS>>>(cfg,h,sym,sym+3*n,x,v,critical,boxes,event_capacity,event_count,device_events);
            check(cudaMemcpy(count,event_count,sizeof(int),cudaMemcpyDeviceToHost));
            if(*count<=event_capacity) break;
            reserve_events(*count);
        }
        if(*count>cap) throw std::runtime_error("Hybrid encounter capacity exceeded");
        host_events.resize(*count);
        if(*count) check(cudaMemcpy(host_events.data(),device_events,*count*sizeof(MercuryEvent),cudaMemcpyDeviceToHost));
        std::sort(host_events.begin(),host_events.end(),[](const MercuryEvent& a,const MercuryEvent& b){return a.i<b.i||(a.i==b.i&&a.j<b.j);});
        selected.clear();
        if(*count) {
            std::fill(ce,ce+n,0);
            for(int k=0;k<*count;k++) { auto e=host_events[k]; pi[k]=e.i+1; pj[k]=e.j+1; ce[e.i]=ce[e.j]=2; }
            for(int j=1;j<n;j++) if(ce[j]) selected.push_back(j);
            int ns=selected.size(); packed.resize(6*ns);
            check(cudaMemcpy(selected_device,selected.data(),ns*sizeof(int),cudaMemcpyHostToDevice));
            selected_states<<<blocks(ns),THREADS>>>(n,ns,selected_device,x,v,transfer,0,x,v,sym,sym+3*n);
            check(cudaMemcpy(packed.data(),transfer,packed.size()*sizeof(double),cudaMemcpyDeviceToHost));
            for(int a=0;a<ns;a++) for(int k=0;k<3;k++) { xx[3*selected[a]+k]=packed[6*a+k]; vv[3*selected[a]+k]=packed[6*a+3+k]; }
        }
        int bad; check(cudaMemcpy(&bad,fault,sizeof(int),cudaMemcpyDeviceToHost));
        if(bad) throw std::runtime_error("Hybrid force failed");
        *forces=nforces-before; return 0;
    } catch(const std::exception& e) { return error(e); }
}
extern "C" int mercury_cuda_hybrid_finish(double h,const double* m,const double* xx,const double* vv,int64_t* forces) {
    try {
        int64_t before=nforces; int ns=selected.size(),n=cfg.n;
        if(ns) {
            for(int a=0;a<ns;a++) for(int k=0;k<3;k++) { packed[6*a+k]=xx[3*selected[a]+k]; packed[6*a+3+k]=vv[3*selected[a]+k]; }
            check(cudaMemcpy(transfer,packed.data(),packed.size()*sizeof(double),cudaMemcpyHostToDevice));
            selected_states<<<blocks(ns),THREADS>>>(n,ns,selected_device,x,v,transfer,1,x,v,sym,sym+3*n);
            check(cudaMemcpy(mass,m,n*sizeof(double),cudaMemcpyHostToDevice));
        }
        solar_drift(h*.5); regular_hybrid_force();
        kick<<<blocks(3*n),THREADS>>>(n,h*.5,sym+9*n,v);
        sym_reset=false;
        int bad; check(cudaMemcpy(&bad,fault,sizeof(int),cudaMemcpyDeviceToHost));
        if(bad) throw std::runtime_error("Hybrid final force failed");
        *forces=nforces-before; return 0;
    } catch(const std::exception& e) { return error(e); }
}
extern "C" int mercury_cuda_encounter_enter(int n,int nb,const double* m,const double* xx,const double* vv,
    const double* crit,const double* limits,const double* radii,int np,const int* pi,const int* pj) {
    try {
        encounter_active=true; other_context.exchange();
        if(mercury_cuda_configure(3)) throw std::runtime_error("Encounter configure failed");
        auto zero=mercury_zero_force_parameters(n); double jc[3]={0,0,0};
        if(mercury_cuda_upload(n,nb,0,0,m,xx,vv,zero.data(),jc,limits,radii)) throw std::runtime_error("Encounter upload failed");
        encounter_mode=true;
        check(cudaMemcpy(critical,crit,n*sizeof(double),cudaMemcpyHostToDevice));
        if(np>pair_capacity) {
            if(pair_capacity) { release(pair_i); release(pair_j); }
            pair_capacity=0; allocate(pair_i,np); allocate(pair_j,np); pair_capacity=np;
        }
        std::vector<int> ids(np);
        for(int k=0;k<np;k++) ids[k]=pi[k]-1;
        check(cudaMemcpy(pair_i,ids.data(),np*sizeof(int),cudaMemcpyHostToDevice));
        for(int k=0;k<np;k++) ids[k]=pj[k]-1;
        check(cudaMemcpy(pair_j,ids.data(),np*sizeof(int),cudaMemcpyHostToDevice));
        pair_count=np; return 0;
    } catch(const std::exception& e) { return error(e); }
}
extern "C" int mercury_cuda_encounter_update(const double* m,const double* xx,const double* vv) {
    try {
        check(cudaMemcpy(mass,m,cfg.n*sizeof(double),cudaMemcpyHostToDevice));
        upload_array(xx,x,3); upload_array(vv,v,3); return 0;
    } catch(const std::exception& e) { return error(e); }
}
extern "C" void mercury_cuda_encounter_exit() { other_context.exchange(); encounter_active=false; }

// Periodic Hill-radius changes do not change the dynamical state or its history.
extern "C" int mercury_cuda_limits(const double* limits) {
    try {
        check(cudaMemcpy(rce,limits,cfg.n*sizeof(double),cudaMemcpyHostToDevice));
        return 0;
    } catch(const std::exception& e) { return error(e); }
}
