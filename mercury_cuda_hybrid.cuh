// Hybrid changeover Hamiltonian. The compact encounter workspace is separate
// from the resident parent system; the Fortran driver still owns collisions.
__device__ double changeover(double d2,double rc,bool inner) {
    double rc2=rc*rc;
    if(!inner&&d2<=.01*rc2) return 0;
    if(inner&&d2>=rc2) return 0;
    double inv=1/sqrt(d2),f=inv*inv*inv;
    if(d2>=rc2||d2<=.01*rc2) return f;
    double q=(1/inv-.1*rc)/(.9*rc),q2=q*q,q3=q*q2,q4=q2*q2,q5=q2*q3;
    return (inner?1-10*q3+15*q4-6*q5:10*q3-15*q4+6*q5)*f;
}
__global__ void hybrid_force(Config c,const double* m,const double* xx,const double* vv,
    const double* crit,const double* ng,const double* ind,double* aa,int* bad,
    bool inner,int pairs,const int* pi,const int* pj) {
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=c.n) return;
    if(j==0) { for(int k=0;k<3;k++) aa[k*c.n]=0; return; }
    double r[3]={xx[j],xx[c.n+j],xx[2*c.n+j]},a[3]={0,0,0};
    int limit=inner?pairs:(j<c.nbig?c.nmass:c.nbig);
    for(int p=inner?0:1;p<limit;p++) {
        int i=p;
        if(inner) { if(pi[p]==j) i=pj[p]; else if(pj[p]==j) i=pi[p]; else continue; }
        if(i==j||m[i]==0) continue;
        double dx=xx[i]-r[0],dy=xx[c.n+i]-r[1],dz=xx[2*c.n+i]-r[2];
        double f=changeover(dx*dx+dy*dy+dz*dz,fmax(crit[i],crit[j]),inner)*m[i];
        a[0]+=f*dx; a[1]+=f*dy; a[2]+=f*dz;
    }
    double r2=r[0]*r[0]+r[1]*r[1]+r[2]*r[2],inv=1/sqrt(r2);
    if(inner) { double f=c.mu*inv*inv*inv; for(int k=0;k<3;k++) a[k]-=f*r[k]; }
    else {
        if(c.j2!=0||c.j4!=0||c.j6!=0) {
            double ao[3]; obl(c,r,inv,ao); for(int k=0;k<3;k++) a[k]+=ao[k]+ind[k];
        }
        if(c.ngflag) {
            double u[3]={vv[j],vv[c.n+j],vv[2*c.n+j]},rv=r[0]*u[0]+r[1]*u[1]+r[2]*u[2];
            nongrav(c,j,r,r2,inv,u,rv,m,ng,a,bad);
        }
    }
    for(int k=0;k<3;k++) { aa[k*c.n+j]=a[k]; if(!isfinite(a[k])) atomicExch(bad,1); }
}
void encounter_force(const double* xx,const double* vv,double* aa) {
    hybrid_force<<<blocks(cfg.n),THREADS>>>(cfg,mass,xx,vv,critical,ngf,indirect,aa,fault,true,pair_count,pair_i,pair_j);
    check(cudaGetLastError()); ++nforces;
}
void regular_hybrid_force() {
    obl_reaction(x);
    hybrid_force<<<blocks(cfg.n),THREADS>>>(cfg,mass,x,v,critical,ngf,indirect,sym+9*cfg.n,fault,false,0,nullptr,nullptr);
    ++nforces;
}
__global__ void momentum(Config c,const double* m,const double* vv,double* out) {
    // Only actual massive sources are visited; massless ensembles cost O(Nbig).
    double sum[3]={0,0,0};
    for(int j=1+threadIdx.x;j<c.nmass;j+=THREADS) for(int k=0;k<3;k++) sum[k]+=m[j]*vv[k*c.n+j];
    for(int k=0;k<3;k++) { double s=block_reduce(sum[k],SumOp()); if(threadIdx.x==0) out[k]=s/c.mu; }
}
__global__ void shift_vector(int n,double h,const double* offset,const double* src,double* dst) {
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=n) return;
    for(int k=0;k<3;k++) dst[k*n+j]=j?src[k*n+j]+h*offset[k]:0;
}
void solar_drift(double h) {
    momentum<<<1,THREADS>>>(cfg,mass,v,indirect);
    shift_vector<<<blocks(cfg.n),THREADS>>>(cfg.n,h,indirect,x,x);
}
void physical_velocity(const double* src,double* dst) {
    momentum<<<1,THREADS>>>(cfg,mass,src,indirect);
    shift_vector<<<blocks(cfg.n),THREADS>>>(cfg.n,1,indirect,src,dst);
}
__global__ void sniff(Config c,double h,const double* ox,const double* ov,
    const double* xx,const double* vv,const double* crit,const double* bb,
    int cap,int* count,MercuryEvent* out) {
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j<1||j>=c.n) return;
    for(int i=1;i<c.nbig&&i<j;i++) {
        if(bb[c.n+i]<bb[j]||bb[c.n+j]<bb[i]||bb[3*c.n+i]<bb[2*c.n+j]||bb[3*c.n+j]<bb[2*c.n+i]) continue;
        double d0=0,d1=0,t0=0,t1=0;
        for(int k=0;k<3;k++) {
            int a=k*c.n+i,b=k*c.n+j; double p=ox[a]-ox[b],q=xx[a]-xx[b];
            d0+=p*p; d1+=q*q; t0+=p*(ov[a]-ov[b]); t1+=q*(vv[a]-vv[b]);
        }
        t0*=2; t1*=2;
        double d=9.9e29,t;
        if(t0*h<=0&&t1*h>=0) minimum(d0,d1,t0,t1,h,d,t);
        double rc=fmax(crit[i],crit[j]);
        if(fmin(d,fmin(d0,d1))<=rc*rc) {
            int z=atomicAdd(count,1); if(z<cap) { MercuryEvent e{}; e.i=i; e.j=j; out[z]=e; }
        }
    }
}
__global__ void selected_states(int n,int count,const int* ids,const double* xx,const double* vv,
    double* data,int scatter,double* dx,double* dv,const double* original_x,const double* original_v) {
    int a=blockIdx.x*blockDim.x+threadIdx.x; if(a>=count) return; int j=ids[a];
    for(int k=0;k<3;k++) {
        int z=k*n+j;
        if(scatter) { dx[z]=data[6*a+k]; dv[z]=data[6*a+3+k]; }
        else {
            // Undo the tentative Kepler drift only for encounter members.
            dx[z]=original_x[z]; dv[z]=original_v[z];
            data[6*a+k]=dx[z]; data[6*a+3+k]=dv[z];
        }
    }
}
std::vector<int> selected;
int* selected_device=nullptr;
std::vector<double> packed;

// Swap complete device contexts, retaining allocations for the next encounter.
// The scalar force counter is intentionally shared across contexts.
#define CONTEXT_FIELDS(X) \
 X(cfg) X(pn_enabled) X(algorithm) X(capacity) X(event_capacity) \
 X(x) X(v) X(oldx) X(oldv) X(wx) X(wv) X(ex) X(ev) X(acc) X(acc0) \
 X(table) X(scale) X(partial) X(maximum) X(mass) X(ngf) X(rce) X(rphys) \
 X(boxes) X(indirect) X(transfer) X(fault) X(event_count) X(device_events) \
 X(host_events) X(allocations) X(sym) X(sym_reset) X(ra_reset) \
 X(critical) X(encounter_mode) X(pair_i) X(pair_j) X(pair_count) X(pair_capacity) X(selected_device)
struct SavedContext {
#define DECLARE_FIELD(name) decltype(name) name##_saved{};
    CONTEXT_FIELDS(DECLARE_FIELD)
#undef DECLARE_FIELD
    void exchange() {
#define SWAP_FIELD(name) std::swap(name,name##_saved);
        CONTEXT_FIELDS(SWAP_FIELD)
#undef SWAP_FIELD
    }
} other_context;
#undef CONTEXT_FIELDS
bool encounter_active=false;
void free_other_context() {
    if(encounter_active) return;
    for(void* p:other_context.allocations_saved) cudaFree(p);
    other_context.allocations_saved.clear();
    if(other_context.device_events_saved) cudaFree(other_context.device_events_saved);
    other_context.device_events_saved=nullptr;
    other_context.capacity_saved=0; other_context.event_capacity_saved=0; other_context.pair_capacity_saved=0;
}
