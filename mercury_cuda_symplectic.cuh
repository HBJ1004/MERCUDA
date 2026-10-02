// MVS state stays in the same (corrected heliocentric) representation as MDT_MVS.
#include "mercury_cuda_kepler.cuh"
double* sym=nullptr;
bool sym_reset=true;
__global__ void jacobi(Config c,const double* m,const double* xx,const double* vv,
    double* jx,double* jv,int inverse) {
    if(threadIdx.x||blockIdx.x||c.nbig<2) return;
    double total=m[1],rx[3],rv[3],f=inverse?m[1]/(total+m[0]):m[1];
    for(int k=0;k<3;k++) { int z=k*c.n+1; jx[z]=xx[z]; jv[z]=vv[z]; rx[k]=f*xx[z]; rv[k]=f*vv[z]; }
    for(int j=2;j<c.nbig;j++) {
        double fac=1/(total+m[0]); total+=m[j];
        for(int k=0;k<3;k++) {
            int z=k*c.n+j;
            if(inverse) { jx[z]=xx[z]+rx[k]; jv[z]=vv[z]+rv[k]; rx[k]+=m[j]/(total+m[0])*xx[z]; rv[k]+=m[j]/(total+m[0])*vv[z]; }
            else { jx[z]=xx[z]-fac*rx[k]; jv[z]=vv[z]-fac*rv[k]; rx[k]+=m[j]*xx[z]; rv[k]+=m[j]*vv[z]; }
        }
    }
}
// MDT_MVS, MDT_HY and MCO_MVS2H discard drift_one's iflag and keep its
// best-effort state, so a nonconverged drift is not a fault here either.
__global__ void kepler_drift(Config c,double h,const double* m,double* xx,double* vv,
    double* jx,double* jv,int use_jacobi) {
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j==0||j>=c.n) return;
    double mu=c.mu; double *px=xx,*pv=vv;
    if(use_jacobi&&j<c.nbig) {
        px=jx; pv=jv; double inside=m[0];
        for(int k=1;k<j;k++) inside+=m[k];
        mu=m[0]*(inside+m[j])/inside;
    }
    int flag=0;
    drift_one__(&mu,px+j,px+c.n+j,px+2*c.n+j,pv+j,pv+c.n+j,pv+2*c.n+j,&h,&flag);
}
__global__ void mvs_prefix(Config c,const double* m,const double* xx,const double* jx,
    double* a1,double* a2,double* terms) {
    if(threadIdx.x||blockIdx.x) return;
    for(int k=0;k<6;k++) terms[k]=0;
    double inside=0,previous[3]={0,0,0};
    if(c.nbig>1) for(int k=0;k<3;k++) { a1[k*c.n+1]=0; a2[k*c.n+1]=0; }
    for(int j=2;j<c.nbig;j++) {
        inside+=m[j-1]; double r2=0,j2=0;
        for(int k=0;k<3;k++) { r2+=xx[k*c.n+j]*xx[k*c.n+j]; j2+=jx[k*c.n+j]*jx[k*c.n+j]; }
        double inv=1/sqrt(r2),ij=1/sqrt(j2),f0=m[j]*inv*inv*inv;
        double f12=m[0]*ij*ij*ij,f2=m[j]*f12/(inside+m[0]);
        double q=(r2-j2)*.5/j2,q2=q*q,q3=q*q2,q4=q2*q2,q5=q2*q3,q6=q3*q3,q7=q3*q4;
        double f1=402.1875*q7-187.6875*q6+86.625*q5-39.375*q4+17.5*q3-7.5*q2+3*q-1;
        for(int k=0;k<3;k++) {
            int z=k*c.n+j; terms[k]-=f0*xx[z];
            a1[z]=f12*(jx[z]+f1*xx[z]); previous[k]+=f2*jx[z]; a2[z]=previous[k];
        }
    }
    if(c.nbig>1) {
        double r2=0; for(int k=0;k<3;k++) r2+=xx[k*c.n+1]*xx[k*c.n+1];
        double inv=1/sqrt(r2),f=m[1]*inv*inv*inv;
        for(int k=0;k<3;k++) terms[3+k]=terms[k]-f*xx[k*c.n+1];
    }
}
__global__ void mvs_acceleration(Config c,const double* m,const double* xx,const double* vv,
    const double* ng,const double* a1,const double* a2,const double* terms,const double* ind,
    double* aa,int* bad) {
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=c.n) return;
    if(j==0) { for(int k=0;k<3;k++) aa[k*c.n]=0; return; }
    double r[3]={xx[j],xx[c.n+j],xx[2*c.n+j]},a[3]={0,0,0};
    for(int i=1;i<c.nbig;i++) if(i!=j) {
        double d[3]={xx[i]-r[0],xx[c.n+i]-r[1],xx[2*c.n+i]-r[2]};
        double inv=1/sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]),f=m[i]*inv*inv*inv;
        for(int k=0;k<3;k++) a[k]+=f*d[k];
    }
    for(int k=0;k<3;k++) a[k]=j<c.nbig?terms[k]+a1[k*c.n+j]+a2[k*c.n+j]+a[k]:terms[3+k]+a[k];
    double r2=r[0]*r[0]+r[1]*r[1]+r[2]*r[2],inv=1/sqrt(r2);
    if(c.j2!=0||c.j4!=0||c.j6!=0) {
        double ao[3]; obl(c,r,inv,ao);
        for(int k=0;k<3;k++) a[k]+=ao[k]+ind[k];
    }
    if(c.ngflag) {
        double u[3]={vv[j],vv[c.n+j],vv[2*c.n+j]},rv=r[0]*u[0]+r[1]*u[1]+r[2]*u[2];
        nongrav(c,j,r,r2,inv,u,rv,m,ng,a,bad);
    }
    for(int k=0;k<3;k++) { aa[k*c.n+j]=a[k]; if(!isfinite(a[k])) atomicExch(bad,1); }
}
__global__ void kick(int n,double h,const double* aa,double* vv) {
    int z=blockIdx.x*blockDim.x+threadIdx.x; if(z<3*n) vv[z]+=h*aa[z];
}
__global__ void obl_indirect(Config c,const double* m,const double* xx,double* result) {
    double sum[3]={0,0,0};
    for(int j=1+threadIdx.x;j<c.nmass;j+=THREADS) {
        if(m[j]==0) continue;
        double r[3]={xx[j],xx[c.n+j],xx[2*c.n+j]},a[3];
        obl(c,r,1/sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]),a);
        for(int k=0;k<3;k++) sum[k]+=m[j]/c.mu*a[k];
    }
    for(int k=0;k<3;k++) { double s=block_reduce(sum[k],SumOp()); if(threadIdx.x==0) result[k]=s; }
}
// Central-body oblateness reaction from massive bodies, into indirect.
void obl_reaction(const double* xx) {
    if(cfg.j2!=0||cfg.j4!=0||cfg.j6!=0) obl_indirect<<<1,THREADS>>>(cfg,mass,xx,indirect);
}
void mvs_force(double* xx,double* vv,double* aa,bool jacobi_ready=false) {
    if(!jacobi_ready) jacobi<<<1,1>>>(cfg,mass,xx,vv,wx,wv,0);
    mvs_prefix<<<1,1>>>(cfg,mass,xx,wx,ex,ev,table);
    obl_reaction(xx);
    mvs_acceleration<<<blocks(cfg.n),THREADS>>>(cfg,mass,xx,vv,ngf,ex,ev,table,indirect,aa,fault);
    ++nforces;
}
void mvs_drift(double* xx,double* vv,double h) {
    jacobi<<<1,1>>>(cfg,mass,xx,vv,wx,wv,0);
    kepler_drift<<<blocks(cfg.n),THREADS>>>(cfg,h,mass,xx,vv,wx,wv,1);
    jacobi<<<1,1>>>(cfg,mass,wx,wv,xx,vv,1);
}
void corrector(double* xx,double* vv,double h,bool inverse) {
    double r=sqrt(10.0),ha[2]={h*r*3/10,h*r/5},hb[2]={-h*r/72,h*r/24};
    if(inverse) { ha[0]=-h*r/5; ha[1]=-h*r*3/10; hb[0]=-h*r/24; hb[1]=h*r/72; }
    for(int k=0;k<2;k++) {
        mvs_drift(xx,vv,ha[k]); mvs_force(xx,vv,acc,true);
        kick<<<blocks(3*cfg.n),THREADS>>>(cfg.n,-hb[k],acc,vv);
        mvs_drift(xx,vv,-2*ha[k]); mvs_force(xx,vv,acc,true);
        kick<<<blocks(3*cfg.n),THREADS>>>(cfg.n,hb[k],acc,vv);
        mvs_drift(xx,vv,ha[k]);
    }
}
void copy_vector(double* dst,const double* src) { check(cudaMemcpyAsync(dst,src,sizeof(double)*3*cfg.n,cudaMemcpyDeviceToDevice)); }
void mvs_step(double h) {
    int n=cfg.n;
    copy_vector(oldx,x); copy_vector(oldv,v);
    if(sym_reset) mvs_force(x,v,sym+9*n);
    kick<<<blocks(3*n),THREADS>>>(n,.5*h,sym+9*n,v);
    copy_vector(sym,x); copy_vector(sym+3*n,v);
    mvs_drift(x,v,h); copy_vector(sym+6*n,v);
    mvs_force(x,v,sym+9*n,true);
    kick<<<blocks(3*n),THREADS>>>(n,.5*h,sym+9*n,v);
    sym_reset=false;
    int bad; check(cudaMemcpy(&bad,fault,sizeof(int),cudaMemcpyDeviceToHost));
    if(bad) throw std::runtime_error("MVS force failed");
}
