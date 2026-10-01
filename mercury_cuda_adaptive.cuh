// Resident conservative Bulirsch-Stoer, following MDT_BS2's recurrence.
__global__ void bs2_begin(int n,double h,const double* ox,const double* ov,
    const double* a0,double* xx,double* b,double* c) {
    int z=blockIdx.x*blockDim.x+threadIdx.x;
    if(z>=3*n) return;
    b[z]=.5*a0[z]; c[z]=0;
    xx[z]=.5*h*h*a0[z]+h*ov[z]+ox[z];
}
__global__ void bs2_substep(int n,int j,double h,const double* ox,const double* ov,
    const double* a0,const double* aa,double* xx,double* b,double* c) {
    int z=blockIdx.x*blockDim.x+threadIdx.x;
    if(z>=3*n) return;
    b[z]+=aa[z]; c[z]+=b[z];
    xx[z]=h*h*c[z]+.5*h*h*a0[z]+h*j*ov[z]+ox[z];
}
__global__ void bs2_endpoint(int n,int level,double h,const double* xx,
    const double* ov,const double* b,const double* aa,double* d) {
    int z=blockIdx.x*blockDim.x+threadIdx.x;
    if(z>=3*n) return;
    d[level*6*n+z]=xx[z];
    d[(level*6+3)*n+z]=h*b[z]+.5*h*aa[z]+ov[z];
}
void bs2_step(double time,double* h,double* hdid,double tol,int64_t* rejected) {
    int n=cfg.n;
    begin_step<<<blocks(n),THREADS>>>(n,x,v,oldx,oldv,scale);
    force(oldx,oldv,acc0);
    for(;;) {
        if(time+*h==time) throw std::runtime_error("BS2 step cannot advance time");
        double h2[12];
        for(int level=0;level<12;level++) {
            int order=level+1; double hs=*h/order; h2[level]=hs*hs;
            bs2_begin<<<blocks(3*n),THREADS>>>(n,hs,oldx,oldv,acc0,wx,wv,ex);
            for(int j=2;j<=order;j++) {
                force(wx,oldv,acc);
                bs2_substep<<<blocks(3*n),THREADS>>>(n,j,hs,oldx,oldv,acc0,acc,wx,wv,ex);
            }
            force(wx,oldv,acc);
            bs2_endpoint<<<blocks(3*n),THREADS>>>(n,level,hs,wx,oldv,wv,acc,table);
            for(int col=level-1;col>=0;col--) {
                double inv=1.0/(h2[col]-h2[level]);
                extrapolate<<<blocks(6*n),THREADS>>>(n,level,col,inv*h2[col+1],inv*h2[level],table);
            }
            if(order>3) {
                estimate_error<<<blocks(n),THREADS>>>(n,table,scale,fault,partial);
                reduce_max<<<1,THREADS>>>(blocks(n),partial,maximum);
                double err; check(cudaMemcpy(&err,maximum,sizeof(double),cudaMemcpyDeviceToHost));
                if(!std::isfinite(err)) throw std::runtime_error("Nonfinite BS2 force or extrapolation");
                if(err<=tol*tol) {
                    accept<<<blocks(3*n),THREADS>>>(n,order,table,x,v);
                    check(cudaGetLastError()); *hdid=*h;
                    if(order>=8) *h*=.55;
                    if(order<7) *h*=1.3;
                    return;
                }
            }
        }
        *h*=.5; ++*rejected;
    }
}
