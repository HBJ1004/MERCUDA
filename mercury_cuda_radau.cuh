// Everhart RA15. Coefficients, iteration counts and predictor updates follow
// MDT_RA15; this is not a substitution of IAS15 or its timestep controller.
struct RaConstants { double h[8],xc[8],vc[7],c[21],d[21],r[28]; };
__constant__ RaConstants ra;
bool ra_reset=true;
void initialize_radau() {
    RaConstants a={{0,.0562625605369221,.1802406917368924,.3526247171131696,
        .5471536263305554,.7342101772154105,.8853209468390958,.9775206135612875},
        {.5,.1666666666666667,.08333333333333333,.05,.03333333333333333,
         .02380952380952381,.01785714285714286,.01388888888888889},
        {.5,.3333333333333333,.25,.2,.1666666666666667,.1428571428571429,.125},{},{},{}};
    int n=0;
    for(int j=1;j<8;j++) for(int k=0;k<j;k++) a.r[n++]=1/(a.h[j]-a.h[k]);
    a.c[0]=-a.h[1]; a.d[0]=a.h[1]; n=1;
    for(int j=3;j<=7;j++) {
        ++n; a.c[n-1]=-a.h[j-1]*a.c[n-j+1]; a.d[n-1]=a.h[1]*a.d[n-j+1];
        for(int k=3;k<j;k++) {
            ++n; a.c[n-1]=a.c[n-j]-a.h[j-1]*a.c[n-j+1];
            a.d[n-1]=a.d[n-j]+a.h[k-1]*a.d[n-j+1];
        }
        ++n; a.c[n-1]=a.c[n-j]-a.h[j-1]; a.d[n-1]=a.d[n-j]+a.h[j-1];
    }
    check(cudaMemcpyToSymbol(ra,&a,sizeof(a)));
}
__global__ void ra_start(int n,const double* b,double* g) {
    int z=blockIdx.x*blockDim.x+threadIdx.x; if(z>=3*n) return;
    for(int k=0;k<7;k++) {
        double sum=0;
        for(int l=6;l>k;l--) sum+=b[l*3*n+z]*ra.d[l*(l-1)/2+k];
        g[k*3*n+z]=sum+b[k*3*n+z];
    }
}
template<bool VELOCITY> __global__ void ra_predict(int n,int node,double t,
    const double* ox,const double* ov,const double* a0,const double* b,double* xx,double* vv) {
    int z=blockIdx.x*blockDim.x+threadIdx.x; if(z>=3*n) return;
    double u=ra.h[node],s[9];
    s[0]=t*u; s[1]=s[0]*s[0]*.5; s[2]=s[1]*u*.3333333333333333;
    s[3]=s[2]*u*.5; s[4]=s[3]*u*.6; s[5]=s[4]*u*.6666666666666667;
    s[6]=s[5]*u*.7142857142857143; s[7]=s[6]*u*.75; s[8]=s[7]*u*.7777777777777778;
    double sum=s[8]*b[6*3*n+z];
    for(int k=5;k>=0;k--) sum+=s[k+2]*b[k*3*n+z];
    xx[z]=sum+s[1]*a0[z]+s[0]*ov[z]+ox[z];
    if(VELOCITY) {
        s[1]=s[0]*u*.5; s[2]=s[1]*u*.6666666666666667;
        s[3]=s[2]*u*.75; s[4]=s[3]*u*.8; s[5]=s[4]*u*.8333333333333333;
        s[6]=s[5]*u*.8571428571428571; s[7]=s[6]*u*.875;
        sum=s[7]*b[6*3*n+z];
        for(int k=5;k>=0;k--) sum+=s[k+1]*b[k*3*n+z];
        vv[z]=sum+s[0]*a0[z]+ov[z];
    }
}
__global__ void ra_correct(int n,int level,const double* aa,const double* a0,double* b,double* g) {
    int z=blockIdx.x*blockDim.x+threadIdx.x; if(z>=3*n) return;
    int start=level*(level+1)/2;
    double val=(aa[z]-a0[z])*ra.r[start];
    for(int k=1;k<=level;k++) val=(val-g[(k-1)*3*n+z])*ra.r[start+k];
    double previous=g[level*3*n+z],delta=val-previous; g[level*3*n+z]=val;
    for(int k=0;k<level;k++) b[k*3*n+z]+=delta*ra.c[level*(level-1)/2+k];
    // MDT_RA15's first stage uses (b + g_new) - g_old, in that order.
    if(level==0) b[z]=(b[z]+val)-previous;
    else b[level*3*n+z]+=delta;
}
__global__ void ra_error(int n,const double* b,const int* bad,double* out) {
    int j=blockIdx.x*blockDim.x+threadIdx.x; double val=0;
    if(j>0&&j<n) for(int k=0;k<3;k++) {
        double a=b[18*n+k*n+j];
        val=isfinite(a)?fmax(val,fabs(a)):INFINITY;
    }
    if(*bad) val=INFINITY;
    val=block_reduce(val,MaxOp());
    if(threadIdx.x==0) out[blockIdx.x]=val;
}
__global__ void ra_finish(int n,double t,double next,const double* ox,const double* ov,
    const double* a0,double* b,double* e,double* xx,double* vv) {
    int z=blockIdx.x*blockDim.x+threadIdx.x; if(z>=3*n) return;
    double p[7],corr[7];
    for(int k=0;k<7;k++) { p[k]=b[k*3*n+z]; corr[k]=p[k]-e[k*3*n+z]; }
    double sx=ra.xc[7]*p[6],sv=ra.vc[6]*p[6];
    for(int k=5;k>=0;k--) { sx+=ra.xc[k+1]*p[k]; sv+=ra.vc[k]*p[k]; }
    xx[z]=(sx+ra.xc[0]*a0[z])*(t*t)+ov[z]*t+ox[z];
    vv[z]=(sv+a0[z])*t+ov[z];
    double q=next/t,q2=q*q,q3=q*q2,q4=q2*q2,q5=q2*q3,q6=q3*q3,q7=q3*q4;
    double pred[7]={
        q*(p[6]*7+p[5]*6+p[4]*5+p[3]*4+p[2]*3+p[1]*2+p[0]),
        q2*(p[6]*21+p[5]*15+p[4]*10+p[3]*6+p[2]*3+p[1]),
        q3*(p[6]*35+p[5]*20+p[4]*10+p[3]*4+p[2]),
        q4*(p[6]*35+p[5]*15+p[4]*5+p[3]),q5*(p[6]*21+p[5]*6+p[4]),
        q6*(p[6]*7+p[5]),q7*p[6]};
    for(int k=0;k<7;k++) { e[k*3*n+z]=pred[k]; b[k*3*n+z]=pred[k]+corr[k]; }
}
void radau_step(double time,double* h,double* hdid,double tol,int64_t* rejected) {
    int n=cfg.n; double* b=table; double* e=b+21*n; double* g=e+21*n;
    begin_step<<<blocks(n),THREADS>>>(n,x,v,oldx,oldv,scale);
    for(;;) {
        if(time+*h==time) throw std::runtime_error("RADAU step cannot advance time");
        if(ra_reset) check(cudaMemset(b,0,sizeof(double)*42*n));
        force(oldx,oldv,acc0);
        ra_start<<<blocks(3*n),THREADS>>>(n,b,g);
        for(int iteration=0;iteration<(ra_reset?6:2);iteration++) for(int node=1;node<8;node++) {
            if(pn_enabled||cfg.ngflag) ra_predict<true><<<blocks(3*n),THREADS>>>(n,node,*h,oldx,oldv,acc0,b,wx,wv);
            else ra_predict<false><<<blocks(3*n),THREADS>>>(n,node,*h,oldx,oldv,acc0,b,wx,wv);
            force(wx,wv,acc);
            ra_correct<<<blocks(3*n),THREADS>>>(n,node-1,acc,acc0,b,g);
        }
        ra_error<<<blocks(n),THREADS>>>(n,b,fault,partial);
        reduce_max<<<1,THREADS>>>(blocks(n),partial,maximum);
        double err; check(cudaMemcpy(&err,maximum,sizeof(double),cudaMemcpyDeviceToHost));
        if(!std::isfinite(err)) throw std::runtime_error("Nonfinite RADAU force or predictor");
        err/=72*pow(fabs(*h),7.0);
        double next=err==0?*h*1.4:copysign(pow(tol/err,1.0/9.0),*h);
        if(ra_reset&&fabs(next/ *h)<1) { *h=next*.8; ++*rejected; continue; }
        if(fabs(next/ *h)>1.4) next=*h*1.4;
        ra_finish<<<blocks(3*n),THREADS>>>(n,*h,next,oldx,oldv,acc0,b,e,x,v);
        check(cudaGetLastError()); *hdid=*h; *h=next; ra_reset=false; return;
    }
}
