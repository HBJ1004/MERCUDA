#pragma once
#include <stdint.h>

// Plain C ABI consumed by ISO_C_BINDING. Arrays at the boundary retain
// Mercury's Fortran layout; the device uses component-major storage.
struct MercuryEvent {
    int i, j, kind, now;
    double distance, time;
    double xi[6], xj[6];
};
extern "C" {
int mercury_cuda_available();
int mercury_cuda_configure(int algorithm);
void mercury_cuda_reset(int flag);
int mercury_cuda_limits(const double* rce);
int mercury_cuda_export(double h,int physical,double* x,double* v);
int mercury_cuda_upload(int n, int nbig, int pn, int ngflag,
    const double *m, const double *x, const double *v,
    const double *ngf, const double *jcen, const double *rce,
    const double *rphys);
int mercury_cuda_step(double time, double *h, double *hdid, double tol,
    int64_t *forces, int64_t *rejected);
int mercury_cuda_download(double *x, double *v, int previous);
int mercury_cuda_events(double time, double h, double rcen,
    const MercuryEvent **events, int *count);
int mercury_cuda_force(double *a); // regression/benchmark interface
int mercury_cuda_hybrid_begin(double,const double*,int,int,int*,int*,int*,int*,double*,double*,int64_t*);
int mercury_cuda_hybrid_finish(double,const double*,const double*,const double*,int64_t*);
int mercury_cuda_encounter_enter(int,int,const double*,const double*,const double*,const double*,const double*,const double*,int,const int*,const int*);
int mercury_cuda_encounter_update(const double*,const double*,const double*);
void mercury_cuda_encounter_exit();
void mercury_cuda_free();
}
