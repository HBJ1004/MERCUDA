#include "mercury_cuda.h"
extern "C" {
int mercury_cuda_available() { return 0; }
int mercury_cuda_configure(int) { return 1; }
void mercury_cuda_reset(int) {}
int mercury_cuda_export(double,int,double*,double*) { return 1; }
int mercury_cuda_upload(int,int,int,int,const double*,const double*,
    const double*,const double*,const double*,const double*,const double*) { return 1; }
int mercury_cuda_step(double,double*,double*,double,int64_t*,int64_t*) { return 1; }
int mercury_cuda_download(double*,double*,int) { return 1; }
int mercury_cuda_events(double,double,double,const MercuryEvent**,int*) { return 1; }
int mercury_cuda_force(double*) { return 1; }
int mercury_cuda_hybrid_begin(double,const double*,int,int,int*,int*,int*,int*,double*,double*,int64_t*) { return 1; }
int mercury_cuda_hybrid_finish(double,const double*,const double*,const double*,int64_t*) { return 1; }
int mercury_cuda_encounter_enter(int,int,const double*,const double*,const double*,const double*,const double*,const double*,int,const int*,const int*) { return 1; }
int mercury_cuda_encounter_update(const double*,const double*,const double*) { return 1; }
void mercury_cuda_encounter_exit() {}
void mercury_cuda_free() {}
}
