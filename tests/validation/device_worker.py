"""Standalone CUDA lifecycle process, suitable for Compute Sanitizer."""
from pathlib import Path
import ctypes as C
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from cases import MU
from force_library import load_gpu,arr,I,D


def exercise(method):
    gpu = load_gpu()
    if gpu is None: raise RuntimeError('CUDA unavailable; sanitizer validation is incomplete')
    gpu.mercury_cuda_encounter_enter.argtypes = [I,I,*([C.POINTER(D)]*6),I,C.POINTER(I),C.POINTER(I)]
    gpu.mercury_cuda_encounter_update.argtypes = [C.POINTER(D)]*3
    gpu.mercury_cuda_limits.argtypes = [C.POINTER(D)]
    for count in (1,2,33,257,17,513,2):
        n = count+1
        assert gpu.mercury_cuda_configure(method) == 0
        m = arr([MU]+[0.]*count)
        x = arr([0.,0.,0.]+[value for j in range(count) for value in (1+j*.001,0.,0.)])
        v = arr([0.,0.,0.]+[value for j in range(count) for value in (0.,math.sqrt(MU/(1+j*.001)),.001)])
        ng = arr([0.]*(5*n)); jc = arr([0.,0.,0.]); rc = arr([.001]*n)
        assert gpu.mercury_cuda_upload(n,1,0,0,m,x,v,ng,jc,rc,rc) == 0
        force = arr([0.]*(3*n)); assert gpu.mercury_cuda_force(force) == 0
        assert all(math.isfinite(z) for z in force)
        assert gpu.mercury_cuda_limits(rc) == 0
        xx,vv = arr([0.]*(3*n)),arr([0.]*(3*n))
        assert gpu.mercury_cuda_download(xx,vv,0) == 0
        assert list(xx) == list(x) and list(vv) == list(v)
        if method in (2,3,4):
            h,done = D(.001),D(0); forces,rejected = C.c_int64(),C.c_int64()
            assert gpu.mercury_cuda_step(0.,C.byref(h),C.byref(done),1e-12,C.byref(forces),C.byref(rejected)) == 0
            assert done.value > 0
    gpu.mercury_cuda_free()
    if method == 10:
        # Grow and shrink compact encounter buffers without growing body arrays.
        n = 4; m = arr([MU,1e-8*MU,1e-8*MU,0.])
        x = arr([0,0,0,1,0,0,1.001,.002,0,1.1,.01,0]); v = arr([0,0,0,0,.017,0,0,.017,0,0,.016,0])
        ng,jc,rc = arr([0.]*(5*n)),arr([0.,0.,0.]),arr([.01]*n)
        assert gpu.mercury_cuda_configure(10) == 0
        assert gpu.mercury_cuda_upload(n,3,0,0,m,x,v,ng,jc,rc,rc) == 0
        for count in (1,3,2,64,17,257,1):
            pi,pj = (I*count)(*([2]*count)),(I*count)(*([3]*count))
            assert gpu.mercury_cuda_encounter_enter(n,3,m,x,v,rc,rc,rc,count,pi,pj) == 0
            assert gpu.mercury_cuda_encounter_update(m,x,v) == 0
            gpu.mercury_cuda_encounter_exit()
        gpu.mercury_cuda_free()


if __name__ == '__main__':
    exercise(int(sys.argv[1]))
