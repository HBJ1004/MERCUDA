"""Build isolated force routines for numerical tests (no production test hooks)."""
from cases import ROOT
from pathlib import Path
import ctypes as C
import re
import os
import subprocess
import shlex

D=C.c_double; I=C.c_int; P=C.POINTER(D)
def arr(values): return (D*len(values))(*values)
def routine(source,name):
    return re.search(r'^      subroutine '+name+r'\b.*?^      end\s*$',source,re.M|re.S).group()

def load_cpu():
    selected=Path(os.environ.get('MERCURY_TEST_BUILD',str(ROOT/'build')))
    build=selected/'tests'; build.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'mercury6_2.for').read_text()
    text='\n'.join(routine(source,name) for name in
                  ['mfo_all','mfo_grav','mfo_obl','mfo_ngf','mfo_pr','mfo_pn','mfo_user'])
    text+='\n'+routine((ROOT/'tests/fixtures/original_pr.for').read_text(),'mfo_pr').replace('subroutine mfo_pr','subroutine original_pr')+'\n'
    (build/'forces.for').write_text(text)
    flags=shlex.split(os.environ.get('MERCURY_TEST_FFLAGS',
                     '-O2 -ffp-contract=off -ffixed-line-length-none'))
    subprocess.run(['gfortran','-shared','-fPIC',*flags,
                    '-I'+str(ROOT),'-I'+str(selected),'-J'+str(build),
                    str(ROOT/'mercury_support.f90'),str(build/'forces.for'),'-o',str(build/'forces.so')],check=True,capture_output=True)
    return C.CDLL(str(build/'forces.so'))

def load_gpu():
    if os.environ.get('MERCURY_TEST_CUDA')=='0': return None
    obj=Path(os.environ.get('MERCURY_TEST_BUILD',str(ROOT/'build')))/'mercury_cuda.o'
    if not obj.exists(): return None
    # Link the same production object; locate the toolkit used by the Makefile.
    env=os.environ.copy()
    for key in ['MAKEFLAGS','MFLAGS','MAKELEVEL']: env.pop(key,None)
    output=subprocess.check_output(['make','--no-print-directory','-s','--eval=print-cuda: ; @echo $(CUDA_LIB)','print-cuda'],cwd=ROOT,text=True,env=env).strip()
    if not output: return None
    so=obj.parent/'tests/cuda.so'; so.parent.mkdir(exist_ok=True)
    subprocess.run(['g++','-shared',str(obj),output,'-ldl','-lrt','-pthread','-o',str(so)],check=True,capture_output=True)
    lib=C.CDLL(str(so))
    if not lib.mercury_cuda_available(): return None
    lib.mercury_cuda_upload.argtypes=[I,I,I,I,P,P,P,P,P,P,P]
    lib.mercury_cuda_force.argtypes=[P]
    lib.mercury_cuda_step.argtypes=[D,P,P,D,C.POINTER(C.c_int64),C.POINTER(C.c_int64)]
    lib.mercury_cuda_download.argtypes=[P,P,I]
    return lib


def cpu_force(lib,m,x,v,ng,jcen=(0,0,0),nbig=2,pn=False,ngflag=0):
    n=len(m); dims=(I*2).in_dll(lib,'mercury_sizes_'); dims[:]=[n,max(5000,n)]
    n_,nb_=I(n),I(nbig); flag=I(ngflag); opt=(I*8)(0,1,1,2,0,1,int(pn),0)
    a=arr([0.]*(3*n)); dummy=arr([0.]*(3*n)); stat=(I*n)(); empty=I(0)
    lib.mfo_all_(C.byref(D(0)),arr(jcen),C.byref(n_),C.byref(nb_),arr(m),arr(x),arr(v),dummy,dummy,a,stat,arr(ng),C.byref(flag),opt,C.byref(empty),C.byref(empty),C.byref(empty))
    return list(a)
