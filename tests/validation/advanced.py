"""Encounters, custom forces, fault fixtures and sanitizers."""
from pathlib import Path
import math
import os
import re
import sys
import tempfile

import numpy as np
from cases import ROOT, body, prepare, dump
from force_library import routine
from scenarios import METHODS,accuracy,checked_reference,tilted
from operations import epoch


def events(c):
    # Non-chaotic strong scattering, with and without all permitted extra forces.
    for enabled in (False,True):
        big = [body('A',mass=1e-5,a=1,e=.01),body('B',mass=1e-5,a=1.02,phase=.02,e=.01)]
        small = [tilted(body('P',a=1.3,e=.2))]
        if enabled:
            for obj in big+small: obj['params'].update(a1=1e-11,a2=-2e-11,a3=3e-11,yar=4e-11,b=.001)
        options = dict(pn=enabled,jcen=(1e-6,-3e-8,4e-10) if enabled else (0,0,0))
        for direction in (-1,1):
            refs,floor = checked_reference(big,small,[direction*t for t in (1.,4.,8.)],**options)
            for profile in c.profiles:
                for method in ('BS','RADAU') if enabled else ('BS','BS2','RADAU','HYBRID'):
                    def scattering():
                        maximum = 0.; sweep = []
                        for time,ref in zip([direction*t for t in (1.,4.,8.)],refs):
                            errors = []
                            for level in range(3):
                                state,_ = c.simulate(profile,big=big,small=small,algorithm=method,
                                    backend='cuda' if profile.cuda else 'cpu',stop=time,interval=abs(time),
                                    step=.04/2**level,tol=(1e-10,1e-12,1e-13)[level],**options)
                                errors.append(max(oracle_errors(state,ref)))
                            accuracy(state,ref,limit=1e-6)
                            if errors[0] > 10*max(floor,1e-13) and errors[-1] > errors[0]/2:
                                raise AssertionError('Encounter refinement did not improve: '+str(errors))
                            maximum = max(maximum,errors[-1]); sweep.append({'time':time,'errors':errors})
                        return {'maximum_error':maximum,'convergence':sweep}
                    c.record('events',f'{profile.name}/{method}/scattering/{enabled}/{direction}',scattering,method=method)
    for profile in c.profiles:
        for method in METHODS:
            for stop_on in (False,True):
                def stop_encounter():
                    planet = body('PLANET',mass=1e-12,e=0,r=3)
                    small = [dict(name='FLYBY',mass=0,x=[1.00001,-.005,0],v=[0,.1,0],params={})]
                    def inspect(path,state,info):
                        records = [row for row in (path/'ce.out').read_bytes().split(b'\n') if row.startswith(b'\x0c6b')]
                        assert records, 'Expected encounter not reported'
                        actual = epoch(path)
                        if stop_on: assert actual < .2
                        else: assert abs(actual-.2) < 1e-12
                        return {'encounters':len(records),'final_epoch':actual}
                    return c.simulate(profile,big=[planet],small=small,algorithm=method,
                        backend='cuda' if profile.cuda else 'cpu',stop=.2,step=.001,interval=.01,
                        stop_encounter=stop_on,inspect=inspect)
                c.record('events',f'{profile.name}/{method}/stop-encounter/{stop_on}',stop_encounter)
    # Boundaries and simultaneous removal: identifiers and coefficients survive.
    for profile in c.profiles:
        for method in METHODS:
            for collisions in (False,True):
                def removal():
                    infall = body('IMPACT'); infall['x']=[.006,0,0]; infall['v']=[-.1,0,0]
                    escape = body('ESCAPE'); escape['x']=[99.9,0,0]; escape['v']=[10,0,0]
                    keep = body('KEEP',a=2,**({'yar':1e-11,'a2':-2e-11,'b':.001} if method in ('BS','RADAU') else {}))
                    state,info = c.simulate(profile,small=[infall,escape,keep],algorithm=method,
                        backend='cuda' if profile.cuda else 'cpu',stop=.3,step=.001,interval=.05,
                        collisions=collisions,periodic_interval=1)
                    assert set(state)=={'KEEP'}
                    for key,value in keep['params'].items(): assert state['KEEP']['params'][key]==value
                    return {'survivors':sorted(state)}
                c.record('events',f'{profile.name}/{method}/removal/{collisions}',removal)
    # Collision-off pair survives; collision-on merges and conserves mass.
    for profile in c.profiles:
        for method in ('BS','BS2','RADAU','HYBRID'):
            for collisions in (False,True):
                def merge():
                    a = body('A',mass=1e-8,e=0,d=1e-6)
                    b = body('B',mass=2e-8,e=0,d=1e-6); b['x'][0]+=.0003
                    state,_ = c.simulate(profile,big=[a,b],small=[],algorithm=method,
                        backend='cuda' if profile.cuda else 'cpu',collisions=collisions,stop=.01,
                        step=.001,interval=.001)
                    assert set(state)==({'B'} if collisions else {'A','B'})
                    assert abs(sum(o['mass'] for o in state.values())-3e-8)<1e-20
                c.record('events',f'{profile.name}/{method}/merge/{collisions}',merge)


def oracle_errors(state,ref):
    import oracle
    return oracle.errors(state,ref)


def lifecycle(c):
    from device_worker import exercise
    for profile in c.profiles:
        if not profile.cuda: continue
        for method in (1,2,3,4,9,10):
            def check():
                profile.select(); exercise(method)
                return {'algorithm':method,'grow_shrink_cycles':7,'compact_pair_cycles':7 if method==10 else 0}
            c.record('lifecycle',f'{profile.name}/{method}',check)
    if c.args.cpu_only:
        c.record('lifecycle','cpu-only-capacity',lambda:{'device_scope':'not requested; CPU allocation is covered by stress tests'})


def custom(c):
    import oracle
    # Build copies with a representative force, never modify production source.
    for profile in c.profiles:
        if profile.cuda: continue
        for law in ('time','drag'):
            with tempfile.TemporaryDirectory(prefix='mercuda-custom-build-') as tmp:
                tmp = Path(tmp)
                source = (ROOT/'mercury6_2.for').read_text()
                old = routine(source,'mfo_user')
                expression = 'a(2,j)=1.d-9*sin(.03d0*time)' if law=='time' else 'a(:,j)=-1.d-7*v(:,j)'
                replacement = '''      subroutine mfo_user(time,jcen,nbod,nbig,m,x,v,a)
      implicit none
      integer nbod,nbig,j
      real*8 time,jcen(3),m(nbod),x(3,nbod),v(3,nbod),a(3,nbod)
      a=0.d0
      do j=2,nbod
        '''+expression+'''
      end do
      end
'''
                (tmp/'custom.for').write_text(source.replace(old,replacement))
                # Link against the CPU-only profile's runtime/backend object.
                c.command(['gfortran',*profile.flags.split(),f'-I{ROOT}',f'-I{profile.build}',
                    '-c',str(tmp/'custom.for'),'-o',str(tmp/'custom.o')])
                c.command(['gfortran',*profile.flags.split(),str(tmp/'custom.o'),
                    *[str(profile.build/name) for name in ('mercury_support.o','mercury_gpu.o','mercury_cuda_stub.o')],
                    '-lstdc++','-o',str(tmp/'mercury6')])
                for direction in (-1,1):
                    custom_force = (lambda t,x,v:np.array([0,1e-9*math.sin(.03*t),0])) if law=='time' else (lambda t,x,v:-1e-7*v)
                    reference,floor = checked_reference([], [body()], [direction*32.],custom=custom_force)
                    for method in ('BS','RADAU','MVS','HYBRID'):
                        def check():
                            with tempfile.TemporaryDirectory() as case:
                                path = prepare(case,user_force=True,algorithm=method,stop=direction*32.,interval=3.7,tol=1e-13,step=.0625)
                                env = profile.environment(); env.pop('MERCURY_TEST_BIN',None)
                                c.command([str(tmp/'mercury6')],cwd=path,env=env)
                                return accuracy(dump(path),reference[0])
                        c.record('custom',f'{profile.name}/{law}/{method}/{direction}',check)
                for backend in ('cuda','auto'):
                    def backend_check():
                        with tempfile.TemporaryDirectory() as case:
                            path = prepare(case,user_force=True,backend=backend,stop=.125)
                            result = c.command([str(tmp/'mercury6')],cwd=path,check=False,env=profile.environment())
                            if backend=='cuda': assert result.returncode>0
                            else:
                                assert result.returncode==0
                                assert 'Execution backend: CPU' in (path/'info.out').read_text()
                    c.record('custom',f'{profile.name}/{law}/{backend}',backend_check)


def sanitize(c):
    profile = next(p for p in c.profiles if p.name=='cuda-debug')
    profile.select()
    worker = ROOT/'tests/validation/device_worker.py'
    for tool in ('memcheck','initcheck','synccheck','racecheck'):
        for method in (1,2,3,4,10):
            def check(tool=tool,method=method):
                options = ['--tool',tool,'--error-exitcode','99','--target-processes','all']
                if tool=='memcheck': options += ['--leak-check','full']
                env = dict(profile.environment(),MERCURY_TEST_GPU_LIBRARY_READY='1')
                result = c.command([c.sanitizer,*options,sys.executable,str(worker),str(method)],
                                   env=env,timeout=240,check=False)
                text = result.stdout+result.stderr
                if "Failed to initialize WDDM debugger interface" in text:
                    raise c.unavailable("Windows CUDA debugger interface is disabled; run NVIDIA EnableDebuggerInterface.bat as administrator and rerun test-sanitize")
                if result.returncode:
                    raise AssertionError("Sanitizer exit "+str(result.returncode)+": "+text[-4000:])
                if not re.search(r'(ERROR SUMMARY: 0 errors|RACECHECK SUMMARY: 0 hazards)',text):
                    raise AssertionError('Sanitizer did not produce a clean error summary: '+text[-4000:])
                if re.search(r'========= (WARNING|ERROR):',text):
                    raise AssertionError('Sanitizer warning/error: '+text[-4000:])
                return {'tool':tool,'algorithm':method,'error_summary':0}
            c.record('sanitize',f'{tool}/{method}',check)
    # Instrument host-side vector packing as well as Fortran array accesses.
    def address_sanitizer():
        build = ROOT/'build/validation/host-asan'; build.mkdir(parents=True,exist_ok=True)
        c.command(['g++','-O1','-g','-std=c++17','-fsanitize=address,undefined',
            '-fno-omit-frame-pointer','-I'+str(ROOT),str(ROOT/'tests/validation/encounter_asan.cpp'),
            '-o',str(build/'encounter-asan')],timeout=120)
        env = dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1')
        c.command([str(build/'encounter-asan')],env=env,timeout=90)
        return {'host_address_errors':0,'host_undefined_behavior_errors':0,'host_leak_detection':True,
                'scope':'production buffer constructor/copy; CUDA runtime checked by Compute Sanitizer'}
    c.record('sanitize','host-address/compact-encounter',address_sanitizer)


def execute(c,group):
    functions = {'events':events,'custom':custom,'lifecycle':lifecycle,'sanitize':sanitize}
    if group not in functions: raise RuntimeError('No scenarios registered for '+group)
    return functions[group](c)
