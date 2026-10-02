"""Deterministic claim-based validation scenarios; all simulations are temporary."""
import copy
import math
import random
import tempfile

import numpy as np
from cases import MU, body, prepare, run
from force_library import load_cpu, load_gpu, cpu_force, arr
import oracle

METHODS = ('BS','BS2','RADAU','MVS','HYBRID')
BITS = ('pn','b','yar','a1','a2','a3','j2','j4','j6')
COUNTS = (0,1,31,32,33,255,256,257,1999,2000,2001,4095,4096,4097)


def tilted(obj, angle=.37):
    obj = copy.deepcopy(obj)
    for key in ('x','v'):
        x,y,z = obj[key]
        obj[key] = [x,math.cos(angle)*y-math.sin(angle)*z,
                       math.sin(angle)*y+math.cos(angle)*z]
    return obj


def force_setup(mask):
    values = dict(zip(BITS,[bool(mask & (1<<i)) for i in range(9)]))
    params = {key:value for key,value in [('b',.001 if values['b'] else 0),
        ('yar',-4e-11 if values['yar'] else 0),('a1',1e-11 if values['a1'] else 0),
        ('a2',-2e-11 if values['a2'] else 0),('a3',3e-11 if values['a3'] else 0)]}
    jcen = tuple(value if values[key] else 0 for key,value in
                 [('j2',.1),('j4',-.01),('j6',.001)])
    big = [tilted(body('PLANET',mass=1e-7,a=1.7,e=.05,phase=.4),-.2)]
    small = [tilted(body('PARTICLE',a=.7,e=.2,phase=.8,**params))]
    return big,small,dict(pn=values['pn'],jcen=jcen)


def supported(method,mask):
    if method in ('BS','RADAU'): return True
    if mask & 7: return False  # PN/PR/yar
    if method == 'BS2' and mask & 63: return False
    return True


def reject(profile, settings, edit=None, expected=None):
    profile.select()
    with tempfile.TemporaryDirectory(prefix='mercuda-invalid-') as tmp:
        path = prepare(tmp,**settings)
        if edit: edit(path)
        result = run(path,check=False,timeout=15)
        output = result.stdout+result.stderr
        if result.returncode == 0: raise AssertionError('Invalid setup exited successfully: '+output)
        if result.returncode < 0: raise AssertionError('Invalid setup crashed instead of reporting an error: '+output)
        if expected and expected.lower() not in output.lower():
            raise AssertionError('Missing diagnostic '+expected+': '+output)
        return {'exit_code':result.returncode}


def accuracy(actual,reference, *, limit=1e-8, distance=1., mu=MU):
    position,velocity = oracle.errors(actual,reference,distance,mu)
    if max(position,velocity) > limit:
        raise AssertionError(f'Trajectory error position={position:.6g}, velocity={velocity:.6g}, limit={limit}')
    return {'position_error':position,'velocity_error':velocity,'limit':limit}


def checked_reference(big,small,times,**options):
    coarse = oracle.trajectory(big,small,times,epsilon=1e-13,**options)
    fine = oracle.trajectory(big,small,times,epsilon=1e-15,**options)
    error = 0.
    for a,b in zip(coarse,fine):
        wrapped = {name:{'x':state[:3].tolist(),'v':state[3:].tolist()} for name,state in a.items()}
        error = max(error,*oracle.errors(wrapped,b))
    if error > 1e-9: raise AssertionError('Independent reference did not converge: '+str(error))
    return fine,error


def force_values(c):
    rng = random.Random(1729)
    n = 7; nbig = 3
    masses = [MU,1e-3*MU,3e-6*MU,1e-8*MU,0.,0.,0.]
    positions = [[0.,0.,0.]]+[[rng.uniform(-2,2) for _ in range(3)] for _ in range(n-1)]
    velocities = [[0.,0.,0.]]+[[rng.uniform(-.03,.03) for _ in range(3)] for _ in range(n-1)]
    libraries = {}
    for p in c.profiles:
        p.select(); libraries[p.name] = (load_cpu(),load_gpu() if p.cuda else None)
    for mask in range(512):
        _,small,options = force_setup(mask)
        params = small[0]['params']; ng = [[0.]*5]+[[params.get(key,0)*(-1 if j%2 else 1)
                  for key in ('a1','a2','a3','b','yar')] for j in range(1,n)]
        expected,scale = oracle.high_precision_force(masses,positions,velocities,ng,options['jcen'],nbig,options['pn'])
        flat = lambda rows:[z for row in rows for z in row]
        flag = (1 if any(params.get(k,0) for k in ('a1','a2','a3','yar')) else 0)+(2 if params.get('b',0) else 0)
        for profile in c.profiles:
            cpu,gpu = libraries[profile.name]
            def check(cpu=cpu,gpu=gpu):
                actual = np.array(cpu_force(cpu,masses,flat(positions),flat(velocities),flat(ng),
                        options['jcen'],nbig,options['pn'],flag)).reshape(n,3)
                ratio = np.max(np.abs(actual[1:]-expected[1:])/(1e-12*scale[1:]+1e-30))
                metrics = {'cpu_bound_fraction':float(ratio)}
                if ratio > 1: raise AssertionError('CPU independent force bound exceeded: '+str(ratio))
                if gpu:
                    rc = arr([.001]*n); output = arr([0.]*(3*n))
                    assert gpu.mercury_cuda_upload(n,nbig,int(options['pn']),flag,arr(masses),arr(flat(positions)),
                           arr(flat(velocities)),arr(flat(ng)),arr(options['jcen']),rc,rc) == 0
                    assert gpu.mercury_cuda_force(output) == 0
                    device = np.array(list(output)).reshape(n,3)
                    ratio = np.max(np.abs(device[1:]-expected[1:])/(1e-12*scale[1:]+1e-30))
                    metrics['cuda_bound_fraction'] = float(ratio)
                    if ratio > 1: raise AssertionError('CUDA independent force bound exceeded: '+str(ratio))
                return metrics
            c.record('force-values',f'{profile.name}/mask-{mask:03d}',check,mask=mask)
    for _,gpu in libraries.values():
        if gpu: gpu.mercury_cuda_free()


def force_matrix(c):
    for mask in range(512):
        big,small,options = force_setup(mask)
        references = {}
        for direction in (-1,1):
            # References advance monotonically from the input epoch.
            references[direction],floor = checked_reference(big,small,[direction*4.],**options)
        for profile in c.profiles:
            backend = 'cuda' if profile.cuda else 'cpu'
            for method in METHODS:
                for direction in (-1,1):
                    settings = dict(big=big,small=small,algorithm=method,backend=backend,
                                    stop=direction*4.,step=.0625,interval=4.,tol=1e-13,**options)
                    def check(settings=settings,method=method,direction=direction):
                        if not supported(method,mask): return reject(profile,settings)
                        state,_ = c.simulate(profile,**settings)
                        return accuracy(state,references[direction][0])
                    c.record('force-matrix',f'{profile.name}/{method}/mask-{mask:03d}/{direction}',check,
                             mask=mask,method=method,direction=direction,supported=supported(method,mask))


def scientific(c):
    from extended import conservation, secular_forces
    conservation(c)
    secular_forces(c)
    # Multi-epoch analytic truth, including the no-big-body MVS configuration.
    for mu_scale,a,e in [(1.,.7,0.),(1.,.7,.2),(.01,.2,.2),(3e-6,.01,.1),
                        (1.,1.,.9),(1.,1.,.99)]:
        obj = tilted(body(a=a,e=e,phase=.6))
        obj['v'] = [z*math.sqrt(mu_scale) for z in obj['v']]
        period = 2*math.pi*math.sqrt(a**3/(MU*mu_scale))
        for profile in c.profiles:
            for method in METHODS:
                for direction in (-1,1):
                    # Pericentre-resolving fixed steps are necessary for eccentric maps.
                    step = period*(1-e)**1.5/128
                    horizon = period*(2 if e < .9 else .2)
                    for fraction in (.1,.37,.73,1.):
                        time = direction*round(horizon*fraction/step)*step
                        ref = {'PARTICLE':oracle.kepler(obj,time,MU*mu_scale)}
                        def check(time=time,ref=ref):
                            state,_ = c.simulate(profile,small=[obj],algorithm=method,
                                backend='cuda' if profile.cuda else 'cpu',stop=time,step=step,
                                interval=abs(time),tol=1e-13,central_mass=mu_scale,
                                rcen=min(.005,a*(1-e)*.01),rmax=max(100,4*a))
                            return accuracy(state,ref,distance=a,mu=MU*mu_scale)
                        c.record('scientific',f'kepler/{profile.name}/{method}/{mu_scale}/{e}/{direction}/{fraction}',
                                 check,method=method,eccentricity=e,time=time,central_mass=mu_scale)
    # Full heliocentric interaction law, including semi-active small bodies.
    for population in ('massless','semi-active','all-big'):
        big = [tilted(body('E',mass=3e-6,a=1,e=.016),.3),body('J',mass=.001,a=5.2,e=.04)]
        small = [tilted(body('S',mass=1e-8 if population=='semi-active' else 0,a=2.1,e=.15),-.4)]
        if population == 'all-big':
            small[0]['mass'] = 1e-8; big += small; small = []
        for direction in (-1,1):
            times = [direction*t for t in (3.,17.,64.,365.)]
            refs,floor = checked_reference(big,small,times)
            for profile in c.profiles:
                for method in METHODS:
                    if method == 'MVS' and population == 'semi-active': continue
                    def check(method=method):
                        maximum = 0.; measured = []
                        for time,ref in zip(times,refs):
                            fine_metrics = None; sweep = []
                            for level in range(3):
                                tolerance = (1e-10,1e-12,1e-13)[level]
                                step = .5/2**level if method in ('MVS','HYBRID') else .5
                                state,_ = c.simulate(profile,big=big,small=small,algorithm=method,
                                    backend='cuda' if profile.cuda else 'cpu',stop=time,interval=abs(time),
                                    step=step,tol=tolerance)
                                error = max(oracle.errors(state,ref))
                                sweep.append(error); fine_metrics = accuracy(state,ref) if level==2 else None
                            if sweep[0] > 10*max(floor,1e-13) and sweep[2] > sweep[0]/2:
                                raise AssertionError('Refinement did not improve error: '+str(sweep))
                            measured.append(dict(time=time,errors=sweep,reference_floor=floor))
                            maximum = max(maximum,sweep[-1])
                        return {'maximum_error':maximum,'convergence':measured}
                    c.record('scientific',f'nbody/{profile.name}/{method}/{population}/{direction}',check,
                             method=method,population=population,direction=direction)
    # Force-enabled accuracy versus an independently coded callback, both directions.
    for direction in (-1,1):
        big,small,options = force_setup(511)
        small[0]['params']['a2'] = 2e-9
        times = [direction*t for t in (1.,11.,64.,365.)]
        refs,floor = checked_reference(big,small,times,**options)
        for profile in c.profiles:
            for method in ('BS','RADAU'):
                def check():
                    measured = []
                    for time,ref in zip(times,refs):
                        errors = []
                        for tolerance in (1e-10,1e-12,1e-13):
                            state,_ = c.simulate(profile,big=big,small=small,algorithm=method,
                                backend='cuda' if profile.cuda else 'cpu',stop=time,interval=abs(time),tol=tolerance,**options)
                            errors.append(max(oracle.errors(state,ref)))
                        accuracy(state,ref)
                        if errors[0] > 10*max(floor,1e-13) and errors[2] > errors[0]/2:
                            raise AssertionError('Force-enabled refinement did not improve: '+str(errors))
                        measured.append(dict(time=time,errors=errors,reference_floor=floor))
                    return {'convergence':measured}
                c.record('scientific',f'full-forces/{profile.name}/{method}/{direction}',check,method=method,direction=direction)


def execute(c,group):
    functions = {'force-values':force_values,'force-matrix':force_matrix,'scientific':scientific}
    if group in functions: return functions[group](c)
    from operations import execute as operations
    return operations(c,group)
