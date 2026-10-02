"""Input, workflow, event and device lifecycle checks for the full campaign."""
from pathlib import Path
import itertools
import math
import re
import shutil
import tempfile

import numpy as np
from cases import ROOT, MU, body, prepare, run, dump
import oracle
from scenarios import METHODS, COUNTS, tilted, reject, accuracy, checked_reference


def replace_setting(path,pattern,value,file='param.in'):
    text = (path/file).read_text()
    updated,count = re.subn(r'(^[ \t]*'+pattern+r'.*?=)\s*[^\n]+',lambda m:m[1]+' '+str(value),text,flags=re.I|re.M)
    if count != 1: raise AssertionError(f'Setting {pattern} matched {count} rows')
    (path/file).write_text(updated)


def epoch(path):
    lines = [line for line in (path/'big.dmp').read_text().splitlines() if line.strip() and not line.startswith(')')]
    return float(lines[1].split('=')[-1].replace('D','E'))


def workflow(c):
    for profile in c.profiles:
        def first_run():
            profile.select()
            with tempfile.TemporaryDirectory(prefix='mercuda-guide-') as tmp:
                path = Path(tmp)
                for sample in ROOT.glob('*.in.sample'): shutil.copy(sample,path/sample.name)
                c.command(['make','-f',str(ROOT/'Makefile'),'gen-in'],cwd=path)
                before = (path/'big.in').read_bytes()
                c.command(['make','-f',str(ROOT/'Makefile'),'gen-in'],cwd=path)
                assert (path/'big.in').read_bytes() == before
                for key,value in [('algorithm','bs'),('start time',2458400.5),('stop time',2458765.5),
                                  ('output interval',30.),('timestep',1.),('accuracy parameter','1d-11')]:
                    replace_setting(path,key,value)
                with (path/'param.in').open('a') as file:
                    file.write('\n execution backend = '+('cuda' if profile.cuda else 'cpu')+'\n')
                run(path,timeout=180)
                assert (path/'xv.out').stat().st_size > 0
                replace_setting(path,'minimum interval',30.,file='element.in')
                run(path,executable=ROOT/'element6'); run(path,executable=ROOT/'close6')
                assert list(path.glob('*.aei'))
                return {'postprocessed_bodies':len(list(path.glob('*.aei'))),'final_epoch':epoch(path)}
        c.record('workflow',profile.name+'/guide-example',first_run)
    for target in ('clean-build','rm-gen','clean'):
        def cleanup(target=target):
            with tempfile.TemporaryDirectory(prefix='mercuda-clean-') as tmp:
                path = Path(tmp); build = path/'compiled'; binary = path/'bin'
                build.mkdir(); binary.mkdir(); (build/'sentinel.o').write_text('compiled')
                for name in ('mercury6','element6','close6'): (binary/name).write_text('program')
                files = ['sentinel.in','sentinel.aei','sentinel.clo','sentinel.out','sentinel.dmp','sentinel.tmp']
                for name in files: (path/name).write_text('keep or remove according to target')
                c.command(['make','-f',str(ROOT/'Makefile'),f'BUILD={build}',f'BIN={binary}',f'EXEC={path}',target])
                assert (path/'sentinel.in').exists() == (target != 'clean')
                for name in files[1:]: assert (path/name).exists() == (target == 'clean-build')
                assert build.exists() == (target == 'rm-gen')
                assert (binary/'mercury6').exists() == (target == 'rm-gen')
        c.record('workflow','cleanup/'+target,cleanup)
    # Ignored legacy switches must not silently change dynamics.
    for profile in c.profiles:
        for units,relative,precision in itertools.product(('days','years'),(False,True),('low','medium','high')):
            def settings_check():
                baseline,_ = c.simulate(profile,stop=4.,interval=1.,fragmentation=False)
                state,info = c.simulate(profile,stop=4.,interval=1.,fragmentation=True,
                    time_units=units,relative_time=relative,precision=precision,dump_interval=2,periodic_interval=1)
                metrics = accuracy(state,{name:np.array(o['x']+o['v']) for name,o in baseline.items()})
                return metrics
            c.record('workflow',f'{profile.name}/{units}/{relative}/{precision}',settings_check)


def formats(c):
    from extended import massive_formats
    massive_formats(c)
    for profile in c.profiles:
        for central in (1.,.01,3e-6):
            for e in (0.,.2,.9,1.,1.2):
                for style in ('Cartesian','Asteroidal','Cometary'):
                    if e == 1 and style == 'Asteroidal': continue
                    q = .5 if central >= .01 else .01
                    a = q/(1-e) if e != 1 else None
                    obj = dict(name='INPUT',mass=0.,params={'yar':0.,'a2':0.},
                        x=[q*math.cos(.6),q*math.sin(.6)*math.cos(.37),q*math.sin(.6)*math.sin(.37)],
                        v=[-math.sqrt(MU*central*(1+e)/q)*math.sin(.6),
                           math.sqrt(MU*central*(1+e)/q)*math.cos(.6)*math.cos(.37),
                           math.sqrt(MU*central*(1+e)/q)*math.cos(.6)*math.sin(.37)])
                    def check():
                        profile.select()
                        with tempfile.TemporaryDirectory(prefix='mercuda-format-') as tmp:
                            path = prepare(tmp,small=[obj],central_mass=central,rcen=q*.001,
                                backend='cuda' if profile.cuda else 'cpu',stop=.125,interval=.125,step=.01)
                            if style != 'Cartesian':
                                header = (path/'small.in').read_text().splitlines()[:3]
                                header[1] = ' style = '+style
                                # Both element forms represent the same perihelion state.
                                values = [a if style=='Asteroidal' else q,e,math.degrees(.37),math.degrees(.6),0,0,0,0,0]
                                (path/'small.in').write_text('\n'.join(header)+'\n'+' '.join(f'{v:.17e}' for v in values)+'\n')
                            run(path)
                            return accuracy(dump(path),{'INPUT':oracle.kepler(obj,.125,MU*central)},distance=q,mu=MU*central)
                    c.record('formats',f'{profile.name}/{style}/e-{e}/central-{central}',check,style=style,eccentricity=e)
    # Case aliases, D exponents, comments, defaults and maximum-length identifiers.
    for profile in c.profiles:
        def syntax():
            profile.select()
            with tempfile.TemporaryDirectory() as tmp:
                obj = body('ABCDEFGHIJKLMNOPQRSTUVWXY',yar=2e-12,a2=-3e-12)
                path = prepare(tmp,small=[obj],stop=.125,interval=.125)
                text = (path/'small.in').read_text().replace('yar=','YaR=').replace('a2=','A2=').replace('e-','D-').replace('e+','D+')
                (path/'small.in').write_text(text+'\n) trailing comment\n')
                run(path); result = dump(path)[obj['name']]
                assert result['params']['yar'] == 2e-12 and result['params']['a2'] == -3e-12
                assert result['mass'] == 0.
        c.record('formats',profile.name+'/syntax',syntax)


def epochs(c):
    for profile in c.profiles:
        for method in METHODS:
            for initial in (0.,2451545.):
                for preparation in (-3.25,0.,3.25):
                    for direction in (-1,1):
                        for cadence in (.125,.5,2.):
                            start = initial+preparation; stop = start+direction*4.
                            obj = tilted(body(a=.7,e=.2,**({'yar':-2e-10,'b':.001,'a2':3e-10} if method in ('BS','RADAU') else {})))
                            if obj['params']:
                                # Propagate separately in each time direction if preparation changes direction.
                                prep = oracle.trajectory([], [obj], [preparation],epsilon=1e-15)[0]
                                at_start = {**obj,'x':prep[obj['name']][:3].tolist(),'v':prep[obj['name']][3:].tolist()}
                                ref = oracle.trajectory([], [at_start], [direction*4.],epsilon=1e-15)[0]
                            else: ref = {obj['name']:oracle.kepler(obj,stop-initial)}
                            def check(ref=ref):
                                state,info = c.simulate(profile,small=[obj],epoch=initial,start=start,stop=stop,
                                    algorithm=method,step=.125,interval=cadence,tol=1e-13,
                                    backend='cuda' if profile.cuda else 'cpu')
                                return accuracy(state,ref)
                            c.record('epochs',f'{profile.name}/{method}/{initial}/{preparation}/{direction}/{cadence}',check)
            def zero():
                obj = body(); state,_ = c.simulate(profile,small=[obj],algorithm=method,
                    start=0,stop=0,backend='cuda' if profile.cuda else 'cpu')
                return accuracy(state,{obj['name']:np.array(obj['x']+obj['v'])},limit=1e-12)
            c.record('epochs',f'{profile.name}/{method}/zero-duration',zero)
            def fixed_grid():
                def inspect(path,state,info):
                    actual = epoch(path)
                    if method in ('MVS','HYBRID'):
                        assert abs(actual-3.3) <= .250001
                    else: assert abs(actual-3.3) < 1e-12
                    return accuracy(state,{'PARTICLE':oracle.kepler(body(),actual)})
                return c.simulate(profile,algorithm=method,stop=3.3,step=.5,interval=.1,inspect=inspect)
            c.record('epochs',f'{profile.name}/{method}/off-grid-stop',fixed_grid)
    # Mixed epochs in each ordering, each force coefficient attached to its body.
    for profile in c.profiles:
        for order in (1,-1):
            objects = [tilted(body('A',ep=order*2.,a2=-2e-10,yar=3e-11)),
                       tilted(body('B',a=1.3,ep=-order*2.,a2=4e-10,yar=-5e-11,b=.001))]
            def mixed():
                state,_ = c.simulate(profile,small=objects,start=-3.,stop=4.,tol=1e-13,
                                    backend='cuda' if profile.cuda else 'cpu')
                maximum = 0.
                for obj in objects:
                    reference = oracle.trajectory([], [obj], [4-obj['params']['ep']],epsilon=1e-15)[0]
                    maximum = max(maximum,*oracle.errors({obj['name']:state[obj['name']]},reference))
                    assert state[obj['name']]['params']['a2'] == obj['params']['a2']
                    assert state[obj['name']]['params']['yar'] == obj['params']['yar']
                assert maximum < 1e-8
                return {'maximum_error':maximum}
            c.record('epochs',f'{profile.name}/mixed/{order}',mixed)


def restarts(c):
    for profile in c.profiles:
        for method in METHODS:
            for direction in (-1,1):
                for version in (0,1,2):
                    obj = body(**({'a2':2e-11,'yar':-3e-11,'b':.001} if method in ('BS','RADAU') else {'a2':2e-11} if method in ('MVS','HYBRID') else {}))
                    if version == 1: obj['params'].pop('a2',None)
                    if version == 0: obj['params'].pop('yar',None)
                    def check():
                        profile.select()
                        with tempfile.TemporaryDirectory(prefix='mercuda-restart-') as tmp:
                            backend = 'cuda' if profile.cuda else 'cpu'
                            path = prepare(tmp,small=[obj],algorithm=method,backend=backend,stop=direction*2.,step=.125,interval=1.)
                            run(path)
                            replace_setting(path,'stop time',direction*4.,file='param.dmp')
                            if version == 0:
                                (path/'param.dmp').write_text(re.sub(r'^.*force model version.*\n','',(path/'param.dmp').read_text(),flags=re.M))
                            if version == 1:
                                replace_setting(path,'force model version',1,file='param.dmp')
                                (path/'small.dmp').write_text((path/'small.dmp').read_text().replace('yar=','a2='))
                            # Ordinary inputs cannot overwrite saved dynamics.
                            replace_setting(path,'include relativity','yes' if method not in ('BS','RADAU') else 'no')
                            (path/'small.in').write_text('deliberately unavailable initial conditions\n')
                            run(path)
                            reference = oracle.trajectory([], [obj], [direction*4.],epsilon=1e-15)[0]
                            return accuracy(dump(path),reference)
                    c.record('restarts',f'{profile.name}/{method}/{direction}/version-{version}',check)
    # Unknown, ambiguous and incompatible restart models must fail.
    for profile in c.profiles:
        for kind in ('unknown','ambiguous','legacy-pn','truncated'):
            def bad():
                profile.select()
                with tempfile.TemporaryDirectory() as tmp:
                    path = prepare(tmp,small=[body(yar=1e-12)],pn=True,stop=1,interval=1)
                    run(path)
                    if kind=='unknown': replace_setting(path,'force model version',999,file='param.dmp')
                    elif kind=='ambiguous': replace_setting(path,'force model version',1,file='param.dmp')
                    elif kind=='legacy-pn': (path/'param.dmp').write_text(re.sub(r'^.*force model version.*\n','',(path/'param.dmp').read_text(),flags=re.M))
                    else: (path/'big.dmp').write_text('truncated\n')
                    result = run(path,check=False)
                    assert result.returncode > 0
            c.record('restarts',f'{profile.name}/reject/{kind}',bad)


def numeric_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        try: values = [float(v.replace('D','E')) for v in line.split()]
        except ValueError: continue
        if values:
            if not all(math.isfinite(v) for v in values): raise AssertionError('Nonfinite postprocessor output')
            rows.append(values)
    return rows


def postprocessing(c):
    for profile in c.profiles:
        for frame,precision,units,relative in itertools.product(('Central','Barycentric','Jacobi'),('low','medium','high'),('days','years'),(False,True)):
            def check():
                profile.select()
                with tempfile.TemporaryDirectory(prefix='mercuda-postprocess-') as tmp:
                    big = [body('PLANET',mass=.001,a=5.2,e=.04)]; small = [body('P')]
                    path = prepare(tmp,big=big,small=small,epoch=2451545.,start=2451545.,stop=2451549.,interval=1,precision=precision,
                                   backend='cuda' if profile.cuda else 'cpu')
                    run(path)
                    text = (ROOT/'element.in.sample').read_text()
                    text = text.replace('= Central','= '+frame).replace('365.2d1','0').replace('= years','= '+units)
                    text = text.replace('start time = yes','start time = '+('yes' if relative else 'no'))
                    text = text.replace(' p13e l13.5e a8.5 e8.6 i8.4 g8.4 n8.4 m13e ',' x24.16 y24.16 z24.16 u24.16 v24.16 w24.16 ')
                    (path/'element.in').write_text(text+'P\nABSENT\n')
                    run(path,executable=ROOT/'element6')
                    assert (path/'P.aei').exists()
                    if (path/'ABSENT.aei').exists(): assert not numeric_rows(path/'ABSENT.aei')
                    rows = numeric_rows(path/'P.aei')
                    columns = 9 if units=='years' and not relative else 7
                    assert len(rows)==5 and all(len(row)==columns for row in rows), rows
                    for t,row in enumerate(rows):
                        if columns==9: assert row[:3]==[2000.,1.,1.5+t]
                        else:
                            clock=t/365.25 if units=='years' else t if relative else 2451545.+t
                            assert abs(row[0]-clock)<1e-6
                    if frame in ('Central','Barycentric','Jacobi'):
                        expected = oracle.trajectory(big,small,[0,1,2,3,4],epsilon=1e-15)
                        limit = {'low':.02,'medium':1e-6,'high':1e-8}[precision]
                        maximum = 0.
                        for row,reference in zip(rows,expected):
                            state = {'P':{'x':row[-6:-3],'v':row[-3:]}}
                            target = reference['P'].copy()
                            if frame == 'Barycentric':
                                target -= .001/(1+.001)*reference['PLANET']
                            # Original Mercury's Jacobi output keeps Small bodies heliocentric.
                            maximum = max(maximum,*oracle.errors(state,{'P':target}))
                        assert maximum < limit, (maximum,limit)
                    shutil.copy(ROOT/'close.in.sample',path/'close.in')
                    run(path,executable=ROOT/'close6')
                    return {'records':len(rows),'precision':precision,'frame':frame}
            c.record('postprocessing',f'{profile.name}/{frame}/{precision}/{units}/{relative}',check)


def backends(c):
    for profile in c.profiles:
        for count in (0,1,4095,4096,4097):
            small = [body('P'+str(j),a=2+j*.00001) for j in range(count)]
            for backend in ('cpu','auto','cuda'):
                def check():
                    settings = dict(small=small,stop=.125,interval=.125,backend=backend)
                    if backend=='cuda' and not profile.cuda:
                        return reject(profile,settings,expected='CUDA')
                    state,info = c.simulate(profile,**settings)
                    expected = 'CUDA' if profile.cuda and (backend=='cuda' or backend=='auto' and count>=4096) else 'CPU'
                    assert 'Execution backend: '+expected in info
                    assert len(state)==count
                    return {'selected':expected,'count':count}
                c.record('backends',f'{profile.name}/{backend}/{count}',check)
        def no_backend():
            profile.select()
            with tempfile.TemporaryDirectory() as tmp:
                path = prepare(tmp,stop=1)
                (path/'param.in').write_text(re.sub(r'^.*execution backend.*\n','',(path/'param.in').read_text(),flags=re.M))
                run(path); assert 'Execution backend: CPU' in (path/'info.out').read_text()
        c.record('backends',profile.name+'/omitted',no_backend)


def invalid(c):
    specs = [
        ('duplicate',dict(small=[body('DUP'),body('DUP')]),None,'Duplicate'),
        ('nan-state',dict(small=[{**body(),'x':[float('nan'),1,0]}]),None,'Nonfinite'),
        ('zero-state',dict(small=[{**body(),'x':[0,0,0]}]),None,'state'),
        ('negative-mass',dict(small=[body(mass=-1e-8)]),None,'mass'),
        ('zero-density',dict(small=[body(d=0)]),None,'density'),
        ('zero-interval',dict(interval=0),None,'interval'),
        ('zero-step',dict(step=0),None,'Step'),
        ('zero-tolerance',dict(tol=0),None,'tolerance'),
        ('zero-central',dict(central_mass=0),None,'mass'),
        ('bad-period',dict(periodic_interval=0),None,'interval'),
        ('bad-dump',dict(dump_interval=0),None,'interval'),
        ('bad-backend',dict(backend='nonsense'),None,'backend'),
        ('close-binary',dict(algorithm='CLOSE'),None,'driver'),
        ('wide-binary',dict(algorithm='WIDE'),None,'driver'),
        ('massive-mvs',dict(algorithm='MVS',small=[body(mass=1e-8)]),None,None),
        ('mixed-massive-epochs',dict(small=[body(mass=1e-8,ep=1)]),None,None),
        ('truncated-coordinates',{},lambda p:(p/'small.in').write_text(')O+_06\n style = Cartesian\n P m=0\n 1 0\n'),'coordinates'),
        ('unknown-setting',{},lambda p:(p/'param.in').write_text((p/'param.in').read_text()+' unknown setting = 1\n'),'Unknown'),
        ('unknown-field',{},lambda p:(p/'small.in').write_text((p/'small.in').read_text().replace('m=','unknown=')),None),
        ('bad-style',{},lambda p:(p/'small.in').write_text((p/'small.in').read_text().replace('Cartesian','nonsense')),None),
        ('missing-file',{},lambda p:(p/'small.in').unlink(),None),
        ('missing-message',{},lambda p:(p/'message.in').unlink(),None),
        ('message-index',{},lambda p:(p/'message.in').write_text('999  1 X\n'),'message.in'),
        ('message-length',{},lambda p:(p/'message.in').write_text('  1 99 X\n'),'message.in'),
        ('message-truncated',{},lambda p:(p/'message.in').write_text('  1  1 X\n'),'message.in'),
    ]
    for key in ('a1','a2','a3','b','yar','m','d','ep'):
        specs.append(('nonfinite-'+key,dict(small=[body(**{key:float('inf')})]),None,'Nonfinite'))
    for key in ('a2','yar','a3'):
        obj = body(**{key:1e-10}); obj['v'] = [.01,0,0]
        specs.append(('undefined-'+key,dict(small=[obj]),None,'Undefined'))
    for method in METHODS:
        for backend in ('cpu','cuda'):
            specs.append(('singular-'+method+'-'+backend,
                          dict(big=[body('A',mass=1e-8),body('B',mass=1e-8)],small=[],algorithm=method,backend=backend),
                          None,'coincident'))
    for profile in c.profiles:
        for name,settings,edit,expected in specs:
            c.record('invalid',profile.name+'/'+name,lambda settings=settings,edit=edit,expected=expected:
                     reject(profile,settings,edit,expected))


def stress(c):
    from extended import capacity
    capacity(c)
    for profile in c.profiles:
        for method in METHODS:
            for count in COUNTS+(100000,):
                def check():
                    objects = [body('P'+str(j),a=2+j%1024*.0001,phase=j*2.399963229728653) for j in range(count)]
                    state,info = c.simulate(profile,small=objects,algorithm=method,
                        backend='cuda' if profile.cuda else 'cpu',stop=.25,step=.125,interval=.25)
                    assert len(state)==count
                    assert all(math.isfinite(z) for obj in state.values() for z in obj['x']+obj['v'])
                    maximum = 0.
                    for j in range(0,count,max(1,count//64)):
                        obj = objects[j]; expected = {obj['name']:oracle.kepler(obj,.25)}
                        maximum = max(maximum,*oracle.errors({obj['name']:state[obj['name']]},expected))
                    assert maximum < 1e-8
                    return {'count':count,'sampled_error':maximum,'finite_states':len(state)}
                c.record('stress',f'{profile.name}/{method}/{count}',check,method=method,count=count)
    # Several hundred massive bodies exercise prefixes and reduction tails.
    big = [tilted(body('B'+str(j),mass=1e-12,a=1+j*.01,phase=j*.37)) for j in range(257)]
    refs,floor = checked_reference(big,[],[.125])
    for profile in c.profiles:
        for method in METHODS:
            def massive():
                state,_ = c.simulate(profile,big=big,small=[],algorithm=method,stop=.125,step=.125,interval=.125,
                                     backend='cuda' if profile.cuda else 'cpu')
                metrics=accuracy(state,refs[0])
                return dict(metrics,massive_bodies=257,reference_floor=floor)
            c.record('stress',f'{profile.name}/{method}/massive-257',massive)


def execute(c,group):
    functions = {'workflow':workflow,'formats':formats,'epochs':epochs,'restarts':restarts,
                 'postprocessing':postprocessing,'backends':backends,'invalid':invalid,'stress':stress}
    if group in functions: return functions[group](c)
    from advanced import execute as advanced
    return advanced(c,group)
