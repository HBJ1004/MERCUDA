"""Required close6 checks: independent invariants, real writers and host memory."""
from pathlib import Path
import itertools
import math
import os
import random
import shutil
import subprocess
import tempfile
import unittest
import close_reference as ref
from cases import ROOT, body, prepare, run
from test_close import Close


def invoke(path,executable=None,check=True,timeout=30):
    if executable is None: return run(path,executable=ROOT/'close6',check=check,timeout=timeout)
    result=subprocess.run([str(executable)],cwd=path,capture_output=True,text=True,timeout=timeout,
                          env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1'))
    if check and result.returncode: raise AssertionError(result.stdout+result.stderr)
    return result


def fixture_check(records, *, names=(), files=None, units='days',relative=False, executable=None):
    with tempfile.TemporaryDirectory(prefix='mercuda-close-validation-') as tmp:
        path=ref.fixture(tmp,records=records,names=names,files=files,units=units,relative=relative)
        invoke(path,executable=executable,timeout=180)
        total=0; errors={}
        inputs=records if files is None else [row for rows in files.values() for row in rows]
        for output in path.glob('*.clo'):
            # Resolve original body IDs, including sanitized filenames.
            name=output.read_text().splitlines()[1].strip()
            metrics=ref.validate_rows(output,inputs,name,absolute_days=units=='days' and not relative)
            total+=metrics.pop('rows')
            for key,value in metrics.items(): errors[key]=max(errors.get(key,0),value)
            assert not any('*' in token for row in ref.rows(output) for token in row[-7:])
        return {'output_files':len(list(path.glob('*.clo'))),'validated_rows':total,**errors}


def reject_records(records, *, executable=None, edit=None):
    with tempfile.TemporaryDirectory(prefix='mercuda-close-invalid-') as tmp:
        path=ref.fixture(tmp,records=records)
        if edit: edit(path)
        result=invoke(path,executable=executable,check=False)
        assert result.returncode>0,(result.returncode,result.stdout,result.stderr)
        assert 'MERCURY:' in result.stderr,result.stderr
        assert not any(text in result.stderr for text in ('SIGFPE','SIGSEGV','AddressSanitizer:','runtime error:'))
        assert not list(path.glob('*.clo'))
        return {'controlled_rejection':True}


def calendar_reference(jd):
    # Independent integer calendar algorithm (Julian/Gregorian JDN inversion).
    z=math.floor(jd+.5); fraction=jd+.5-z
    if z>=2299161:
        a=z+32044; b=(4*a+3)//146097; c=a-(146097*b)//4
    else:
        b=0; c=z+32082
    d=(4*c+3)//1461; e=c-(1461*d)//4; m=(5*e+2)//153
    day=e-(153*m+2)//5+1+fraction
    month=m+3-12*(m//10)
    year=(100*b if z>=2299161 else 0)+d-4800+m//10
    return year,month,day


def close6(c):
    writer_results={}
    for profile in c.profiles:
        profile.select()
        # Expose individual fast regressions in the full report as well as quick CI.
        for method in unittest.defaultTestLoader.getTestCaseNames(Close):
            def regression(method=method):
                result=unittest.TestResult(); Close(method).run(result)
                assert not result.errors and not result.failures,(result.errors,result.failures)
                assert result.testsRun==1 and not result.skipped
                return {'tests':1}
            c.record('close6',profile.name+'/regression/'+method,regression)
        for central,ratio,precision in itertools.product((1,.01,3e-6),(0,.001,.1),(1,2,3)):
            mass=central*ratio
            records=ref.header(central=central,masses=[mass,0],precision=precision)
            records += [ref.encounter(first=ref.orbit(e=e,inclination=inclination,mass=mass,central=central),central=central,time=2451545+j)
                for j,(e,inclination) in enumerate(itertools.product((0,.1,.9,.999999,1-1e-8,1+1e-8,1.2,20),(0,15,90,165,180)))]
            c.record('close6',f'{profile.name}/elements/{central}/{mass}/{precision}',lambda r=records:fixture_check(r),central_mass=central,body_mass=mass,precision=precision)
        rng=random.Random(1729)
        records=ref.header()
        for j in range(256):
            first=ref.orbit(e=rng.uniform(0,2),inclination=rng.uniform(0,180),phase=rng.uniform(0,2*math.pi),q=10**rng.uniform(-1,1))
            records.append(ref.encounter(first=first,time=2451545+j))
        c.record('close6',profile.name+'/random-256',lambda:fixture_check(records),seed=1729)
        overflow=ref.header(rmax=1e12)+[ref.encounter(first=ref.orbit(q=1e7),rmax=1e12,time=1e25,distance=123)]
        c.record('close6',profile.name+'/scientific-notation',lambda:fixture_check(overflow))
        clocks=(-999999999999.25,-1000000.25,-1.25,0,.25,2299159.75,2299160.75,2415079.75,2451603.75,2451545.75,999999999999.25)
        records=ref.header()+[ref.encounter(time=time) for time in clocks]
        for units,relative in itertools.product(('days','years'),(False,True)):
            def clock_check(units=units,relative=relative):
                with tempfile.TemporaryDirectory() as tmp:
                    path=ref.fixture(tmp,records=records,units=units,relative=relative)
                    invoke(path); actual=ref.rows(path/'PLANET.clo'); refs=ref.references(records)
                    origin=float(ref.floating(records[0][5:13])); max_error=0.
                    assert len(actual)==len(clocks)
                    for row,reference in zip(actual,refs):
                        time=reference['time']
                        if units=='years' and not relative:
                            year,month,day=calendar_reference(time)
                            assert int(row[0])==year and int(row[1])==month,(time,row,(year,month,day))
                            error=abs(float(row[2])-day); limit=5e-6+64*math.ulp(time)
                        else:
                            expected=(time-origin) if relative else time
                            if units=='years': expected/=365.25
                            token=row[0]; error=abs(float(token)-expected)
                            resolution=10.**(int(token.split('E')[-1])-8) if 'E' in token else 10.**(-7 if units=='years' else -5)
                            limit=.5*resolution+64*math.ulp(expected)
                        assert error<=limit,(row,time,error,limit)
                        max_error=max(max_error,error)
                    return {'time_formats':1,'rows':len(actual),'maximum_time_error':max_error}
            c.record('close6',f'{profile.name}/time/{units}/{relative}',clock_check)
        for count in (0,1,255,256,257,1999,2000,2001,4096,4097,100000):
            def capacity(count=count):
                names=['P'+str(j) for j in range(count)]
                records=ref.header(names=names,masses=[0]*count)
                if count>1: records += [ref.encounter(codes=(1,count))]
                selected=['ABSENT'] if count==0 else [names[0]] if count==1 else [names[0],names[-1]]
                return fixture_check(records,names=selected)
            c.record('close6',f'{profile.name}/capacity/{count}',capacity,bodies=count)
        for count in (255,256,257,513):
            def batching(count=count):
                names=['P'+str(j) for j in range(count)]
                records=ref.header(names=names,masses=[0]*count)+[ref.encounter(codes=(1,count))]
                metrics=fixture_check(records)
                assert metrics['output_files']==count and metrics['validated_rows']==2,metrics
                return metrics
            c.record('close6',f'{profile.name}/batch/{count}',batching)
        def sparse_updates():
            first=ref.header(names=['A','B'],codes=[4097,9000],masses=[0,0])+[ref.encounter(codes=(4097,9000))]
            second=ref.header(names=['C','A','D'],codes=[2,4097,19],masses=[0,.1,0],time=2451546)
            second+=[ref.encounter(first=ref.orbit(mass=.1),codes=(4097,19),time=2451547)]
            return fixture_check(first+second)
        c.record('close6',profile.name+'/sparse-changing-population',sparse_updates)
        for count in (1,2,50):
            files={f'{j}.ce':ref.header(time=2451545+j)+[ref.encounter(time=2451545+j+.5)] for j in range(count)}
            c.record('close6',f'{profile.name}/input-files/{count}',lambda f=files:fixture_check(None,files=f,relative=True))
        def names_and_sanitization():
            records=ref.header(names=['ABCDEFGHIJKLMNOPQRSTUVWXY','A/B.C:D&E*'])+[ref.encounter()]
            metrics=fixture_check(records)
            assert metrics['output_files']==2
            return metrics
        c.record('close6',profile.name+'/names-and-sanitization',names_and_sanitization)
        records=ref.header()+[ref.encounter()]
        edits={
            'zero-files':lambda p:(p/'close.in').write_text(')O+_06\n0\n'),
            'too-many-files':lambda p:(p/'close.in').write_text(')O+_06\n51\n'),
            'missing-setting':lambda p:(p/'close.in').write_text(')O+_06\n1\nce.out\n'),
            'invalid-units':lambda p:(p/'close.in').write_text(')O+_06\n1\nce.out\ncenturies\nno\n'),
            'invalid-relative':lambda p:(p/'close.in').write_text(')O+_06\n1\nce.out\ndays\nmaybe\n'),
            'long-name':lambda p:(p/'close.in').write_text(')O+_06\n1\nce.out\ndays\nno\n'+'A'*26+'\n'),
            'missing-input':lambda p:(p/'ce.out').unlink(),
            'missing-config':lambda p:(p/'close.in').unlink(),
            'missing-message':lambda p:(p/'message.in').unlink(),
            'corrupt-message':lambda p:(p/'message.in').write_text('999  1 x\n'),
        }
        for name,edit in edits.items():
            c.record('close6',profile.name+'/invalid/'+name,lambda edit=edit:reject_records(records,edit=edit))
        cases={
            'over-capacity-count': [records[0][:13]+ref.digits(224**3-1,3)*2+records[0][19:]],
            'duplicate-name': [records[0],records[1],records[2][:3]+records[1][3:28]+records[2][28:]],
            'zero-metadata-code': [records[0],b' '*3+records[1][3:],records[2]],
            'empty-body-name': [records[0],records[1][:3]+b' '*25+records[1][28:],records[2]],
            'negative-density': [records[0],records[1][:60]+ref.float_bytes(-1),records[2]],
            'invalid-radial-range': ref.header(rcen=100,rmax=100),
            'negative-central': ref.header(central=-1),
            'invalid-algorithm':[records[0][:3]+b'99'+records[0][5:],*records[1:]],
            'negative-distance':records[:3]+[records[3][:17]+ref.float_bytes(-1)+records[3][25:]],
            'old-header': [b'\x0c5a'+records[0][3:]],
            'unknown-type':records[:3]+[b'\x0c6x'+records[3][3:]],
            'missing-body':records[:2],
            'inactive-code':records+ref.header(names=['A'],codes=[9],masses=[0])+[ref.encounter()],
            'calendar-range':ref.header()+[ref.encounter(time=1e20)],
        }
        for name,data in cases.items():
            if name=='calendar-range':
                edit=lambda p:(p/'close.in').write_text(')O+_06\n1\nce.out\nyears\nno\n')
            else: edit=None
            c.record('close6',profile.name+'/invalid/'+name,lambda data=data,edit=edit:reject_records(data,edit=edit))
        rng=random.Random(9831)
        event=records[-1]
        for j in range(256):
            if j%3==0: mutated=event[:rng.randrange(73)]
            elif j%3==1:
                at=rng.choice(list(range(3,11))+list(range(17,73)))
                mutated=event[:at]+bytes([rng.randrange(1,10)])+event[at+1:]
            else: mutated=event[:11]+ref.digits(100+rng.randrange(10000),3)+event[14:]
            c.record('close6',f'{profile.name}/mutation/{j}',lambda m=mutated:reject_records(records[:3]+[m]),seed=9831)
        # Real producer matrix: no assertion that MVS resolves close encounters.
        for algorithm,sign,precision in itertools.product(('BS','BS2','RADAU','MVS','HYBRID'),(-1,1),('low','medium','high')):
            def writer(algorithm=algorithm,sign=sign,precision=precision,forces=False):
                profile.select()
                planet=body('PLANET',mass=1e-6 if algorithm=='HYBRID' else 1e-12,e=0,r=10000)
                small=body('PARTICLE',**({'a1':1e-12,'a2':2e-12,'a3':1e-12,'yar':3e-12,'b':1e-4} if forces else {}))
                small['x']=[1.001,-sign*.005,0]; small['v']=[0,.1,0]
                with tempfile.TemporaryDirectory() as tmp:
                    path=prepare(tmp,algorithm=algorithm,big=[planet],small=[small],backend='cuda' if profile.cuda else 'cpu',
                                 stop=sign*.1,step=.001,interval=.1,precision=precision,pn=forces)
                    run(path,timeout=180)
                    records=(path/'ce.out').read_bytes().split(b'\n')
                    records=[r for r in records if r]
                    refs=ref.references(records)
                    if algorithm!='MVS': assert refs,'Fixture did not record an encounter'
                    (path/'close.in').write_text(')O+_06\n1\nce.out\ndays\nno\n')
                    invoke(path)
                    for name in ('PLANET','PARTICLE'): ref.validate_rows(path/(name+'.clo'),records,name,absolute_days=True)
                    if algorithm=='HYBRID':
                        # Continue using real dumps; new headers must retain a valid ID map.
                        text=(path/'param.dmp').read_text(); import re
                        text=re.sub(r'(stop time.*?=)\s*[^\n]+',r'\g<1> '+str(sign*.2),text,flags=re.I)
                        (path/'param.dmp').write_text(text); run(path,timeout=180)
                        for output in path.glob('*.clo'): output.unlink()
                        records=[r for r in (path/'ce.out').read_bytes().split(b'\n') if r]; invoke(path)
                        for name in ('PLANET','PARTICLE'): ref.validate_rows(path/(name+'.clo'),records,name,absolute_days=True)
                    refs=ref.references(records)
                    writer_results[(profile.name,algorithm,sign,precision,forces)]=refs
                    return {'algorithm':algorithm,'recorded_encounters':len(refs),'forces':forces}
            c.record('close6',f'{profile.name}/writer/{algorithm}/{sign}/{precision}',writer)
            if algorithm in ('BS','RADAU') and precision=='high':
                c.record('close6',f'{profile.name}/writer-forces/{algorithm}/{sign}',lambda writer=writer:writer(forces=True))
        for algorithm in ('BS','BS2','RADAU','HYBRID'):
            def removal_writer(algorithm=algorithm):
                profile.select()
                a=body('A',mass=1e-8,e=0,d=1e-6); b=body('B',mass=2e-8,e=0,d=1e-6)
                b['x'][0]+=.0003
                impact=body('IMPACT'); impact['x']=[.006,0,0]; impact['v']=[-.1,0,0]
                escape=body('ESCAPE'); escape['x']=[99.9,0,0]; escape['v']=[10,0,0]
                with tempfile.TemporaryDirectory() as tmp:
                    path=prepare(tmp,algorithm=algorithm,big=[a,b],small=[impact,escape,body('KEEP',a=2)],
                                 backend='cuda' if profile.cuda else 'cpu',collisions=True,
                                 stop=.03,step=.001,interval=.01,periodic_interval=1)
                    run(path,timeout=180)
                    from cases import dump
                    assert set(dump(path,'big.dmp'))=={'B'}
                    assert set(dump(path))=={'KEEP'}
                    records=[r for r in (path/'ce.out').read_bytes().split(b'\n') if r]
                    return fixture_check(records)
            c.record('close6',f'{profile.name}/writer-removals/{algorithm}',removal_writer)
        def golden():
            directory=ROOT/'tests/fixtures/close6'
            records=[r for r in (directory/'upstream.ce').read_bytes().split(b'\n') if r]
            return fixture_check(records)
        c.record('close6',profile.name+'/upstream-golden',golden)
    if not c.args.cpu_only:
        for kind,algorithm,sign,forces in itertools.product(('opt','debug'),('BS','BS2','RADAU','MVS','HYBRID'),(-1,1),(False,True)):
            if forces and algorithm not in ('BS','RADAU'): continue
            def consistency(kind=kind,algorithm=algorithm,sign=sign,forces=forces):
                first=writer_results[('cpu-'+kind,algorithm,sign,'high',forces)]
                second=writer_results[('cuda-'+kind,algorithm,sign,'high',forces)]
                assert len(first)==len(second)
                maximum=0.
                for a,b in zip(first,second):
                    assert a['names']==b['names']
                    assert abs(a['time']-b['time'])<=1e-6
                    assert abs(a['distance']-b['distance'])<=1e-6
                    for sa,sb in zip(a['states'],b['states']):
                        error=max(abs(x-y)/(1 if j<3 else math.sqrt(ref.MU)) for j,(x,y) in enumerate(zip(sa,sb)))
                        assert error<=1e-6,(algorithm,error)
                        maximum=max(maximum,error)
                return {'paired_records':len(first),'maximum_scaled_state_difference':maximum}
            c.record('close6',f'consistency/{kind}/{algorithm}/{sign}/{forces}',consistency)
    host_memory(c)


def host_memory(c):
    build=ROOT/'build/validation/close-asan'
    flags='-O1 -g -ffixed-line-length-none -ffp-contract=off -fcheck=all -ffpe-trap=invalid,zero,overflow -fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie -no-pie'
    def compile_():
        c.command(['make','CUDA=0',f'BUILD={build}',f'BIN={build}',f'FFLAGS={flags}',str(build/'close6')])
        return {'instrumented':'close6 + mercury_close + mercury_support','address_and_undefined':True}
    row=c.record('close6','host-memory/build',compile_)
    if row['status']!='pass': return
    binary=build/'close6'
    records=ref.header()+[ref.encounter(),ref.encounter(first=([0,0,1],[0,0,.01])),
        ref.encounter(first=([1,2,3],[.01,.02,.03]))]
    c.record('close6','host-memory/valid-and-radial',lambda:fixture_check(records,executable=binary))
    records=ref.header(names=['P'+str(j) for j in range(513)],masses=[0]*513)+[ref.encounter(codes=(1,513))]
    c.record('close6','host-memory/batching',lambda:fixture_check(records,executable=binary))
    for j in range(16):
        records=ref.header()+[ref.encounter()[:j*4]]
        c.record('close6',f'host-memory/truncated/{j}',lambda r=records:reject_records(r,executable=binary))
