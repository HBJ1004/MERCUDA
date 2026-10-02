"""Conservation, massive element formats and output-capacity arithmetic."""
from pathlib import Path
import math
import tempfile
import numpy as np
from cases import MU, body, prepare, run, dump
import oracle
from scenarios import METHODS,accuracy


def capacity(c):
    for profile in c.profiles:
        def check():
            directory = profile.build/'tests'; directory.mkdir(exist_ok=True)
            source = directory/'capacity.f90'
            source.write_text("program probe\ninteger n,c\ncommon /mercury_sizes/ n,c\ncall mercury_capacity('element')\nprint *,n\nend program\n")
            exe = directory/'capacity'
            c.command(['gfortran',*profile.flags.split(),str(source),str(profile.build/'mercury_support.o'),'-o',str(exe)])
            def encoded(n):
                return bytes([32+n//224**2,32+n//224%224,32+n%224])
            count = 0
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)
                (path/'element.in').write_text('input files = 1\nxv.out\n')
                for nb,ns in [(0,0),(1,0),(0,1),(257,4097),(0,11239423),(11239423,0),(1,11239423)]:
                    (path/'xv.out').write_bytes(b'\x0c6a'+b' '*10+encoded(nb)+encoded(ns)+b'X\n')
                    result=c.command([str(exe)],cwd=path,check=False)
                    if nb+ns+1>11239424:
                        assert result.returncode>0 and 'encoding' in result.stderr
                    else:
                        assert result.returncode==0 and int(result.stdout)==nb+ns+1
                    count+=1
                for data in (b'\x0c6a\n',b'\x0c6a'+b' '*10+b'\x00'+b' '*5+b'\n',b'corrupt\n'):
                    (path/'xv.out').write_bytes(data)
                    result=c.command([str(exe)],cwd=path,check=False)
                    assert result.returncode>0
                    count+=1
                for nfiles in (0,51):
                    (path/'element.in').write_text(f'input files = {nfiles}\n')
                    assert c.command([str(exe)],cwd=path,check=False).returncode>0
                    count+=1
            return {'arithmetic_cases':count,'maximum_capacity':11239424,'large_arrays_allocated':False}
        c.record('stress',profile.name+'/encoding-capacity',check)


def massive_formats(c):
    mass=3e-6; q=.5
    for profile in c.profiles:
        for e in (0.,.2,.9,1.,1.2):
            for style in ('Cartesian','Asteroidal','Cometary'):
                if e==1 and style=='Asteroidal': continue
                obj=dict(name='BIG',mass=mass,params={},x=[q,0,0],v=[0,math.sqrt(MU*(1+mass)*(1+e)/q),0])
                def check():
                    profile.select()
                    with tempfile.TemporaryDirectory() as tmp:
                        path=prepare(tmp,big=[obj],small=[],stop=.125,interval=.125,backend='cuda' if profile.cuda else 'cpu')
                        if style!='Cartesian':
                            first=q if style=='Cometary' else q/(1-e)
                            (path/'big.in').write_text(f')O+_06\n style = {style}\n epoch = 0\n BIG m={mass}\n {first:.17e} {e} 0 0 0 0 0 0 0\n')
                        run(path)
                        return accuracy(dump(path,'big.dmp'),{'BIG':oracle.kepler(obj,.125,MU*(1+mass))})
                c.record('formats',f'{profile.name}/massive/{style}/{e}',check)


def conservation(c):
    # A pure two-body orbit has exact invariants, without dissipative forces.
    initial=body(a=.7,e=.4,phase=.6)
    x=np.array(initial['x']); v=np.array(initial['v'])
    energy=np.dot(v,v)/2-MU/np.linalg.norm(x); angular=np.cross(x,v)
    period=2*math.pi*math.sqrt(.7**3/MU)
    for profile in c.profiles:
        for method in METHODS:
            for direction in (-1,1):
                def check():
                    step=period/2048
                    maximum_energy=maximum_angular=0.
                    for turns in (1,3,8):
                        time=direction*turns*period
                        state,_=c.simulate(profile,small=[initial],algorithm=method,stop=time,step=step,
                                          interval=abs(time),tol=1e-13,backend='cuda' if profile.cuda else 'cpu')
                        x=np.array(state['PARTICLE']['x']); v=np.array(state['PARTICLE']['v'])
                        de=abs((np.dot(v,v)/2-MU/np.linalg.norm(x))/energy-1)
                        dh=np.linalg.norm(np.cross(x,v)-angular)/np.linalg.norm(angular)
                        maximum_energy=max(maximum_energy,de); maximum_angular=max(maximum_angular,dh)
                    assert max(maximum_energy,maximum_angular)<1e-8
                    return {'energy_relative_error':float(maximum_energy),'angular_momentum_relative_error':float(maximum_angular)}
                c.record('scientific',f'{profile.name}/{method}/invariants/{direction}',check)


def secular_forces(c):
    for profile in c.profiles:
        for method in ('BS','RADAU'):
            def precession():
                a=.387; e=.2056; revolutions=30
                period=2*math.pi*math.sqrt(a**3/MU)
                state,_=c.simulate(profile,small=[body(a=a,e=e)],algorithm=method,pn=True,
                                  backend='cuda' if profile.cuda else 'cpu',stop=revolutions*period,
                                  interval=period,tol=1e-13)
                x=np.array(state['PARTICLE']['x']); v=np.array(state['PARTICLE']['v'])
                eccentric=np.cross(v,np.cross(x,v))/MU-x/np.linalg.norm(x)
                observed=math.atan2(eccentric[1],eccentric[0])
                expected=revolutions*6*math.pi*MU/(a*(1-e*e)*oracle.LIGHT**2)
                discrepancy=abs(observed/expected-1)
                assert discrepancy<.002
                return {'relative_precession_error':discrepancy,'expected_radians':expected}
            c.record('scientific',f'{profile.name}/{method}/PN-precession',precession)
            for direction in (-1,1):
                for sign in (-1,1):
                    def drift():
                        e=.1; coefficient=sign*1e-10; period=2*math.pi/math.sqrt(MU)
                        duration=direction*10*period
                        state,_=c.simulate(profile,small=[body(e=e,yar=coefficient)],algorithm=method,
                                          backend='cuda' if profile.cuda else 'cpu',stop=duration,
                                          interval=period,tol=1e-13,step=3)
                        x=np.array(state['PARTICLE']['x']); v=np.array(state['PARTICLE']['v'])
                        axis=1/(2/np.linalg.norm(x)-np.dot(v,v)/MU)
                        expected=2*coefficient/(math.sqrt(MU)*(1-e*e))*duration
                        discrepancy=abs((axis-1)/expected-1)
                        assert discrepancy<.003
                        return {'relative_drift_error':float(discrepancy),'axis_change_au':float(axis-1),
                                'expected_change_au':expected,'direction':direction,'coefficient':coefficient}
                    c.record('scientific',f'{profile.name}/{method}/Yarkovsky/{direction}/{sign}',drift)


def exact_parabolic(c):
    for profile in c.profiles:
        for method in METHODS:
            for preparation in (-.125,0.,.125):
                for direction in (-1,1):
                    def check():
                        profile.select()
                        stop=preparation+direction*.125
                        with tempfile.TemporaryDirectory() as tmp:
                            path=prepare(tmp,small=[],central_mass=1/MU,algorithm=method,start=preparation,stop=stop,
                                         interval=.125,step=.125/16,tol=1e-13,backend='cuda' if profile.cuda else 'cpu')
                            (path/'small.in').write_text(')O+_06\n style = Cometary\n EXACT m=0 ep=0\n .5 1 0 0 0 0 0 0 0\n')
                            run(path)
                            initial=dict(x=[.5,0,0],v=[0,2,0])
                            return accuracy(dump(path),{'EXACT':oracle.kepler(initial,stop,1.)},mu=1.)
                    c.record('formats',f'{profile.name}/{method}/exact-parabolic/{preparation}/{direction}',check)
