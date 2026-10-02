import ctypes as C
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
import unittest
from cases import ROOT, MU, body, prepare, run, dump
from force_library import load_cpu, load_gpu, cpu_force, arr, I, D, routine

class Forces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu=load_cpu(); cls.gpu=load_gpu()

    def test_pr_preserved(self):
        self.assertEqual(routine((ROOT/'mercury6_2.for').read_text(),'mfo_pr'),
                         routine((ROOT/'tests/fixtures/original_pr.for').read_text(),'mfo_pr'))
        rng=random.Random(7); n=64
        m=arr([MU]+[0 if i%2 else 1e-9 for i in range(n-1)])
        x=arr([rng.uniform(-4,4) for _ in range(3*n)])
        v=arr([rng.uniform(-.1,.1) for _ in range(3*n)])
        ng=arr([rng.uniform(-.01,.1) for _ in range(4*n)])
        actual,expected=arr([0.]*(3*n)),arr([0.]*(3*n))
        self.cpu.mfo_pr_(C.byref(I(n)),C.byref(I(5)),m,x,v,actual,ng)
        self.cpu.original_pr_(C.byref(I(n)),C.byref(I(5)),m,x,v,expected,ng)
        self.assertEqual(bytes(actual),bytes(expected))

    def test_pn_analytic_and_reversal(self):
        x=[0,0,0,.4,.3,.1]; v=[0,0,0,-.01,.025,.003]; m=[MU,0]; n=I(2)
        a=arr([0.]*6)
        self.cpu.mfo_pn_(C.byref(n),C.byref(n),arr(m),arr(x),arr(v),a)
        r=math.sqrt(sum(k*k for k in x[3:])); rv=sum(a*b for a,b in zip(x[3:],v[3:])); v2=sum(k*k for k in v[3:]); light=299792458*86400/1.495978707e11
        expected=[MU/(light**2*r**3)*((4*MU/r-v2)*p+4*rv*u) for p,u in zip(x[3:],v[3:])]
        for val,ref in zip(a[3:],expected): self.assertAlmostEqual(val/ref,1.,places=14)
        a2=arr([0.]*6)
        self.cpu.mfo_pn_(C.byref(n),C.byref(n),arr(m),arr(x),arr([-k for k in v]),a2)
        self.assertEqual(bytes(a),bytes(a2))

    def test_exact_parabolic_mean_anomaly(self):
        # D=+/-1 gives x=0, y=+/-1 and Barker mean D+D^3/3.
        for sign in (-1,1):
            for name in ('mco_x2el_','element_x2el_'):
                values=[D(x) for x in [1,0,sign,0,-sign,1,0]+[0]*6]
                getattr(self.cpu,name)(*[C.byref(x) for x in values])
                self.assertEqual(values[7].value,.5)
                self.assertEqual(values[8].value,1.)
                self.assertAlmostEqual(values[-1].value,sign*4/3,places=14)

    def test_hyperbolic_mean_anomaly_is_unbounded(self):
        e=1.2; axis=2.5
        for anomaly in (-3.,3.):
            den=e*math.cosh(anomaly)-1
            inputs=[1,axis*(e-math.cosh(anomaly)),axis*math.sqrt(e*e-1)*math.sinh(anomaly),0,
                    -math.sqrt(1/axis)*math.sinh(anomaly)/den,
                    math.sqrt(1/axis)*math.sqrt(e*e-1)*math.cosh(anomaly)/den,0]
            for name in ('mco_x2el_','element_x2el_'):
                values=[D(x) for x in inputs+[0]*6]
                getattr(self.cpu,name)(*[C.byref(x) for x in values])
                self.assertAlmostEqual(values[-1].value,e*math.sinh(anomaly)-anomaly,delta=1e-12)

    def test_yar_inverse_square_and_direction(self):
        # Includes beyond the cometary cutoff and a massive big body.
        for r in [1.,2.,20.]:
            for sign in [-1,1]:
                a=arr([0.]*6); n=I(2); A2=sign*1e-10
                self.cpu.mfo_ngf_(C.byref(n),arr([0,0,0,r,0,0]),arr([0,0,0,.01,.02,0]),a,arr([0,0,0,0,0,0,0,0,0,A2]))
                self.assertEqual(a[3],0.)
                self.assertAlmostEqual(a[4]/(A2/r**2),1.,places=14)
                self.assertEqual(a[5],0.)

    def test_cometary_a2_and_yar_are_independent(self):
        # At 1 AU both laws are nearly normalized alike; at 2/20 AU they differ.
        for radius in [1.,2.,20.]:
            for comet in [0.,-2e-10,2e-7]:
                for thermal in [0.,3e-11]:
                    ng=[0.,0.,0.,0.,0.,0.,comet,0.,0.,thermal]
                    a=arr([0.]*6); n=I(2)
                    self.cpu.mfo_ngf_(C.byref(n),arr([0,0,0,radius,0,0]),
                                     arr([0,0,0,.01,.02,0]),a,arr(ng))
                    q=radius/2.808
                    g=.111262*q**(-2.15)*(1+q**5.093)**(-4.6142)
                    expected=(comet*g if radius**2<88 or abs(comet)>1e-7 else 0)+thermal/radius**2
                    self.assertAlmostEqual(a[4],expected,delta=max(abs(expected)*2e-14,1e-30))
                    self.assertEqual(a[3],0.);self.assertEqual(a[5],0.)

    def test_gpu_force_matrix(self):
        if self.gpu is None: self.skipTest('CUDA unavailable')
        rng=random.Random(19); n=200
        m=[MU,3e-6*MU,1e-3*MU]+[0.]*(n-3)
        # Semi-active bodies perturb the planets and central body, but not
        # each other. Keep massless entries too, to exercise PR mass gating.
        for j in range(3,n,7): m[j]=1e-8*MU
        x=[0.,0.,0.]+[rng.uniform(-4,4) for _ in range(3*(n-1))]
        v=[0.,0.,0.]+[rng.uniform(-.02,.02) for _ in range(3*(n-1))]
        for pn in [False,True]:
            for flag in range(4):
                ng=[0.,0.,0.,0.,0.]
                for j in range(1,n): ng.extend([1e-12 if flag&1 else 0,(-1)**j*2e-11 if flag&1 else 0,3e-13 if flag&1 else 0,.001 if flag&2 else 0,(-1)**j*4e-11 if flag&1 else 0])
                jcen=[1e-7,-1e-10,1e-13]
                expected=cpu_force(self.cpu,m,x,v,ng,jcen,3,pn,flag)
                rc=arr([.01]*n); actual=arr([0.]*(3*n))
                self.assertEqual(self.gpu.mercury_cuda_upload(n,3,int(pn),flag,arr(m),arr(x),arr(v),arr(ng),arr(jcen),rc,rc),0)
                self.assertEqual(self.gpu.mercury_cuda_force(actual),0)
                error=max(abs(a-b)/max(abs(b),1e-20) for a,b in zip(actual[3:],expected[3:]))
                self.assertLess(error,2e-12,(pn,flag,error))
        self.gpu.mercury_cuda_free()

class Integration(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.base=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def case(self,name,**kw):
        p=prepare(self.base/name,**kw); run(p); return p
    def assertState(self,a,b,tol=3e-11,reverse=False):
        self.assertEqual(set(a),set(b))
        for name in a:
            for key in ['x','v']:
                for val,ref in zip(a[name][key],b[name][key]):
                    if reverse and key=='v': ref=-ref
                    self.assertLess(abs(val-ref),tol,(name,key,val,ref))

    def test_backward_velocity_reversal_all_methods(self):
        for method in ['BS','BS2','RADAU','MVS','HYBRID']:
            with self.subTest(method=method):
                o=body(); rv=body(); rv['v']=[-v for v in o['v']]
                planet=body('JUPITER',mass=.001,a=5.2,e=.05,phase=.4)
                reversed_planet={**planet,'v':[-v for v in planet['v']]}
                b=self.case(method+'b',big=[planet],small=[o],stop=-31,interval=1.,algorithm=method)
                f=self.case(method+'f',big=[reversed_planet],small=[rv],stop=31,interval=1.,algorithm=method)
                self.assertState(dump(b),dump(f),tol=2e-10,reverse=True)

    def test_roundtrip_and_preparation(self):
        for method in ['BS','BS2','RADAU','MVS','HYBRID']:
            with self.subTest(method=method):
                f=self.case(method+'f',stop=40,algorithm=method,interval=1.)
                b=self.case(method+'b',small=list(dump(f).values()),epoch=40,start=40,stop=0,algorithm=method,interval=1.)
                self.assertState(dump(b),{'PARTICLE':body()},tol=2e-9)
                prep=self.case(method+'p',epoch=0,start=-10.3,stop=-20.3,interval=1.,algorithm=method)
                ref=self.case(method+'r',epoch=0,start=0,stop=-20.3,interval=1.,algorithm='BS')
                self.assertState(dump(prep),dump(ref),tol=2e-9)

    def test_cpu_gpu_full_forces(self):
        if load_gpu() is None: self.skipTest('CUDA unavailable')
        big=[body('PLANET',mass=1e-6,a=2,e=.03)]
        small=[body('P'+str(i),a=.5+.07*i,e=.15,phase=.3*i,yar=(-1)**i*1e-11,b=1e-4) for i in range(20)]
        for direction in [-1,1]:
            cpu=self.case('cpu'+str(direction),big=big,small=small,stop=direction*80.,pn=True)
            gpu=self.case('gpu'+str(direction),big=big,small=small,stop=direction*80.,pn=True,backend='cuda')
            self.assertState(dump(cpu),dump(gpu),tol=1e-10)
            self.assertState(dump(cpu,'big.dmp'),dump(gpu,'big.dmp'),tol=1e-11)

    def test_pn_precession(self):
        a=.387; e=.2056; periods=30; period=2*math.pi*math.sqrt(a**3/MU)
        p=self.case('pn',small=[body(a=a,e=e)],stop=periods*period,pn=True,interval=period,tol=1e-13,step=1)
        o=dump(p)['PARTICLE']; x,y,_=o['x']; vx,vy,_=o['v']; h=x*vy-y*vx; r=math.hypot(x,y)
        ex=vy*h/MU-x/r; ey=-vx*h/MU-y/r; angle=math.atan2(ey,ex)
        c=299792458*86400/1.495978707e11
        expected=periods*6*math.pi*MU/(a*(1-e*e)*c*c)
        self.assertLess(abs(angle/expected-1),.002)

    def test_yarkovsky_secular_drift(self):
        A2=1e-10; period=2*math.pi/math.sqrt(MU); duration=10*period; e=.1
        for sign in [-1,1]:
            p=self.case('drift'+str(sign),small=[body(yar=sign*A2)],stop=duration,interval=period,step=3,tol=1e-12)
            o=dump(p)['PARTICLE']; r=math.sqrt(sum(v*v for v in o['x'])); vsq=sum(v*v for v in o['v'])
            afinal=1/(2/r-vsq/MU); expected=sign*2*A2/(math.sqrt(MU)*(1-e*e))*duration
            self.assertLess(abs((afinal-1)/expected-1),.003)

    def test_input_guards_and_big_yar(self):
        p=self.case('big_a2',big=[body('BIG',mass=1e-15,yar=3.4e-14)],small=[],stop=3)
        self.assertEqual(dump(p,'big.dmp')['BIG']['params']['yar'],3.4e-14)
        for name,kw,message in [('bad_backend',dict(backend='cuda',user_force=True),'CUDA requires'),('bad_force',dict(pn=True,algorithm='BS2'),'require BS'),('duplicate',dict(small=[body(),body()]),'Duplicate body'),('zero_interval',dict(interval=0),'interval')]:
            p=prepare(self.base/name,**kw); res=run(p,check=False)
            self.assertNotEqual(res.returncode,0,(name,res.stdout)); self.assertIn(message,(res.stdout+res.stderr))

    def test_pr_rejected_for_symplectic_methods(self):
        for method in ['MVS','HYBRID']:
            for backend in ['cpu','cuda']:
                with self.subTest(method=method,backend=backend):
                    p=prepare(self.base/(method+backend),algorithm=method,backend=backend,
                              small=[body(b=.001)],stop=1)
                    result=run(p,check=False)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('PN, yar and PR require BS or RADAU',result.stdout+result.stderr)

    def test_distinct_inputs_and_restart_migration(self):
        obj=body('BOTH',a2=-2e-11,yar=3e-12,b=1e-4)
        p=self.case('both',small=[obj],stop=5,interval=1)
        params=dump(p)['BOTH']['params']
        self.assertEqual(params['a2'],-2e-11); self.assertEqual(params['yar'],3e-12)
        self.assertIn('force model version = 2',(p/'param.dmp').read_text())
        # Simulate an earlier MERCUDA dump: A2 then meant Yarkovsky.
        old=self.case('v1',small=[body(yar=3e-12,b=1e-4)],stop=5,interval=1)
        text=(old/'param.dmp').read_text().replace('version = 2','version = 1')
        text=re.sub(r'(stop time.*?=)\s*[^\n]+',r'\g<1> 10',text,flags=re.I)
        (old/'param.dmp').write_text(text)
        (old/'small.dmp').write_text((old/'small.dmp').read_text().replace(' yar=',' a2='))
        run(old)
        ref=self.case('v1ref',small=[body(yar=3e-12,b=1e-4)],stop=10,interval=1)
        self.assertState(dump(old),dump(ref),tol=1e-10)
        self.assertNotIn('a2',dump(old)['PARTICLE']['params'])
        self.assertEqual(dump(old)['PARTICLE']['params']['yar'],3e-12)
        self.assertIn('Migrated force model 1',(old/'info.out').read_text())

    def test_cometary_a2_legacy_restart_and_symplectic_support(self):
        for method in ['BS','MVS','HYBRID']:
            for backend in ['cpu','cuda']:
                if backend=='cuda' and load_gpu() is None: continue
                p=self.case(method+backend+'comet',algorithm=method,backend=backend,
                            small=[body(a2=2e-12)],stop=5,step=.5,interval=1)
                text=(p/'param.dmp').read_text()
                text=re.sub(r'^.*force model version.*\n','',text,flags=re.M)
                text=re.sub(r'(stop time.*?=)\s*[^\n]+',r'\g<1> 10',text,flags=re.I)
                (p/'param.dmp').write_text(text);run(p)
                ref=self.case(method+backend+'cometref',algorithm=method,backend=backend,
                              small=[body(a2=2e-12)],stop=10,step=.5,interval=1)
                self.assertState(dump(p),dump(ref),tol=1e-10)
                self.assertEqual(dump(p)['PARTICLE']['params']['a2'],2e-12)

    def test_yar_rejected_for_incompatible_methods(self):
        for method in ['BS2','MVS','HYBRID']:
            for backend in ['cpu','cuda']:
                p=prepare(self.base/(method+backend+'yar'),algorithm=method,backend=backend,
                          small=[body(yar=1e-12)],stop=1)
                result=run(p,check=False)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('PN, yar and PR require BS or RADAU',result.stdout+result.stderr)

    def test_restart_and_postprocessors(self):
        p=self.case('restart',pn=True,small=[body(yar=1e-12)],stop=10,interval=1)
        text=(p/'param.dmp').read_text(); text=re.sub(r'(stop time.*?=)\s*[^\n]+',r'\g<1> 20',text,flags=re.I); (p/'param.dmp').write_text(text)
        run(p)
        ref=self.case('full',pn=True,small=[body(yar=1e-12)],stop=20,interval=1)
        self.assertState(dump(p),dump(ref),tol=1e-10)
        for prog in ['element','close']:
            shutil.copy(ROOT/(prog+'.in.sample'),p/(prog+'.in'))
            result=run(p,executable=ROOT/(prog+'6'))
            self.assertNotIn('ERROR',result.stdout.upper())
        self.assertTrue((p/'PARTICLE.aei').exists())

if __name__=='__main__': unittest.main()
