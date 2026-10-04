import math
import re
import shutil
import test_mercury as core
from cases import ROOT, MU, body, prepare, run, dump
from force_library import load_gpu
import close_reference as close_ref

# Reuse the fixture/assertion helpers without running Integration's tests twice.
import unittest
class Events(unittest.TestCase):
    setUp=core.Integration.setUp
    tearDown=core.Integration.tearDown
    case=core.Integration.case
    assertState=core.Integration.assertState

    def backends(self): return ['cpu','cuda'] if load_gpu() else ['cpu']

    def test_encounter_buffer_growth(self):
        planet=body('PLANET',mass=1e-12,e=0,r=10000)
        small=[]
        for i in range(4100):
            p=body('P'+str(i)); p['x']=[1.001+i*1e-10,-.005,0.]; p['v']=[0.,.1,0.]; small.append(p)
        results=[]
        for backend in self.backends():
            p=self.case(backend,big=[planet],small=small,backend=backend,stop=.1,step=.1,interval=.1)
            records=[r[3:] for r in (p/'ce.out').read_bytes().split(b'\n') if r.startswith(b'\x0c6b')]
            self.assertEqual(len(records),4100)
            results.append(records)
            shutil.copy(ROOT/'close.in.sample',p/'close.in')
            with (p/'close.in').open('a') as f: f.write('PLANET\n')
            run(p,executable=ROOT/'close6')
            decoded=close_ref.rows(p/'PLANET.clo')
            self.assertEqual(len(decoded),4100)
            for row,record in zip(decoded,records):
                code=0
                for digit in record[11:14]: code=code*224+digit-32
                self.assertEqual(row[-8],'P'+str(code-2))
            indices=sorted(set([0,4099]+[j*4100//64 for j in range(64)]))
            sampled=p/'sample.clo'
            sampled.write_text('\n'.join(' '.join(decoded[j]) for j in indices)+'\n')
            metadata=[r for r in (p/'ce.out').read_bytes().split(b'\n') if r][:2+4100]
            # This fixture has one header, one massive body and 4100 particles.
            sample_records=metadata+[b'\x0c6b'+records[j] for j in indices]
            close_ref.validate_rows(sampled,sample_records,'PLANET')
        if len(results)==2:
            for a,b in zip(*results):
                self.assertEqual(a[8:14],b[8:14]) # identities and deterministic ordering
                def real8(data):
                    mant=sum((c-32)/224**(i+1) for i,c in enumerate(data[:7]))
                    return (2*mant-1)*10.**(data[7]-32-112)
                self.assertLess(abs(real8(a[:8])-real8(b[:8])),1e-10)
                self.assertLess(abs(real8(a[14:22])-real8(b[14:22])),1e-10)

    def test_massive_merge_and_further_steps(self):
        for algorithm in ['BS','BS2','RADAU','HYBRID']:
            with self.subTest(algorithm=algorithm): self.merge_case(algorithm)

    def merge_case(self,algorithm):
        a=body('A',mass=1e-8,e=0,d=1e-6); b=body('B',mass=2e-8,e=0,d=1e-6)
        b['x'][0]+=.0003
        results=[]
        for backend in self.backends():
            p=self.case(algorithm+backend,algorithm=algorithm,big=[a,b],small=[body('P',a=2)],backend=backend,stop=.2,step=.01,interval=.05,collisions=True)
            big=dump(p,'big.dmp'); self.assertEqual(set(big),{'B'}); self.assertAlmostEqual(big['B']['mass'],3e-8,places=20)
            results.append((dump(p),big))
        if len(results)==2:
            self.assertState(results[0][0],results[1][0],tol=2e-10)
            self.assertState(results[0][1],results[1][1],tol=2e-10)

    def test_hybrid_merge_and_central_impact(self):
        a=body('A',mass=1e-8,e=0,d=1e-6)
        b=body('B',mass=2e-8,e=0,d=1e-6); b['x'][0]+=.0003
        impact=body('IMPACT'); impact['x']=[.00505,0,0]; impact['v']=[-.1,0,0]
        states=[]
        for backend in self.backends():
            p=self.case('combined'+backend,algorithm='HYBRID',backend=backend,
                        big=[a,b],small=[impact,body('KEEP',a=2)],collisions=True,
                        stop=.01,step=.001,interval=.001)
            self.assertEqual(set(dump(p,'big.dmp')),{'B'})
            self.assertEqual(set(dump(p)),{'KEEP'})
            states.append({**dump(p,'big.dmp'),**dump(p)})
        if len(states)==2:self.assertState(*states,tol=2e-10)

    def test_central_impact_and_ejection(self):
        for algorithm in ['BS','BS2','RADAU','MVS','HYBRID']:
            with self.subTest(algorithm=algorithm): self.impact_case(algorithm)

    def impact_case(self,algorithm):
        infall=body('IMPACT'); infall['x']=[.006,0,0]; infall['v']=[-.1,0,0]
        escaping=body('ESCAPE'); escaping['x']=[99.9,0,0]; escaping['v']=[10.,0,0]
        results=[]
        for backend in self.backends():
            p=prepare(self.base/(algorithm+backend),algorithm=algorithm,small=[infall,escaping,body('KEEP',**({'a2':2e-11,'yar':3e-12,'b':1e-4} if algorithm in ['BS','RADAU'] else {}))],backend=backend,stop=.3,step=.001,interval=.05)
            text=(p/'param.in').read_text(); text=re.sub(r'(periodic effects.*?=)\s*[^\n]+',r'\g<1> 10',text,flags=re.I)
            (p/'param.in').write_text(text); run(p)
            state=dump(p); self.assertEqual(set(state),{'KEEP'}); results.append(state)
            if algorithm in ['BS','RADAU']:
                for key,value in [('a2',2e-11),('yar',3e-12),('b',1e-4)]:
                    self.assertEqual(state['KEEP']['params'][key],value)
            log=(p/'info.out').read_text(); self.assertIn('IMPACT',log); self.assertIn('ESCAPE',log)
        if len(results)==2: self.assertState(*results,tol=1e-10)

    def test_fixed_step_dense_cadence(self):
        for sign in [-1,1]:
            p=self.case(str(sign),algorithm='MVS',step=10,interval=1,stop=sign*30)
            records=(p/'xv.out').read_bytes().split(b'\n')
            self.assertEqual(sum(r.startswith(b'\x0c6b') for r in records),4)

    def test_nonfinite_scalar_rejected(self):
        for key in ['a2','yar','m','b']:
            p=prepare(self.base/key,small=[body(**{key:float('nan')})])
            result=run(p,check=False); self.assertNotEqual(result.returncode,0); self.assertIn('Nonfinite',result.stderr)
