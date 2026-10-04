"""Release-audit regressions: encoded dates and tight output-cadence accuracy."""
import math
from pathlib import Path
import re
import tempfile
import unittest
from cases import ROOT, MU, body, prepare, run, dump
from force_library import load_gpu

class Release(unittest.TestCase):
    def backends(self):
        return ['cpu','cuda'] if load_gpu() else ['cpu']

    def test_element_julian_date_cadence(self):
        for sign in (-1,1):
            for interval,span in ((1,200),(100,1200)):
                with self.subTest(sign=sign,interval=interval), tempfile.TemporaryDirectory() as tmp:
                    epoch=2451545.5
                    path=prepare(tmp,epoch=epoch,start=epoch,stop=epoch+sign*span,interval=interval)
                    run(path)
                    config=(ROOT/'element.in.sample').read_text()
                    config=re.sub(r'(minimum interval[^=]*=).*',rf'\g<1> {interval}',config)
                    config=re.sub(r'(express time in[^=]*=).*',r'\g<1> days',config)
                    config=re.sub(r'(express time relative[^=]*=).*',r'\g<1> no',config)
                    (path/'element.in').write_text(config)
                    run(path,executable=ROOT/'element6')
                    rows=[l.split() for l in (path/'PARTICLE.aei').read_text().splitlines()
                          if l.split() and l.split()[0][0] in '-0123456789']
                    self.assertEqual(len(rows),span//interval+1)
                    for n,row in enumerate(rows):
                        self.assertAlmostEqual(float(row[0]),epoch+sign*n*interval,delta=1e-5)

    def test_radau_frequent_output_accuracy(self):
        # Exact circular solution; a tighter bound detects loss hidden by 1e-8 checks.
        for backend in self.backends():
            for sign in (-1,1):
                for tol in (1e-10,1e-12):
                    with self.subTest(backend=backend,sign=sign,tol=tol), tempfile.TemporaryDirectory() as tmp:
                        obj=body(e=0)
                        stop=sign*1000
                        path=prepare(tmp,algorithm='RADAU',small=[obj],stop=stop,interval=10,tol=tol,backend=backend)
                        run(path); actual=dump(path)['PARTICLE']
                        angle=math.sqrt(MU)*stop
                        expected=[math.cos(angle),math.sin(angle),0,-math.sqrt(MU)*math.sin(angle),math.sqrt(MU)*math.cos(angle),0]
                        self.assertLess(max(abs(a-b) for a,b in zip(actual['x'],expected[:3])),1e-12)
                        self.assertLess(max(abs(a-b)/math.sqrt(MU) for a,b in zip(actual['v'],expected[3:])),1e-12)

    def test_hybrid_merger_momentum_survives_central_redo(self):
        # Negligible pair masses isolate collision momentum from gravity.
        a=body('A',mass=1e-20,e=0,d=1e-18)
        b=body('B',mass=2e-20,e=0,d=1e-18)
        b['x'][0]+=.000001
        a['v']=[.02,.01,0]; b['v']=[-.01,.04,0]
        merged=body('B',mass=3e-20,e=0,d=1e-18)
        merged['x']=[(x+2*y)/3 for x,y in zip(a['x'],b['x'])]
        merged['v']=[(x+2*y)/3 for x,y in zip(a['v'],b['v'])]
        impact=body('IMPACT');impact['x']=[.00505,0,0];impact['v']=[-.1,0,0]
        for backend in self.backends():
            with tempfile.TemporaryDirectory() as tmp:
                states=[]
                for name,big in [('pair',[a,b]),('reference',[merged])]:
                    path=prepare(Path(tmp)/name,algorithm='HYBRID',big=big,small=[impact],
                                 backend=backend,collisions=True,stop=.001,step=.001,interval=.001)
                    run(path);self.assertEqual(set(dump(path,'big.dmp')),{'B'})
                    self.assertEqual(dump(path),{})
                    states.append(dump(path,'big.dmp')['B'])
                self.assertLess(max(abs(a-b) for key in ['x','v'] for a,b in zip(states[0][key],states[1][key])),2e-12)
