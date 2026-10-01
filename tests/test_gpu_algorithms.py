"""Backend parity of complete integrations, including semi-active bodies."""
import math
import tempfile
import unittest
from pathlib import Path
from cases import MU, body, prepare, run, dump
from force_library import load_gpu


class GPUAlgorithms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if load_gpu() is None: raise unittest.SkipTest('CUDA unavailable')

    def compare(self, algorithm, *, massive=False, pn=False, stop=32., interval=8., params=None, big=None, count=33):
        with tempfile.TemporaryDirectory() as tmp:
            states=[]
            for backend in ['cpu','cuda']:
                small=[body('S'+str(j),mass=1e-8 if massive and j%3==0 else 0,
                            a=1.6+j*.012,e=.12,phase=j*.37,**(params or {})) for j in range(count)]
                p=prepare(Path(tmp)/backend,algorithm=algorithm,backend=backend,
                          big=big if big is not None else [body('E',mass=3e-6,a=1,e=.016),
                              body('J',mass=.001,a=5.2,e=.04),body('S',mass=.0003,a=9.5,e=.05)],small=small,
                          stop=stop,interval=interval,step=.5,pn=pn)
                run(p)
                states.append({**dump(p,'big.dmp'),**dump(p)})
            self.assertEqual(states[0].keys(),states[1].keys())
            for name in states[0]:
                for key in ['x','v']:
                    for a,b in zip(states[0][name][key],states[1][name][key]):
                        self.assertLess(abs(a-b),1e-10,(algorithm,name,key,a,b))

    def test_mvs(self):
        for direction in [-1,1]:
            self.compare('MVS',stop=direction*128,interval=7.3)
        for n in [0,1,257]: self.compare('MVS',big=[],count=n)
        self.compare('TEST',stop=32)

    def test_kepler_regimes(self):
        for speed in [.7,1.,1.2]:
            with tempfile.TemporaryDirectory() as tmp:
                states=[]
                for backend in ['cpu','cuda']:
                    obj=dict(name='K',mass=0.,x=[.5,0,0],v=[0,speed*math.sqrt(2*MU/.5),0])
                    p=prepare(Path(tmp)/backend,algorithm='MVS',backend=backend,small=[obj],
                              stop=32,step=.5,interval=32)
                    run(p); states.append(dump(p)['K'])
                for k in ['x','v']:
                    self.assertLess(max(abs(a-b) for a,b in zip(states[0][k],states[1][k])),1e-10)

    def test_radau(self):
        for direction in [-1,1]:
            for pn in [False,True]:
                self.compare('RADAU',massive=True,pn=pn,stop=direction*32,interval=7.3,
                             params={'a2':1e-12,'b':.001} if pn else None)

    def test_bs2(self):
        for massive in [False,True]:
            for direction in [-1,1]:
                self.compare('BS2',massive=massive,stop=direction*32)

    def test_bs_semi_active(self):
        for direction in [-1,1]:
            self.compare('BS',massive=True,pn=True,stop=direction*32,
                         params={'a2':1e-12,'b':.001})


if __name__=='__main__': unittest.main()
