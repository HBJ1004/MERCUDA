"""Backend parity of complete integrations, including semi-active bodies."""
import tempfile
import unittest
from pathlib import Path
from cases import body, prepare, run, dump
from force_library import load_gpu


class GPUAlgorithms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if load_gpu() is None: raise unittest.SkipTest('CUDA unavailable')

    def compare(self, algorithm, *, massive=False, pn=False, stop=32., interval=8., params=None):
        with tempfile.TemporaryDirectory() as tmp:
            states=[]
            for backend in ['cpu','cuda']:
                small=[body('S'+str(j),mass=1e-8 if massive and j%3==0 else 0,
                            a=1.6+j*.012,e=.12,phase=j*.37,**(params or {})) for j in range(33)]
                p=prepare(Path(tmp)/backend,algorithm=algorithm,backend=backend,
                          big=[body('J',mass=.001,a=5.2,e=.04)],small=small,
                          stop=stop,interval=interval,step=.5,pn=pn)
                run(p)
                states.append({**dump(p,'big.dmp'),**dump(p)})
            self.assertEqual(states[0].keys(),states[1].keys())
            for name in states[0]:
                for key in ['x','v']:
                    for a,b in zip(states[0][name][key],states[1][name][key]):
                        self.assertLess(abs(a-b),1e-10,(algorithm,name,key,a,b))

    def test_bs_semi_active(self):
        for direction in [-1,1]:
            self.compare('BS',massive=True,pn=True,stop=direction*32,
                         params={'a2':1e-12,'b':.001})


if __name__=='__main__': unittest.main()
