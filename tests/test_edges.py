"""Scheduling, restart, and allocation regression cases."""
import re
import shutil
import unittest
import test_mercury as core
from cases import ROOT, body, prepare, run, dump
from force_library import load_gpu

class Edges(unittest.TestCase):
    setUp=core.Integration.setUp
    tearDown=core.Integration.tearDown
    case=core.Integration.case
    assertState=core.Integration.assertState

    def test_mixed_epochs_keep_coefficients(self):
        objs=[body('LATE',a2=1e-10,ep=2),body('EARLY',a2=-3e-10,b=.001,ep=-2)]
        p=self.case('mixed',small=objs,start=0,stop=10,interval=.7)
        merged=dump(p)
        for obj in objs:
            ref=self.case(obj['name'],small=[obj],start=0,stop=10,interval=.7)
            self.assertState({obj['name']:merged[obj['name']]},dump(ref),tol=2e-10)
            self.assertEqual(merged[obj['name']]['params']['a2'],obj['params']['a2'])

    def test_dissipative_roundtrip_and_radau(self):
        obj=body(a=.7,e=.2,a2=1e-9,b=.001)
        bs=self.case('bs',small=[obj],stop=20,interval=1,pn=True)
        ra=self.case('ra',small=[obj],stop=20,interval=1,pn=True,algorithm='RADAU')
        self.assertState(dump(bs),dump(ra),tol=2e-10)
        back=self.case('back',small=list(dump(bs).values()),epoch=20,start=20,stop=0,interval=1,pn=True)
        self.assertState(dump(back),{'PARTICLE':obj},tol=2e-10)

    def test_gpu_rejections_and_repeated_execution(self):
        if load_gpu() is None: self.skipTest('CUDA unavailable')
        objects=[body('P'+str(i),a=1.5) for i in range(513)]
        objects[-1]=body('P512',a=.3,e=.97)
        states=[]
        for back in ['cpu','cuda','cuda']:
            p=self.case(back+str(len(states)),small=objects,stop=1,step=1,interval=1,backend=back,tol=1e-13)
            log=(p/'info.out').read_text()
            self.assertGreater(int(re.search(r'Rejected attempts:\s*(\d+)',log).group(1)),0)
            states.append(dump(p))
        self.assertState(states[0],states[1],tol=1e-10)
        self.assertEqual(states[1],states[2])

    def test_restart_switches_backend(self):
        if load_gpu() is None: self.skipTest('CUDA unavailable')
        p=self.case('restart',small=[body(a2=2e-12,b=1e-4)],stop=5,interval=1,pn=True)
        (p/'param.dmp').write_text(re.sub(r'(stop time.*?=)\s*[^\n]+',r'\g<1> 15',(p/'param.dmp').read_text(),flags=re.I))
        (p/'param.in').write_text((p/'param.in').read_text().replace('backend = cpu','backend = cuda'))
        run(p)
        reference=self.case('reference',small=[body(a2=2e-12,b=1e-4)],stop=15,interval=1,pn=True)
        self.assertState(dump(p),dump(reference),tol=1e-10)
        self.assertIn('Execution backend: CUDA',(p/'info.out').read_text())

    def test_postprocessor_union_of_names(self):
        a=self.case('first',small=[body('A'+str(i),phase=i*.2) for i in range(10)],stop=1)
        b=self.case('second',small=[body('B'+str(i),phase=i*.2) for i in range(10)],stop=1)
        for prog,file in [('element','xv.out'),('close','ce.out')]:
            shutil.copy(b/file,a/('second_'+file))
            config=(ROOT/(prog+'.in.sample')).read_text().replace('input files = 1','input files = 2').replace(' '+file+'\n',' '+file+'\n second_'+file+'\n')
            config+='ABSENT\nB9\n'
            (a/(prog+'.in')).write_text(config)
            run(a,executable=ROOT/(prog+'6'))
        self.assertTrue((a/'B9.aei').exists())

    def test_singular_input_and_oversized_metadata_fail(self):
        for name,objects in [('singular',[body('A',mass=1e-8),body('B',mass=1e-8)]),('fields',[body()])]:
            p=prepare(self.base/name,big=objects if name=='singular' else [],small=[] if name=='singular' else objects,stop=1)
            if name=='fields':
                text=(p/'small.in').read_text().replace('PARTICLE ', 'PARTICLE '+'a1=0 '*60)
                (p/'small.in').write_text(text)
            result=run(p,check=False)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Nonfinite' if name=='singular' else 'Too many fields',result.stderr)
