"""Scheduling, restart, and allocation regression cases."""
import re
import shutil
import os
import subprocess
import unittest
from pathlib import Path
import test_mercury as core
from cases import ROOT, body, prepare, run, dump
from force_library import load_gpu

class Edges(unittest.TestCase):
    setUp=core.Integration.setUp
    tearDown=core.Integration.tearDown
    case=core.Integration.case
    assertState=core.Integration.assertState

    def test_auto_upload_failure_logs_cpu_fallback(self):
        # Fault injection at the existing C ABI: a device is available, but its
        # first upload fails. The production Fortran driver must log and use CPU.
        build=ROOT/os.environ.get('MERCURY_TEST_BUILD','build')
        source=(ROOT/'mercury_cuda_stub.cpp').read_text()
        source=source.replace('mercury_cuda_available() { return 0;',
                              'mercury_cuda_available() { return 1;')
        source=source.replace('mercury_cuda_configure(int) { return 1;',
                              'mercury_cuda_configure(int) { return 0;')
        stub=self.base/'fault.cpp';stub.write_text(source)
        obj=self.base/'fault.o';exe=self.base/'mercury6'
        subprocess.run(['g++','-I'+str(ROOT),'-c',str(stub),'-o',str(obj)],check=True,capture_output=True)
        subprocess.run(['gfortran','-o',str(exe),*[str(build/name) for name in
                        ['mercury6.o','mercury_support.o','mercury_gpu.o']],str(obj),'-lstdc++'],
                       check=True,capture_output=True)
        p=prepare(self.base/'fallback',backend='auto',stop=.01,interval=.01,
                  small=[body('P'+str(j),a=2+j*.001) for j in range(4096)])
        result=subprocess.run([str(exe)],cwd=p,text=True,capture_output=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        log=(p/'info.out').read_text()
        backend_lines=[line for line in log.splitlines() if 'Execution backend:' in line]
        self.assertIn('CUDA',backend_lines[0])
        self.assertIn('CPU (CUDA initialization failed; auto fallback)',backend_lines[-1])
        self.assertEqual(len(dump(p)),4096)

    def test_auto_configure_failure_falls_back(self):
        build=Path(os.environ.get('MERCURY_TEST_BUILD',str(ROOT/'build')))
        source=(ROOT/'mercury_cuda_stub.cpp').read_text().replace(
            'mercury_cuda_available() { return 0;', 'mercury_cuda_available() { return 1;')
        stub=self.base/'configure-fault.cpp'; stub.write_text(source)
        obj=self.base/'configure-fault.o'; exe=self.base/'configure-fault'
        subprocess.run(['g++','-I'+str(ROOT),'-c',str(stub),'-o',str(obj)],check=True,capture_output=True)
        subprocess.run(['gfortran','-o',str(exe),*[str(build/name) for name in
                        ['mercury6.o','mercury_support.o','mercury_gpu.o']],str(obj),'-lstdc++'],
                       check=True,capture_output=True)
        for backend in ('auto','cuda'):
            p=prepare(self.base/('configure-'+backend),backend=backend,stop=.01,interval=.01,
                      small=[body('P'+str(j),a=2+j*.001) for j in range(4096)])
            result=subprocess.run([str(exe)],cwd=p,text=True,capture_output=True,timeout=60)
            if backend=='cuda':
                self.assertGreater(result.returncode,0)
            else:
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('CPU', (p/'info.out').read_text())
                self.assertEqual(len(dump(p)),4096)

    def test_mixed_epochs_keep_coefficients(self):
        objs=[body('LATE',a2=-2e-11,yar=1e-10,ep=2),body('EARLY',a2=4e-11,yar=-3e-10,b=.001,ep=-2)]
        p=self.case('mixed',small=objs,start=0,stop=10,interval=.7)
        merged=dump(p)
        for obj in objs:
            ref=self.case(obj['name'],small=[obj],start=0,stop=10,interval=.7)
            self.assertState({obj['name']:merged[obj['name']]},dump(ref),tol=2e-10)
            for key in ['a2','yar']:
                self.assertEqual(merged[obj['name']]['params'][key],obj['params'][key])

    def test_dissipative_roundtrip_and_radau(self):
        obj=body(a=.7,e=.2,yar=1e-9,b=.001)
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
        p=self.case('restart',small=[body(yar=2e-12,b=1e-4)],stop=5,interval=1,pn=True)
        (p/'param.dmp').write_text(re.sub(r'(stop time.*?=)\s*[^\n]+',r'\g<1> 15',(p/'param.dmp').read_text(),flags=re.I))
        (p/'param.in').write_text((p/'param.in').read_text().replace('backend = cpu','backend = cuda'))
        run(p)
        reference=self.case('reference',small=[body(yar=2e-12,b=1e-4)],stop=15,interval=1,pn=True)
        self.assertState(dump(p),dump(reference),tol=1e-10)
        self.assertIn('Execution backend: CUDA',(p/'info.out').read_text())

    def test_restart_all_algorithms(self):
        if load_gpu() is None: self.skipTest('CUDA unavailable')
        for algorithm in ['BS','BS2','RADAU','MVS','HYBRID']:
            for first,second in [('cpu','cuda'),('cuda','cpu')]:
                with self.subTest(algorithm=algorithm,first=first):
                    p=self.case(algorithm+first,algorithm=algorithm,backend=first,
                                small=[body()],stop=8,step=.5,interval=2)
                    (p/'param.dmp').write_text(re.sub(r'(stop time.*?=)\s*[^\n]+',
                        r'\g<1> 16',(p/'param.dmp').read_text(),flags=re.I))
                    (p/'param.in').write_text((p/'param.in').read_text().replace('backend = '+first,'backend = '+second))
                    run(p)
                    ref=self.case(algorithm+first+'ref',algorithm=algorithm,backend='cpu',
                                  small=[body()],stop=16,step=.5,interval=2)
                    self.assertState(dump(p),dump(ref),tol=1e-10)

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
