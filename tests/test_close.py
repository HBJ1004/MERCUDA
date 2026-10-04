"""Fast independent close6 regressions; expanded matrices live in validation."""
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from cases import ROOT, MU, run
import close_reference as ref

class Close(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mercuda-close-'); self.base=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def execute(self,records=None,**options):
        path=ref.fixture(self.base,records=records,**options)
        run(path,executable=ROOT/'close6')
        return path
    def test_massive_circular_and_radial(self):
        radial=([0,0,1],[0,0,.01])
        records=ref.header()+[ref.encounter(),ref.encounter(first=radial,time=2451546),
                              ref.encounter(first=([1.,2.,3.],[.01,.02,.03]),time=2451547)]
        path=self.execute(records)
        ref.validate_rows(path/'PLANET.clo',records,'PLANET')
        ref.validate_rows(path/'PARTICLE.clo',records,'PARTICLE')
        first=ref.rows(path/'PLANET.clo')[0]
        self.assertEqual(float(first[-6]),1.)
        self.assertEqual(float(first[-5]),0.)
        self.assertTrue(math.isnan(float(ref.rows(path/'PLANET.clo')[1][-4])))
        self.assertTrue(math.isnan(float(ref.rows(path/'PLANET.clo')[2][-4])))
    def test_orbits_and_precision(self):
        for precision in (1,2,3):
            with tempfile.TemporaryDirectory(dir=self.base) as tmp:
                records=ref.header(precision=precision)+[ref.encounter(first=ref.orbit(e=e,inclination=i),time=2451545+k)
                    for k,(e,i) in enumerate([(0,0),(.1,15),(.9,90),(.999999,165),(1-1e-8,0),(1+1e-8,180),(1.2,45),(20,90)])]
                path=ref.fixture(tmp,records=records); run(path,executable=ROOT/'close6')
                ref.validate_rows(path/'PLANET.clo',records,'PLANET')
                self.assertNotIn('*',(path/'PLANET.clo').read_text())
    def test_all_time_formats(self):
        expected={('days',False):[2451545.5],('days',True):[.5],('years',True):[.5/365.25],('years',False):[2000,1,2]}
        for (units,relative),clock in expected.items():
            with tempfile.TemporaryDirectory(dir=self.base) as tmp:
                path=ref.fixture(tmp,units=units,relative=relative); run(path,executable=ROOT/'close6')
                row=ref.rows(path/'PLANET.clo')[0]
                for token,value in zip(row,clock): self.assertAlmostEqual(float(token),value,delta=5.1e-6)
    def test_selections_and_existing_file(self):
        records=ref.header()+[ref.encounter()]
        path=self.execute(records,names=['PLANET','PLANET','ABSENT'])
        self.assertEqual(len(ref.rows(path/'PLANET.clo')),1)
        self.assertFalse((path/'PARTICLE.clo').exists()); self.assertEqual(ref.rows(path/'ABSENT.clo'),[])
        old=(path/'PLANET.clo').read_bytes(); result=run(path,executable=ROOT/'close6')
        self.assertIn('existing output skipped',result.stdout); self.assertEqual((path/'PLANET.clo').read_bytes(),old)
    def test_header_updates_and_file_origins(self):
        first=ref.header()+[ref.encounter()]+ref.header(codes=[20,9],time=2451546)+[ref.encounter(codes=(20,9),time=2451547)]
        second=ref.header(codes=[17,99],time=2451548)+[ref.encounter(codes=(17,99),time=2451549)]
        path=self.execute(files={'a.ce':first,'b.ce':second},relative=True)
        self.assertEqual([float(r[0]) for r in ref.rows(path/'PLANET.clo')],[.5,2.,1.])
        ref.validate_rows(path/'PLANET.clo',first+second,'PLANET')
    def test_line_endings_and_empty_history(self):
        for newline in (b'\n',b'\r\n'):
            for final in (True,False):
                with tempfile.TemporaryDirectory(dir=self.base) as tmp:
                    path=ref.fixture(tmp,newline=newline,final_newline=final); run(path,executable=ROOT/'close6')
                    self.assertEqual(len(ref.rows(path/'PLANET.clo')),1)
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            path=ref.fixture(tmp,records=ref.header()); run(path,executable=ROOT/'close6')
            self.assertEqual(ref.rows(path/'PLANET.clo'),[])
    def test_invalid_records_stop_before_outputs(self):
        header=ref.header(); event=ref.encounter()
        cases=[[],[event],header+[event[:-1]],header+[b'x'+event[1:]],header+[event[:30]+b'\x01'+event[31:]],
               header+[event[:11]+ref.digits(0,3)+event[14:]],header+[event[:11]+ref.digits(99,3)+event[14:]],
               header+[event[:14]+event[11:14]+event[17:]],header+[event[:37]+b' '*4+event[41:]],
               [header[0][:-1]], [header[0][:-1]+b'9',*header[1:]],
               [header[0],header[1],header[1]],ref.header(central=0),ref.header(rcen=0),ref.header(masses=[-1,0])]
        for j,records in enumerate(cases):
            with tempfile.TemporaryDirectory(dir=self.base) as tmp:
                path=ref.fixture(tmp,records=records); result=run(path,executable=ROOT/'close6',check=False)
                self.assertGreater(result.returncode,0,(j,result.stderr))
                self.assertIn('ce.out: record',result.stderr); self.assertEqual(list(path.glob('*.clo')),[])
    def test_empty_population(self):
        path=self.execute(ref.header(names=[],masses=[]))
        self.assertEqual(list(path.glob('*.clo')),[])
    def test_filename_collisions(self):
        path=ref.fixture(self.base,records=ref.header(names=['A/B','A_B'])+[ref.encounter()])
        result=run(path,executable=ROOT/'close6',check=False)
        self.assertGreater(result.returncode,0); self.assertIn('filename collision',result.stderr)
        self.assertEqual(list(path.glob('*.clo')),[])
    def test_batch_boundary(self):
        names=['P'+str(j) for j in range(257)]
        records=ref.header(names=names,masses=[0]*257)+[ref.encounter(codes=(1,257))]
        path=self.execute(records)
        self.assertEqual(len(list(path.glob('*.clo'))),257)
        ref.validate_rows(path/'P0.clo',records,'P0'); ref.validate_rows(path/'P256.clo',records,'P256')
    def test_direct_elements(self):
        build=Path(os.environ.get('MERCURY_TEST_BUILD',ROOT/'build'))
        source=self.base/'direct.f90'; exe=self.base/'direct'
        source.write_text('''program direct
use mercury_close, only: close_elements, close_field
implicit none
real(8):: a,e,i,x(3),v(3),mu
integer:: ios
do
read(*,*,iostat=ios) mu,x,v
if(ios/=0) exit
call close_elements(mu,x,v,a,e,i)
write(*,'(3(es26.17e3,1x),3(a,1x))') a,e,i,close_field(a,'(f9.4)'),close_field(e,'(f8.6)'),close_field(i,'(f7.3)')
end do
end program
''')
        flags=os.environ.get('MERCURY_TEST_FFLAGS','-O3 -ffp-contract=off').split()+['-ffree-line-length-none']
        # The legacy header routine is the only external dependency; do not link a second program.
        header_source=self.base/'header.for'
        text=(ROOT/'close6.for').read_text(); start=text.index('      subroutine m_formce'); end=text.index('c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',start)
        header_source.write_text(text[start:end])
        result=subprocess.run(['gfortran',*flags,'-I'+str(build),str(source),str(header_source),str(build/'mercury_close.o'),str(build/'mercury_support.o'),'-o',str(exe)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        states=[(1,[.5,0,0],[0,2,0]),(1,[1,0,0],[1,0,0]),(.5,[1,0,0],[1,0,0]),(1,[1,0,0],[0,0,0]),(1,[1,2,3],[.01,.02,.03]),
                (1,[1,2,3],[-1,-2,-3]),(1,[1,2,3],[.01,.02,.0300000001])]
        rng=random.Random(1729)
        for _ in range(32): states.append((1,[rng.uniform(.2,2),rng.uniform(-1,1),rng.uniform(-1,1)],[rng.uniform(-1,1) for _ in range(3)]))
        data='\n'.join(' '.join(str(n) for n in [mu,*x,*v]) for mu,x,v in states)+'\n'
        result=subprocess.run([str(exe)],input=data,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        tokens=[line.split() for line in result.stdout.splitlines()]
        actual=[[float(t) for t in line[:3]] for line in tokens]
        self.assertEqual(tokens[0][3],'Infinity')
        self.assertEqual(tokens[1][5],'NaN')
        self.assertFalse(any('*' in token for line in tokens for token in line))
        self.assertTrue(math.isinf(actual[0][0])); self.assertEqual(actual[1][:2],[1,1]); self.assertTrue(math.isnan(actual[1][2]))
        from decimal import Decimal as D,localcontext
        for (mu,x,v),values in zip(states,actual):
            with localcontext() as ctx:
                ctx.prec=80; expected,r=ref.elements(D(str(mu)),list(map(lambda t:D(str(t)),x)),list(map(lambda t:D(str(t)),v)))
            for j,(val,want) in enumerate(zip(values,expected)):
                if math.isnan(want): self.assertTrue(math.isnan(val))
                elif math.isinf(want): self.assertEqual(val,want)
                elif j==0: self.assertLessEqual(abs(r/val-r/want),1e-12)
                else: self.assertAlmostEqual(val,want,delta=1e-10 if j==2 else 1e-12*max(1,abs(want)))
