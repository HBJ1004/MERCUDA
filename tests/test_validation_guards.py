"""Regression tests for failures found by the independent validation campaign."""
import math
import tempfile
import unittest
from pathlib import Path
from cases import ROOT,MU,body,prepare,run,dump


class ValidationGuards(unittest.TestCase):
    def test_missing_message_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=prepare(tmp,stop=.1)
            (path/'message.in').unlink()
            result=run(path,check=False)
            self.assertGreater(result.returncode,0)
            self.assertIn('message.in',result.stdout+result.stderr)

    def test_parabolic_cometary_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=prepare(tmp,small=[],stop=.125,interval=.125)
            (path/'small.in').write_text(')O+_06\n style = Cometary\n PARABOLA m=0\n .5 1 0 0 0 0 0 0 0\n')
            run(path)
            result=dump(path)['PARABOLA']
            # Barker's equation supplies an independent parabolic reference.
            mean=math.sqrt(MU/(2*.5**3))*.125
            d=mean
            for _ in range(8): d-=(d+d**3/3-mean)/(1+d*d)
            speed=math.sqrt(2*MU/.5)/(1+d*d)
            expected=[.5*(1-d*d),d,0,-d*speed,speed,0]
            for got,truth in zip(result['x']+result['v'],expected):
                self.assertAlmostEqual(got,truth,delta=1e-12)

    def test_invalid_asteroidal_elements_rejected(self):
        for elements in ['1 -0.1 0 0 0 0 0 0 0','1 1 0 0 0 0 0 0 0','-1 .1 0 0 0 0 0 0 0']:
            with tempfile.TemporaryDirectory() as tmp:
                path=prepare(tmp,small=[],stop=.125)
                (path/'small.in').write_text(')O+_06\n style = Asteroidal\n INVALID m=0\n'+elements+'\n')
                result=run(path,check=False)
                self.assertGreater(result.returncode,0)
                self.assertIn('Invalid orbital elements',result.stdout+result.stderr)

    def test_hyperbolic_perihelion_time_is_not_wrapped(self):
        e=1.2; axis=2.5; frequency=math.sqrt(MU/axis**3)
        for perihelion in (-2000.,2000.):
            with tempfile.TemporaryDirectory() as tmp:
                path=prepare(tmp,small=[],stop=0)
                (path/'small.in').write_text(')O+_06\n style = Cometary\n HYPER m=0\n .5 1.2 0 0 0 '+str(perihelion)+' 0 0 0\n')
                run(path); result=dump(path)['HYPER']
                mean=-frequency*perihelion
                anomaly=math.asinh(mean/e)
                for _ in range(20):
                    anomaly-=(e*math.sinh(anomaly)-anomaly-mean)/(e*math.cosh(anomaly)-1)
                den=e*math.cosh(anomaly)-1
                expected=[axis*(e-math.cosh(anomaly)),axis*math.sqrt(e*e-1)*math.sinh(anomaly),0,
                          -math.sqrt(MU/axis)*math.sinh(anomaly)/den,
                          math.sqrt(MU/axis)*math.sqrt(e*e-1)*math.cosh(anomaly)/den,0]
                for got,truth in zip(result['x']+result['v'],expected):
                    self.assertAlmostEqual(got,truth,delta=1e-12)

    def test_corrupt_message_table_rejected(self):
        for content in ('999  1 X\n','  1 99 X\n','  1  1 X\n'):
            with tempfile.TemporaryDirectory() as tmp:
                path=prepare(tmp,stop=.125)
                (path/'message.in').write_text(content)
                result=run(path,check=False)
                self.assertGreater(result.returncode,0)
                self.assertIn('message.in',result.stderr)

    def test_jacobi_postprocessing_with_small_bodies(self):
        with tempfile.TemporaryDirectory() as tmp:
            objects=[body('INNER',mass=1e-4),body('OUTER',mass=1e-3,a=3)]
            particle=body('SMALL',a=2)
            path=prepare(tmp,big=objects,small=[particle],stop=0)
            run(path)
            text=(ROOT/'element.in.sample').read_text().replace('= Central','= Jacobi').replace('365.2d1','0')
            text=text.replace(' p13e l13.5e a8.5 e8.6 i8.4 g8.4 n8.4 m13e ',
                              ' x24.16 y24.16 z24.16 u24.16 v24.16 w24.16 ')
            (path/'element.in').write_text(text)
            result=run(path,executable=ROOT/'element6',check=False)
            self.assertEqual(result.returncode,0,result.stderr)
            rows=[]
            for line in (path/'SMALL.aei').read_text().splitlines():
                try: values=list(map(float,line.split()))
                except ValueError: continue
                if len(values)==7: rows.append(values)
            self.assertEqual(len(rows),1)
            for got,truth in zip(rows[0][1:],particle['x']+particle['v']):
                self.assertAlmostEqual(got,truth,delta=1e-8)

    def test_stop_encounter_writes_record(self):
        for method in ('BS','BS2','RADAU','MVS','HYBRID'):
            with tempfile.TemporaryDirectory() as tmp:
                planet=body('PLANET',mass=1e-12,e=0,r=3)
                particle=dict(name='FLYBY',mass=0,x=[1.00001,-.005,0],v=[0,.1,0],params={})
                path=prepare(tmp,big=[planet],small=[particle],algorithm=method,stop=.2,
                             step=.001,interval=.01,stop_encounter=True)
                run(path)
                records=[r for r in (path/'ce.out').read_bytes().split(b'\n') if r.startswith(b'\x0c6b')]
                self.assertGreater(len(records),0,method)
                self.assertIn('Stopping integration due to an encounter', (path/'info.out').read_text())

    def test_coincident_interacting_bodies_rejected_for_every_method(self):
        for method in ('BS','BS2','RADAU','MVS','HYBRID'):
            for backend in ('cpu','cuda'):
                with tempfile.TemporaryDirectory() as tmp:
                    path=prepare(tmp,big=[body('A',mass=1e-8),body('B',mass=1e-8)],small=[],
                                 algorithm=method,backend=backend,stop=.125)
                    result=run(path,check=False)
                    self.assertGreater(result.returncode,0)
                    self.assertIn('coincident interacting bodies',result.stderr)
