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
