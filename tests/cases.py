"""Small, isolated Mercury cases shared by regression tests and benchmarks."""
from pathlib import Path
import math
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MU = 2.959122082855911e-4


def body(name="PARTICLE", mass=0.0, a=1.0, e=0.1, phase=0.0, **params):
    r = a * (1-e)
    speed = math.sqrt(MU*(1+e)/r)
    c,s = math.cos(phase),math.sin(phase)
    return dict(name=name, mass=mass, x=[r*c,r*s,0.0],
                v=[-speed*s,speed*c,0.0], params=params)


def prepare(path, small=None, big=None, *, epoch=0.0, start=0.0, stop=100.0,
            algorithm="BS", step=1.0, interval=7.3, tol=1e-12,
            pn=False, backend="cpu", collisions=False, user_force=False,
            jcen=(0.0, 0.0, 0.0)):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name in ["message.in", "files.in"]:
        shutil.copyfile(ROOT/(name+".sample"), path/name)
    values = [algorithm, start, stop, interval, step, tol, "no",
              "yes" if collisions else "no", "no", "days", "no", "high",
              "unused", "yes" if pn else "no", "yes" if user_force else "no", 100.0, 0.005, 1.0,
              *jcen, "unused", "unused", 3.0, 5000, 100]
    labels = [line.split("=")[0] for line in (ROOT/"param.in.sample").read_text().splitlines()
              if line and not line.startswith(")")][:26]
    (path/"param.in").write_text(
        ")O+_06 Integration parameters\n" +
        "\n".join(f"{label} = {value}" for label,value in zip(labels,values)) +
        f"\n execution backend = {backend}\n")
    for filename,objects,is_big in [("big.in",big or [],True),
                                   ("small.in",small if small is not None else [body()],False)]:
        with (path/filename).open("w") as f:
            f.write(")O+_06 Initial data\n style = Cartesian\n")
            if is_big: f.write(f" epoch = {epoch:.17e}\n")
            for obj in objects:
                params={"m":obj["mass"], **obj.get("params",{})}
                f.write(obj["name"]+" "+" ".join(f"{k}={v:.17e}" for k,v in params.items())+"\n")
                f.write(" ".join(f"{v:.17e}" for v in obj["x"]+obj["v"]+[0.,0.,0.])+"\n")
    return path


def run(path, executable=None, timeout=60, check=True):
    exe=Path(executable or ROOT/"mercury6").resolve()
    if os.environ.get("MERCURY_TEST_BIN"):
        exe=Path(os.environ["MERCURY_TEST_BIN"])/exe.name
    result=subprocess.run([str(exe)],cwd=path,text=True,capture_output=True,timeout=timeout)
    if check and result.returncode:
        raise AssertionError(result.stdout+result.stderr)
    return result


def dump(path, filename="small.dmp"):
    lines=[line.strip() for line in (Path(path)/filename).read_text().splitlines()
           if line.strip() and not line.startswith(")")]
    lines=lines[2 if filename.startswith("big") else 1:]
    result={}
    for k in range(0,len(lines),4):
        tokens=lines[k].replace("="," = ").split()
        params={tokens[i-1]:float(tokens[i+1].replace("D","E"))
                for i,t in enumerate(tokens) if t=="="}
        xyz=[float(v.replace("D","E")) for line in lines[k+1:k+3] for v in line.split()[:3]]
        result[tokens[0]]=dict(name=tokens[0],mass=params.pop("m",0.0),
                               x=xyz[:3],v=xyz[3:],params=params)
    return result
