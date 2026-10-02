#!/usr/bin/env python3
"""Strict, isolated validation campaign. Results are never part of the package."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tests'),str(Path(__file__).parent)]
from cases import prepare, run, dump

OPT = '-O3 -g -ffixed-line-length-none -ffp-contract=off'
DEBUG = '-O0 -g -ffixed-line-length-none -ffp-contract=off -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'
GROUPS = {'build','workflow','quick','force-values','force-matrix','scientific',
          'epochs','restarts','formats','postprocessing','events','stress',
          'invalid','custom','backends','lifecycle','sanitize'}


class Unavailable(RuntimeError):
    """Required validation could not execute on this environment."""


class Profile:
    def __init__(self, name):
        self.name = name
        self.cuda = name.startswith('cuda')
        self.debug = name.endswith('debug')
        self.build = ROOT/'build/validation'/name
        self.flags = DEBUG if self.debug else OPT

    def environment(self):
        return dict(os.environ, MERCURY_TEST_BIN=str(self.build),
                    MERCURY_TEST_BUILD=str(self.build),MERCURY_TEST_CUDA=str(int(self.cuda)),
                    MERCURY_TEST_FFLAGS=self.flags)

    def select(self):
        os.environ.update({key:value for key,value in self.environment().items() if key.startswith('MERCURY_TEST_')})


class Campaign:
    def __init__(self,args):
        self.args = args
        self.unavailable = Unavailable
        self.catalog = json.loads((Path(__file__).parent/'catalog.json').read_text())
        self.output = ROOT/'results/validation'/('cpu' if args.cpu_only else 'full')
        if args.sanitize_only: self.output = ROOT/'results/validation/sanitize'
        if args.groups: self.output = ROOT/'results/validation/development'/args.groups.replace(',','-')
        self.output.mkdir(parents=True,exist_ok=True)
        self.rows = []; self.started = time.perf_counter()
        self.profiles = [Profile(name) for name in ['cpu-opt','cpu-debug']+
                         ([] if args.cpu_only else ['cuda-opt','cuda-debug'])]
        self.completed_groups = set()
        self.sanitizer = self.find_sanitizer(args.sanitizer)
        self.provenance = {'utc':datetime.now(timezone.utc).isoformat(),
            'head':self.command(['git','rev-parse','HEAD']).stdout.strip(),
            'working_tree_dirty':bool(self.command(['git','status','--porcelain']).stdout.strip()),
            'platform':platform.platform(),'python':sys.version,
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted([*ROOT.glob('*.for'),*ROOT.glob('*.f90'),*ROOT.glob('*.cu'),
                                 *ROOT.glob('*.cuh'),*ROOT.glob('*.h'),*ROOT.glob('*.cpp'),*ROOT.glob('*.inc'),ROOT/'Makefile',
                                 *ROOT.glob('tests/*.py'),*ROOT.glob('tests/validation/*')]) if p.is_file()},
            'scope':'CPU only' if args.cpu_only else 'CPU and CUDA',
            'compiler':self.command(['gfortran','--version']).stdout.splitlines()[0],
            'cxx':self.command(['g++','--version']).stdout.splitlines()[0],
            'gpu':self.command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],check=False).stdout.strip(),
            'thresholds':self.catalog['limits'],'dependencies':{},'sanitizer':self.sanitizer}

    @staticmethod
    def command(command, *, check=True, timeout=600, cwd=ROOT, env=None):
        try:
            result = subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
        except FileNotFoundError:
            if check: raise
            return subprocess.CompletedProcess(command,127,'','Command unavailable')
        if check and result.returncode:
            raise AssertionError(f'{command}: exit {result.returncode}\n'+result.stdout[-5000:]+result.stderr[-5000:])
        return result

    @staticmethod
    def find_sanitizer(name):
        candidate = shutil.which(name)
        if candidate: return candidate
        if Path(name).is_file(): return str(Path(name).resolve())
        if name == 'compute-sanitizer':
            tools = sorted((Path.home()/'.local/opt').glob('cuda_sanitizer_api-*/bin/compute-sanitizer'))
            if tools: return str(tools[-1])
        return None

    def record(self, group, name, function, **settings):
        start = time.perf_counter()
        row = {'group':group,'case':name,'settings':settings}
        try:
            metrics = function() or {}
            row.update(status='pass',metrics=metrics)
        except Unavailable as exc:
            row.update(status='incomplete',error=str(exc))
            print(f'INCOMPLETE {group}/{name}: {exc}',flush=True)
        except Exception as exc:
            row.update(status='fail',error=str(exc),traceback=traceback.format_exc())
            print(f'FAIL {group}/{name}: {exc}',flush=True)
        row['seconds'] = time.perf_counter()-start
        self.rows.append(row)
        if row['status'] != 'pass' or len(self.rows)%100 == 0:
            self.save()
            print(f'{len(self.rows)} cases, {sum(r["status"]=="fail" for r in self.rows)} failures',flush=True)
        return row

    def simulate(self, profile, *, big=None, small=None, inspect=None, **settings):
        profile.select()
        with tempfile.TemporaryDirectory(prefix='mercuda-validation-') as tmp:
            path = prepare(tmp,big=big,small=small,**settings)
            result = run(path,timeout=180)
            # STOP without a successful final dump is not a successful calculation.
            if not (path/'small.dmp').is_file() or not (path/'big.dmp').is_file():
                raise AssertionError('Run ended without final dumps: '+result.stdout+result.stderr)
            state = {**dump(path,'big.dmp'),**dump(path)}
            info = (path/'info.out').read_text()
            if inspect: return inspect(path,state,info)
            return state,info

    def build(self,profile):
        flags = profile.flags
        nvflags = '-std=c++17 -arch=native --fmad=false -Xcompiler -fPIC '+('-O0 -G -lineinfo' if profile.debug else '-O3 -lineinfo')
        dependencies = [*ROOT.glob('*.for'),*ROOT.glob('*.f90'),*ROOT.glob('*.cu'),
                        *ROOT.glob('*.cuh'),*ROOT.glob('*.h'),*ROOT.glob('*.cpp'),*ROOT.glob('*.inc'),ROOT/'Makefile']
        digest = hashlib.sha256(b''.join(path.read_bytes() for path in sorted(dependencies))).hexdigest()
        desired = {'fortran':flags,'cuda':nvflags,'enabled':profile.cuda,'source':digest}
        marker = profile.build/'.validation_flags.json'
        if marker.exists() and json.loads(marker.read_text()) == desired:
            return desired
        if profile.build.exists() and (not marker.exists() or json.loads(marker.read_text()) != desired):
            shutil.rmtree(profile.build)
        self.command(['make','-j4',f'CUDA={"auto" if profile.cuda else "0"}',
            f'BUILD={profile.build}',f'BIN={profile.build}',f'FFLAGS={flags}',f'NVCCFLAGS={nvflags}','build'])
        marker.write_text(json.dumps(desired))
        if profile.cuda:
            profile.select()
            from force_library import load_gpu
            if load_gpu() is None: raise self.unavailable('CUDA profile requires a usable GPU and toolkit')
        return desired

    def quick(self,profile):
        result = self.command([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py','-v'],env=profile.environment())
        output = result.stdout+result.stderr
        if profile.cuda and ('skipped=' in output or '... skipped' in output):
            raise AssertionError('GPU regression suite skipped required checks')
        match = re.search(r'Ran (\d+) tests',output)
        return {'tests':int(match.group(1)) if match else 0,'cpu_expected_gpu_skips':not profile.cuda,'summary':output[-160:]}

    def preflight(self):
        for name,version in [('rebound','5.2.1'),('mpmath','1.3.0'),('matplotlib','3.10.8')]:
            found = importlib.metadata.version(name)
            if found != version: raise RuntimeError(f'{name} {version} required, found {found}')
            self.provenance['dependencies'][name] = found
        self.provenance['dependencies']['numpy'] = importlib.metadata.version('numpy')
        nvcc = self.command(['make','--no-print-directory','-s','--eval=validation-nvcc: ; @echo $(NVCC)','validation-nvcc'],check=False).stdout.strip()
        self.provenance['cuda_compiler'] = self.command([nvcc,'--version'],check=False).stdout.strip() if nvcc else None
        if self.sanitizer:
            self.provenance['sanitizer_version'] = self.command([self.sanitizer,'--version'],check=False).stdout.strip()
        registered = GROUPS
        for claim in self.catalog['claims']:
            if not claim['groups'] or not set(claim['groups']) <= registered:
                raise RuntimeError('Uncovered claim: '+claim['id'])
            if claim['section'].lower() not in (ROOT/claim['document']).read_text().lower():
                raise RuntimeError('Documentation anchor changed: '+claim['id'])
        if not self.args.cpu_only and not self.sanitizer:
            raise Unavailable('Compute Sanitizer is required for complete GPU validation')

    def save(self,final=False):
        required = GROUPS-({'sanitize'} if self.args.cpu_only else set())
        if self.args.sanitize_only: required = {'build','sanitize'}
        missing = sorted(required-self.completed_groups)
        failures = sum(row['status']=='fail' for row in self.rows)
        unavailable = sum(row['status']=='incomplete' for row in self.rows)
        status = 'failed' if failures else ('passed' if final and not missing and not unavailable and not self.args.groups else 'incomplete')
        report = {'status':status,'elapsed_seconds':time.perf_counter()-self.started,
            'provenance':self.provenance,'required_groups':sorted(required),
            'completed_groups':sorted(self.completed_groups),'missing_groups':missing,
            'passed_cases':sum(row['status']=='pass' for row in self.rows),'failed_cases':failures,
            'incomplete_cases':unavailable,
            'limitations':self.catalog['limitations'],'claims':self.catalog['claims'],'cases':self.rows}
        temporary = self.output/'report.json.tmp'
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(self.output/'report.json')
        if final:
            with (self.output/'table.csv').open('w',newline='') as file:
                writer = csv.writer(file); writer.writerow(['group','case','status','seconds','settings','metrics','error'])
                for row in self.rows:
                    writer.writerow([row['group'],row['case'],row['status'],row['seconds'],
                        json.dumps(row['settings']),json.dumps(row.get('metrics',{})),row.get('error','')])
            lines = ['# MERCUDA validation',f'\nStatus: **{status.upper()}**.',
                f'\n{report["passed_cases"]} passed cases; {failures} failed cases; {unavailable} incomplete cases. Scope: {self.provenance["scope"]}.',
                '\n| Group | Passed | Failed | Incomplete |','| --- | ---: | ---: | ---: |']
            for group in sorted(self.completed_groups):
                subset = [r for r in self.rows if r['group']==group]
                lines.append(f'| {group} | {sum(r["status"]=="pass" for r in subset)} | {sum(r["status"]=="fail" for r in subset)} | {sum(r["status"]=="incomplete" for r in subset)} |')
            lines += ['\nMissing groups: '+(', '.join(missing) or 'none'),
                '\nSee report.json for measured errors, frozen thresholds, source hashes and tool versions.',
                '\n## Limits',*['\n- '+limit for limit in self.catalog['limitations']]]
            (self.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
        return status


def main():
    signal.signal(signal.SIGINT,signal.default_int_handler)
    signal.signal(signal.SIGTERM,signal.default_int_handler)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--sanitize-only',action='store_true')
    parser.add_argument('--sanitizer',default='compute-sanitizer')
    parser.add_argument('--groups',help='Comma-separated development subset; never reported as full validation')
    args = parser.parse_args()
    campaign = Campaign(args)
    selected = set(args.groups.split(',')) if args.groups else GROUPS.copy()
    if args.cpu_only: selected.discard('sanitize')
    if args.sanitize_only: selected = {'build','sanitize'}
    if selected-GROUPS: parser.error('Unknown groups: '+str(selected-GROUPS))
    try:
        campaign.preflight()
        for profile in campaign.profiles:
            print('Building '+profile.name,flush=True)
            row = campaign.record('build',profile.name,lambda p=profile:campaign.build(p))
            if row['status'] == 'incomplete': raise campaign.unavailable('Required build could not be validated')
            if row['status'] != 'pass': raise RuntimeError('Required build failed')
        campaign.completed_groups.add('build')
        if 'quick' in selected:
            for profile in campaign.profiles:
                campaign.record('quick',profile.name,lambda p=profile:campaign.quick(p))
            campaign.completed_groups.add('quick')
        from scenarios import execute
        order = (['workflow'] if 'workflow' in selected else [])+sorted(selected-{'build','quick','workflow'})
        for group in order:
            print('Running '+group,flush=True)
            execute(campaign,group)
            campaign.completed_groups.add(group); campaign.save()
    except KeyboardInterrupt:
        campaign.rows.append({'group':'campaign','case':'interrupted','status':'incomplete',
                              'error':'Campaign interrupted before completion','settings':{},'seconds':0})
    except Exception as exc:
        campaign.record('preflight','required-tools-or-campaign',lambda:(_ for _ in ()).throw(exc))
    status = campaign.save(final=True)
    if campaign.rows:
        from figures import plot
        plot(campaign.output)
    print(f'{status.upper()}: {campaign.output}/REPORT.md',flush=True)
    return 0 if status == 'passed' else (1 if status=='failed' else 2)


if __name__ == '__main__':
    raise SystemExit(main())
