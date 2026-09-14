"""Build a separate perception128 source with a strictly observed non-TIME KO timer tail."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from .perception128_draw_identity import (INPUT_NAMES,PACKAGE,SCHEMA,PROTOCOL,
                                                digest,derive,validate_parent,validate_build)
HERE=Path(__file__).resolve().parent


def build(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    parent,captured=validate_parent(source)
    inputs={n:(HERE/n).read_bytes() for n in INPUT_NAMES};derived=derive(captured,inputs)
    for n,b in derived.items():
        if n.endswith('.py'):compile(b,n,'exec')
    output.mkdir(parents=True,exist_ok=False);ignore=shutil.ignore_patterns('__pycache__','*.pyc')
    shutil.copytree(source,output/'settlement-parent',ignore=ignore)
    shutil.copytree(source/'src',output/'src',ignore=ignore)
    package=output/PACKAGE;package.mkdir()
    for n,b in derived.items():(package/n).write_bytes(b)
    (output/'settlement-inputs').mkdir()
    for n,b in inputs.items():(output/'settlement-inputs'/n).write_bytes(b)
    launcher=(source/'launch.py').read_text().replace(parent['package'],PACKAGE)
    (output/'launch.py').write_text(launcher)
    frozen={p.relative_to(output).as_posix():digest(p.read_bytes())
            for name in ('settlement-parent','settlement-inputs','src')
            for p in (output/name).rglob('*') if p.is_file()}
    frozen['launch.py']=digest(launcher.encode())
    manifest=dict(parent)
    manifest.update(schema=SCHEMA,package=PACKAGE,settlement_protocol=PROTOCOL,
                    status='candidate',native_validated=False,parent_build_sha256=digest((source/'build.json').read_bytes()),
                    derived_sha256={n:digest(b) for n,b in derived.items()},frozen_files_sha256=frozen,
                    scope='Result-only observed non-TIME KO timer tail with mature native pip confirmation; all policy/input/reward/optimizer bytes unchanged.')
    (output/'build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate_build(manifest,package)
    if validate_parent(source)[1]!=captured:raise RuntimeError('Parent changed during KO build')
    return manifest


def bootstrap(output):
    from .perception128_timing_builder import bootstrap as make_parent
    with tempfile.TemporaryDirectory(prefix='astra-native-ko-') as folder:
        parent=Path(folder)/'parent';make_parent(parent);return build(parent,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=build(args.source,args.output) if args.source else bootstrap(args.output)
    print(json.dumps({k:result[k] for k in ('schema','package','settlement_protocol','parent_build_sha256')}))


if __name__=='__main__':main()
