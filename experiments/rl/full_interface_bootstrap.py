"""Build the full85 experiment from a clean source checkout, without private data.

Construct the public actions16 -> round-chain -> settlement-v5 -> pulsed-normal
ancestry in a temporary directory. The final builder retains the source and
manifests required for validation. No MAME, ROM, checkpoint, model, or training
history is read, and no emulator or PPO job is launched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .actions16_builder import build as build_actions16, PACKAGE as ACTIONS_PACKAGE
from .round_chain_builder import build as build_chain, PACKAGE as CHAIN_PACKAGE
from .pulsed_normals_builder import build as build_pulsed
from .full_interface_builder import build as build_full

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _settlement_v5(chain_root):
    """Use the existing chain revision contract; never relabel an old runtime."""
    manifest_path = chain_root/'build.json'
    previous = manifest_path.read_bytes()
    manifest = json.loads(previous)
    package = chain_root/manifest['package']
    old = manifest['derived_sha256']['settlement.lua']
    source = (HERE/'settlement_v5.lua').read_bytes()
    new = hashlib.sha256(source).hexdigest()
    (chain_root/'revision-parent-build.json').write_bytes(previous)
    (package/'settlement.lua').write_bytes(source)
    manifest['derived_sha256']['settlement.lua'] = new
    manifest['settlement_revision'] = {
        'parent_manifest': 'revision-parent-build.json',
        'parent_build_sha256': hashlib.sha256(previous).hexdigest(),
        'old_sha256': old, 'new_sha256': new,
        'scope': 'Public settlement_v5.lua; native locked round results and timeout boundary.'}
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


def build(output):
    """Return the validated final manifest; output must be a fresh directory."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output already exists: '+str(output))
    with tempfile.TemporaryDirectory(prefix='astra-full85-bootstrap-') as temporary:
        work = Path(temporary)
        build_actions16(work/'actions16')
        build_chain(work/'chain', work/'actions16'/ACTIONS_PACKAGE)
        _settlement_v5(work/'chain')
        shutil.copytree(REPO/'src', work/'chain/src',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        build_pulsed(work/'pulse', work/'chain'/CHAIN_PACKAGE)
        return build_full(work/'pulse', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    manifest = build(**vars(parser.parse_args()))
    print(json.dumps({k: manifest[k] for k in
                     ('schema', 'package', 'actions', 'observations')}, indent=2))


if __name__ == '__main__':
    main()
