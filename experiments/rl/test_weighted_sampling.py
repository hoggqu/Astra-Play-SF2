"""Offline fixed-sampler identity and reset-choice tests; never construct MAME."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from .actions16_builder import build as build16
from .round_chain_builder import build as build_chain, PACKAGE as CHAIN
from .weighted_sampling_builder import build, PACKAGE
from .fixed_sampling import OPPONENTS, validate_config


def config():
    total = sum(range(1, 12))
    return {'schema': 'astra.rl-fixed-opponent-sampling.v1',
            'probabilities': {str(op): (i+1)/total for i, op in enumerate(OPPONENTS)},
            'formula': {'kind': 'synthetic_test_weights'},
            'statistics_source': {'kind': 'offline_test_fixture'}}


class WeightedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        build16(self.root/'sixteen')
        build_chain(self.root/'chain', self.root/'sixteen'/'astra_sf2_rl16')
        self.source = self.root/'chain'/CHAIN
        self.weights = self.root/'weights.json'
        self.weights.write_text(json.dumps(config()))

    def tearDown(self):
        for name in list(sys.modules):
            if name == PACKAGE or name.startswith(PACKAGE+'.'):
                del sys.modules[name]
        self.temp.cleanup()

    def build_import(self):
        manifest = build(self.root/'candidate', self.source, self.weights)
        sys.path.insert(0, str(self.root/'candidate'))
        self.addCleanup(sys.path.remove, str(self.root/'candidate'))
        return manifest, self.root/'candidate'/PACKAGE

    def test_reset_changes_only_opponent_choice_probabilities(self):
        manifest, package = self.build_import()
        module = importlib.import_module(PACKAGE+'.batch_env')
        calls = []
        class Random:
            def choice(self, values, **kwargs):
                calls.append((list(values), kwargs))
                return values[0]
        fake = SimpleNamespace(checkpoint_groups={op: [op*10, op*10+1] for op in OPPONENTS}, np_random=Random())
        result = module.BatchEnv.reset_choice(fake)
        self.assertEqual(result, {'checkpoint': 0, 'lead': 0})
        self.assertEqual(calls[0], (list(OPPONENTS), {'p': list(config()['probabilities'].values())}))
        self.assertEqual(calls[1], ([0, 1], {}))
        self.assertEqual(calls[2][1], {})
        fake.checkpoint_groups.pop(11)
        with self.assertRaises(ValueError):
            module.BatchEnv.reset_choice(fake)
        self.assertFalse(manifest['native_validated'])

    def test_gameplay_and_lua_preserved_and_dependencies_audited(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir() if p.is_file()}
        manifest, package = self.build_import()
        for name, body in before.items():
            if name.endswith('.lua'):
                self.assertEqual((package/name).read_bytes(), body)
        for name in ('continuous.py', 'native_continuous.py', 'export.py'):
            self.assertEqual((package/name).read_text(), before[name].decode().replace(CHAIN, PACKAGE))
        identity = importlib.import_module(PACKAGE+'.weighted_identity')
        identity.validate_weighted_build(manifest, package)
        metadata = identity.sampling_metadata(package)
        self.assertEqual(metadata['configuration'], config())
        support = importlib.import_module(PACKAGE+'.support')
        self.assertIs(support.validate_chain_build, identity.validate_weighted_build)
        campaign = importlib.import_module(PACKAGE+'.native_campaign')
        paths = campaign.source_paths()
        for name in ('fixed_sampling.py', 'sampling_config.py', 'weighted_identity.py'):
            self.assertIn(package/name, paths.values())
        for name in ('build.json', 'parent-chain-build.json', 'parent-build.json'):
            self.assertIn(package.parent/name, paths.values())
        self.assertIn("'opponent_sampling': sampling_metadata", (package/'batch_train.py').read_text())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir() if p.is_file()})

    def test_configuration_or_parent_tamper_rejected(self):
        manifest, package = self.build_import()
        identity = importlib.import_module(PACKAGE+'.weighted_identity')
        for path in (package/'sampling_config.py', package/'fixed_sampling.py', package.parent/'parent-chain-build.json', package.parent/'parent-build.json'):
            before = path.read_bytes()
            path.write_bytes(before+b'\n#tamper')
            with self.assertRaises((RuntimeError, ValueError)):
                identity.validate_weighted_build(manifest, package)
            path.write_bytes(before)
        identity.validate_weighted_build(manifest, package)

    def test_bad_probabilities_rejected_without_build(self):
        for mode in ('missing', 'zero', 'nan', 'bool', 'sum'):
            value = config()
            if mode == 'missing': value['probabilities'].pop('11')
            else: value['probabilities']['0'] = {'zero': 0, 'nan': float('nan'), 'bool': True, 'sum': .5}[mode]
            with self.assertRaises(ValueError): validate_config(value)

    def test_parent_package_mutation_rejected(self):
        path = self.source/'settlement.lua'
        path.write_text(path.read_text()+'\n-- changed')
        with self.assertRaises(ValueError):
            build(self.root/'candidate', self.source, self.weights)


if __name__ == '__main__':
    unittest.main()
