# Lint as: python3
# Copyright 2020 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for lib.py."""
import unittest
from frozendict import frozendict

from src.test.py.bazel import test_base
import tools.ctexplain.bazel_api as bazel_api
import tools.ctexplain.lib as lib
from tools.ctexplain.types import Configuration
from tools.ctexplain.types import ConfiguredTarget
from tools.ctexplain.types import HostConfiguration
from tools.ctexplain.types import NullConfiguration


class LibTest(test_base.TestBase):

  _bazel: bazel_api.BazelApi = None

  def setUp(self):
    test_base.TestBase.setUp(self)
    self._bazel = bazel_api.BazelApi(self.RunBazel)
    self.ScratchFile('WORKSPACE')
    self.CreateWorkspaceWithDefaultRepos('repo/WORKSPACE')

  def tearDown(self):
    test_base.TestBase.tearDown(self)

  def testAnalyzeBuild(self):
    self.ScratchFile('testapp/defs.bzl', [
        'def _impl(ctx):',
        '    pass',
        'rule_with_host_dep = rule(',
        '    implementation = _impl,',
        '    attrs = { "host_deps": attr.label_list(cfg = "host") })',
    ])
    self.ScratchFile('testapp/BUILD', [
        'load("//testapp:defs.bzl", "rule_with_host_dep")',
        'rule_with_host_dep(name = "a", host_deps = [":h"])',
        'filegroup(name = "h", srcs = ["h.src"])'
    ])
    cts = lib.analyze_build(self._bazel, ('//testapp:a',), ())
    # Remove boilerplate deps to focus on targets declared here.
    cts = [ct for ct in cts if ct.label.startswith('//testapp')]

    self.assertListEqual([ct.label for ct in cts],
                         ['//testapp:a', '//testapp:h', '//testapp:h.src'])
    # Don't use assertIsInstance because we don't want to match subclasses.
    self.assertEqual(Configuration, type(cts[0].config))
    self.assertEqual('HOST', cts[1].config_hash)
    self.assertIsInstance(cts[1].config, HostConfiguration)
    self.assertEqual('null', cts[2].config_hash)
    self.assertIsInstance(cts[2].config, NullConfiguration)

  def testAnalyzeBuildNoRepeats(self):
    self.ScratchFile('testapp/defs.bzl', [
        'def _impl(ctx):',
        '    pass',
        'rule_with_host_dep = rule(',
        '    implementation = _impl,',
        '    attrs = { "host_deps": attr.label_list(cfg = "host") })',
    ])
    self.ScratchFile('testapp/BUILD', [
        'load("//testapp:defs.bzl", "rule_with_host_dep")',
        'rule_with_host_dep(name = "a", host_deps = [":h", ":other"])',
        'rule_with_host_dep(name = "other")',
        'filegroup(name = "h", srcs = ["h.src", ":other"])'
    ])
    cts = lib.analyze_build(self._bazel, ('//testapp:a',), ())
    # Remove boilerplate deps to focus on targets declared here.
    cts = [ct for ct in cts if ct.label.startswith('//testapp')]

    # Even though the build references //testapp:other twice, it only appears
    # once.
    self.assertListEqual(
        [ct.label for ct in cts],
        ['//testapp:a', '//testapp:h', '//testapp:other', '//testapp:h.src'])

  def testBasicTrimming(self):
    fragments = frozendict({
        'FooFragment': ('FooOptions',),
        'BarFragment': ('BarOptions',),
    })
    options1 = frozendict({
        'FooOptions': frozendict({'foo_opt': 'foo_val1'}),
        'BarOptions': frozendict({'bar_opt': 'bar_val1'}),
    })
    options2 = frozendict({
        'FooOptions': frozendict({'foo_opt': 'foo_val2'}),
        'BarOptions': frozendict({'bar_opt': 'bar_val1'}),
    })
    options3 = frozendict({
        'FooOptions': frozendict({'foo_opt': 'foo_val1'}),
        'BarOptions': frozendict({'bar_opt': 'bar_val2'}),
    })

    config1 = Configuration(fragments, options1)
    config2 = Configuration(fragments, options2)
    config3 = Configuration(fragments, options3)

    ct1 = ConfiguredTarget('//foo', config1, 'hash1', ('FooFragment',))
    ct2 = ConfiguredTarget('//foo', config2, 'hash2', ('FooFragment',))
    ct3 = ConfiguredTarget('//foo', config3, 'hash3', ('FooFragment',))

    get_foo_opt = lambda x: x.config.options['FooOptions']['foo_opt']
    trimmed_cts = sorted(lib.trim_configured_targets((ct1, ct2, ct3)).items(),
                         key=lambda x: get_foo_opt(x[0]))

    self.assertEqual(len(trimmed_cts), 2)
    self.assertEqual(get_foo_opt(trimmed_cts[0][0]), 'foo_val1')
    self.assertListEqual(trimmed_cts[0][1], [ct1, ct3])
    self.assertEqual(get_foo_opt(trimmed_cts[1][0]), 'foo_val2')
    self.assertListEqual(trimmed_cts[1][1], [ct2])

  # Requirements are FragmentOptions, not Fragments.
  def testTrimRequiredOptions(self):
    config = Configuration(
        fragments=frozendict({
            'FooFragment': ('FooOptions',),
            'BarFragment': ('BarOptions',),
            'GreedyFragment': ('FooOptions', 'BarOptions'),
        }),
        options=frozendict({
            'FooOptions': frozendict({'foo_opt': 'foo_val'}),
            'BarOptions': frozendict({'bar_opt': 'bar_val'}),
        })
    )

    ct = ConfiguredTarget('//foo', config, 'hash', ('FooOptions',))
    trimmed_cts = lib.trim_configured_targets((ct,))
    trimmed_ct = list(trimmed_cts.keys())[0]

    self.assertEqual(len(trimmed_cts), 1)
    # Currently expect to keep the requiring fragment (of all that require
    # 'FooOptions') with the smallest number of total requirements.
    self.assertTupleEqual(tuple(trimmed_ct.config.fragments.keys()),
                          ('FooFragment',))
    self.assertEqual(trimmed_ct.config.options,
                     frozendict(
                         {'FooOptions': frozendict({'foo_opt': 'foo_val'})}
                     ))

  def testTrimUserDefinedFlags(self):
    config = Configuration(
        fragments=frozendict({'FooFragment': ('FooOptions',)}),
        options=frozendict({
            'FooOptions': frozendict({}),
            'user-defined': frozendict({
                '--define:foo': 'foo_val',
                '--define:bar': 'bar_val',
                '--//starlark_foo_flag': 'starlark_foo',
                '--//starlark_bar_flag': 'starlark_bar',
            }),
        })
    )

    required = ('FooFragment', '--define:foo', '--//starlark_bar_flag')
    ct = ConfiguredTarget('//foo', config, 'hash', required)
    trimmed_cts = lib.trim_configured_targets((ct,))
    trimmed_ct = list(trimmed_cts.keys())[0]

    self.assertEqual(len(trimmed_cts), 1)
    self.assertTupleEqual(tuple(trimmed_ct.config.fragments.keys()),
                          ('FooFragment',))
    self.assertEqual(trimmed_ct.config.options,
                     frozendict({
                         'FooOptions': frozendict({}),
                         'user-defined': frozendict({
                             '--define:foo': 'foo_val',
                             '--//starlark_bar_flag': 'starlark_bar',
                         }),
                     }))

  def testTrimUnnecessaryCoreOptions(self):
    config = Configuration(
        fragments=frozendict({}),
        options=frozendict({
            'CoreOptions': frozendict({
                'affected by starlark transition': 'drop this',
                'keep this': 'keep val',
                'transition directory name fragment': 'drop this too',
            })
        }))

    ct = ConfiguredTarget('//foo', config, 'hash', ())
    trimmed_cts = lib.trim_configured_targets((ct,))
    trimmed_ct = list(trimmed_cts.keys())[0]

    self.assertEqual(len(trimmed_cts), 1)
    self.assertEqual(trimmed_ct.config.options,
                     frozendict({
                         'CoreOptions': frozendict({'keep this': 'keep val'}),
                     }))

if __name__ == '__main__':
  unittest.main()
