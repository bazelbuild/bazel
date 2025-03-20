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
"""Tests for bazel_api.py."""
import os
import unittest
from src.test.py.bazel import test_base
from tools.ctexplain.bazel_api import BazelApi
from tools.ctexplain.types import HostConfiguration
from tools.ctexplain.types import NullConfiguration


class BazelApiTest(test_base.TestBase):

  _bazel_api: BazelApi = None

  def setUp(self):
    test_base.TestBase.setUp(self)
    self._bazel_api = BazelApi(self.RunBazel)

  def tearDown(self):
    test_base.TestBase.tearDown(self)

  def testBasicCquery(self):
    self.ScratchFile('testapp/BUILD', [
        'filegroup(name = "fg", srcs = ["a.file"])',
    ])
    res = self._bazel_api.cquery(['//testapp:all'])
    success = res[0]
    cts = res[2]
    self.assertTrue(success)
    self.assertEqual(len(cts), 1)
    self.assertEqual(cts[0].label, '//testapp:fg')
    self.assertIsNone(cts[0].config)
    self.assertGreater(len(cts[0].config_hash), 10)
    self.assertIn('PlatformConfiguration', cts[0].transitive_fragments)

  def testFailedCquery(self):
    self.ScratchFile('testapp/BUILD', [
        'filegroup(name = "fg", srcs = ["a.file"])',
    ])
    (success, stderr, cts) = self._bazel_api.cquery(['//testapp:typo'])
    self.assertFalse(success)
    self.assertEqual(len(cts), 0)
    self.assertIn("target 'typo' not declared in package 'testapp'",
                  os.linesep.join(stderr))

  def testTransitiveFragmentsAccuracy(self):
    self.ScratchFile('testapp/BUILD', [
        'filegroup(name = "fg", srcs = ["a.file"])',
        'filegroup(name = "ccfg", srcs = [":ccbin"])',
        'cc_binary(name = "ccbin", srcs = ["ccbin.cc"])'
    ])
    cts1 = self._bazel_api.cquery(['//testapp:fg'])[2]
    self.assertNotIn('CppConfiguration', cts1[0].transitive_fragments)
    cts2 = self._bazel_api.cquery(['//testapp:ccfg'])[2]
    self.assertIn('CppConfiguration', cts2[0].transitive_fragments)

  def testGetTargetConfig(self):
    self.ScratchFile('testapp/BUILD', [
        'filegroup(name = "fg", srcs = ["a.file"])',
    ])
    cts = self._bazel_api.cquery(['//testapp:fg'])[2]
    config = self._bazel_api.get_config(cts[0].config_hash)
    expected_fragments = ['PlatformConfiguration', 'JavaConfiguration']
    for exp in expected_fragments:
      self.assertIn(exp, config.fragments.keys())
    core_options = config.options['CoreOptions']
    self.assertIsNotNone(core_options)
    self.assertIn(('stamp', 'false'), core_options.items())

  def testGetHostConfig(self):
    self.ScratchFile('testapp/BUILD', [
        'genrule(',
        '    name = "g",',
        '    srcs = [],',
        '    cmd = "",',
        '    outs = ["g.out"],',
        '    tools = [":fg"])',
        'filegroup(name = "fg", srcs = ["a.file"])',
    ])
    query = ['//testapp:fg', '--universe_scope=//testapp:g']
    cts = self._bazel_api.cquery(query)[2]
    config = self._bazel_api.get_config(cts[0].config_hash)
    self.assertIsInstance(config, HostConfiguration)
    # We don't currently populate or read a host configuration's details.
    self.assertEqual(len(config.fragments), 0)
    self.assertEqual(len(config.options), 0)

  def testGetNullConfig(self):
    self.ScratchFile('testapp/BUILD', [
        'filegroup(name = "fg", srcs = ["a.file"])',
    ])
    cts = self._bazel_api.cquery(['//testapp:a.file'])[2]
    config = self._bazel_api.get_config(cts[0].config_hash)
    self.assertIsInstance(config, NullConfiguration)
    # Null configurations have no information by definition.
    self.assertEqual(len(config.fragments), 0)
    self.assertEqual(len(config.options), 0)

  def testConfigFragmentsMap(self):
    self.ScratchFile('testapp/BUILD', [
        'filegroup(name = "fg", srcs = ["a.file"])',
    ])
    cts = self._bazel_api.cquery(['//testapp:fg'])[2]
    fragments_map = self._bazel_api.get_config(cts[0].config_hash).fragments
    self.assertIn('PlatformOptions', fragments_map['PlatformConfiguration'])
    self.assertIn('ShellConfiguration$Options',
                  fragments_map['ShellConfiguration'])

  def testConfigWithDefines(self):
    self.ScratchFile('testapp/BUILD', [
        'filegroup(name = "fg", srcs = ["a.file"])',
    ])
    cquery_args = ['//testapp:fg', '--define', 'a=b']
    cts = self._bazel_api.cquery(cquery_args)[2]
    config = self._bazel_api.get_config(cts[0].config_hash)
    user_defined_options = config.options['user-defined']
    self.assertIsNotNone(user_defined_options)
    self.assertDictEqual(user_defined_options._dict, {'--define:a': 'b'})

  def testConfigWithStarlarkFlags(self):
    self.ScratchFile('testapp/defs.bzl', [
        'def _flag_impl(settings, attr):',
        '  pass',
        'string_flag = rule(',
        '  implementation = _flag_impl,',
        '  build_setting = config.string(flag = True)',
        ')',
    ])
    self.ScratchFile('testapp/BUILD', [
        'load(":defs.bzl", "string_flag")',
        'string_flag(name = "my_flag", build_setting_default = "nada")',
        'filegroup(name = "fg", srcs = ["a.file"])',
    ])
    cquery_args = ['//testapp:fg', '--//testapp:my_flag', 'algo']
    cts = self._bazel_api.cquery(cquery_args)[2]
    config = self._bazel_api.get_config(cts[0].config_hash)
    user_defined_options = config.options['user-defined']
    self.assertIsNotNone(user_defined_options)
    self.assertDictEqual(user_defined_options._dict,
                         {'//testapp:my_flag': 'algo'})


if __name__ == '__main__':
  unittest.main()
