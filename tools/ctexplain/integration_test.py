# Lint as: python3
# Copyright 2021 The Bazel Authors. All rights reserved.
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
"""Tests expected results over actual builds."""
from typing import Mapping
from typing import Tuple
import unittest

from src.test.py.bazel import test_base
import tools.ctexplain.analyses.summary as summary
import tools.ctexplain.bazel_api as bazel_api
import tools.ctexplain.lib as lib
from tools.ctexplain.types import ConfiguredTarget

Cts = Tuple[ConfiguredTarget, ...]
TrimmedCts = Mapping[ConfiguredTarget, Cts]


class IntegrationTest(test_base.TestBase):

  _bazel_api: bazel_api.BazelApi = None

  def setUp(self):
    test_base.TestBase.setUp(self)
    self._bazel_api = bazel_api.BazelApi(self.RunBazel)
    self.ScratchFile('WORKSPACE')
    self.CreateWorkspaceWithDefaultRepos('repo/WORKSPACE')
    self.ScratchFile('tools/allowlists/function_transition_allowlist/BUILD', [
        'package_group(',
        '    name = "function_transition_allowlist",',
        '    packages = ["//..."])',
    ])

  def tearDown(self):
    test_base.TestBase.tearDown(self)

  def _get_cts(self, labels: Tuple[str, ...],
               build_flags: Tuple[str, ...]) -> Tuple[Cts, TrimmedCts]:
    """Returns a build's configured targets.

    Args:
      labels: The targets to build.
      build_flags: The build flags to use.

    Returns:
      Tuple (cts, trimmed_cts) where cts is the set of untrimmed configured
      targets for the build and trimmed_cts maps trimmed configured targets to
      their untrimmed variants.
    """
    cts = lib.analyze_build(self._bazel_api, labels, build_flags)
    trimmed_cts = lib.trim_configured_targets(cts)
    return (cts, trimmed_cts)

  # Simple example of a build where trimming makes a big diffrence.
  #
  # Building ":split" builds ":dep1" and its subgraph in two distinct
  # configurations: one with --python_version="PY3" (the default) and
  # one with --python_version="PY2".
  #
  # None of these rules need the Python configuration. So they should
  # all be reducible to the same trimmed equivalent.
  def testHighlyTrimmableBuild(self):
    self.ScratchFile('testapp/defs.bzl', ['''
def _rule_impl(ctx):
    pass
simple_rule = rule(
    implementation = _rule_impl,
    attrs = { "deps": attr.label_list(), },
)

def _my_transition_impl(settings, attr):
    return {"//command_line_option:python_version": "PY2"}
_my_transition = transition(
    implementation = _my_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:python_version"],
)

splitter_rule = rule(
    implementation = _rule_impl,
    attrs = {
        "_allowlist_function_transition": attr.label(
            default = "//tools/allowlists/function_transition_allowlist",
        ),
        "cfg1_deps": attr.label_list(),
        "cfg2_deps": attr.label_list(cfg = _my_transition),
    },
)'''])
    self.ScratchFile('testapp/BUILD', ['''
load(":defs.bzl", "simple_rule", "splitter_rule")
simple_rule(
    name = "dep1",
    deps = [":dep2"],
)
simple_rule(name = "dep2")
splitter_rule(
    name = "buildme",
    cfg1_deps = [":dep1"],
    cfg2_deps = [":dep1"],
)
'''])
    cts, trimmed_cts = self._get_cts(('//testapp:buildme',), ())
    stats = summary.analyze(cts, trimmed_cts)
    self.assertEqual(stats.configurations, 3)
    # It's hard to guess the exact number of configured targets since the
    # dependency graph has a bunch of (often changing) implicit deps. But we
    # expect it to be substantailly greater than the number of targets.
    self.assertLess(stats.targets, stats.configured_targets / 1.5)
    self.assertEqual(stats.targets, stats.trimmed_configured_targets)
    self.assertGreater(stats.repeated_targets, 6)


if __name__ == '__main__':
  unittest.main()
