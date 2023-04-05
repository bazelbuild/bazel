// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.python;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@code <tools repo>//tools/python:srcs_version.bzl%find_requirements}. */
@RunWith(JUnit4.class)
public class PythonSrcsVersionAspectTest extends BuildViewTestCase {

  private static String join(String... args) {
    return String.join("\n", args);
  }

  /**
   * Returns the contents of the {@code -pyversioninfo.txt} file that would be produced by running
   * the aspect on the given target.
   */
  private String evaluateAspectFor(String label) throws Exception {
    scratch.file(
        "asp/BUILD",
        "load('" + TestConstants.TOOLS_REPOSITORY + "//tools/python:srcs_version.bzl', ",
        "     'apply_find_requirements_for_testing')",
        "apply_find_requirements_for_testing(",
        "    name = 'asp',",
        "    target = '" + label + "',",
        "    out = 'out',",
        ")");
    ConfiguredTarget ct = getConfiguredTarget("//asp");
    assertThat(ct).isNotNull();
    Artifact out = getBinArtifact("out", ct);
    Action action = getGeneratingAction(out);
    assertThat(action).isInstanceOf(FileWriteAction.class);
    return ((FileWriteAction) action).getFileContents();
  }

  @Test
  public void noRequirements() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "py_library(",
        "    name = 'lib',",
        "    srcs = ['lib.py'],",
        ")",
        "py_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.py'],",
        "    deps = [':lib'],",
        ")");
    String result = evaluateAspectFor("//pkg:bin");
    String golden =
        join(
            "Python 2-only deps:",
            "<None>",
            "",
            "Python 3-only deps:",
            "<None>",
            "",
            "Paths to these deps:",
            "<None>",
            "");
    assertThat(result).isEqualTo(golden);
  }

  @Test
  public void requirementNotPropagated() throws Exception {
    // A <- B <- C <- bin, A introduces the requirement, but B doesn't propagate it.
    // dummy_rule propagates sources but nothing else. It also has a srcs_version attr that is
    // ignored because the provider field is false.
    scratch.file(
        "pkg/rules.bzl",
        "def _dummy_rule_impl(ctx):",
        "    info = PyInfo(",
        "        transitive_sources = depset(",
        "            transitive=[d[PyInfo].transitive_sources for d in ctx.attr.deps],",
        "            order='postorder'))",
        "    return [info]",
        "dummy_rule = rule(",
        "    implementation = _dummy_rule_impl,",
        "    attrs = {",
        "        'deps': attr.label_list(),",
        "        'srcs_version': attr.string(),",
        "    },",
        ")");
    scratch.file(
        "pkg/BUILD",
        "load(':rules.bzl', 'dummy_rule')",
        "py_library(",
        "    name = 'libA',",
        "    srcs = ['libA.py'],",
        "    srcs_version = 'PY3ONLY',",
        ")",
        "dummy_rule(",
        "    name = 'libB',",
        "    deps = [':libA'],",
        "    srcs_version = 'PY3ONLY',",
        ")",
        "py_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.py'],",
        "    deps = [':libB'],",
        ")");
    String result = evaluateAspectFor("//pkg:bin");
    String golden =
        join(
            "Python 2-only deps:",
            "<None>",
            "",
            "Python 3-only deps:",
            "<None>",
            "",
            "Paths to these deps:",
            "<None>",
            "");
    assertThat(result).isEqualTo(golden);
  }

  @Test
  public void toleratesTargetsWithoutDepsAttr() throws Exception {
    scratch.file(
        "pkg/rules.bzl",
        "def _dummy_rule_impl(ctx):",
        "    info = PyInfo(transitive_sources = depset([]))",
        "    return [info]",
        "dummy_rule = rule(",
        "    implementation = _dummy_rule_impl,",
        ")");
    scratch.file(
        "pkg/BUILD",
        "load(':rules.bzl', 'dummy_rule')",
        "dummy_rule(",
        "    name = 'lib',",
        ")",
        "py_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.py'],",
        "    deps = [':lib'],",
        ")");
    evaluateAspectFor("//pkg:bin");
  }
}
