// Copyright 2018 The Bazel Authors. All rights reserved.
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

import org.junit.Test;

/** Tests that are common to {@code py_binary} and {@code py_test}. */
public abstract class PyExecutableConfiguredTargetTestBase extends PyBaseConfiguredTargetTestBase {

  private final String ruleName;

  protected PyExecutableConfiguredTargetTestBase(String ruleName) {
    super(ruleName);
    this.ruleName = ruleName;
  }

  @Test
  public void unknownDefaultPythonVersionValue() throws Exception {
    checkError("pkg", "foo",
        // error:
        "invalid value in 'default_python_version' attribute: "
            + "has to be one of 'PY2' or 'PY3' instead of 'doesnotexist'",
        // build file:
        ruleName + "(",
        "     name = 'foo',",
        "     default_python_version = 'doesnotexist',",
        "     srcs = ['foo.py'])");
  }

  @Test
  public void badDefaultPythonVersionValue() throws Exception {
    checkError("pkg", "foo",
        // error:
        "invalid value in 'default_python_version' attribute: "
            + "has to be one of 'PY2' or 'PY3' instead of 'PY2AND3'",
        // build file:
        ruleName + "(",
        "     name = 'foo',",
        "     default_python_version = 'PY2AND3',",
        "     srcs = ['foo.py'])");
  }

  @Test
  public void goodDefaultPythonVersionValue() throws Exception {
    scratch.file("foo/BUILD",
        ruleName + "(",
        "     name = 'foo',",
        "     default_python_version = 'PY2',",
        "     srcs = ['foo.py'])");
    getConfiguredTarget("//foo:foo");
    assertNoEvents();
  }

  @Test
  public void versionIs3WhenSetByDefaultPythonVersion() throws Exception {
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = ['foo.py'],",
        "    default_python_version = 'PY3')");
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo"))).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void versionIs2WhenSetByDefaultPythonVersion() throws Exception {
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = ['foo.py'],",
        "    default_python_version = 'PY2')");
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo"))).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void canBuildTwoTargetsSpecifyingDifferentVersions() throws Exception {
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo_v2',",
        "    srcs = ['foo_v2.py'],",
        "    default_python_version = 'PY2')",
        ruleName + "(",
        "    name = 'foo_v3',",
        "    srcs = ['foo_v3.py'],",
        "    default_python_version = 'PY3')");

    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo_v2"))).isEqualTo(PythonVersion.PY2);
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo_v3"))).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void flagOverridesDefaultPythonVersionFrom2To3() throws Exception {
    useConfiguration("--force_python=PY3");
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = ['foo.py'],",
        "    default_python_version = 'PY2')");
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo"))).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void flagOverridesDefaultPythonVersionFrom3To2() throws Exception {
    useConfiguration("--force_python=PY2");
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = ['foo.py'],",
        "    default_python_version = 'PY3')");
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo"))).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void canBuildTwoTargetsSpecifyingDifferentVersions_ForcedTo2() throws Exception {
    useConfiguration("--force_python=PY2");
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo_v2',",
        "    srcs = ['foo_v2.py'],",
        "    default_python_version = 'PY2')",
        ruleName + "(",
        "    name = 'foo_v3',",
        "    srcs = ['foo_v3.py'],",
        "    default_python_version = 'PY3')");
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo_v2"))).isEqualTo(PythonVersion.PY2);
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo_v3"))).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void canBuildTwoTargetsSpecifyingDifferentVersions_ForcedTo3() throws Exception {
    useConfiguration("--force_python=PY3");
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo_v2',",
        "    srcs = ['foo_v2.py'],",
        "    default_python_version = 'PY2')",
        ruleName + "(",
        "    name = 'foo_v3',",
        "    srcs = ['foo_v3.py'],",
        "    default_python_version = 'PY3')");
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo_v2"))).isEqualTo(PythonVersion.PY3);
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo_v3"))).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void srcsVersionClashesWithDefaultVersionAttr() throws Exception {
    checkError("pkg", "foo",
        // error:
        "'//pkg:foo' can only be used with Python 2",
        // build file:
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = [':foo.py'],",
        "    srcs_version = 'PY2ONLY',",
        "    default_python_version = 'PY3')");
  }

  @Test
  public void srcsVersionClashesWithDefaultVersionAttr_Implicitly() throws Exception {
    // Canary assertion: This'll fail when we flip the default to PY3. At that point change this
    // test to use srcs_version = 'PY2ONLY' instead.
    assertThat(PythonVersion.DEFAULT_TARGET_VALUE).isEqualTo(PythonVersion.PY2);

    // Fails because default_python_version is PY2 by default, so the config is set to PY2
    // regardless of srcs_version.
    checkError("pkg", "foo",
        // error:
        "'//pkg:foo' can only be used with Python 3",
        // build file:
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = [':foo.py'],",
        "    srcs_version = 'PY3ONLY')");
  }
}
