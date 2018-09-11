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

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;

/** Tests that are common to {@code py_binary} and {@code py_test}. */
public abstract class PyExecutableConfiguredTargetTestBase extends BuildViewTestCase {

  private final String ruleName;

  protected PyExecutableConfiguredTargetTestBase(String ruleName) {
    this.ruleName = ruleName;
  }

  @Before
  public final void setUpPython() throws Exception {
    analysisMock.pySupport().setup(mockToolsConfig);
  }

  private PythonVersion getPythonVersion(ConfiguredTarget ct) {
    return getConfiguration(ct).getOptions().get(PythonOptions.class).getPythonVersion();
  }

  @Test
  public void badDefaultPythonVersion() throws Exception {
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
  public void goodDefaultPythonVersion() throws Exception {
    scratch.file("foo/BUILD",
        ruleName + "(",
        "     name = 'foo',",
        "     default_python_version = 'PY2',",
        "     srcs = ['foo.py'])");
    getConfiguredTarget("//foo:foo");
    assertNoEvents();
  }

  @Test
  public void badSrcsVersion() throws Exception {
    checkError("pkg", "foo",
        // error:
        "invalid value in 'srcs_version' attribute: "
            + "has to be one of 'PY2', 'PY3', 'PY2AND3', 'PY2ONLY' "
            + "or 'PY3ONLY' instead of 'invalid'",
        // build file:
        ruleName + "(",
        "    name = 'foo',",
        "    srcs_version = 'invalid',",
        "    srcs = ['foo.py'])");
  }

  @Test
  public void goodSrcsVersion() throws Exception {
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo',",
        "    srcs_version = 'PY2',",
        "    srcs = ['foo.py'])");
    getConfiguredTarget("//foo:foo");
    assertNoEvents();
  }

  @Test
  public void pythonVersionWith3AsDefault() throws Exception {
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = ['foo.py'],",
        "    default_python_version = 'PY3')");
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo"))).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void pythonVersionWith2AsDefault() throws Exception {
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = ['foo.py'],",
        "    default_python_version = 'PY2')");
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo"))).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void pythonVersionDefaultForBuildIs2() throws Exception {
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = ['foo.py'])");
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo"))).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void pythonVersionsWithMixedDefaults() throws Exception {
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
  public void forcePython3Version() throws Exception {
    useConfiguration("--force_python=PY3");
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = ['foo.py'],",
        "    default_python_version = 'PY2')");
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo"))).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void forcePython2Version() throws Exception {
    useConfiguration("--force_python=PY2");
    scratch.file("foo/BUILD",
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = ['foo.py'],",
        "    default_python_version = 'PY3')");
    assertThat(getPythonVersion(getConfiguredTarget("//foo:foo"))).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void forcePython2VersionMultiple() throws Exception {
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
  public void forcePython3VersionMultiple() throws Exception {
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

}
