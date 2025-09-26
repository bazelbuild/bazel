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
import static com.google.devtools.build.lib.rules.python.PythonTestUtils.getPyLoad;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@code py_runtime}. */
@RunWith(JUnit4.class)
public class PyRuntimeConfiguredTargetTest extends BuildViewTestCase {

  @Before
  public final void setUpPython() throws Exception {
    analysisMock.pySupport().setup(mockToolsConfig);
  }

  @Test
  public void nonhermeticRuntime() throws Exception {
    scratch.file(
        "pkg/BUILD",
        getPyLoad("py_runtime"),
        "py_runtime(",
        "    name = 'myruntime',",
        "    interpreter_path = '/system/interpreter',",
        "    python_version = 'PY3',",
        ")");
    PyRuntimeInfo info = PyRuntimeInfo.fromTarget(getConfiguredTarget("//pkg:myruntime"));

    assertThat(info.getInterpreterPathString()).isEqualTo("/system/interpreter");
    assertThat(info.getInterpreter()).isNull();
    assertThat(info.getFiles()).isNull();
    assertThat(info.getPythonVersion()).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void cannotUseBothInterpreterAndPath() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "pkg/BUILD",
        getPyLoad("py_runtime"),
        "py_runtime(",
        "    name = 'myruntime',",
        "    interpreter = ':myinterpreter',",
        "    interpreter_path = '/system/interpreter',",
        "    python_version = 'PY3',",
        ")");
    getConfiguredTarget("//pkg:myruntime");

    assertContainsEvent(
        "exactly one of the 'interpreter' or 'interpreter_path' attributes must be specified");
  }

  @Test
  public void mustUseEitherInterpreterOrPath() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "pkg/BUILD", //
        getPyLoad("py_runtime"),
        "py_runtime(",
        "    name = 'myruntime',",
        "    python_version = 'PY3',",
        ")");
    getConfiguredTarget("//pkg:myruntime");

    assertContainsEvent(
        "exactly one of the 'interpreter' or 'interpreter_path' attributes must be specified");
  }

  @Test
  public void interpreterPathMustBeAbsolute() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "pkg/BUILD",
        getPyLoad("py_runtime"),
        "py_runtime(",
        "    name = 'myruntime',",
        "    interpreter_path = 'some/relative/path',",
        "    python_version = 'PY3',",
        ")");
    getConfiguredTarget("//pkg:myruntime");

    assertContainsEvent("must be an absolute path");
  }

  @Test
  public void cannotSpecifyFilesForNonhermeticRuntime() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "pkg/BUILD",
        getPyLoad("py_runtime"),
        "py_runtime(",
        "    name = 'myruntime',",
        "    files = [':myfile'],",
        "    interpreter_path = '/system/interpreter',",
        "    python_version = 'PY3',",
        ")");
    getConfiguredTarget("//pkg:myruntime");

    assertContainsEvent("if 'interpreter_path' is given then 'files' must be empty");
  }

  @Test
  public void badPythonVersionAttribute() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "pkg/BUILD",
        getPyLoad("py_runtime"),
        "py_runtime(",
        "    name = 'myruntime',",
        "    interpreter_path = '/system/interpreter',",
        "    python_version = 'not a Python version',",
        ")");
    getConfiguredTarget("//pkg:myruntime");

    assertContainsEvent("invalid value in 'python_version' attribute");
  }
}
