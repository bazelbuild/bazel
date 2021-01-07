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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.starlark.util.BazelEvaluationTestCase;
import com.google.devtools.build.lib.vfs.PathFragment;
import net.starlark.java.syntax.Location;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PyRuntimeInfo}. */
@RunWith(JUnit4.class)
public class PyRuntimeInfoTest extends BuildViewTestCase {

  private final BazelEvaluationTestCase ev = new BazelEvaluationTestCase();

  private Artifact dummyInterpreter;
  private Artifact dummyFile;

  @Before
  public void setUp() throws Exception {
    dummyInterpreter = getSourceArtifact("dummy_interpreter");
    dummyFile = getSourceArtifact("dummy_file");
    ev.update("PyRuntimeInfo", PyRuntimeInfo.PROVIDER);
    ev.update("dummy_interpreter", dummyInterpreter);
    ev.update("dummy_file", dummyFile);
  }

  /** We need this because {@code NestedSet}s don't have value equality. */
  private static void assertHasOrderAndContainsExactly(
      NestedSet<?> set, Order order, Object... values) {
    assertThat(set.getOrder()).isEqualTo(order);
    assertThat(set.toList()).containsExactly(values);
  }

  @Test
  public void factoryMethod_InBuildRuntime() throws Exception {
    NestedSet<Artifact> files = NestedSetBuilder.create(Order.STABLE_ORDER, dummyFile);
    PyRuntimeInfo inBuildRuntime =
        PyRuntimeInfo.createForInBuildRuntime(dummyInterpreter, files, PythonVersion.PY2);

    assertThat(inBuildRuntime.getCreationLocation()).isEqualTo(Location.BUILTIN);
    assertThat(inBuildRuntime.getInterpreterPath()).isNull();
    assertThat(inBuildRuntime.getInterpreterPathString()).isNull();
    assertThat(inBuildRuntime.getInterpreter()).isEqualTo(dummyInterpreter);
    assertThat(inBuildRuntime.getFiles()).isEqualTo(files);
    assertThat(inBuildRuntime.getFilesForStarlark().getSet(Artifact.class)).isEqualTo(files);
    assertThat(inBuildRuntime.getPythonVersion()).isEqualTo(PythonVersion.PY2);
    assertThat(inBuildRuntime.getPythonVersionForStarlark()).isEqualTo("PY2");
  }

  @Test
  public void factoryMethod_PlatformRuntime() {
    PathFragment path = PathFragment.create("/system/interpreter");
    PyRuntimeInfo platformRuntime = PyRuntimeInfo.createForPlatformRuntime(path, PythonVersion.PY2);

    assertThat(platformRuntime.getCreationLocation()).isEqualTo(Location.BUILTIN);
    assertThat(platformRuntime.getInterpreterPath()).isEqualTo(path);
    assertThat(platformRuntime.getInterpreterPathString()).isEqualTo("/system/interpreter");
    assertThat(platformRuntime.getInterpreter()).isNull();
    assertThat(platformRuntime.getFiles()).isNull();
    assertThat(platformRuntime.getFilesForStarlark()).isNull();
    assertThat(platformRuntime.getPythonVersion()).isEqualTo(PythonVersion.PY2);
    assertThat(platformRuntime.getPythonVersionForStarlark()).isEqualTo("PY2");
  }

  @Test
  public void starlarkConstructor_InBuildRuntime() throws Exception {
    ev.exec(
        "info = PyRuntimeInfo(",
        "    interpreter = dummy_interpreter,",
        "    files = depset([dummy_file]),",
        "    python_version = 'PY2',",
        ")");
    PyRuntimeInfo info = (PyRuntimeInfo) ev.lookup("info");
    assertThat(info.getCreationLocation().toString()).isEqualTo(":1:21");
    assertThat(info.getInterpreterPath()).isNull();
    assertThat(info.getInterpreter()).isEqualTo(dummyInterpreter);
    assertHasOrderAndContainsExactly(info.getFiles(), Order.STABLE_ORDER, dummyFile);
    assertThat(info.getPythonVersion()).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void starlarkConstructor_PlatformRuntime() throws Exception {
    ev.exec(
        "info = PyRuntimeInfo(", //
        "    interpreter_path = '/system/interpreter',",
        "    python_version = 'PY2',",
        ")");
    PyRuntimeInfo info = (PyRuntimeInfo) ev.lookup("info");
    assertThat(info.getCreationLocation().toString()).isEqualTo(":1:21");
    assertThat(info.getInterpreterPath()).isEqualTo(PathFragment.create("/system/interpreter"));
    assertThat(info.getInterpreter()).isNull();
    assertThat(info.getFiles()).isNull();
    assertThat(info.getPythonVersion()).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void starlarkConstructor_FilesDefaultsToEmpty() throws Exception {
    ev.exec(
        "info = PyRuntimeInfo(", //
        "    interpreter = dummy_interpreter,",
        "    python_version = 'PY2',",
        ")");
    PyRuntimeInfo info = (PyRuntimeInfo) ev.lookup("info");
    assertHasOrderAndContainsExactly(info.getFiles(), Order.STABLE_ORDER);
  }

  @Test
  public void starlarkConstructorErrors_InBuildXorPlatform() throws Exception {
    ev.checkEvalErrorContains(
        "exactly one of the 'interpreter' or 'interpreter_path' arguments must be specified",
        "PyRuntimeInfo(",
        "    python_version = 'PY2',",
        ")");
    ev.checkEvalErrorContains(
        "exactly one of the 'interpreter' or 'interpreter_path' arguments must be specified",
        "PyRuntimeInfo(",
        "    interpreter_path = '/system/interpreter',",
        "    interpreter = dummy_interpreter,",
        "    python_version = 'PY2',",
        ")");
  }

  @Test
  public void starlarkConstructorErrors_Files() throws Exception {
    ev.checkEvalErrorContains(
        "got value of type 'string', want 'depset or NoneType'",
        "PyRuntimeInfo(",
        "    interpreter = dummy_interpreter,",
        "    files = 'abc',",
        "    python_version = 'PY2',",
        ")");
    ev.checkEvalErrorContains(
        "got a depset of 'string', expected a depset of 'File'",
        "PyRuntimeInfo(",
        "    interpreter = dummy_interpreter,",
        "    files = depset(['abc']),",
        "    python_version = 'PY2',",
        ")");
    ev.checkEvalErrorContains(
        "cannot specify 'files' if 'interpreter_path' is given",
        "PyRuntimeInfo(",
        "    interpreter_path = '/system/interpreter',",
        "    files = depset([dummy_file]),",
        "    python_version = 'PY2',",
        ")");
  }

  @Test
  public void starlarkConstructorErrors_PythonVersion() throws Exception {
    ev.checkEvalErrorContains(
        "missing 1 required named argument: python_version",
        "PyRuntimeInfo(",
        "    interpreter_path = '/system/interpreter',",
        ")");
    ev.checkEvalErrorContains(
        "illegal value for 'python_version': 'not a Python version' is not a valid Python major "
            + "version. Expected 'PY2' or 'PY3'.",
        "PyRuntimeInfo(",
        "    interpreter_path = '/system/interpreter',",
        "    python_version = 'not a Python version',",
        ")");
  }
}
