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

package com.google.devtools.build.lib.bazel.rules.python;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Bazel-specific tests for {@code py_binary}. */
@RunWith(JUnit4.class)
public class BazelPyBinaryConfiguredTargetTest extends BuildViewTestCase {

  /**
   * Given a {@code py_binary} or {@code py_test} target, returns the path of the Python interpreter
   * used by the generated stub script.
   *
   * <p>This works by casting the stub script's generating action to a template expansion action and
   * looking for the expansion key for the Python interpreter. It's therefore linked to the
   * implementation of the rule, but that's the cost we pay for avoiding an execution-time test.
   */
  private String getInterpreterPathFromStub(ConfiguredTarget pyExecutableTarget) {
    // First find the stub script. Normally this is just the executable associated with the target.
    // But for Windows the executable is a separate launcher with an ".exe" extension, and the stub
    // script artifact has the same base name with the extension ".temp" instead. (At least, when
    // --build_python_zip is enabled, which is the default on Windows.)
    Artifact executable = pyExecutableTarget.getProvider(FilesToRunProvider.class).getExecutable();
    Artifact stub;
    if (OS.getCurrent() == OS.WINDOWS) {
      stub =
          getDerivedArtifact(
              FileSystemUtils.replaceExtension(executable.getRootRelativePath(), ".temp"),
              executable.getRoot(),
              executable.getArtifactOwner());
    } else {
      stub = executable;
    }
    assertThat(stub).isNotNull();
    // Now grab its generating action, which should be a template action, and get the key for the
    // binary path.
    Action generatingAction = getGeneratingAction(stub);
    assertThat(generatingAction).isInstanceOf(TemplateExpansionAction.class);
    TemplateExpansionAction templateAction = (TemplateExpansionAction) generatingAction;
    for (Substitution sub : templateAction.getSubstitutions()) {
      if (sub.getKey().equals("%python_binary%")) {
        return sub.getValue();
      }
    }
    throw new AssertionError(
        "Failed to find the '%python_binary%' key in the stub script's template expansion action");
  }

  @Test
  public void runtimeSetByPythonTop() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "py_runtime(",
        "    name = 'my_py_runtime',",
        "    interpreter_path = '/system/python2',",
        "    python_version = 'PY2',",
        ")",
        "py_binary(",
        "    name = 'pybin',",
        "    srcs = ['pybin.py'],",
        ")");
    String pythonTop =
        analysisMock.pySupport().createPythonTopEntryPoint(mockToolsConfig, "//pkg:my_py_runtime");
    useConfiguration("--python_top=" + pythonTop);
    String path = getInterpreterPathFromStub(getConfiguredTarget("//pkg:pybin"));
    assertThat(path).isEqualTo("/system/python2");
  }

  @Test
  public void runtimeSetByPythonPath() throws Exception {
    scratch.file(
        "pkg/BUILD", //
        "py_binary(",
        "    name = 'pybin',",
        "    srcs = ['pybin.py'],",
        ")");
    useConfiguration("--python_path=/system/python2");
    String path = getInterpreterPathFromStub(getConfiguredTarget("//pkg:pybin"));
    assertThat(path).isEqualTo("/system/python2");
  }

  @Test
  public void runtimeDefaultsToPythonSystemCommand() throws Exception {
    scratch.file(
        "pkg/BUILD", //
        "py_binary(",
        "    name = 'pybin',",
        "    srcs = ['pybin.py'],",
        ")");
    String path = getInterpreterPathFromStub(getConfiguredTarget("//pkg:pybin"));
    assertThat(path).isEqualTo("python");
  }

  @Test
  public void pythonTopTakesPrecedenceOverPythonPath() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "py_runtime(",
        "    name = 'my_py_runtime',",
        "    interpreter_path = '/system/python2',",
        "    python_version = 'PY2',",
        ")",
        "py_binary(",
        "    name = 'pybin',",
        "    srcs = ['pybin.py'],",
        ")");
    String pythonTop =
        analysisMock.pySupport().createPythonTopEntryPoint(mockToolsConfig, "//pkg:my_py_runtime");
    useConfiguration("--python_top=" + pythonTop, "--python_path=/better/not/be/this/one");
    String path = getInterpreterPathFromStub(getConfiguredTarget("//pkg:pybin"));
    assertThat(path).isEqualTo("/system/python2");
  }
}
