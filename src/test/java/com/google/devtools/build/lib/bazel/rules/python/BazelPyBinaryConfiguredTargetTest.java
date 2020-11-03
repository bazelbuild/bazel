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
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Bazel-specific tests for {@code py_binary}. */
@RunWith(JUnit4.class)
public class BazelPyBinaryConfiguredTargetTest extends BuildViewTestCase {

  private static final String TOOLCHAIN_BZL =
      TestConstants.TOOLS_REPOSITORY + "//tools/python:toolchain.bzl";

  private static final String TOOLCHAIN_TYPE =
      TestConstants.TOOLS_REPOSITORY + "//tools/python:toolchain_type";

  private static String join(String... lines) {
    return String.join("\n", lines);
  }

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

  // TODO(#8169): Delete tests of the legacy --python_top / --python_path behavior.

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
    useConfiguration("--incompatible_use_python_toolchains=false", "--python_top=" + pythonTop);
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
    useConfiguration("--incompatible_use_python_toolchains=false", "--python_path=/system/python2");
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
    useConfiguration("--incompatible_use_python_toolchains=false");
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
    useConfiguration(
        "--incompatible_use_python_toolchains=false",
        "--python_top=" + pythonTop,
        "--python_path=/better/not/be/this/one");
    String path = getInterpreterPathFromStub(getConfiguredTarget("//pkg:pybin"));
    assertThat(path).isEqualTo("/system/python2");
  }

  // TODO(brandjon): Move generic toolchain tests that don't access legacy behavior to
  // PyExecutableConfiguredtargetTestBase. Asserting on the chosen PyRuntimeInfo is problematic to
  // do at analysis time though. It's easier in this test because we know the PythonSemantics is
  // BazelPythonSemantics.

  /** Adds toolchain definitions to a //toolchains package, for user by the below tests. */
  private void defineToolchains() throws Exception {
    scratch.file(
        "toolchains/BUILD",
        "load('" + TOOLCHAIN_BZL + "', 'py_runtime_pair')",
        "py_runtime(",
        "    name = 'py2_runtime',",
        "    interpreter_path = '/system/python2',",
        "    python_version = 'PY2',",
        ")",
        "py_runtime(",
        "    name = 'py3_runtime',",
        "    interpreter_path = '/system/python3',",
        "    python_version = 'PY3',",
        ")",
        "py_runtime_pair(",
        "    name = 'py_runtime_pair',",
        "    py2_runtime = ':py2_runtime',",
        "    py3_runtime = ':py3_runtime',",
        ")",
        "toolchain(",
        "    name = 'py_toolchain',",
        "    toolchain = ':py_runtime_pair',",
        "    toolchain_type = '" + TOOLCHAIN_TYPE + "',",
        ")",
        "py_runtime_pair(",
        "    name = 'py_runtime_pair_for_py2_only',",
        "    py2_runtime = ':py2_runtime',",
        ")",
        "toolchain(",
        "    name = 'py_toolchain_for_py2_only',",
        "    toolchain = ':py_runtime_pair_for_py2_only',",
        "    toolchain_type = '" + TOOLCHAIN_TYPE + "',",
        ")");
  }

  @Test
  public void runtimeObtainedFromToolchain() throws Exception {
    defineToolchains();
    scratch.file(
        "pkg/BUILD",
        "py_binary(",
        "    name = 'py2_bin',",
        "    srcs = ['py2_bin.py'],",
        "    python_version = 'PY2',",
        ")",
        "py_binary(",
        "    name = 'py3_bin',",
        "    srcs = ['py3_bin.py'],",
        "    python_version = 'PY3',",
        ")");
    useConfiguration(
        "--incompatible_use_python_toolchains=true",
        "--extra_toolchains=//toolchains:py_toolchain");

    String py2Path = getInterpreterPathFromStub(getConfiguredTarget("//pkg:py2_bin"));
    String py3Path = getInterpreterPathFromStub(getConfiguredTarget("//pkg:py3_bin"));
    assertThat(py2Path).isEqualTo("/system/python2");
    assertThat(py3Path).isEqualTo("/system/python3");
  }

  @Test
  public void toolchainCanOmitUnusedRuntimeVersion() throws Exception {
    defineToolchains();
    scratch.file(
        "pkg/BUILD",
        "py_binary(",
        "    name = 'py2_bin',",
        "    srcs = ['py2_bin.py'],",
        "    python_version = 'PY2',",
        ")");
    useConfiguration(
        "--incompatible_use_python_toolchains=true",
        "--extra_toolchains=//toolchains:py_toolchain_for_py2_only");

    String path = getInterpreterPathFromStub(getConfiguredTarget("//pkg:py2_bin"));
    assertThat(path).isEqualTo("/system/python2");
  }

  @Test
  public void toolchainTakesPrecedenceOverLegacyFlags() throws Exception {
    defineToolchains();
    scratch.file(
        "pkg/BUILD",
        "py_binary(",
        "    name = 'py2_bin',",
        "    srcs = ['py2_bin.py'],",
        "    python_version = 'PY2',",
        ")");
    useConfiguration(
        "--incompatible_use_python_toolchains=true",
        "--extra_toolchains=//toolchains:py_toolchain",
        "--python_path=/better/not/be/this/one");

    String path = getInterpreterPathFromStub(getConfiguredTarget("//pkg:py2_bin"));
    assertThat(path).isEqualTo("/system/python2");
  }

  @Test
  public void toolchainIsMissingNeededRuntime() throws Exception {
    defineToolchains();
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "pkg/BUILD",
        "py_binary(",
        "    name = 'py3_bin',",
        "    srcs = ['py3_bin.py'],",
        "    python_version = 'PY3',",
        ")");
    useConfiguration(
        "--incompatible_use_python_toolchains=true",
        "--extra_toolchains=//toolchains:py_toolchain_for_py2_only");

    getConfiguredTarget("//pkg:py3_bin");
    assertContainsEvent("The Python toolchain does not provide a runtime for Python version PY3");
  }

  /**
   * Creates a custom toolchain at //toolchains:custom that has the given lines in its rule
   * implementation function.
   */
  private void defineCustomToolchain(String... lines) throws Exception {
    String indentedBody;
    if (lines.length == 0) {
      indentedBody = "    pass";
    } else {
      indentedBody = "    " + join(lines).replace("\n", "\n    ");
    }
    scratch.file(
        "toolchains/rules.bzl",
        "def _custom_impl(ctx):",
        indentedBody,
        "custom = rule(",
        "    implementation = _custom_impl",
        ")");
    scratch.file(
        "toolchains/BUILD",
        "load(':rules.bzl', 'custom')",
        "custom(",
        "    name = 'custom',",
        ")",
        "toolchain(",
        "    name = 'custom_toolchain',",
        "    toolchain = ':custom',",
        "    toolchain_type = '" + TOOLCHAIN_TYPE + "',",
        ")");
  }

  /**
   * Defines a PY2 py_binary target at //pkg:pybin, configures it to use the custom toolchain
   * //toolchains:custom, and attempts to retrieve it with {@link #getConfiguredTarget}.
   */
  private void analyzePy2BinaryTargetUsingCustomToolchain() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "py_binary(",
        "    name = 'pybin',",
        "    srcs = ['pybin.py'],",
        "    python_version = 'PY2',",
        ")");
    useConfiguration(
        "--incompatible_use_python_toolchains=true",
        "--extra_toolchains=//toolchains:custom_toolchain");
    getConfiguredTarget("//pkg:pybin");
  }

  @Test
  public void toolchainInfoFieldIsMissing() throws Exception {
    reporter.removeHandler(failFastHandler);
    defineCustomToolchain(
        "return platform_common.ToolchainInfo(",
        "    py2_runtime = PyRuntimeInfo(",
        "        interpreter_path = '/system/python2',",
        "        python_version = 'PY2')",
        ")");
    // Use PY2 binary to test that we still validate the PY3 field even when it's not needed.
    analyzePy2BinaryTargetUsingCustomToolchain();
    assertContainsEvent(
        "Error parsing the Python toolchain's ToolchainInfo: field 'py3_runtime' is missing");
  }

  @Test
  public void toolchainInfoFieldHasBadType() throws Exception {
    reporter.removeHandler(failFastHandler);
    defineCustomToolchain(
        "return platform_common.ToolchainInfo(",
        "    py2_runtime = PyRuntimeInfo(",
        "        interpreter_path = '/system/python2',",
        "        python_version = 'PY2'),",
        "    py3_runtime = 'abc',",
        ")");
    // Use PY2 binary to test that we still validate the PY3 field even when it's not needed.
    analyzePy2BinaryTargetUsingCustomToolchain();
    assertContainsEvent(
        "Error parsing the Python toolchain's ToolchainInfo: Expected a PyRuntimeInfo in field "
            + "'py3_runtime', but got 'string'");
  }

  @Test
  public void toolchainInfoFieldHasBadVersion() throws Exception {
    reporter.removeHandler(failFastHandler);
    defineCustomToolchain(
        "return platform_common.ToolchainInfo(",
        "    py2_runtime = PyRuntimeInfo(",
        "        interpreter_path = '/system/python2',",
        "        python_version = 'PY2'),",
        "    py3_runtime = PyRuntimeInfo(",
        "        interpreter_path = '/system/python3',",
        // python_version is erroneously set to PY2 for the PY3 field.
        "        python_version = 'PY2'),",
        ")");
    // Use PY2 binary to test that we still validate the PY3 field even when it's not needed.
    analyzePy2BinaryTargetUsingCustomToolchain();
    assertContainsEvent(
        "Error retrieving the Python runtime from the toolchain: Expected field 'py3_runtime' to "
            + "have a runtime with python_version = 'PY3', but got python_version = 'PY2'");
  }

  @Test
  public void explicitInitPy_CanBeGloballyEnabled() throws Exception {
    scratch.file(
        "pkg/BUILD",
        join(
            "py_binary(", //
            "    name = 'foo',",
            "    srcs = ['foo.py'],",
            ")"));
    useConfiguration("--incompatible_default_to_explicit_init_py=true");
    assertThat(getDefaultRunfiles(getConfiguredTarget("//pkg:foo")).getEmptyFilenames().toList())
        .isEmpty();
  }

  @Test
  public void explicitInitPy_CanBeSelectivelyDisabled() throws Exception {
    scratch.file(
        "pkg/BUILD",
        join(
            "py_binary(", //
            "    name = 'foo',",
            "    srcs = ['foo.py'],",
            "    legacy_create_init = True,",
            ")"));
    useConfiguration("--incompatible_default_to_explicit_init_py=true");
    assertThat(getDefaultRunfiles(getConfiguredTarget("//pkg:foo")).getEmptyFilenames().toList())
        .containsExactly("pkg/__init__.py");
  }

  @Test
  public void explicitInitPy_CanBeGloballyDisabled() throws Exception {
    scratch.file(
        "pkg/BUILD",
        join(
            "py_binary(", //
            "    name = 'foo',",
            "    srcs = ['foo.py'],",
            ")"));
    useConfiguration("--incompatible_default_to_explicit_init_py=false");
    assertThat(getDefaultRunfiles(getConfiguredTarget("//pkg:foo")).getEmptyFilenames().toList())
        .containsExactly("pkg/__init__.py");
  }

  @Test
  public void explicitInitPy_CanBeSelectivelyEnabled() throws Exception {
    scratch.file(
        "pkg/BUILD",
        join(
            "py_binary(", //
            "    name = 'foo',",
            "    srcs = ['foo.py'],",
            "    legacy_create_init = False,",
            ")"));
    useConfiguration("--incompatible_default_to_explicit_init_py=false");
    assertThat(getDefaultRunfiles(getConfiguredTarget("//pkg:foo")).getEmptyFilenames().toList())
        .isEmpty();
  }

  @Test
  public void packageNameCannotHaveHyphen() throws Exception {
    checkError(
        "pkg-hyphenated",
        "foo",
        // error:
        "paths to Python packages may not contain '-'",
        // build file:
        "py_binary(",
        "    name = 'foo',",
        "    srcs = ['foo.py'],",
        ")");
  }

  @Test
  public void srcsPackageNameCannotHaveHyphen() throws Exception {
    scratch.file(
        "pkg-hyphenated/BUILD", //
        "exports_files(['bar.py'])");
    checkError(
        "otherpkg",
        "foo",
        // error:
        "paths to Python packages may not contain '-'",
        // build file:
        "py_binary(",
        "    name = 'foo',",
        "    srcs = ['foo.py', '//pkg-hyphenated:bar.py'],",
        ")");
  }
}
