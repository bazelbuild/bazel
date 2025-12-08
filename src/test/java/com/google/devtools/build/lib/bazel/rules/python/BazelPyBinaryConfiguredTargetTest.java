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
import static com.google.devtools.build.lib.rules.python.PythonTestUtils.getPyLoad;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Bazel-specific tests for {@code py_binary}. */
@RunWith(JUnit4.class)
public class BazelPyBinaryConfiguredTargetTest extends BuildViewTestCase {

  private static final String TOOLCHAIN_TYPE =
      TestConstants.TOOLS_REPOSITORY + "//tools/python:toolchain_type";

  private static String join(String... lines) {
    return String.join("\n", lines);
  }

  // TODO(brandjon): Move generic toolchain tests that don't access legacy behavior to
  // PyExecutableConfiguredtargetTestBase. Asserting on the chosen PyRuntimeInfo is problematic to
  // do at analysis time though. It's easier in this test because we know the PythonSemantics is
  // BazelPythonSemantics.

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
        getPyLoad("PyRuntimeInfo"),
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
   * Defines a py_binary target at //pkg:pybin, configures it to use the custom toolchain
   * //toolchains:custom, and attempts to retrieve it with {@link #getConfiguredTarget}.
   */
  private void analyzePyBinaryTargetUsingCustomToolchain() throws Exception {
    scratch.file(
        "pkg/BUILD",
        getPyLoad("py_binary"),
        """
        py_binary(
            name = "pybin",
            srcs = ["pybin.py"],
            python_version = "PY3",
        )
        """);
    useConfiguration("--extra_toolchains=//toolchains:custom_toolchain");
    getConfiguredTarget("//pkg:pybin");
  }

  @Test
  public void toolchainInfoFieldIsMissing() throws Exception {
    reporter.removeHandler(failFastHandler);
    defineCustomToolchain("return platform_common.ToolchainInfo()", "");
    analyzePyBinaryTargetUsingCustomToolchain();
    assertContainsEvent(Pattern.compile("py3_runtime.*missing"));
  }

  @Test
  public void toolchainInfoFieldHasBadVersion() throws Exception {
    reporter.removeHandler(failFastHandler);
    defineCustomToolchain(
        "return platform_common.ToolchainInfo(",
        "    py3_runtime = PyRuntimeInfo(",
        "        interpreter_path = '/system/python3',",
        // python_version is erroneously set to PY2 for the PY3 field.
        "        python_version = 'PY2'),",
        ")");
    analyzePyBinaryTargetUsingCustomToolchain();
    assertContainsEvent(Pattern.compile("py3_runtime.*python_version.*got.*PY2"));
  }

  @Test
  public void packageNameCanHaveHyphen() throws Exception {
    scratch.file(
        "pkg-hyphenated/BUILD",
        getPyLoad("py_binary"),
        """
        py_binary(
            name = "foo",
            srcs = ["foo.py"],
        )
        """);
    assertThat(getConfiguredTarget("//pkg-hyphenated:foo")).isNotNull();
    assertNoEvents();
  }

  @Test
  public void srcsPackageNameCanHaveHyphen() throws Exception {
    scratch.file(
        "pkg-hyphenated/BUILD", //
        "exports_files(['bar.py'])");
    scratch.file(
        "otherpkg/BUILD",
        getPyLoad("py_binary"),
        """
        py_binary(
            name = "foo",
            srcs = [
                "foo.py",
                "//pkg-hyphenated:bar.py",
            ],
        )
        """);
    assertThat(getConfiguredTarget("//otherpkg:foo")).isNotNull();
    assertNoEvents();
  }
}
