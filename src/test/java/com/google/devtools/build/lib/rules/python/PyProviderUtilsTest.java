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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.testutil.MoreAsserts.ThrowingRunnable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PyProviderUtils}. */
@RunWith(JUnit4.class)
public class PyProviderUtilsTest extends BuildViewTestCase {

  private static void assertThrowsEvalExceptionContaining(
      ThrowingRunnable runnable, String message) {
    assertThat(assertThrows(EvalException.class, runnable)).hasMessageThat().contains(message);
  }

  /**
   * Creates {@code //pkg:target} as an instance of a trivial rule having the given implementation
   * function body.
   */
  private void declareTargetWithImplementation(String... lines) throws Exception {
    String body;
    if (lines.length == 0) {
      body = "    pass";
    } else {
      body = "    " + String.join("\n    ", lines);
    }
    scratch.file(
        "pkg/rules.bzl",
        "def _myrule_impl(ctx):",
        body,
        "myrule = rule(",
        "    implementation = _myrule_impl,",
        ")");
    scratch.file(
        "pkg/BUILD", //
        "load(':rules.bzl', 'myrule')", //
        "myrule(", //
        "    name = 'target',", //
        ")");
  }

  /** Retrieves the target defined by {@link #declareTargetWithImplementation}. */
  private ConfiguredTarget getTarget() throws Exception {
    ConfiguredTarget target = getConfiguredTarget("//pkg:target");
    assertThat(target).isNotNull();
    return target;
  }

  @Test
  public void getAndHasProvider_Present() throws Exception {
    declareTargetWithImplementation( //
        "return struct(py=struct())"); // Accessor doesn't actually check fields
    assertThat(PyProviderUtils.hasProvider(getTarget())).isTrue();
    assertThat(PyProviderUtils.getProvider(getTarget())).isNotNull();
  }

  @Test
  public void getAndHasProvider_Absent() throws Exception {
    declareTargetWithImplementation( //
        "return []");
    assertThat(PyProviderUtils.hasProvider(getTarget())).isFalse();
    assertThrowsEvalExceptionContaining(
        () -> PyProviderUtils.getProvider(getTarget()), "Target does not have 'py' provider");
  }

  @Test
  public void getProvider_NotAStruct() throws Exception {
    declareTargetWithImplementation( //
        "return struct(py=123)");
    assertThrowsEvalExceptionContaining(
        () -> PyProviderUtils.getProvider(getTarget()), "'py' provider should be a struct");
  }

  @Test
  public void getTransitiveSources_Provider() throws Exception {
    declareTargetWithImplementation(
        "afile = ctx.actions.declare_file('a.py')",
        "ctx.actions.write(output=afile, content='a')",
        "info = struct(transitive_sources = depset(direct=[afile]))",
        "return struct(py=info)");
    assertThat(PyProviderUtils.getTransitiveSources(getTarget()))
        .containsExactly(getBinArtifact("a.py", getTarget()));
  }

  @Test
  public void getTransitiveSources_NoProvider() throws Exception {
    declareTargetWithImplementation(
        "afile = ctx.actions.declare_file('a.py')",
        "ctx.actions.write(output=afile, content='a')",
        "return [DefaultInfo(files=depset(direct=[afile]))]");
    assertThat(PyProviderUtils.getTransitiveSources(getTarget()))
        .containsExactly(getBinArtifact("a.py", getTarget()));
  }

  @Test
  public void getUsesSharedLibraries_Provider() throws Exception {
    declareTargetWithImplementation(
        "info = struct(transitive_sources = depset([]), uses_shared_libraries=True)",
        "return struct(py=info)");
    assertThat(PyProviderUtils.getUsesSharedLibraries(getTarget())).isTrue();
  }

  @Test
  public void getUsesSharedLibraries_NoProvider() throws Exception {
    declareTargetWithImplementation(
        "afile = ctx.actions.declare_file('a.so')",
        "ctx.actions.write(output=afile, content='a')",
        "return [DefaultInfo(files=depset(direct=[afile]))]");
    assertThat(PyProviderUtils.getUsesSharedLibraries(getTarget())).isTrue();
  }

  // TODO(#7054): Add tests for getImports once we actually read the Python provider instead of the
  // specialized PythonImportsProvider.

  @Test
  public void getHasPy2OnlySources_Provider() throws Exception {
    declareTargetWithImplementation(
        "info = struct(transitive_sources = depset([]), has_py2_only_sources=True)",
        "return struct(py=info)");
    assertThat(PyProviderUtils.getHasPy2OnlySources(getTarget())).isTrue();
  }

  @Test
  public void getHasPy2OnlySources_NoProvider() throws Exception {
    declareTargetWithImplementation( //
        "return []");
    assertThat(PyProviderUtils.getHasPy2OnlySources(getTarget())).isFalse();
  }

  @Test
  public void getHasPy3OnlySources_Provider() throws Exception {
    declareTargetWithImplementation(
        "info = struct(transitive_sources = depset([]), has_py3_only_sources=True)",
        "return struct(py=info)");
    assertThat(PyProviderUtils.getHasPy3OnlySources(getTarget())).isTrue();
  }

  @Test
  public void getHasPy3OnlySources_NoProvider() throws Exception {
    declareTargetWithImplementation( //
        "return []");
    assertThat(PyProviderUtils.getHasPy3OnlySources(getTarget())).isFalse();
  }
}
