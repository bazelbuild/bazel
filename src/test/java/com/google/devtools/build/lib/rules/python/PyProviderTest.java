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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.testutil.MoreAsserts.ThrowingRunnable;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PyProvider}. */
@RunWith(JUnit4.class)
public class PyProviderTest extends BuildViewTestCase {

  private static final StructImpl EMPTY_STRUCT =
      StructProvider.STRUCT.create(ImmutableMap.of(), "No such attribute '%s'");

  private static final StructImpl WRONG_TYPES_STRUCT =
      StructProvider.STRUCT.create(
          ImmutableMap.<String, Object>of(
              PyProvider.TRANSITIVE_SOURCES,
              123,
              PyProvider.USES_SHARED_LIBRARIES,
              123,
              PyProvider.IMPORTS,
              123),
          "No such attribute '%s'");

  private StructImpl getGoodStruct() {
    return StructProvider.STRUCT.create(
        ImmutableMap.<String, Object>of(
            PyProvider.TRANSITIVE_SOURCES,
            SkylarkNestedSet.of(
                Artifact.class,
                NestedSetBuilder.create(Order.STABLE_ORDER, getSourceArtifact("dummy_artifact"))),
            PyProvider.USES_SHARED_LIBRARIES,
            true,
            PyProvider.IMPORTS,
            SkylarkNestedSet.of(String.class, NestedSetBuilder.create(Order.STABLE_ORDER, "abc"))),
        "No such attribute '%s'");
  }

  /** Defines //pytarget, a target that returns a py provider with some arbitrary field values. */
  private void definePyTarget() throws IOException {
    scratch.file("rules/BUILD");
    scratch.file(
        "rules/pyrule.bzl",
        "def _pyrule_impl(ctx):",
        "    info = struct(",
        "        transitive_sources = depset(direct=ctx.files.transitive_sources),",
        "        uses_shared_libraries = ctx.attr.uses_shared_libraries,",
        "        imports = depset(direct=ctx.attr.imports),",
        "    )",
        "    return struct(py=info)",
        "",
        "pyrule = rule(",
        "    implementation = _pyrule_impl,",
        "    attrs = {",
        "        'transitive_sources': attr.label_list(allow_files=True),",
        "        'uses_shared_libraries': attr.bool(default=False),",
        "        'imports': attr.string_list(),",
        "    },",
        ")");
    scratch.file(
        "pytarget/BUILD",
        "load('//rules:pyrule.bzl', 'pyrule')",
        "",
        "pyrule(",
        "    name = 'pytarget',",
        "    transitive_sources = ['a.py'],",
        "    uses_shared_libraries = True,",
        "    imports = ['b']",
        ")");
    scratch.file("pytarget/a.py");
  }

  /** Defines //dummytarget, a target that returns no py provider. */
  private void defineDummyTarget() throws IOException {
    scratch.file("rules/BUILD");
    scratch.file(
        "rules/dummytarget.bzl",
        "def _dummyrule_impl(ctx):",
        "    pass",
        "",
        "dummyrule = rule(",
        "    implementation = _dummyrule_impl,",
        ")");
    scratch.file(
        "dummytarget/BUILD",
        "load('//rules:dummytarget.bzl', 'dummyrule')",
        "",
        "dummyrule(",
        "    name = 'dummytarget',",
        ")");
  }

  @Test
  public void hasProvider_True() throws Exception {
    definePyTarget();
    assertThat(PyProvider.hasProvider(getConfiguredTarget("//pytarget"))).isTrue();
  }

  @Test
  public void hasProvider_False() throws Exception {
    defineDummyTarget();
    assertThat(PyProvider.hasProvider(getConfiguredTarget("//dummytarget"))).isFalse();
  }

  @Test
  public void getProvider_Present() throws Exception {
    definePyTarget();
    StructImpl info = PyProvider.getProvider(getConfiguredTarget("//pytarget"));
    // If we got this far, it's present. getProvider() should never be null, but check just in case.
    assertThat(info).isNotNull();
  }

  @Test
  public void getProvider_Absent() throws Exception {
    defineDummyTarget();
    EvalException ex =
        assertThrows(
            EvalException.class,
            () -> PyProvider.getProvider(getConfiguredTarget("//dummytarget")));
    assertThat(ex).hasMessageThat().contains("Target does not have 'py' provider");
  }

  @Test
  public void getProvider_WrongType() throws Exception {
    // badtyperule() returns a "py" provider that has the wrong type.
    scratch.file("rules/BUILD");
    scratch.file(
        "rules/badtyperule.bzl",
        "def _badtyperule_impl(ctx):",
        "    return struct(py='abc')",
        "",
        "badtyperule = rule(",
        "    implementation = _badtyperule_impl",
        ")");
    scratch.file(
        "badtypetarget/BUILD",
        "load('//rules:badtyperule.bzl', 'badtyperule')",
        "",
        "badtyperule(",
        "    name = 'badtypetarget',",
        ")");
    EvalException ex =
        assertThrows(
            EvalException.class,
            () -> PyProvider.getProvider(getConfiguredTarget("//badtypetarget")));
    assertThat(ex).hasMessageThat().contains("'py' provider should be a struct");
  }

  private static void assertThrowsEvalExceptionContaining(
      ThrowingRunnable runnable, String message) {
    assertThat(assertThrows(EvalException.class, runnable)).hasMessageThat().contains(message);
  }

  private static void assertHasMissingFieldMessage(ThrowingRunnable access, String fieldName) {
    assertThrowsEvalExceptionContaining(
        access, String.format("\'py' provider missing '%s' field", fieldName));
  }

  private static void assertHasWrongTypeMessage(
      ThrowingRunnable access, String fieldName, String expectedType) {
    assertThrowsEvalExceptionContaining(
        access,
        String.format(
            "\'py' provider's '%s' field should be a %s (got a 'int')", fieldName, expectedType));
  }

  @Test
  public void getTransitiveSources() throws Exception {
    assertHasMissingFieldMessage(
        () -> PyProvider.getTransitiveSources(EMPTY_STRUCT), "transitive_sources");
    assertHasWrongTypeMessage(
        () -> PyProvider.getTransitiveSources(WRONG_TYPES_STRUCT),
        "transitive_sources",
        "depset of Files");
    assertThat(PyProvider.getTransitiveSources(getGoodStruct()))
        .containsExactly(getSourceArtifact("dummy_artifact"));
  }

  @Test
  public void getTransitiveSources_OrderMismatch() throws Exception {
    reporter.removeHandler(failFastHandler);
    // Depset order mismatches should be caught as rule errors.
    scratch.file("rules/BUILD");
    scratch.file(
        "rules/badorderrule.bzl",
        "def _badorderrule_impl(ctx):",
        // Native rules use "compile" / "postorder", so using "preorder" here creates a conflict.
        "    info = struct(transitive_sources=depset(direct=[], order='preorder'))",
        "    return struct(py=info)",
        "",
        "badorderrule = rule(",
        "    implementation = _badorderrule_impl",
        ")");
    scratch.file(
        "badordertarget/BUILD",
        "load('//rules:badorderrule.bzl', 'badorderrule')",
        "",
        "badorderrule(",
        "    name = 'badorderdep',",
        ")",
        "py_library(",
        "    name = 'pylib',",
        "    srcs = ['pylib.py'],",
        ")",
        "py_binary(",
        "    name = 'pybin',",
        "    srcs = ['pybin.py'],",
        "    deps = [':pylib', ':badorderdep'],",
        ")");
    getConfiguredTarget("//badordertarget:pybin");
    assertContainsEvent(
        "Incompatible order for transitive_sources: expected 'default' or 'postorder', got "
            + "'preorder'");
  }

  @Test
  public void getUsesSharedLibraries() throws Exception {
    assertHasMissingFieldMessage(
        () -> PyProvider.getUsesSharedLibraries(EMPTY_STRUCT), "uses_shared_libraries");
    assertHasWrongTypeMessage(
        () -> PyProvider.getUsesSharedLibraries(WRONG_TYPES_STRUCT),
        "uses_shared_libraries",
        "boolean");
    assertThat(PyProvider.getUsesSharedLibraries(getGoodStruct())).isTrue();
  }

  @Test
  public void getImports() throws Exception {
    assertHasMissingFieldMessage(() -> PyProvider.getImports(EMPTY_STRUCT), "imports");
    assertHasWrongTypeMessage(
        () -> PyProvider.getImports(WRONG_TYPES_STRUCT), "imports", "depset of strings");
    assertThat(PyProvider.getImports(getGoodStruct())).containsExactly("abc");
  }
}
