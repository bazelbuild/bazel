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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.testutil.MoreAsserts.ThrowingRunnable;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PyProvider}. */
@RunWith(JUnit4.class)
public class PyProviderTest extends BuildViewTestCase {

  /**
   * Constructs a py provider struct with the given field values and with default values for any
   * field not specified.
   *
   * <p>The struct is constructed directly, rather than using {@link PyProvider.Builder}, so that
   * the resulting instance is suitable for asserting on {@code PyProvider}'s operations over
   * structs with known contents. {@code overrides} is applied directly without validating the
   * fields' names or types.
   */
  private StructImpl makeStruct(Map<String, Object> overrides) {
    Map<String, Object> fields = new LinkedHashMap<>();
    fields.put(
        PyProvider.TRANSITIVE_SOURCES,
        SkylarkNestedSet.of(Artifact.class, NestedSetBuilder.emptySet(Order.COMPILE_ORDER)));
    fields.put(PyProvider.USES_SHARED_LIBRARIES, false);
    fields.put(
        PyProvider.IMPORTS,
        SkylarkNestedSet.of(String.class, NestedSetBuilder.emptySet(Order.COMPILE_ORDER)));
    fields.putAll(overrides);
    return StructProvider.STRUCT.create(fields, "No such attribute '%s'");
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

  /** We need this because {@code NestedSet}s don't have value equality. */
  private static void assertHasOrderAndContainsExactly(
      NestedSet<?> set, Order order, Object... values) {
    assertThat(set.getOrder()).isEqualTo(order);
    assertThat(set).containsExactly(values);
  }

  @Test
  public void getTransitiveSources_Good() throws Exception {
    NestedSet<Artifact> sources =
        NestedSetBuilder.create(Order.COMPILE_ORDER, getSourceArtifact("dummy"));
    StructImpl info =
        makeStruct(
            ImmutableMap.of(
                PyProvider.TRANSITIVE_SOURCES, SkylarkNestedSet.of(Artifact.class, sources)));
    assertThat(PyProvider.getTransitiveSources(info)).isSameAs(sources);
  }

  @Test
  public void getTransitiveSources_Missing() {
    StructImpl info = StructProvider.STRUCT.createEmpty(null);
    assertHasMissingFieldMessage(() -> PyProvider.getTransitiveSources(info), "transitive_sources");
  }

  @Test
  public void getTransitiveSources_WrongType() {
    StructImpl info = makeStruct(ImmutableMap.of(PyProvider.TRANSITIVE_SOURCES, 123));
    assertHasWrongTypeMessage(
        () -> PyProvider.getTransitiveSources(info), "transitive_sources", "depset of Files");
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
  public void getUsesSharedLibraries_Good() throws Exception {
    StructImpl info = makeStruct(ImmutableMap.of(PyProvider.USES_SHARED_LIBRARIES, true));
    assertThat(PyProvider.getUsesSharedLibraries(info)).isTrue();
  }

  @Test
  public void getUsesSharedLibraries_Missing() throws Exception {
    StructImpl info = StructProvider.STRUCT.createEmpty(null);
    assertThat(PyProvider.getUsesSharedLibraries(info)).isFalse();
  }

  @Test
  public void getUsesSharedLibraries_WrongType() {
    StructImpl info = makeStruct(ImmutableMap.of(PyProvider.USES_SHARED_LIBRARIES, 123));
    assertHasWrongTypeMessage(
        () -> PyProvider.getUsesSharedLibraries(info), "uses_shared_libraries", "boolean");
  }

  @Test
  public void getImports_Good() throws Exception {
    NestedSet<String> imports = NestedSetBuilder.create(Order.COMPILE_ORDER, "abc");
    StructImpl info =
        makeStruct(ImmutableMap.of(PyProvider.IMPORTS, SkylarkNestedSet.of(String.class, imports)));
    assertThat(PyProvider.getImports(info)).isSameAs(imports);
  }

  @Test
  public void getImports_Missing() throws Exception {
    StructImpl info = StructProvider.STRUCT.createEmpty(null);
    assertHasOrderAndContainsExactly(PyProvider.getImports(info), Order.COMPILE_ORDER);
  }

  @Test
  public void getImports_WrongType() {
    StructImpl info = makeStruct(ImmutableMap.of(PyProvider.IMPORTS, 123));
    assertHasWrongTypeMessage(() -> PyProvider.getImports(info), "imports", "depset of strings");
  }

  /** Checks values set by the builder. */
  @Test
  public void builder() throws Exception {
    NestedSet<Artifact> sources =
        NestedSetBuilder.create(Order.COMPILE_ORDER, getSourceArtifact("dummy"));
    NestedSet<String> imports = NestedSetBuilder.create(Order.COMPILE_ORDER, "abc");
    StructImpl info =
        PyProvider.builder()
            .setTransitiveSources(sources)
            .setUsesSharedLibraries(true)
            .setImports(imports)
            .build();
    // Assert using struct operations, not PyProvider accessors, which aren't necessarily trusted to
    // be correct.
    assertHasOrderAndContainsExactly(
        ((SkylarkNestedSet) info.getValue(PyProvider.TRANSITIVE_SOURCES)).getSet(Artifact.class),
        Order.COMPILE_ORDER,
        getSourceArtifact("dummy"));
    assertThat((Boolean) info.getValue(PyProvider.USES_SHARED_LIBRARIES)).isTrue();
    assertHasOrderAndContainsExactly(
        ((SkylarkNestedSet) info.getValue(PyProvider.IMPORTS)).getSet(String.class),
        Order.COMPILE_ORDER,
        "abc");
  }

  /** Checks the defaults set by the builder. */
  @Test
  public void builderDefaults() throws Exception {
    // transitive_sources is mandatory, so create a dummy value but no need to assert on it.
    NestedSet<Artifact> sources =
        NestedSetBuilder.create(Order.COMPILE_ORDER, getSourceArtifact("dummy"));
    StructImpl info = PyProvider.builder().setTransitiveSources(sources).build();
    // Assert using struct operations, not PyProvider accessors, which aren't necessarily trusted to
    // be correct.
    assertThat((Boolean) info.getValue(PyProvider.USES_SHARED_LIBRARIES)).isFalse();
    assertHasOrderAndContainsExactly(
        ((SkylarkNestedSet) info.getValue(PyProvider.IMPORTS)).getSet(String.class),
        Order.COMPILE_ORDER);
  }
}
