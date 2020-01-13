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
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.SkylarkInfo;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.MoreAsserts.ThrowingRunnable;
import com.google.devtools.build.lib.vfs.Root;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PyStructUtils}. */
@RunWith(JUnit4.class)
public class PyStructUtilsTest extends FoundationTestCase {

  private Artifact dummyArtifact;

  @Before
  public void setUp() {
    this.dummyArtifact =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(rootDirectory)), "dummy");
  }

  /**
   * Constructs a py provider struct with the given field values and with default values for any
   * field not specified.
   *
   * <p>The struct is constructed directly, rather than using {@link PyStructUtils.Builder}, so that
   * the resulting instance is suitable for asserting on {@code PyStructUtils}'s operations over
   * structs with known contents. {@code overrides} is applied directly without validating the
   * fields' names or types.
   */
  private StructImpl makeStruct(Map<String, Object> overrides) {
    Map<String, Object> fields = new LinkedHashMap<>();
    fields.put(
        PyStructUtils.TRANSITIVE_SOURCES,
        Depset.of(Artifact.TYPE, NestedSetBuilder.emptySet(Order.COMPILE_ORDER)));
    fields.put(PyStructUtils.USES_SHARED_LIBRARIES, false);
    fields.put(
        PyStructUtils.IMPORTS,
        Depset.of(SkylarkType.STRING, NestedSetBuilder.emptySet(Order.COMPILE_ORDER)));
    fields.put(PyStructUtils.HAS_PY2_ONLY_SOURCES, false);
    fields.put(PyStructUtils.HAS_PY3_ONLY_SOURCES, false);
    fields.putAll(overrides);
    return StructProvider.STRUCT.create(fields, "No such attribute '%s'");
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
    NestedSet<Artifact> sources = NestedSetBuilder.create(Order.COMPILE_ORDER, dummyArtifact);
    StructImpl info =
        makeStruct(
            ImmutableMap.of(PyStructUtils.TRANSITIVE_SOURCES, Depset.of(Artifact.TYPE, sources)));
    assertThat(PyStructUtils.getTransitiveSources(info)).isSameInstanceAs(sources);
  }

  private static final StructImpl emptyInfo =
      SkylarkInfo.create(StructProvider.STRUCT, ImmutableMap.of(), /* loc= */ null);

  @Test
  public void getTransitiveSources_Missing() {
    assertHasMissingFieldMessage(
        () -> PyStructUtils.getTransitiveSources(emptyInfo), "transitive_sources");
  }

  @Test
  public void getTransitiveSources_WrongType() {
    StructImpl info = makeStruct(ImmutableMap.of(PyStructUtils.TRANSITIVE_SOURCES, 123));
    assertHasWrongTypeMessage(
        () -> PyStructUtils.getTransitiveSources(info), "transitive_sources", "depset of Files");
  }

  @Test
  public void getTransitiveSources_OrderMismatch() throws Exception {
    NestedSet<Artifact> sources = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    StructImpl info =
        makeStruct(
            ImmutableMap.of(PyStructUtils.TRANSITIVE_SOURCES, Depset.of(Artifact.TYPE, sources)));
    assertThrowsEvalExceptionContaining(
        () -> PyStructUtils.getTransitiveSources(info),
        "Incompatible depset order for 'transitive_sources': expected 'default' or 'postorder', "
            + "but got 'preorder'");
  }

  @Test
  public void getUsesSharedLibraries_Good() throws Exception {
    StructImpl info = makeStruct(ImmutableMap.of(PyStructUtils.USES_SHARED_LIBRARIES, true));
    assertThat(PyStructUtils.getUsesSharedLibraries(info)).isTrue();
  }

  @Test
  public void getUsesSharedLibraries_Missing() throws Exception {
    assertThat(PyStructUtils.getUsesSharedLibraries(emptyInfo)).isFalse();
  }

  @Test
  public void getUsesSharedLibraries_WrongType() {
    StructImpl info = makeStruct(ImmutableMap.of(PyStructUtils.USES_SHARED_LIBRARIES, 123));
    assertHasWrongTypeMessage(
        () -> PyStructUtils.getUsesSharedLibraries(info), "uses_shared_libraries", "boolean");
  }

  @Test
  public void getImports_Good() throws Exception {
    NestedSet<String> imports = NestedSetBuilder.create(Order.COMPILE_ORDER, "abc");
    StructImpl info =
        makeStruct(ImmutableMap.of(PyStructUtils.IMPORTS, Depset.of(SkylarkType.STRING, imports)));
    assertThat(PyStructUtils.getImports(info)).isSameInstanceAs(imports);
  }

  @Test
  public void getImports_Missing() throws Exception {
    assertHasOrderAndContainsExactly(PyStructUtils.getImports(emptyInfo), Order.COMPILE_ORDER);
  }

  @Test
  public void getImports_WrongType() {
    StructImpl info = makeStruct(ImmutableMap.of(PyStructUtils.IMPORTS, 123));
    assertHasWrongTypeMessage(() -> PyStructUtils.getImports(info), "imports", "depset of strings");
  }

  @Test
  public void getHasPy2OnlySources_Good() throws Exception {
    StructImpl info = makeStruct(ImmutableMap.of(PyStructUtils.HAS_PY2_ONLY_SOURCES, true));
    assertThat(PyStructUtils.getHasPy2OnlySources(info)).isTrue();
  }

  @Test
  public void getHasPy2OnlySources_Missing() throws Exception {
    assertThat(PyStructUtils.getHasPy2OnlySources(emptyInfo)).isFalse();
  }

  @Test
  public void getHasPy2OnlySources_WrongType() {
    StructImpl info = makeStruct(ImmutableMap.of(PyStructUtils.HAS_PY2_ONLY_SOURCES, 123));
    assertHasWrongTypeMessage(
        () -> PyStructUtils.getHasPy2OnlySources(info), "has_py2_only_sources", "boolean");
  }

  @Test
  public void getHasPy3OnlySources_Good() throws Exception {
    StructImpl info = makeStruct(ImmutableMap.of(PyStructUtils.HAS_PY3_ONLY_SOURCES, true));
    assertThat(PyStructUtils.getHasPy3OnlySources(info)).isTrue();
  }

  @Test
  public void getHasPy3OnlySources_Missing() throws Exception {
    assertThat(PyStructUtils.getHasPy3OnlySources(emptyInfo)).isFalse();
  }

  @Test
  public void getHasPy3OnlySources_WrongType() {
    StructImpl info = makeStruct(ImmutableMap.of(PyStructUtils.HAS_PY3_ONLY_SOURCES, 123));
    assertHasWrongTypeMessage(
        () -> PyStructUtils.getHasPy3OnlySources(info), "has_py3_only_sources", "boolean");
  }

  /** Checks values set by the builder. */
  @Test
  public void builder() throws Exception {
    NestedSet<Artifact> sources = NestedSetBuilder.create(Order.COMPILE_ORDER, dummyArtifact);
    NestedSet<String> imports = NestedSetBuilder.create(Order.COMPILE_ORDER, "abc");
    StructImpl info =
        PyStructUtils.builder()
            .setTransitiveSources(sources)
            .setUsesSharedLibraries(true)
            .setImports(imports)
            .setHasPy2OnlySources(true)
            .setHasPy3OnlySources(true)
            .build();
    // Assert using struct operations, not PyStructUtils accessors, which aren't necessarily trusted
    // to be correct.
    assertHasOrderAndContainsExactly(
        ((Depset) info.getValue(PyStructUtils.TRANSITIVE_SOURCES)).getSet(Artifact.class),
        Order.COMPILE_ORDER,
        dummyArtifact);
    assertThat((Boolean) info.getValue(PyStructUtils.USES_SHARED_LIBRARIES)).isTrue();
    assertHasOrderAndContainsExactly(
        ((Depset) info.getValue(PyStructUtils.IMPORTS)).getSet(String.class),
        Order.COMPILE_ORDER,
        "abc");
    assertThat((Boolean) info.getValue(PyStructUtils.HAS_PY2_ONLY_SOURCES)).isTrue();
    assertThat((Boolean) info.getValue(PyStructUtils.HAS_PY3_ONLY_SOURCES)).isTrue();
  }

  /** Checks the defaults set by the builder. */
  @Test
  public void builderDefaults() throws Exception {
    // transitive_sources is mandatory, so create a dummy value but no need to assert on it.
    NestedSet<Artifact> sources = NestedSetBuilder.create(Order.COMPILE_ORDER, dummyArtifact);
    StructImpl info = PyStructUtils.builder().setTransitiveSources(sources).build();
    // Assert using struct operations, not PyStructUtils accessors, which aren't necessarily trusted
    // to be correct.
    assertThat((Boolean) info.getValue(PyStructUtils.USES_SHARED_LIBRARIES)).isFalse();
    assertHasOrderAndContainsExactly(
        ((Depset) info.getValue(PyStructUtils.IMPORTS)).getSet(String.class), Order.COMPILE_ORDER);
    assertThat((Boolean) info.getValue(PyStructUtils.HAS_PY2_ONLY_SOURCES)).isFalse();
    assertThat((Boolean) info.getValue(PyStructUtils.HAS_PY3_ONLY_SOURCES)).isFalse();
  }
}
