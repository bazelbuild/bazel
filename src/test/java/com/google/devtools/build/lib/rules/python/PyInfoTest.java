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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PyInfo}. */
@RunWith(JUnit4.class)
public class PyInfoTest extends SkylarkTestCase {

  private static final Location dummyLoc =
      new Location(0, 0) {
        @Override
        public PathFragment getPath() {
          return null;
        }
      };

  private Artifact dummyArtifact;

  @Before
  public void setUp() throws Exception {
    dummyArtifact = getSourceArtifact("dummy");
    update("PyInfo", PyInfo.PROVIDER);
    update("dummy_file", dummyArtifact);
  }

  /** We need this because {@code NestedSet}s don't have value equality. */
  private static void assertHasOrderAndContainsExactly(
      NestedSet<?> set, Order order, Object... values) {
    assertThat(set.getOrder()).isEqualTo(order);
    assertThat(set).containsExactly(values);
  }

  /** Checks values set by the builder. */
  @Test
  public void builderExplicit() throws Exception {
    NestedSet<Artifact> sources = NestedSetBuilder.create(Order.COMPILE_ORDER, dummyArtifact);
    NestedSet<String> imports = NestedSetBuilder.create(Order.COMPILE_ORDER, "abc");
    PyInfo info =
        PyInfo.builder()
            .setLocation(dummyLoc)
            .setTransitiveSources(sources)
            .setUsesSharedLibraries(true)
            .setImports(imports)
            .setHasPy2OnlySources(true)
            .setHasPy3OnlySources(true)
            .build();
    assertThat(info.getCreationLoc()).isEqualTo(dummyLoc);
    assertHasOrderAndContainsExactly(
        info.getTransitiveSources().getSet(Artifact.class), Order.COMPILE_ORDER, dummyArtifact);
    assertThat(info.getUsesSharedLibraries()).isTrue();
    assertHasOrderAndContainsExactly(
        info.getImports().getSet(String.class), Order.COMPILE_ORDER, "abc");
    assertThat(info.getHasPy2OnlySources()).isTrue();
    assertThat(info.getHasPy3OnlySources()).isTrue();
  }

  /** Checks the defaults set by the builder. */
  @Test
  public void builderDefaults() throws Exception {
    // transitive_sources is mandatory, so create a dummy value but no need to assert on it.
    NestedSet<Artifact> sources = NestedSetBuilder.create(Order.COMPILE_ORDER, dummyArtifact);
    PyInfo info = PyInfo.builder().setTransitiveSources(sources).build();
    assertThat(info.getCreationLoc()).isEqualTo(Location.BUILTIN);
    assertThat(info.getUsesSharedLibraries()).isFalse();
    assertHasOrderAndContainsExactly(info.getImports().getSet(String.class), Order.COMPILE_ORDER);
    assertThat(info.getHasPy2OnlySources()).isFalse();
    assertThat(info.getHasPy3OnlySources()).isFalse();
  }

  @Test
  public void starlarkConstructor() throws Exception {
    eval(
        "info = PyInfo(",
        "    transitive_sources = depset(direct=[dummy_file]),",
        "    uses_shared_libraries = True,",
        "    imports = depset(direct=['abc']),",
        "    has_py2_only_sources = True,",
        "    has_py3_only_sources = True,",
        ")");
    PyInfo info = (PyInfo) lookup("info");
    assertThat(info.getCreationLoc().getStartOffset()).isEqualTo(7);
    assertHasOrderAndContainsExactly(
        info.getTransitiveSources().getSet(Artifact.class), Order.STABLE_ORDER, dummyArtifact);
    assertThat(info.getUsesSharedLibraries()).isTrue();
    assertHasOrderAndContainsExactly(
        info.getImports().getSet(String.class), Order.STABLE_ORDER, "abc");
    assertThat(info.getHasPy2OnlySources()).isTrue();
    assertThat(info.getHasPy3OnlySources()).isTrue();
  }

  @Test
  public void starlarkConstructorDefaults() throws Exception {
    eval("info = PyInfo(transitive_sources = depset(direct=[dummy_file]))");
    PyInfo info = (PyInfo) lookup("info");
    assertThat(info.getCreationLoc().getStartOffset()).isEqualTo(7);
    assertHasOrderAndContainsExactly(
        info.getTransitiveSources().getSet(Artifact.class), Order.STABLE_ORDER, dummyArtifact);
    assertThat(info.getUsesSharedLibraries()).isFalse();
    assertHasOrderAndContainsExactly(info.getImports().getSet(String.class), Order.COMPILE_ORDER);
    assertThat(info.getHasPy2OnlySources()).isFalse();
    assertThat(info.getHasPy3OnlySources()).isFalse();
  }

  @Test
  public void starlarkConstructorErrors_TransitiveSources() throws Exception {
    checkEvalErrorContains(
        "'transitive_sources' has no default value", //
        "PyInfo()");
    checkEvalErrorContains(
        "expected value of type 'depset of Files' for parameter 'transitive_sources'",
        "PyInfo(transitive_sources = 'abc')");
    checkEvalErrorContains(
        "expected value of type 'depset of Files' for parameter 'transitive_sources'",
        "PyInfo(transitive_sources = depset(direct=['abc']))");
    checkEvalErrorContains(
        "'transitive_sources' field should be a postorder-compatible depset of Files",
        "PyInfo(transitive_sources = depset(direct=[dummy_file], order='preorder'))");
  }

  @Test
  public void starlarkConstructorErrors_UsesSharedLibraries() throws Exception {
    checkEvalErrorContains(
        "expected value of type 'bool' for parameter 'uses_shared_libraries'",
        "PyInfo(transitive_sources = depset([]), uses_shared_libraries = 'abc')");
  }

  @Test
  public void starlarkConstructorErrors_Imports() throws Exception {
    checkEvalErrorContains(
        "expected value of type 'depset of strings' for parameter 'imports'",
        "PyInfo(transitive_sources = depset([]), imports = 'abc')");
    checkEvalErrorContains(
        "expected value of type 'depset of strings' for parameter 'imports'",
        "PyInfo(transitive_sources = depset([]), imports = depset(direct=[123]))");
  }

  @Test
  public void starlarkConstructorErrors_HasPy2OnlySources() throws Exception {
    checkEvalErrorContains(
        "expected value of type 'bool' for parameter 'has_py2_only_sources'",
        "PyInfo(transitive_sources = depset([]), has_py2_only_sources = 'abc')");
  }

  @Test
  public void starlarkConstructorErrors_HasPy3OnlySources() throws Exception {
    checkEvalErrorContains(
        "expected value of type 'bool' for parameter 'has_py3_only_sources'",
        "PyInfo(transitive_sources = depset([]), has_py3_only_sources = 'abc')");
  }
}
