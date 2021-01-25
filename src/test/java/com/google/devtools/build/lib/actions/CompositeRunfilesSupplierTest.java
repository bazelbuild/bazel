// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for CompositeRunfilesSupplier */
@RunWith(JUnit4.class)
public class CompositeRunfilesSupplierTest {

  private RunfilesSupplier mockFirst;
  private RunfilesSupplier mockSecond;

  private Path execRoot;
  private ArtifactRoot rootDir;

  @Before
  public final void createMocks() throws IOException {
    Scratch scratch = new Scratch();
    execRoot = scratch.getFileSystem().getPath("/");
    rootDir =
        ArtifactRoot.asDerivedRoot(execRoot, false, false, false, "fake", "root", "dont", "matter");

    mockFirst = mock(RunfilesSupplier.class);
    mockSecond = mock(RunfilesSupplier.class);
  }

  @Test
  public void emptySuppliersIgnored() {
    assertThat(
            CompositeRunfilesSupplier.of(
                EmptyRunfilesSupplier.INSTANCE, EmptyRunfilesSupplier.INSTANCE))
        .isSameInstanceAs(EmptyRunfilesSupplier.INSTANCE);
    assertThat(CompositeRunfilesSupplier.of(EmptyRunfilesSupplier.INSTANCE, mockFirst))
        .isSameInstanceAs(mockFirst);
    assertThat(CompositeRunfilesSupplier.of(mockFirst, EmptyRunfilesSupplier.INSTANCE))
        .isSameInstanceAs(mockFirst);
  }

  @Test
  public void fromSuppliersSeleton() {
    assertThat(CompositeRunfilesSupplier.fromSuppliers(ImmutableList.of(mockFirst)))
        .isSameInstanceAs(mockFirst);
  }

  @Test
  public void testGetArtifactsReturnsCombinedArtifacts() {
    when(mockFirst.getArtifacts()).thenReturn(mkArtifacts(rootDir, "first", "shared"));
    when(mockSecond.getArtifacts()).thenReturn(mkArtifacts(rootDir, "second", "shared"));

    RunfilesSupplier underTest = CompositeRunfilesSupplier.of(mockFirst, mockSecond);
    assertThat(underTest.getArtifacts().toList())
        .containsExactlyElementsIn(mkArtifacts(rootDir, "first", "second", "shared").toList());
  }

  @Test
  public void testGetRunfilesDirsReturnsCombinedPaths() {
    PathFragment first = PathFragment.create("first");
    PathFragment second = PathFragment.create("second");
    PathFragment shared = PathFragment.create("shared");

    when(mockFirst.getRunfilesDirs()).thenReturn(ImmutableSet.of(first, shared));
    when(mockSecond.getRunfilesDirs()).thenReturn(ImmutableSet.of(second, shared));

    RunfilesSupplier underTest = CompositeRunfilesSupplier.of(mockFirst, mockSecond);
    assertThat(underTest.getRunfilesDirs()).containsExactly(first, second, shared);
  }

  @Test
  public void testGetMappingsReturnsMappingsWithFirstPrecedenceOverSecond() throws IOException {
    PathFragment first = PathFragment.create("first");
    Map<PathFragment, Artifact> firstMappings = mkMappings(rootDir, "first1", "first2");

    PathFragment second = PathFragment.create("second");
    Map<PathFragment, Artifact> secondMappings = mkMappings(rootDir, "second1", "second2");

    PathFragment shared = PathFragment.create("shared");
    Map<PathFragment, Artifact> firstSharedMappings = mkMappings(rootDir, "shared1", "shared2");
    Map<PathFragment, Artifact> secondSharedMappings = mkMappings(rootDir, "lost1", "lost2");

    when(mockFirst.getMappings())
        .thenReturn(
            ImmutableMap.of(
                first, firstMappings,
                shared, firstSharedMappings));
    when(mockSecond.getMappings())
        .thenReturn(
            ImmutableMap.of(
                second, secondMappings,
                shared, secondSharedMappings));

    // We expect the mappings for shared added by mockSecond to be dropped.
    RunfilesSupplier underTest = CompositeRunfilesSupplier.of(mockFirst, mockSecond);
    assertThat(underTest.getMappings())
        .containsExactly(
            first, firstMappings,
            second, secondMappings,
            shared, firstSharedMappings);
  }

  @Test
  public void testGetMappingsViaListConstructorReturnsMappingsWithFirstPrecedenceOverSecond()
      throws IOException {
    PathFragment first = PathFragment.create("first");
    Map<PathFragment, Artifact> firstMappings = mkMappings(rootDir, "first1", "first2");

    PathFragment second = PathFragment.create("second");
    Map<PathFragment, Artifact> secondMappings = mkMappings(rootDir, "second1", "second2");

    PathFragment shared = PathFragment.create("shared");
    Map<PathFragment, Artifact> firstSharedMappings = mkMappings(rootDir, "shared1", "shared2");
    Map<PathFragment, Artifact> secondSharedMappings = mkMappings(rootDir, "lost1", "lost2");

    when(mockFirst.getMappings())
        .thenReturn(
            ImmutableMap.of(
                first, firstMappings,
                shared, firstSharedMappings));
    when(mockSecond.getMappings())
        .thenReturn(
            ImmutableMap.of(
                second, secondMappings,
                shared, secondSharedMappings));

    // We expect the mappings for shared added by mockSecond to be dropped.
    RunfilesSupplier underTest = CompositeRunfilesSupplier.of(mockFirst, mockSecond);
    assertThat(underTest.getMappings())
        .containsExactly(
            first, firstMappings,
            second, secondMappings,
            shared, firstSharedMappings);
   }

  private static Map<PathFragment, Artifact> mkMappings(ArtifactRoot rootDir, String... paths) {
    ImmutableMap.Builder<PathFragment, Artifact> builder = ImmutableMap.builder();
    for (String path : paths) {
      builder.put(PathFragment.create(path), ActionsTestUtil.createArtifact(rootDir, path));
    }
    return builder.build();
  }

  private static NestedSet<Artifact> mkArtifacts(ArtifactRoot rootDir, String... paths) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    for (String path : paths) {
      builder.add(ActionsTestUtil.createArtifact(rootDir, path));
    }
    return builder.build();
  }
}
