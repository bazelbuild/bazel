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
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.List;
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

  private Root rootDir;

  @Before
  public final void createMocks() throws IOException {
    Scratch scratch = new Scratch();
    rootDir = Root.asDerivedRoot(scratch.dir("/fake/root/dont/matter"));

    mockFirst = mock(RunfilesSupplier.class);
    mockSecond = mock(RunfilesSupplier.class);
  }

  @Test
  public void testGetArtifactsReturnsCombinedArtifacts() {
    when(mockFirst.getArtifacts()).thenReturn(mkArtifacts(rootDir, "first", "shared"));
    when(mockSecond.getArtifacts()).thenReturn(mkArtifacts(rootDir, "second", "shared"));

    CompositeRunfilesSupplier underTest = new CompositeRunfilesSupplier(mockFirst, mockSecond);
    assertThat(underTest.getArtifacts()).containsExactlyElementsIn(
        mkArtifacts(rootDir, "first", "second", "shared"));
  }

  @Test
  public void testGetRunfilesDirsReturnsCombinedPaths() {
    PathFragment first = PathFragment.create("first");
    PathFragment second = PathFragment.create("second");
    PathFragment shared = PathFragment.create("shared");

    when(mockFirst.getRunfilesDirs()).thenReturn(ImmutableSet.of(first, shared));
    when(mockSecond.getRunfilesDirs()).thenReturn(ImmutableSet.of(second, shared));

    CompositeRunfilesSupplier underTest = new CompositeRunfilesSupplier(mockFirst, mockSecond);
    assertThat(underTest.getRunfilesDirs()).containsExactlyElementsIn(
        ImmutableSet.of(first, second, shared));
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

    when(mockFirst.getMappings()).thenReturn(ImmutableMap.of(
        first, firstMappings,
        shared, firstSharedMappings));
    when(mockSecond.getMappings()).thenReturn(ImmutableMap.of(
        second, secondMappings,
        shared, secondSharedMappings));

    // We expect the mappings for shared added by mockSecond to be dropped.
    CompositeRunfilesSupplier underTest = new CompositeRunfilesSupplier(mockFirst, mockSecond);
    assertThat(underTest.getMappings()).containsExactly(
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

    when(mockFirst.getMappings()).thenReturn(ImmutableMap.of(
        first, firstMappings,
        shared, firstSharedMappings));
    when(mockSecond.getMappings()).thenReturn(ImmutableMap.of(
        second, secondMappings,
        shared, secondSharedMappings));

    // We expect the mappings for shared added by mockSecond to be dropped.
    CompositeRunfilesSupplier underTest =
        new CompositeRunfilesSupplier(ImmutableList.of(mockFirst, mockSecond));
    assertThat(underTest.getMappings()).containsExactly(
        first, firstMappings,
        second, secondMappings,
        shared, firstSharedMappings);
   }

  private static Map<PathFragment, Artifact> mkMappings(Root rootDir, String... paths) {
    ImmutableMap.Builder<PathFragment, Artifact> builder = ImmutableMap.builder();
    for (String path : paths) {
      builder.put(PathFragment.create(path), mkArtifact(rootDir, path));
    }
    return builder.build();
  }

  private static Artifact mkArtifact(Root rootDir, String path) {
    return new Artifact(PathFragment.create(path), rootDir);
  }

  private static List<Artifact> mkArtifacts(Root rootDir, String... paths) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    for (String path : paths) {
      builder.add(mkArtifact(rootDir, path));
    }
    return builder.build();
  }
}
