// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.SortedMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RunfilesArtifactValue}. */
@RunWith(JUnit4.class)
public final class RunfilesArtifactValueTest {

  private final Path execRoot =
      new InMemoryFileSystem(DigestHashFunction.SHA256).getPath("/execroot");
  private final ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out");
  private final Artifact file = ActionsTestUtil.createArtifact(root, "pkg/file");
  private final Artifact manifest =
      ActionsTestUtil.createArtifact(root, "pkg/bin.runfiles/MANIFEST");

  private static FileArtifactValue fileValue(int content) {
    return FileArtifactValue.createForNormalFile(new byte[] {(byte) content}, null, 1);
  }

  private static RunfilesTree treeWithMapping(SortedMap<PathFragment, Artifact> mapping) {
    RunfilesTree tree = mock(RunfilesTree.class);
    when(tree.getMapping()).thenReturn(mapping);
    return tree;
  }

  private RunfilesArtifactValue create(
      RunfilesTree tree, boolean hasManifest, FileArtifactValue manifestValue) {
    return new RunfilesArtifactValue(
        tree,
        hasManifest,
        ImmutableList.of(file, manifest),
        ImmutableList.of(fileValue(1), manifestValue),
        ImmutableList.of(),
        ImmutableList.of(),
        ImmutableList.of(),
        ImmutableList.of());
  }

  @Test
  public void withManifest_mappingNotComputed() {
    RunfilesTree tree = mock(RunfilesTree.class);

    RunfilesArtifactValue value = create(tree, /* hasManifest= */ true, fileValue(2));

    assertThat(value.getMetadata().getDigest()).isNotNull();
    verify(tree, never()).getMapping();
  }

  @Test
  public void withManifest_layoutChangeVisibleThroughManifestDigest() {
    RunfilesTree tree = mock(RunfilesTree.class);

    RunfilesArtifactValue value = create(tree, /* hasManifest= */ true, fileValue(2));
    RunfilesArtifactValue sameManifest = create(tree, /* hasManifest= */ true, fileValue(2));
    RunfilesArtifactValue otherManifest = create(tree, /* hasManifest= */ true, fileValue(3));

    assertThat(sameManifest).isEqualTo(value);
    assertThat(sameManifest.getMetadata()).isEqualTo(value.getMetadata());
    assertThat(otherManifest).isNotEqualTo(value);
    assertThat(otherManifest.getMetadata()).isNotEqualTo(value.getMetadata());
  }

  @Test
  public void withoutManifest_layoutChangeVisibleThroughMapping() {
    RunfilesTree tree =
        treeWithMapping(ImmutableSortedMap.of(PathFragment.create("ws/pkg/file"), file));
    RunfilesTree sameLayout =
        treeWithMapping(ImmutableSortedMap.of(PathFragment.create("ws/pkg/file"), file));
    RunfilesTree otherLayout =
        treeWithMapping(ImmutableSortedMap.of(PathFragment.create("ws/pkg/renamed"), file));

    RunfilesArtifactValue value = create(tree, /* hasManifest= */ false, fileValue(2));

    assertThat(create(sameLayout, /* hasManifest= */ false, fileValue(2)).getMetadata())
        .isEqualTo(value.getMetadata());
    assertThat(create(otherLayout, /* hasManifest= */ false, fileValue(2)).getMetadata())
        .isNotEqualTo(value.getMetadata());
    assertThat(create(tree, /* hasManifest= */ true, fileValue(2)).getMetadata())
        .isNotEqualTo(value.getMetadata());
  }
}
