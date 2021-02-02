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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SingleRunfilesSupplier}. */
@RunWith(JUnit4.class)
public final class SingleRunfilesSupplierTest {

  private final ArtifactRoot rootDir =
      ArtifactRoot.asDerivedRoot(
          new InMemoryFileSystem(DigestHashFunction.SHA256).getPath("/"),
          RootType.Output,
          "fake",
          "root",
          "dont",
          "matter");

  @Test
  public void testGetArtifactsWithSingleMapping() {
    List<Artifact> artifacts = mkArtifacts(rootDir, "thing1", "thing2");

    SingleRunfilesSupplier underTest =
        new SingleRunfilesSupplier(
            PathFragment.create("notimportant"),
            mkRunfiles(artifacts),
            /*manifest=*/ null,
            /*buildRunfileLinks=*/ false,
            /*runfileLinksEnabled=*/ false);

    assertThat(underTest.getArtifacts().toList()).containsExactlyElementsIn(artifacts);
  }

  @Test
  public void testGetManifestsWhenNone() {
    RunfilesSupplier underTest =
        new SingleRunfilesSupplier(
            PathFragment.create("ignored"),
            Runfiles.EMPTY,
            /*manifest=*/ null,
            /*buildRunfileLinks=*/ false,
            /*runfileLinksEnabled=*/ false);
    assertThat(underTest.getManifests()).isEmpty();
  }

  @Test
  public void testGetManifestsWhenSupplied() {
    Artifact manifest = ActionsTestUtil.createArtifact(rootDir, "manifest");
    RunfilesSupplier underTest =
        new SingleRunfilesSupplier(
            PathFragment.create("ignored"),
            Runfiles.EMPTY,
            manifest,
            /*buildRunfileLinks=*/ false,
            /*runfileLinksEnabled=*/ false);
    assertThat(underTest.getManifests()).containsExactly(manifest);
  }

  @Test
  public void withOverriddenRunfilesDir() {
    SingleRunfilesSupplier original =
        new SingleRunfilesSupplier(
            PathFragment.create("old"),
            Runfiles.EMPTY,
            ActionsTestUtil.createArtifact(rootDir, "manifest"),
            /*buildRunfileLinks=*/ false,
            /*runfileLinksEnabled=*/ false);
    PathFragment newDir = PathFragment.create("new");

    RunfilesSupplier overridden = original.withOverriddenRunfilesDir(newDir);

    assertThat(overridden.getRunfilesDirs()).containsExactly(newDir);
    assertThat(overridden.getMappings())
        .containsExactly(newDir, Iterables.getOnlyElement(original.getMappings().values()));
    assertThat(overridden.getArtifacts()).isEqualTo(original.getArtifacts());
    assertThat(overridden.getManifests()).isEqualTo(original.getManifests());
  }

  @Test
  public void withOverriddenRunfilesDir_noChange_sameObject() {
    PathFragment dir = PathFragment.create("dir");
    SingleRunfilesSupplier original =
        new SingleRunfilesSupplier(
            dir,
            Runfiles.EMPTY,
            ActionsTestUtil.createArtifact(rootDir, "manifest"),
            /*buildRunfileLinks=*/ false,
            /*runfileLinksEnabled=*/ false);
    assertThat(original.withOverriddenRunfilesDir(dir)).isSameInstanceAs(original);
  }

  private static Runfiles mkRunfiles(Iterable<Artifact> artifacts) {
    return new Runfiles.Builder("TESTING", false).addArtifacts(artifacts).build();
  }

  private static ImmutableList<Artifact> mkArtifacts(ArtifactRoot rootDir, String... paths) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    for (String path : paths) {
      builder.add(ActionsTestUtil.createArtifact(rootDir, path));
    }
    return builder.build();
  }
}
