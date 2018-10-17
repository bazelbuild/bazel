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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.FileNotFoundException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ActionMetadataHandler}. */
@RunWith(JUnit4.class)
public class ActionMetadataHandlerTest {
  private Scratch scratch;
  private ArtifactRoot sourceRoot;
  private ArtifactRoot outputRoot;

  @Before
  public final void setRootDir() throws Exception  {
    scratch = new Scratch();
    sourceRoot = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/workspace")));
    outputRoot = ArtifactRoot.asDerivedRoot(scratch.dir("/output"), scratch.dir("/output/bin"));
  }

  @Test
  public void withNonArtifactInput() throws Exception {
    ActionInput input = ActionInputHelper.fromPath("foo/bar");
    FileArtifactValue metadata = FileArtifactValue.createNormalFile(new byte[] {1,  2, 3}, 10);
    ActionInputMap map = new ActionInputMap(1);
    map.put(input, metadata);
    assertThat(map.getMetadata(input)).isEqualTo(metadata);
    ActionMetadataHandler handler = new ActionMetadataHandler(
        map,
        /* missingArtifactsAllowed= */ false,
        /* outputs= */ ImmutableList.of(),
        /* tsgm= */ null,
        ArtifactPathResolver.IDENTITY,
        new MinimalOutputStore());
    assertThat(handler.getMetadata(input)).isNull();
  }

  @Test
  public void withArtifactInput() throws Exception {
    PathFragment path = PathFragment.create("src/a");
    Artifact artifact = new Artifact(path, sourceRoot);
    FileArtifactValue metadata = FileArtifactValue.createNormalFile(new byte[] {1,  2, 3}, 10);
    ActionInputMap map = new ActionInputMap(1);
    map.put(artifact, metadata);
    ActionMetadataHandler handler = new ActionMetadataHandler(
        map,
        /* missingArtifactsAllowed= */ false,
        /* outputs= */ ImmutableList.of(),
        /* tsgm= */ null,
        ArtifactPathResolver.IDENTITY,
        new MinimalOutputStore());
    assertThat(handler.getMetadata(artifact)).isEqualTo(metadata);
  }

  @Test
  public void withUnknownSourceArtifactAndNoMissingArtifactsAllowed() throws Exception {
    PathFragment path = PathFragment.create("src/a");
    Artifact artifact = new Artifact(path, sourceRoot);
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler = new ActionMetadataHandler(
        map,
        /* missingArtifactsAllowed= */ false,
        /* outputs= */ ImmutableList.of(),
        /* tsgm= */ null,
        ArtifactPathResolver.IDENTITY,
        new MinimalOutputStore());
    try {
      handler.getMetadata(artifact);
      fail();
    } catch (IllegalStateException expected) {
      assertThat(expected).hasMessageThat().contains("null for ");
    }
  }

  @Test
  public void withUnknownSourceArtifact() throws Exception {
    PathFragment path = PathFragment.create("src/a");
    Artifact artifact = new Artifact(path, sourceRoot);
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler = new ActionMetadataHandler(
        map,
        /* missingArtifactsAllowed= */ true,
        /* outputs= */ ImmutableList.of(),
        /* tsgm= */ null,
        ArtifactPathResolver.IDENTITY,
        new MinimalOutputStore());
    assertThat(handler.getMetadata(artifact)).isNull();
  }

  @Test
  public void withUnknownOutputArtifactMissingAllowed() throws Exception {
    PathFragment path = PathFragment.create("foo/bar");
    Artifact artifact = new Artifact(path, outputRoot);
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler = new ActionMetadataHandler(
        map,
        /* missingArtifactsAllowed= */ true,
        /* outputs= */ ImmutableList.of(),
        /* tsgm= */ null,
        ArtifactPathResolver.IDENTITY,
        new MinimalOutputStore());
    assertThat(handler.getMetadata(artifact)).isNull();
  }

  @Test
  public void withUnknownOutputArtifactStatsFile() throws Exception {
    scratch.file("/output/bin/foo/bar", "not empty");
    Artifact artifact = new Artifact(PathFragment.create("foo/bar"), outputRoot);
    assertThat(artifact.getPath().exists()).isTrue();
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler = new ActionMetadataHandler(
        map,
        /* missingArtifactsAllowed= */ false,
        /* outputs= */ ImmutableList.of(artifact),
        /* tsgm= */ null,
        ArtifactPathResolver.IDENTITY,
        new MinimalOutputStore());
    assertThat(handler.getMetadata(artifact)).isNotNull();
  }

  @Test
  public void withUnknownOutputArtifactStatsFileFailsWithException() throws Exception {
    Artifact artifact = new Artifact(PathFragment.create("foo/bar"), outputRoot);
    assertThat(artifact.getPath().exists()).isFalse();
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler = new ActionMetadataHandler(
        map,
        /* missingArtifactsAllowed= */ false,
        /* outputs= */ ImmutableList.of(artifact),
        /* tsgm= */ null,
        ArtifactPathResolver.IDENTITY,
        new MinimalOutputStore());
    try {
      handler.getMetadata(artifact);
      fail();
    } catch (FileNotFoundException expected) {
    }
  }

  @Test
  public void withUnknownOutputArtifactMissingDisallowed() throws Exception {
    PathFragment path = PathFragment.create("foo/bar");
    Artifact artifact = new Artifact(path, outputRoot);
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler = new ActionMetadataHandler(
        map,
        /* missingArtifactsAllowed= */ false,
        /* outputs= */ ImmutableList.of(),
        /* tsgm= */ null,
        ArtifactPathResolver.IDENTITY,
        new MinimalOutputStore());
    try {
      handler.getMetadata(artifact);
      fail();
    } catch (IllegalStateException expected) {
    }
  }

  @Test
  public void withUnknownOutputArtifactMissingAllowedTreeArtifact() throws Exception {
    PathFragment path = PathFragment.create("bin/foo/bar");
    SpecialArtifact treeArtifact =
        new SpecialArtifact(
            outputRoot, path, ArtifactOwner.NullArtifactOwner.INSTANCE, SpecialArtifactType.TREE);
    Artifact artifact = new TreeFileArtifact(treeArtifact, PathFragment.create("baz"));
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler = new ActionMetadataHandler(
        map,
        /* missingArtifactsAllowed= */ true,
        /* outputs= */ ImmutableList.of(),
        /* tsgm= */ null,
        ArtifactPathResolver.IDENTITY,
        new MinimalOutputStore());
    assertThat(handler.getMetadata(artifact)).isNull();
  }

  @Test
  public void withUnknownOutputArtifactStatsFileTreeArtifact() throws Exception {
    scratch.file("/output/bin/foo/bar/baz", "not empty");
    PathFragment path = PathFragment.create("bin/foo/bar");
    SpecialArtifact treeArtifact =
        new SpecialArtifact(
            outputRoot, path, ArtifactOwner.NullArtifactOwner.INSTANCE, SpecialArtifactType.TREE);
    Artifact artifact = new TreeFileArtifact(treeArtifact, PathFragment.create("baz"));
    assertThat(artifact.getPath().exists()).isTrue();
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler = new ActionMetadataHandler(
        map,
        /* missingArtifactsAllowed= */ false,
        /* outputs= */ ImmutableList.of(treeArtifact),
        /* tsgm= */ null,
        ArtifactPathResolver.IDENTITY,
        new MinimalOutputStore());
    assertThat(handler.getMetadata(artifact)).isNotNull();
  }

  @Test
  public void withUnknownOutputArtifactMissingDisallowedTreeArtifact() throws Exception {
    PathFragment path = PathFragment.create("bin/foo/bar");
    SpecialArtifact treeArtifact =
        new SpecialArtifact(
            outputRoot, path, ArtifactOwner.NullArtifactOwner.INSTANCE, SpecialArtifactType.TREE);
    Artifact artifact = new TreeFileArtifact(treeArtifact, PathFragment.create("baz"));
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler = new ActionMetadataHandler(
        map,
        /* missingArtifactsAllowed= */ false,
        /* outputs= */ ImmutableList.of(),
        /* tsgm= */ null,
        ArtifactPathResolver.IDENTITY,
        new MinimalOutputStore());
    try {
      handler.getMetadata(artifact);
      fail();
    } catch (IllegalStateException expected) {
    }
  }
}
