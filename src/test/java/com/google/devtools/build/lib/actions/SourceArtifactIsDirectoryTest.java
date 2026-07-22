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

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.devtools.build.lib.actions.Artifact.ArtifactSerializationContext;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.LabelArtifactOwner;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.Root.RootCodecDependencies;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link SourceArtifact#isDirectory()} with {@link SourceDirectoryIsDirectoryFlag}
 * enabled. The flag is read from a system property at class-load time, so this test has its own
 * target that sets it via {@code jvm_flags}. The default behavior is covered by {@link
 * ArtifactTest}.
 */
@RunWith(JUnit4.class)
public final class SourceArtifactIsDirectoryTest {

  private final Scratch scratch = new Scratch();
  private ArtifactRoot root;

  @Before
  public void createRoot() throws Exception {
    root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/src")));
  }

  @Test
  public void flagEnabled() {
    assertThat(SourceDirectoryIsDirectoryFlag.sourceDirectoryIsDirectory()).isTrue();
  }

  @Test
  public void notDirectoryUntilSet() {
    SourceArtifact artifact = (SourceArtifact) ActionsTestUtil.createArtifact(root, "some_dir");

    assertThat(artifact.isDirectory()).isFalse();
  }

  @Test
  public void setIsDirectory_updatesInBothDirections() {
    SourceArtifact artifact = (SourceArtifact) ActionsTestUtil.createArtifact(root, "some_path");

    artifact.setIsDirectory(true);
    assertThat(artifact.isDirectory()).isTrue();

    artifact.setIsDirectory(false);
    assertThat(artifact.isDirectory()).isFalse();
  }

  @Test
  public void codec_preservesDirectoryBit() throws Exception {
    ArtifactFactory artifactFactory =
        new ArtifactFactory(scratch.dir("/base/exec").getParentDirectory(), "blaze-out");
    ObjectCodecs objectCodecs =
        new ObjectCodecs(
            AutoRegistry.get()
                .getBuilder()
                .addReferenceConstant(scratch.getFileSystem())
                .setAllowDefaultCodec(true)
                .build(),
            ImmutableClassToInstanceMap.builder()
                .put(FileSystem.class, scratch.getFileSystem())
                .put(ArtifactSerializationContext.class, artifactFactory::getSourceArtifact)
                .put(RootCodecDependencies.class, new RootCodecDependencies(root.getRoot()))
                .build());
    SourceArtifact dir =
        new SourceArtifact(
            root,
            PathFragment.create("some_dir"),
            new LabelArtifactOwner(Label.parseCanonicalUnchecked("//foo:bar")));
    dir.setIsDirectory(true);

    SourceArtifact deserialized =
        (SourceArtifact) objectCodecs.deserialize(objectCodecs.serialize(dir));

    assertThat(deserialized.isDirectory()).isTrue();
  }

  @Test
  public void derivedArtifacts_unaffected() throws Exception {
    ArtifactRoot derivedRoot =
        ArtifactRoot.asDerivedRoot(scratch.dir("/base/exec"), RootType.OUTPUT, "root");
    Artifact regular = ActionsTestUtil.createArtifact(derivedRoot, "out/file");
    Artifact tree = ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedRoot, "out/tree");

    assertThat(regular.isDirectory()).isFalse();
    assertThat(tree.isDirectory()).isTrue();
  }
}
