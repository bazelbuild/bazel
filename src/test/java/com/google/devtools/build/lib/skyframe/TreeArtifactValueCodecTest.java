// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactSerializationContext;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestBase;
import com.google.devtools.build.lib.skyframe.serialization.DeserializedSkyValue;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class TreeArtifactValueCodecTest extends BuildViewTestBase {

  @Test
  public void serializationRoundtrip() throws Exception {
    var subjects = ImmutableList.<TreeArtifactValue>builder();
    subjects.add(TreeArtifactValue.empty());

    ArtifactRoot root =
        ArtifactRoot.asDerivedRoot(
            skyframeExecutor.getBlazeDirectoriesForTesting().getOutputBase(),
            RootType.OUTPUT,
            PathFragment.create("bin"));
    SpecialArtifact parent =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            root, PathFragment.create("bin/tree"));
    TreeFileArtifact child1 = TreeFileArtifact.createTreeOutput(parent, "child1");
    TreeFileArtifact child2 = TreeFileArtifact.createTreeOutput(parent, "child2");
    FileArtifactValue metadata1 = metadataWithId(1);
    FileArtifactValue metadata2 = metadataWithId(2);
    subjects.add(
        TreeArtifactValue.newBuilder(parent)
            .putChild(child1, metadata1)
            .putChild(child2, metadata2)
            .build());

    ArchivedTreeArtifact archivedTreeArtifact = ArchivedTreeArtifact.createForTree(parent);
    FileArtifactValue archivedArtifactMetadata = metadataWithId(3);
    subjects.add(
        TreeArtifactValue.newBuilder(parent)
            .putChild(child1, metadata1)
            .putChild(child2, metadata2)
            .setArchivedRepresentation(archivedTreeArtifact, archivedArtifactMetadata)
            .build());

    PathFragment targetPath = PathFragment.create("/some/target/path");
    subjects
        .add(TreeArtifactValue.newBuilder(parent).setResolvedPath(targetPath).build())
        .add(
            TreeArtifactValue.newBuilder(parent)
                .setArchivedRepresentation(archivedTreeArtifact, archivedArtifactMetadata)
                .build());

    new SerializationTester(subjects.build())
        .addDependency(Root.RootCodecDependencies.class, new Root.RootCodecDependencies())
        .addDependency(
            FileSystem.class,
            skyframeExecutor.getBlazeDirectoriesForTesting().getOutputBase().getFileSystem())
        .addDependency(
            ArtifactSerializationContext.class,
            skyframeExecutor.getSkyframeBuildView().getArtifactFactory()::getSourceArtifact)
        .setVerificationFunction(
            (in, out) -> {
              assertThat(in).isEqualTo(out);
              assertThat(out).isInstanceOf(DeserializedSkyValue.class);
            })
        .runTests();
  }

  private static FileArtifactValue metadataWithId(int id) {
    return FileArtifactValue.createForRemoteFile(new byte[] {(byte) id}, id, id);
  }
}
