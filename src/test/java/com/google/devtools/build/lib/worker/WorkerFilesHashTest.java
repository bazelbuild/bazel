// Copyright 2022 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.worker;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactExpander;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.worker.WorkerFilesHash.MissingInputException;
import java.io.IOException;
import java.util.SortedMap;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WorkerFilesHash}. */
@RunWith(JUnit4.class)
public final class WorkerFilesHashTest {

  private final ArtifactRoot outputRoot =
      ArtifactRoot.asDerivedRoot(new Scratch().resolve("/execroot"), RootType.Output, "bazel-out");

  @Test
  public void getWorkerFilesWithDigests_returnsToolsWithCorrectDigests() throws Exception {
    byte[] tool1Digest = "text1".getBytes(UTF_8);
    byte[] tool2Digest = "text2".getBytes(UTF_8);
    InputMetadataProvider inputMetadataProvider =
        createMetadataProvider(
            ImmutableMap.of(
                "tool1", fileArtifactValue(tool1Digest), "tool2", fileArtifactValue(tool2Digest)));
    Spawn spawn =
        new SpawnBuilder()
            .withTool(ActionInputHelper.fromPath("tool1"))
            .withTool(ActionInputHelper.fromPath("tool2"))
            .build();

    SortedMap<PathFragment, byte[]> filesWithDigests =
        WorkerFilesHash.getWorkerFilesWithDigests(
            spawn, treeArtifact -> ImmutableSortedSet.of(), inputMetadataProvider);

    assertThat(filesWithDigests)
        .containsExactly(
            PathFragment.create("tool1"), tool1Digest, PathFragment.create("tool2"), tool2Digest)
        .inOrder();
  }

  @Test
  public void getWorkerFilesWithDigests_treeArtifactTool_returnsExpanded() throws Exception {
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact child1 = TreeFileArtifact.createTreeOutput(tree, "child1");
    TreeFileArtifact child2 = TreeFileArtifact.createTreeOutput(tree, "child2");
    byte[] child1Digest = "text1".getBytes(UTF_8);
    byte[] child2Digest = "text2".getBytes(UTF_8);
    InputMetadataProvider inputMetadataProvider =
        createMetadataProvider(
            ImmutableMap.of(
                child1.getExecPathString(),
                fileArtifactValue(child1Digest),
                child2.getExecPathString(),
                fileArtifactValue(child2Digest)));
    Spawn spawn = new SpawnBuilder().withTool(tree).build();
    ArtifactExpander expander =
        treeArtifact ->
            treeArtifact.equals(tree)
                ? ImmutableSortedSet.of(
                    TreeFileArtifact.createTreeOutput(tree, "child1"),
                    TreeFileArtifact.createTreeOutput(tree, "child2"))
                : ImmutableSortedSet.of();

    SortedMap<PathFragment, byte[]> filesWithDigests =
        WorkerFilesHash.getWorkerFilesWithDigests(spawn, expander, inputMetadataProvider);

    assertThat(filesWithDigests)
        .containsExactly(child1.getExecPath(), child1Digest, child2.getExecPath(), child2Digest)
        .inOrder();
  }

  @Test
  public void getWorkerFilesWithDigests_spawnWithInputsButNoTools_returnsEmpty() throws Exception {
    InputMetadataProvider inputMetadataProvider = createMetadataProvider(ImmutableMap.of());
    Spawn spawn = new SpawnBuilder().withInputs("file1", "file2").build();

    SortedMap<PathFragment, byte[]> filesWithDigests =
        WorkerFilesHash.getWorkerFilesWithDigests(
            spawn, treeArtifact -> ImmutableSortedSet.of(), inputMetadataProvider);

    assertThat(filesWithDigests).isEmpty();
  }

  @Test
  public void getWorkerFilesWithDigests_missingDigestForTool_fails() {
    InputMetadataProvider inputMetadataProvider = createMetadataProvider(ImmutableMap.of());
    Spawn spawn = new SpawnBuilder().withTool(ActionInputHelper.fromPath("tool")).build();

    assertThrows(
        MissingInputException.class,
        () ->
            WorkerFilesHash.getWorkerFilesWithDigests(
                spawn, treeArtifact -> ImmutableSortedSet.of(), inputMetadataProvider));
  }

  @Test
  public void getWorkerFilesWithDigests_ioExceptionForToolMetadata_fails() {
    IOException injected = new IOException("oh no");
    InputMetadataProvider inputMetadataProvider =
        createMetadataProvider(ImmutableMap.of("tool", injected));
    Spawn spawn = new SpawnBuilder().withTool(ActionInputHelper.fromPath("tool")).build();

    IOException thrown =
        assertThrows(
            IOException.class,
            () ->
                WorkerFilesHash.getWorkerFilesWithDigests(
                    spawn, treeArtifact -> ImmutableSortedSet.of(), inputMetadataProvider));

    assertThat(thrown).isSameInstanceAs(injected);
  }

  private static InputMetadataProvider createMetadataProvider(
      ImmutableMap<String, Object> inputMetadataOrExceptions) {
    return new InputMetadataProvider() {

      @Nullable
      @Override
      public FileArtifactValue getInputMetadata(ActionInput input) throws IOException {
        @Nullable
        Object metadataOrException = inputMetadataOrExceptions.get(input.getExecPathString());
        if (metadataOrException == null) {
          return null;
        }
        if (metadataOrException instanceof IOException ioException) {
          throw ioException;
        }
        if (metadataOrException instanceof FileArtifactValue fileArtifactValue) {
          return fileArtifactValue;
        }
        throw new AssertionError("Unexpected value: " + metadataOrException);
      }

      @Override
      @Nullable
      public RunfilesArtifactValue getRunfilesMetadata(ActionInput input) {
        throw new UnsupportedOperationException();
      }

      @Override
      public ImmutableList<RunfilesTree> getRunfilesTrees() {
        throw new UnsupportedOperationException();
      }

      @Nullable
      @Override
      public ActionInput getInput(String execPath) {
        throw new UnsupportedOperationException();
      }
    };
  }

  private static FileArtifactValue fileArtifactValue(byte[] digest) {
    FileArtifactValue value = mock(FileArtifactValue.class);
    when(value.getDigest()).thenReturn(digest);
    return value;
  }

  private SpecialArtifact createTreeArtifact(String rootRelativePath) {
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        outputRoot, outputRoot.getExecPath().getRelative(rootRelativePath));
  }
}
