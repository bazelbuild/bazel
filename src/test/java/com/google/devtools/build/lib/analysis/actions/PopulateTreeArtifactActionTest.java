// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.cache.Md5Digest;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.ActionTester.ActionCombinationFactory;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link PopulateTreeArtifactAction}. */
@RunWith(JUnit4.class)
public class PopulateTreeArtifactActionTest extends BuildViewTestCase {
  private static class TestMetadataHandler implements MetadataHandler {
    private final List<Artifact> storingExpandedTreeFileArtifacts;

    TestMetadataHandler(List<Artifact> storingExpandedTreeFileArtifacts) {
      this.storingExpandedTreeFileArtifacts = storingExpandedTreeFileArtifacts;
    }

    @Override
    public void addExpandedTreeOutput(TreeFileArtifact output) {
      storingExpandedTreeFileArtifacts.add(output);
    }

    @Override
    public Iterable<TreeFileArtifact> getExpandedOutputs(Artifact artifact) {
      throw new UnsupportedOperationException(artifact.prettyPrint());
    }

    @Override
    public Metadata getMetadata(Artifact artifact) {
      throw new UnsupportedOperationException(artifact.prettyPrint());
    }

    @Override
    public void setDigestForVirtualArtifact(Artifact artifact, Md5Digest md5Digest) {
      throw new UnsupportedOperationException(artifact.prettyPrint() + ": " + md5Digest);
    }

    @Override
    public void injectDigest(ActionInput output, FileStatus statNoFollow, byte[] digest) {
      throw new UnsupportedOperationException(output.toString());
    }

    @Override
    public void markOmitted(ActionInput output) {
      throw new UnsupportedOperationException(output.toString());
    }

    @Override
    public boolean isRegularFile(Artifact artifact) {
      throw new UnsupportedOperationException(artifact.prettyPrint());
    }

    @Override
    public boolean artifactOmitted(Artifact artifact) {
      throw new UnsupportedOperationException(artifact.prettyPrint());
    }

    @Override
    public void discardOutputMetadata() {
      throw new UnsupportedOperationException();
    }
  };

  private Root root;

  @Before
  public void setRootDir() throws Exception  {
    root = Root.asDerivedRoot(scratch.dir("/exec/root"));
  }

  @Test
  public void testActionOutputs() throws Exception {
    Action action = createPopulateTreeArtifactAction();
    assertThat(Artifact.toExecPaths(action.getOutputs())).containsExactly("test/archive_member");
  }

  @Test
  public void testActionInputs() throws Exception {
    Action action = createPopulateTreeArtifactAction();
    assertThat(Artifact.toExecPaths(action.getInputs())).containsExactly(
        "myArchive.zip",
        "archiveManifest.txt",
        "unzipBinary");
  }

  @Test
  public void testSpawnOutputs() throws Exception {
    PopulateTreeArtifactAction action = createPopulateTreeArtifactAction();
    Spawn spawn = action.createSpawn();
    Iterable<Artifact> outputs = actionInputsToArtifacts(spawn.getOutputFiles());
    assertThat(Artifact.toExecPaths(outputs)).containsExactly(
        "test/archive_member/archive_members/1.class",
        "test/archive_member/archive_members/2.class",
        "test/archive_member/archive_members/txt/text.txt");
  }

  @Test
  public void testSpawnInputs() throws Exception {
    PopulateTreeArtifactAction action = createPopulateTreeArtifactAction();
    Spawn spawn = action.createSpawn();
    Iterable<Artifact> inputs = actionInputsToArtifacts(spawn.getInputFiles());
    assertThat(Artifact.toExecPaths(inputs)).containsExactly(
        "myArchive.zip",
        "archiveManifest.txt",
        "unzipBinary");
  }

  @Test
  public void testSpawnArguments() throws Exception {
    PopulateTreeArtifactAction action = createPopulateTreeArtifactAction();
    BaseSpawn spawn = (BaseSpawn) action.createSpawn();
    assertThat(spawn.getArguments()).containsExactly(
        "unzipBinary",
        "x",
        "myArchive.zip",
        "-d",
        "test/archive_member",
        "@archiveManifest.txt").inOrder();
  }

  @Test
  public void testTreeArtifactPopulated() throws Exception {
    ArrayList<Artifact> treefileArtifacts = new ArrayList<Artifact>();
    PopulateTreeArtifactAction action = createPopulateTreeArtifactAction();
    ActionExecutionContext executionContext = actionExecutionContext(treefileArtifacts);
    action.execute(executionContext);

    assertThat(Artifact.toExecPaths(treefileArtifacts)).containsExactly(
        "test/archive_member/archive_members/1.class",
        "test/archive_member/archive_members/2.class",
        "test/archive_member/archive_members/txt/text.txt");
  }

  @Test
  public void testInvalidManifestEntryPaths() throws Exception {
    Action action = createPopulateTreeArtifactAction();
    scratch.overwriteFile(
        "archiveManifest.txt",
        "archive_members/1.class",
        "../invalid_relative_path/myfile.class");
    ActionExecutionContext executionContext = actionExecutionContext(new ArrayList<Artifact>());

    try {
      action.execute(executionContext);
      fail("Invalid manifest entry paths, expected exception");
    } catch (ActionExecutionException e) {
      // Expect ActionExecutionException
    }
  }

  @Test
  public void testTreeFileArtifactPathPrefixConflicts() throws Exception {
    Action action = createPopulateTreeArtifactAction();
    scratch.overwriteFile(
        "archiveManifest.txt",
        "archive_members/conflict",
        "archive_members/conflict/1.class");
    ActionExecutionContext executionContext = actionExecutionContext(new ArrayList<Artifact>());

    try {
      action.execute(executionContext);
      fail("Artifact path prefix conflicts, expected exception");
    } catch (ActionExecutionException e) {
      // Expect ActionExecutionException
    }
  }

  @Test
  public void testEmptyTreeArtifactInputAndOutput() throws Exception {
    Action action = createPopulateTreeArtifactAction();
    scratch.overwriteFile("archiveManifest.txt", "");

    ArrayList<Artifact> treeFileArtifacts = new ArrayList<Artifact>();
    ActionExecutionContext executionContext = actionExecutionContext(treeFileArtifacts);

    action.execute(executionContext);

    assertThat(treeFileArtifacts).isEmpty();
  }

  @Test
  public void testOutputTreeFileArtifactDirsCreated() throws Exception {
    Action action = createPopulateTreeArtifactAction();
    scratch.overwriteFile(
        "archiveManifest.txt",
        "archive_members/dirA/memberA",
        "archive_members/dirB/memberB");

    ArrayList<Artifact> treeFileArtifacts = new ArrayList<Artifact>();
    ActionExecutionContext executionContext = actionExecutionContext(treeFileArtifacts);
    action.execute(executionContext);

    // We check whether the parent directory structures of output TreeFileArtifacts exist even
    // though the spawn is not executed (the SpawnActionContext is mocked out).
    assertThat(treeFileArtifacts).hasSize(2);
    for (Artifact treeFileArtifact : treeFileArtifacts) {
      assertThat(treeFileArtifact.getPath().getParentDirectory().exists()).isTrue();
      assertThat(treeFileArtifact.getPath().exists()).isFalse();
    }
  }

  private enum KeyAttributes {
    ARCHIVE,
    TREE_ARTIFACT,
    ARCHIVE_MANIFEST,
    ZIPPER
  }

  @Test
  public void testComputeKey() throws Exception {
    final Artifact archiveA = getSourceArtifact("myArchiveA.zip");
    final Artifact archiveB = getSourceArtifact("myArchiveB.zip");
    final Artifact treeArtifactToPopulateA = createTreeArtifact("testA/archive_member");
    final Artifact treeArtifactToPopulateB = createTreeArtifact("testB/archive_member");
    final Artifact archiveManifestA = getSourceArtifact("archiveManifestA.txt");
    final Artifact archiveManifestB = getSourceArtifact("archiveManifestB.txt");
    final FilesToRunProvider zipperA = FilesToRunProvider.fromSingleExecutableArtifact(
        getSourceArtifact("unzipBinaryA"));
    final FilesToRunProvider zipperB = FilesToRunProvider.fromSingleExecutableArtifact(
        getSourceArtifact("unzipBinaryB"));

    ActionTester.runTest(
        KeyAttributes.class,
        new ActionCombinationFactory<KeyAttributes>() {
          @Override
          public Action generate(ImmutableSet<KeyAttributes> attributesToFlip) {
            Artifact archive =
                attributesToFlip.contains(KeyAttributes.ARCHIVE) ? archiveA : archiveB;
            Artifact treeArtifactToPopulate =
                attributesToFlip.contains(KeyAttributes.TREE_ARTIFACT)
                    ? treeArtifactToPopulateA
                    : treeArtifactToPopulateB;
            Artifact archiveManifest =
                attributesToFlip.contains(KeyAttributes.ARCHIVE_MANIFEST)
                    ? archiveManifestA
                    : archiveManifestB;
            FilesToRunProvider zipper =
                attributesToFlip.contains(KeyAttributes.ZIPPER) ? zipperA : zipperB;

            return new PopulateTreeArtifactAction(
                ActionsTestUtil.NULL_ACTION_OWNER,
                archive,
                archiveManifest,
                treeArtifactToPopulate,
                zipper);
          }
        });
  }

  private PopulateTreeArtifactAction createPopulateTreeArtifactAction() throws Exception {
    Artifact archive = getSourceArtifact("myArchive.zip");
    Artifact treeArtifactToPopulate = createTreeArtifact("test/archive_member");
    Artifact archiveManifest = getSourceArtifact("archiveManifest.txt");
    FilesToRunProvider unzip = FilesToRunProvider.fromSingleExecutableArtifact(
        getSourceArtifact("unzipBinary"));

    scratch.file(
        "archiveManifest.txt",
        "archive_members/1.class",
        "archive_members/2.class",
        "archive_members/txt/text.txt");

    return new PopulateTreeArtifactAction(
        ActionsTestUtil.NULL_ACTION_OWNER,
        archive,
        archiveManifest,
        treeArtifactToPopulate,
        unzip);
  }

  private ActionExecutionContext actionExecutionContext(
      List<Artifact> storingExpandedTreeFileArtifacts) throws Exception {
    Executor executor = new TestExecutorBuilder(directories, null)
        .setExecution(PopulateTreeArtifactAction.MNEMONIC, mock(SpawnActionContext.class))
        .build();

    return new ActionExecutionContext(
        executor,
        null,
        ActionInputPrefetcher.NONE,
        new TestMetadataHandler(storingExpandedTreeFileArtifacts),
        null,
        ImmutableMap.<String, String>of(),
        null);
  }

  private Artifact createTreeArtifact(String rootRelativePath) {
    PathFragment relpath = PathFragment.create(rootRelativePath);
    return new SpecialArtifact(
        root.getPath().getRelative(relpath),
        root,
        root.getExecPath().getRelative(relpath),
        ArtifactOwner.NULL_OWNER,
        SpecialArtifactType.TREE);
  }

  private Iterable<Artifact> actionInputsToArtifacts(Iterable<? extends ActionInput> files) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.<Artifact>builder();
    for (ActionInput file : files) {
      builder.add((Artifact) file);
    }
    return builder.build();
  }
}
