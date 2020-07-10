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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.HasDigest;
import com.google.devtools.build.lib.actions.HasDigest.ByteStringDigest;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.FileNotFoundException;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ActionMetadataHandler}. */
@RunWith(JUnit4.class)
public final class ActionMetadataHandlerTest {

  private final Scratch scratch = new Scratch();
  private final TimestampGranularityMonitor tsgm =
      new TimestampGranularityMonitor(new ManualClock());

  private final ArtifactRoot sourceRoot =
      ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
  private final ArtifactRoot outputRoot =
      ArtifactRoot.asDerivedRoot(scratch.resolve("/output"), "bin");

  @Before
  public void createRootDirs() throws Exception {
    sourceRoot.getRoot().asPath().createDirectoryAndParents();
    outputRoot.getRoot().asPath().createDirectoryAndParents();
  }

  @Test
  public void withNonArtifactInput() throws Exception {
    ActionInput input = ActionInputHelper.fromPath("foo/bar");
    FileArtifactValue metadata =
        FileArtifactValue.createForNormalFile(
            new byte[] {1, 2, 3}, /*proxy=*/ null, 10L, /*isShareable=*/ true);
    ActionInputMap map = new ActionInputMap(1);
    map.putWithNoDepOwner(input, metadata);
    assertThat(map.getMetadata(input)).isEqualTo(metadata);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            map,
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());
    assertThat(handler.getMetadata(input)).isNull();
  }

  @Test
  public void withArtifactInput() throws Exception {
    PathFragment path = PathFragment.create("src/a");
    Artifact artifact = ActionsTestUtil.createArtifactWithRootRelativePath(sourceRoot, path);
    FileArtifactValue metadata =
        FileArtifactValue.createForNormalFile(
            new byte[] {1, 2, 3}, /*proxy=*/ null, 10L, /*isShareable=*/ true);
    ActionInputMap map = new ActionInputMap(1);
    map.putWithNoDepOwner(artifact, metadata);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            map,
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());
    assertThat(handler.getMetadata(artifact)).isEqualTo(metadata);
  }

  @Test
  public void withUnknownSourceArtifactAndNoMissingArtifactsAllowed() {
    PathFragment path = PathFragment.create("src/a");
    Artifact artifact = ActionsTestUtil.createArtifactWithRootRelativePath(sourceRoot, path);
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            map,
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());
    Exception e = assertThrows(IllegalStateException.class, () -> handler.getMetadata(artifact));
    assertThat(e).hasMessageThat().contains(artifact + " is not present in declared outputs");
  }

  @Test
  public void withUnknownSourceArtifact() throws Exception {
    PathFragment path = PathFragment.create("src/a");
    Artifact artifact = ActionsTestUtil.createArtifactWithRootRelativePath(sourceRoot, path);
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            map,
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ true,
            /*outputs=*/ ImmutableSet.of(),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());
    assertThat(handler.getMetadata(artifact)).isNull();
  }

  @Test
  public void withUnknownOutputArtifactMissingAllowed() throws Exception {
    PathFragment path = PathFragment.create("foo/bar");
    Artifact artifact = ActionsTestUtil.createArtifactWithRootRelativePath(outputRoot, path);
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            map,
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ true,
            /*outputs=*/ ImmutableSet.of(),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());
    assertThat(handler.getMetadata(artifact)).isNull();
  }

  @Test
  public void withUnknownOutputArtifactStatsFile() throws Exception {
    scratch.file("/output/bin/foo/bar", "not empty");
    Artifact artifact = ActionsTestUtil.createArtifact(outputRoot, "foo/bar");
    assertThat(artifact.getPath().exists()).isTrue();
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            map,
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(artifact),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());
    assertThat(handler.getMetadata(artifact)).isNotNull();
  }

  @Test
  public void withUnknownOutputArtifactStatsFileFailsWithException() {
    Artifact artifact = ActionsTestUtil.createArtifact(outputRoot, "foo/bar");
    assertThat(artifact.getPath().exists()).isFalse();
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            map,
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(artifact),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());
    assertThrows(FileNotFoundException.class, () -> handler.getMetadata(artifact));
  }

  @Test
  public void withUnknownOutputArtifactMissingDisallowed() {
    PathFragment path = PathFragment.create("foo/bar");
    Artifact artifact = ActionsTestUtil.createArtifactWithRootRelativePath(outputRoot, path);
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            map,
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());
    assertThrows(IllegalStateException.class, () -> handler.getMetadata(artifact));
  }

  @Test
  public void withUnknownOutputArtifactMissingAllowedTreeArtifact() throws Exception {
    PathFragment path = PathFragment.create("bin/foo/bar");
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputRoot, path);
    Artifact artifact = TreeFileArtifact.createTreeOutput(treeArtifact, "baz");
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            map,
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ true,
            /*outputs=*/ ImmutableSet.of(),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());
    assertThat(handler.getMetadata(artifact)).isNull();
  }

  @Test
  public void withUnknownOutputArtifactStatsFileTreeArtifact() throws Exception {
    scratch.file("/output/bin/foo/bar/baz", "not empty");
    PathFragment path = PathFragment.create("bin/foo/bar");
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputRoot, path);
    Artifact artifact = TreeFileArtifact.createTreeOutput(treeArtifact, "baz");
    assertThat(artifact.getPath().exists()).isTrue();
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            map,
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(treeArtifact),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());
    assertThat(handler.getMetadata(artifact)).isNotNull();
  }

  @Test
  public void withUnknownOutputArtifactMissingDisallowedTreeArtifact() {
    PathFragment path = PathFragment.create("bin/foo/bar");
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputRoot, path);
    Artifact artifact = TreeFileArtifact.createTreeOutput(treeArtifact, "baz");
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            map,
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());
    assertThrows(IllegalStateException.class, () -> handler.getMetadata(artifact));
  }

  @Test
  public void createsTreeArtifactValueFromFilesystem() throws Exception {
    scratch.file("/output/bin/foo/bar/child1", "child1");
    scratch.file("/output/bin/foo/bar/child2", "child2");
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, PathFragment.create("bin/foo/bar"));
    TreeFileArtifact child1 = TreeFileArtifact.createTreeOutput(treeArtifact, "child1");
    TreeFileArtifact child2 = TreeFileArtifact.createTreeOutput(treeArtifact, "child2");
    assertThat(child1.getPath().exists()).isTrue();
    assertThat(child2.getPath().exists()).isTrue();

    OutputStore store = new OutputStore();
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            new ActionInputMap(1),
            /*expandedFilesets=*/ ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(treeArtifact),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            store,
            outputRoot.getRoot().asPath());

    FileArtifactValue treeMetadata = handler.getMetadata(treeArtifact);
    FileArtifactValue child1Metadata = handler.getMetadata(child1);
    FileArtifactValue child2Metadata = handler.getMetadata(child2);
    TreeArtifactValue tree = store.getTreeArtifactData(treeArtifact);

    assertThat(tree.getMetadata()).isEqualTo(treeMetadata);
    assertThat(tree.getChildValues())
        .containsExactly(child1, child1Metadata, child2, child2Metadata);
    assertThat(store.getAllArtifactData()).isEmpty(); // All data should be in treeArtifactData.
  }

  @Test
  public void resettingOutputs() throws Exception {
    scratch.file("/output/bin/foo/bar", "not empty");
    PathFragment path = PathFragment.create("foo/bar");
    Artifact artifact = ActionsTestUtil.createArtifactWithRootRelativePath(outputRoot, path);
    ActionInputMap map = new ActionInputMap(1);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            map,
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ true,
            /*outputs=*/ ImmutableSet.of(artifact),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());
    handler.discardOutputMetadata();

    // The handler doesn't have any info. It'll stat the file and discover that it's 10 bytes long.
    assertThat(handler.getMetadata(artifact).getSize()).isEqualTo(10);

    // Inject a remote file of size 42.
    handler.injectFile(
        artifact, new RemoteFileArtifactValue(new byte[] {1, 2, 3}, 42, 0, "ultimate-answer"));
    assertThat(handler.getMetadata(artifact).getSize()).isEqualTo(42);

    // Reset this output, which will make the handler stat the file again.
    handler.resetOutputs(ImmutableList.of(artifact));
    assertThat(handler.getMetadata(artifact).getSize()).isEqualTo(10);
  }

  @Test
  public void injectRemoteArtifactMetadata() throws Exception {
    PathFragment path = PathFragment.create("foo/bar");
    Artifact artifact = ActionsTestUtil.createArtifactWithRootRelativePath(outputRoot, path);
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            /* inputArtifactData= */ new ActionInputMap(0),
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(artifact),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            new OutputStore(),
            outputRoot.getRoot().asPath());
    handler.discardOutputMetadata();

    byte[] digest = new byte[] {1, 2, 3};
    int size = 10;
    handler.injectFile(
        artifact, new RemoteFileArtifactValue(digest, size, /*locationIndex=*/ 1, "action-id"));

    FileArtifactValue v = handler.getMetadata(artifact);
    assertThat(v).isNotNull();
    assertThat(v.getDigest()).isEqualTo(digest);
    assertThat(v.getSize()).isEqualTo(size);
  }

  @Test
  public void cannotInjectTreeArtifactChildIndividually() {
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, PathFragment.create("bin/foo/bar"));
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(treeArtifact, "child");

    OutputStore store = new OutputStore();
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            /*inputArtifactData=*/ new ActionInputMap(1),
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(treeArtifact),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            store,
            outputRoot.getRoot().asPath());
    handler.discardOutputMetadata();

    RemoteFileArtifactValue childValue = new RemoteFileArtifactValue(new byte[] {1, 2, 3}, 5, 1);

    assertThrows(IllegalArgumentException.class, () -> handler.injectFile(child, childValue));
    assertThat(store.getAllArtifactData()).isEmpty();
    assertThat(store.getAllTreeArtifactData()).isEmpty();
  }

  @Test
  public void canInjectTemplateExpansionOutput() {
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, PathFragment.create("bin/foo/bar"));
    TreeFileArtifact output =
        TreeFileArtifact.createTemplateExpansionOutput(
            treeArtifact, "output", ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER);

    OutputStore store = new OutputStore();
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            /*inputArtifactData=*/ new ActionInputMap(1),
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(treeArtifact),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            store,
            outputRoot.getRoot().asPath());
    handler.discardOutputMetadata();

    RemoteFileArtifactValue value = new RemoteFileArtifactValue(new byte[] {1, 2, 3}, 5, 1);
    handler.injectFile(output, value);

    assertThat(store.getAllArtifactData()).containsExactly(output, value);
    assertThat(store.getAllTreeArtifactData()).isEmpty();
  }

  @Test
  public void injectRemoteTreeArtifactMetadata() throws Exception {
    PathFragment path = PathFragment.create("bin/dir");
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputRoot, path);
    OutputStore store = new OutputStore();
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            /* inputArtifactData= */ new ActionInputMap(0),
            ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(treeArtifact),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            store,
            outputRoot.getRoot().asPath());
    handler.discardOutputMetadata();

    RemoteFileArtifactValue fooValue =
        new RemoteFileArtifactValue(new byte[] {1, 2, 3}, 5, 1, "foo");
    RemoteFileArtifactValue barValue =
        new RemoteFileArtifactValue(new byte[] {4, 5, 6}, 10, 1, "bar");
    Map<TreeFileArtifact, FileArtifactValue> children =
        ImmutableMap.of(
            TreeFileArtifact.createTreeOutput(treeArtifact, "foo"), fooValue,
            TreeFileArtifact.createTreeOutput(treeArtifact, "bar"), barValue);

    handler.injectDirectory(treeArtifact, children);

    FileArtifactValue value = handler.getMetadata(treeArtifact);
    assertThat(value).isNotNull();
    TreeArtifactValue treeValue = store.getTreeArtifactData(treeArtifact);
    assertThat(treeValue).isNotNull();
    assertThat(treeValue.getDigest()).isEqualTo(value.getDigest());

    assertThat(treeValue.getChildPaths())
        .containsExactly(PathFragment.create("foo"), PathFragment.create("bar"));
    assertThat(treeValue.getChildValues().values()).containsExactly(fooValue, barValue);

    // Make sure that all children are transferred properly into the ActionExecutionValue. If any
    // child is missing, getExistingFileArtifactValue will throw.
    ActionExecutionValue actionExecutionValue =
        ActionExecutionValue.createFromOutputStore(handler.getOutputStore(), null, null, false);
    treeValue.getChildren().forEach(actionExecutionValue::getExistingFileArtifactValue);
  }

  @Test
  public void getMetadataFromFilesetMapping() throws Exception {
    FileArtifactValue directoryFav = FileArtifactValue.createForDirectoryWithMtime(10L);
    FileArtifactValue regularFav =
        FileArtifactValue.createForVirtualActionInput(new byte[] {1, 2, 3, 4}, 10L);
    HasDigest.ByteStringDigest byteStringDigest = new ByteStringDigest(new byte[] {2, 3, 4});

    ImmutableList<FilesetOutputSymlink> symlinks =
        ImmutableList.of(
            createFilesetOutputSymlink(directoryFav, "dir"),
            createFilesetOutputSymlink(regularFav, "file"),
            createFilesetOutputSymlink(byteStringDigest, "bytes"));

    Artifact artifact =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("foo/bar"));
    ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> filesetMap =
        ImmutableMap.of(artifact, symlinks);

    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            new ActionInputMap(0),
            filesetMap,
            /*missingArtifactsAllowed=*/ false,
            /*outputs=*/ ImmutableSet.of(),
            tsgm,
            ArtifactPathResolver.forExecRoot(outputRoot.getRoot().asPath()),
            new MinimalOutputStore(),
            outputRoot.getRoot().asPath());

    // Only the regular FileArtifactValue should have its metadata stored.
    assertThat(handler.getMetadata(createInput("dir"))).isNull();
    assertThat(handler.getMetadata(createInput("file"))).isEqualTo(regularFav);
    assertThat(handler.getMetadata(createInput("bytes"))).isNull();
    assertThat(handler.getMetadata(createInput("does_not_exist"))).isNull();
  }

  private FilesetOutputSymlink createFilesetOutputSymlink(HasDigest digest, String identifier) {
    return FilesetOutputSymlink.create(
        PathFragment.create(identifier + "_symlink"),
        PathFragment.create(identifier),
        digest,
        /*isGeneratedTarget=*/ true,
        outputRoot.getExecPath());
  }

  private ActionInput createInput(String identifier) {
    return ActionInputHelper.fromPath(outputRoot.getRoot().getRelative(identifier).getPathString());
  }

  @Test
  public void omitRegularArtifact() {
    OutputStore store = new MinimalOutputStore();
    Artifact omitted =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("omitted"));
    Artifact consumed =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("consumed"));
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            new ActionInputMap(1),
            /*expandedFilesets=*/ ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            ImmutableSet.of(omitted, consumed),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            store,
            outputRoot.getRoot().asPath());

    handler.discardOutputMetadata();
    handler.markOmitted(omitted);

    assertThat(handler.artifactOmitted(omitted)).isTrue();
    assertThat(handler.artifactOmitted(consumed)).isFalse();
    assertThat(store.getAllArtifactData())
        .containsExactly(omitted, FileArtifactValue.OMITTED_FILE_MARKER);
    assertThat(store.getAllTreeArtifactData()).isEmpty();
  }

  @Test
  public void omitTreeArtifact() {
    OutputStore store = new MinimalOutputStore();
    SpecialArtifact omittedTree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, PathFragment.create("omitted"));
    SpecialArtifact consumedTree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, PathFragment.create("consumed"));
    ActionMetadataHandler handler =
        new ActionMetadataHandler(
            new ActionInputMap(1),
            /*expandedFilesets=*/ ImmutableMap.of(),
            /*missingArtifactsAllowed=*/ false,
            ImmutableSet.of(omittedTree, consumedTree),
            tsgm,
            ArtifactPathResolver.IDENTITY,
            store,
            outputRoot.getRoot().asPath());

    handler.discardOutputMetadata();
    handler.markOmitted(omittedTree);
    handler.markOmitted(omittedTree); // Marking a tree artifact as omitted twice is tolerated.

    assertThat(handler.artifactOmitted(omittedTree)).isTrue();
    assertThat(handler.artifactOmitted(consumedTree)).isFalse();
    assertThat(store.getAllTreeArtifactData())
        .containsExactly(omittedTree, TreeArtifactValue.OMITTED_TREE_MARKER);
    assertThat(store.getAllArtifactData()).isEmpty();
  }
}
