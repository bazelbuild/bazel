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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.HasDigest;
import com.google.devtools.build.lib.actions.HasDigest.ByteStringDigest;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.NullAction;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ActionMetadataHandler}. */
@RunWith(JUnit4.class)
public final class ActionMetadataHandlerTest {

  private final Map<Path, Integer> chmodCalls = Maps.newConcurrentMap();

  private final Scratch scratch =
      new Scratch(
          new InMemoryFileSystem(DigestHashFunction.SHA256) {
            @Override
            public void chmod(PathFragment pathFragment, int mode) throws IOException {
              Path path = getPath(pathFragment);
              if (chmodCalls.containsKey(path)) {
                fail("chmod called on " + path + " twice");
              }
              chmodCalls.put(path, mode);
              super.chmod(pathFragment, mode);
            }
          });

  private final TimestampGranularityMonitor tsgm =
      new TimestampGranularityMonitor(new ManualClock());

  private final ArtifactRoot sourceRoot =
      ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
  private final PathFragment derivedPathPrefix = PathFragment.create("bin");
  private final ArtifactRoot outputRoot =
      ArtifactRoot.asDerivedRoot(scratch.resolve("/output"), RootType.Output, derivedPathPrefix);
  private final Path execRoot = outputRoot.getRoot().asPath();

  @Before
  public void createRootDirs() throws Exception {
    sourceRoot.getRoot().asPath().createDirectoryAndParents();
    execRoot.createDirectoryAndParents();
  }

  private ActionMetadataHandler createHandler(
      ActionInputMap inputMap, ImmutableSet<Artifact> outputs) {
    return ActionMetadataHandler.create(
        inputMap,
        /* archivedTreeArtifactsEnabled= */ false,
        OutputPermissions.READONLY,
        outputs,
        SyscallCache.NO_CACHE,
        tsgm,
        ArtifactPathResolver.IDENTITY,
        execRoot.asFragment(),
        /* expandedFilesets= */ ImmutableMap.of());
  }

  @Test
  public void withNonArtifactInput() throws Exception {
    ActionInput input = ActionInputHelper.fromPath("foo/bar");
    FileArtifactValue metadata =
        FileArtifactValue.createForNormalFile(
            new byte[] {1, 2, 3}, /* proxy= */ null, /* size= */ 10L);
    ActionInputMap map = new ActionInputMap(1);
    map.putWithNoDepOwner(input, metadata);
    assertThat(map.getInputMetadata(input)).isEqualTo(metadata);
    ActionMetadataHandler handler =
        createHandler(map, /* forInputDiscovery= *//* outputs= */ ImmutableSet.of());
    assertThat(handler.getInputMetadata(input)).isNull();
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void withArtifactInput() throws Exception {
    PathFragment path = PathFragment.create("src/a");
    Artifact artifact = ActionsTestUtil.createArtifactWithRootRelativePath(sourceRoot, path);
    FileArtifactValue metadata =
        FileArtifactValue.createForNormalFile(
            new byte[] {1, 2, 3}, /* proxy= */ null, /* size= */ 10L);
    ActionInputMap map = new ActionInputMap(1);
    map.putWithNoDepOwner(artifact, metadata);
    ActionMetadataHandler handler =
        createHandler(map, /* forInputDiscovery= *//* outputs= */ ImmutableSet.of());
    assertThat(handler.getInputMetadata(artifact)).isEqualTo(metadata);
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void unknownSourceArtifactPermittedDuringInputDiscovery() throws Exception {
    PathFragment path = PathFragment.create("src/a");
    Artifact artifact = ActionsTestUtil.createArtifactWithRootRelativePath(sourceRoot, path);
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0), /* forInputDiscovery= *//* outputs= */ ImmutableSet.of());
    assertThat(handler.getInputMetadata(artifact)).isNull();
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void unknownArtifactPermittedDuringInputDiscovery() throws Exception {
    PathFragment path = PathFragment.create("foo/bar");
    Artifact artifact = ActionsTestUtil.createArtifactWithRootRelativePath(outputRoot, path);
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0), /* forInputDiscovery= *//* outputs= */ ImmutableSet.of());
    assertThat(handler.getInputMetadata(artifact)).isNull();
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void withKnownOutputArtifactStatsFile() throws Exception {
    scratch.file("/output/bin/foo/bar", "not empty");
    Artifact artifact = ActionsTestUtil.createArtifact(outputRoot, "foo/bar");
    assertThat(artifact.getPath().exists()).isTrue();
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(artifact));
    assertThat(handler.getOutputMetadata(artifact)).isNotNull();
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void withMissingOutputArtifactStatsFileFailsWithException() {
    Artifact artifact = ActionsTestUtil.createArtifact(outputRoot, "foo/bar");
    assertThat(artifact.getPath().exists()).isFalse();
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(artifact));
    assertThrows(FileNotFoundException.class, () -> handler.getOutputMetadata(artifact));
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void unknownTreeArtifactPermittedDuringInputDiscovery() throws Exception {
    PathFragment path = PathFragment.create("bin/foo/bar");
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputRoot, path);
    Artifact artifact = TreeFileArtifact.createTreeOutput(treeArtifact, "baz");
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0), /* forInputDiscovery= *//* outputs= */ ImmutableSet.of());
    assertThat(handler.getInputMetadata(artifact)).isNull();
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void withUnknownOutputArtifactStatsFileTreeArtifact() throws Exception {
    scratch.file("/output/bin/foo/bar/baz", "not empty");
    PathFragment path = PathFragment.create("bin/foo/bar");
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputRoot, path);
    Artifact artifact = TreeFileArtifact.createTreeOutput(treeArtifact, "baz");
    assertThat(artifact.getPath().exists()).isTrue();
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(treeArtifact));
    assertThat(handler.getOutputMetadata(artifact)).isNotNull();
    assertThat(chmodCalls).isEmpty();
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

    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(treeArtifact));

    FileArtifactValue treeMetadata = handler.getOutputMetadata(treeArtifact);
    FileArtifactValue child1Metadata = handler.getOutputMetadata(child1);
    FileArtifactValue child2Metadata = handler.getOutputMetadata(child2);
    TreeArtifactValue tree = handler.getOutputStore().getTreeArtifactData(treeArtifact);

    assertThat(tree.getMetadata()).isEqualTo(treeMetadata);
    assertThat(tree.getChildValues())
        .containsExactly(child1, child1Metadata, child2, child2Metadata);
    assertThat(handler.getTreeArtifactChildren(treeArtifact)).isEqualTo(tree.getChildren());
    assertThat(handler.getOutputStore().getAllArtifactData()).isEmpty();
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void resettingOutputs() throws Exception {
    PathFragment path = PathFragment.create("foo/bar");
    Artifact artifact = ActionsTestUtil.createArtifactWithRootRelativePath(outputRoot, path);
    Path outputPath = scratch.file(artifact.getPath().getPathString(), "not empty");
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(artifact));
    handler.prepareForActionExecution();

    // The handler doesn't have any info. It'll stat the file and discover that it's 10 bytes long.
    assertThat(handler.getOutputMetadata(artifact).getSize()).isEqualTo(10);
    assertThat(chmodCalls).containsExactly(outputPath, 0555);

    // Inject a remote file of size 42.
    handler.injectFile(
        artifact,
        RemoteFileArtifactValue.create(new byte[] {1, 2, 3}, 42, 0, /* expireAtEpochMilli= */ -1));
    assertThat(handler.getOutputMetadata(artifact).getSize()).isEqualTo(42);

    // Reset this output, which will make the handler stat the file again.
    handler.resetOutputs(ImmutableList.of(artifact));
    chmodCalls.clear();
    assertThat(handler.getOutputMetadata(artifact).getSize()).isEqualTo(10);
    // The handler should not have chmodded the file as it already has the correct permission.
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void injectRemoteArtifactMetadata() throws Exception {
    PathFragment path = PathFragment.create("foo/bar");
    Artifact artifact = ActionsTestUtil.createArtifactWithRootRelativePath(outputRoot, path);
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(artifact));
    handler.prepareForActionExecution();

    byte[] digest = new byte[] {1, 2, 3};
    int size = 10;
    handler.injectFile(
        artifact,
        RemoteFileArtifactValue.create(
            digest, size, /* locationIndex= */ 1, /* expireAtEpochMilli= */ -1));

    FileArtifactValue v = handler.getOutputMetadata(artifact);
    assertThat(v).isNotNull();
    assertThat(v.getDigest()).isEqualTo(digest);
    assertThat(v.getSize()).isEqualTo(size);
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void cannotInjectTreeArtifactChildIndividually() {
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, PathFragment.create("bin/foo/bar"));
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(treeArtifact, "child");

    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(treeArtifact));
    handler.prepareForActionExecution();

    RemoteFileArtifactValue childValue =
        RemoteFileArtifactValue.create(new byte[] {1, 2, 3}, 5, 1, /* expireAtEpochMilli= */ -1);

    assertThrows(IllegalArgumentException.class, () -> handler.injectFile(child, childValue));
    assertThat(handler.getOutputStore().getAllArtifactData()).isEmpty();
    assertThat(handler.getOutputStore().getAllTreeArtifactData()).isEmpty();
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void canInjectTemplateExpansionOutput() {
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, PathFragment.create("bin/foo/bar"));
    TreeFileArtifact output =
        TreeFileArtifact.createTemplateExpansionOutput(
            treeArtifact, "output", ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER);

    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(treeArtifact));
    handler.prepareForActionExecution();

    RemoteFileArtifactValue value =
        RemoteFileArtifactValue.create(new byte[] {1, 2, 3}, 5, 1, /* expireAtEpochMilli= */ -1);
    handler.injectFile(output, value);

    assertThat(handler.getOutputStore().getAllArtifactData()).containsExactly(output, value);
    assertThat(handler.getOutputStore().getAllTreeArtifactData()).isEmpty();
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void injectRemoteTreeArtifactMetadata() throws Exception {
    PathFragment path = PathFragment.create("bin/dir");
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputRoot, path);
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(treeArtifact));
    handler.prepareForActionExecution();

    TreeArtifactValue tree =
        TreeArtifactValue.newBuilder(treeArtifact)
            .putChild(
                TreeFileArtifact.createTreeOutput(treeArtifact, "foo"),
                RemoteFileArtifactValue.create(
                    new byte[] {1, 2, 3}, 5, 1, /* expireAtEpochMilli= */ -1))
            .putChild(
                TreeFileArtifact.createTreeOutput(treeArtifact, "bar"),
                RemoteFileArtifactValue.create(
                    new byte[] {4, 5, 6}, 10, 1, /* expireAtEpochMilli= */ -1))
            .build();

    handler.injectTree(treeArtifact, tree);

    FileArtifactValue value = handler.getOutputMetadata(treeArtifact);
    assertThat(value).isNotNull();
    assertThat(value.getDigest()).isEqualTo(tree.getDigest());
    assertThat(handler.getOutputStore().getTreeArtifactData(treeArtifact)).isEqualTo(tree);
    assertThat(chmodCalls).isEmpty();

    assertThat(handler.getTreeArtifactChildren(treeArtifact)).isEqualTo(tree.getChildren());

    // Make sure that all children are transferred properly into the ActionExecutionValue. If any
    // child is missing, getExistingFileArtifactValue will throw.
    ActionExecutionValue actionExecutionValue =
        ActionExecutionValue.createFromOutputStore(
            handler.getOutputStore(), /* outputSymlinks= */ ImmutableList.of(), new NullAction());
    tree.getChildren().forEach(actionExecutionValue::getExistingFileArtifactValue);
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
    ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets =
        ImmutableMap.of(artifact, symlinks);

    ActionMetadataHandler handler =
        ActionMetadataHandler.create(
            new ActionInputMap(0),
            /* archivedTreeArtifactsEnabled= */ false,
            OutputPermissions.READONLY,
            /* outputs= */ ImmutableSet.of(),
            SyscallCache.NO_CACHE,
            tsgm,
            ArtifactPathResolver.IDENTITY,
            execRoot.asFragment(),
            expandedFilesets);

    // Only the regular FileArtifactValue should have its metadata stored.
    assertThat(handler.getInputMetadata(createInput("dir"))).isNull();
    assertThat(handler.getInputMetadata(createInput("file"))).isEqualTo(regularFav);
    assertThat(handler.getInputMetadata(createInput("bytes"))).isNull();
    assertThat(handler.getInputMetadata(createInput("does_not_exist"))).isNull();
    assertThat(chmodCalls).isEmpty();
  }

  private FilesetOutputSymlink createFilesetOutputSymlink(HasDigest digest, String identifier) {
    return FilesetOutputSymlink.create(
        PathFragment.create(identifier + "_symlink"),
        PathFragment.create(identifier),
        digest,
        outputRoot.getExecPath());
  }

  private ActionInput createInput(String identifier) {
    return ActionInputHelper.fromPath(outputRoot.getRoot().getRelative(identifier).getPathString());
  }

  @Test
  public void omitRegularArtifact() {
    Artifact omitted =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("omitted"));
    Artifact consumed =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("consumed"));
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(omitted, consumed));

    handler.prepareForActionExecution();
    handler.markOmitted(omitted);

    assertThat(handler.artifactOmitted(omitted)).isTrue();
    assertThat(handler.artifactOmitted(consumed)).isFalse();
    assertThat(handler.getOutputStore().getAllArtifactData())
        .containsExactly(omitted, FileArtifactValue.OMITTED_FILE_MARKER);
    assertThat(handler.getOutputStore().getAllTreeArtifactData()).isEmpty();
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void omitTreeArtifact() {
    SpecialArtifact omittedTree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, PathFragment.create("omitted"));
    SpecialArtifact consumedTree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, PathFragment.create("consumed"));
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(omittedTree, consumedTree));

    handler.prepareForActionExecution();
    handler.markOmitted(omittedTree);
    handler.markOmitted(omittedTree); // Marking a tree artifact as omitted twice is tolerated.

    assertThat(handler.artifactOmitted(omittedTree)).isTrue();
    assertThat(handler.artifactOmitted(consumedTree)).isFalse();
    assertThat(handler.getOutputStore().getAllTreeArtifactData())
        .containsExactly(omittedTree, TreeArtifactValue.OMITTED_TREE_MARKER);
    assertThat(handler.getOutputStore().getAllArtifactData()).isEmpty();
    assertThat(chmodCalls).isEmpty();
  }

  @Test
  public void outputArtifactNotPreviouslyInjectedInExecutionMode() throws Exception {
    Artifact output =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("dir/file.out"));
    Path outputPath = scratch.file(output.getPath().getPathString(), "contents");
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0), /* forInputDiscovery= *//* outputs= */ ImmutableSet.of(output));
    handler.prepareForActionExecution();

    FileArtifactValue metadata = handler.getOutputMetadata(output);

    assertThat(metadata.getDigest()).isEqualTo(outputPath.getDigest());
    assertThat(handler.getOutputStore().getAllArtifactData()).containsExactly(output, metadata);
    assertThat(handler.getOutputStore().getAllTreeArtifactData()).isEmpty();
    assertThat(chmodCalls).containsExactly(outputPath, 0555);
  }

  @Test
  public void outputArtifactNotPreviouslyInjectedInExecutionMode_writablePermissions()
      throws Exception {
    Artifact output =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("dir/file.out"));
    Path outputPath = scratch.file(output.getPath().getPathString(), "contents");
    ActionMetadataHandler handler =
        ActionMetadataHandler.create(
            new ActionInputMap(0),
            /* archivedTreeArtifactsEnabled= */ false,
            OutputPermissions.WRITABLE,
            /* outputs= */ ImmutableSet.of(output),
            SyscallCache.NO_CACHE,
            tsgm,
            ArtifactPathResolver.IDENTITY,
            execRoot.asFragment(),
            /* expandedFilesets= */ ImmutableMap.of());
    handler.prepareForActionExecution();

    FileArtifactValue metadata = handler.getOutputMetadata(output);

    assertThat(metadata.getDigest()).isEqualTo(outputPath.getDigest());
    assertThat(handler.getOutputStore().getAllArtifactData()).containsExactly(output, metadata);
    assertThat(handler.getOutputStore().getAllTreeArtifactData()).isEmpty();
    // Permissions preserved in handler, so chmod calls should be empty.
    assertThat(chmodCalls).containsExactly(outputPath, 0755);
  }

  @Test
  public void outputTreeArtifactNotPreviouslyInjectedInExecutionMode() throws Exception {
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, PathFragment.create("bin/foo/bar"));
    TreeFileArtifact child1 = TreeFileArtifact.createTreeOutput(treeArtifact, "child1");
    TreeFileArtifact child2 = TreeFileArtifact.createTreeOutput(treeArtifact, "subdir/child2");
    Path child1Path = scratch.file(child1.getPath().getPathString(), "contents1");
    Path child2Path = scratch.file(child2.getPath().getPathString(), "contents2");
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(treeArtifact));
    handler.prepareForActionExecution();

    FileArtifactValue treeMetadata = handler.getOutputMetadata(treeArtifact);
    FileArtifactValue child1Metadata = handler.getOutputMetadata(child1);
    FileArtifactValue child2Metadata = handler.getOutputMetadata(child2);
    TreeArtifactValue tree = handler.getOutputStore().getTreeArtifactData(treeArtifact);

    assertThat(tree.getMetadata()).isEqualTo(treeMetadata);
    assertThat(tree.getChildValues())
        .containsExactly(child1, child1Metadata, child2, child2Metadata);
    assertThat(handler.getTreeArtifactChildren(treeArtifact)).isEqualTo(tree.getChildren());
    assertThat(handler.getOutputStore().getAllArtifactData()).isEmpty();
    assertThat(chmodCalls)
        .containsExactly(
            treeArtifact.getPath(),
            0555,
            child1Path,
            0555,
            child2Path,
            0555,
            child2Path.getParentDirectory(),
            0555);
  }

  @Test
  public void getTreeArtifactChildren_noData_returnsEmptySet() {
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, PathFragment.create("bin/tree"));
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(treeArtifact));
    assertThat(handler.getTreeArtifactChildren(treeArtifact)).isEmpty();
  }

  @Test
  public void enteringExecutionModeClearsCachedOutputs() throws Exception {
    Artifact artifact =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("bin/output"));
    SpecialArtifact treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, PathFragment.create("bin/tree"));
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(treeArtifact, "child");
    scratch.file(artifact.getPath().getPathString(), "1");
    scratch.file(child.getPath().getPathString(), "1");
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(artifact, treeArtifact));
    OutputStore store = handler.getOutputStore();

    FileArtifactValue artifactMetadata1 = handler.getOutputMetadata(artifact);
    FileArtifactValue treeArtifactMetadata1 = handler.getOutputMetadata(treeArtifact);
    assertThat(artifactMetadata1).isNotNull();
    assertThat(artifactMetadata1).isNotNull();
    assertThat(store.getAllArtifactData().keySet()).containsExactly(artifact);
    assertThat(store.getAllTreeArtifactData().keySet()).containsExactly(treeArtifact);

    // Entering execution mode should clear the cached outputs.
    handler.prepareForActionExecution();
    assertThat(store.getAllArtifactData()).isEmpty();
    assertThat(store.getAllTreeArtifactData()).isEmpty();

    // Updated metadata should be read from the filesystem.
    scratch.overwriteFile(artifact.getPath().getPathString(), "2");
    scratch.overwriteFile(child.getPath().getPathString(), "2");
    FileArtifactValue artifactMetadata2 = handler.getOutputMetadata(artifact);
    FileArtifactValue treeArtifactMetadata2 = handler.getOutputMetadata(treeArtifact);
    assertThat(artifactMetadata2).isNotNull();
    assertThat(treeArtifactMetadata2).isNotNull();
    assertThat(artifactMetadata2).isNotEqualTo(artifactMetadata1);
    assertThat(treeArtifactMetadata2).isNotEqualTo(treeArtifactMetadata1);
  }

  @Test
  public void cannotEnterExecutionModeTwice() {
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0), /* forInputDiscovery= *//* outputs= */ ImmutableSet.of());
    handler.prepareForActionExecution();
    assertThrows(IllegalStateException.class, handler::prepareForActionExecution);
  }

  @Test
  public void fileArtifactValueFromArtifactCompatibleWithGetMetadata_changed() throws Exception {
    Artifact artifact =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("bin/output"));
    scratch.file(artifact.getPath().getPathString(), "1");
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(artifact));

    FileArtifactValue getMetadataResult = handler.getOutputMetadata(artifact);
    assertThat(getMetadataResult).isNotNull();

    scratch.overwriteFile(artifact.getPath().getPathString(), "2");
    FileArtifactValue fileArtifactValueFromArtifactResult =
        ActionMetadataHandler.fileArtifactValueFromArtifact(
            artifact, /* statNoFollow= */ null, SyscallCache.NO_CACHE, /* tsgm= */ null);
    assertThat(fileArtifactValueFromArtifactResult).isNotNull();

    assertThat(fileArtifactValueFromArtifactResult.couldBeModifiedSince(getMetadataResult))
        .isTrue();
  }

  @Test
  public void fileArtifactValueFromArtifactCompatibleWithGetMetadata_notChanged() throws Exception {
    Artifact artifact =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("bin/output"));
    scratch.file(artifact.getPath().getPathString(), "contents");
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(artifact));

    FileArtifactValue getMetadataResult = handler.getOutputMetadata(artifact);
    assertThat(getMetadataResult).isNotNull();

    FileArtifactValue fileArtifactValueFromArtifactResult =
        ActionMetadataHandler.fileArtifactValueFromArtifact(
            artifact, /* statNoFollow= */ null, SyscallCache.NO_CACHE, /* tsgm= */ null);
    assertThat(fileArtifactValueFromArtifactResult).isNotNull();

    assertThat(fileArtifactValueFromArtifactResult.couldBeModifiedSince(getMetadataResult))
        .isFalse();
  }

  @Test
  public void fileArtifactValueForSymlink_readFromCache() throws Exception {
    DigestUtils.configureCache(1);
    Artifact target =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("bin/target"));
    scratch.file(target.getPath().getPathString(), "contents");
    Artifact symlink =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("bin/symlink"));
    scratch
        .getFileSystem()
        .getPath(symlink.getPath().getPathString())
        .createSymbolicLink(scratch.getFileSystem().getPath(target.getPath().getPathString()));
    ActionMetadataHandler handler =
        createHandler(
            new ActionInputMap(0),
            /* forInputDiscovery= */
            /* outputs= */ ImmutableSet.of(target, symlink));
    var targetMetadata = handler.getOutputMetadata(target);
    assertThat(DigestUtils.getCacheStats().hitCount()).isEqualTo(0);

    var symlinkMetadata = handler.getOutputMetadata(symlink);

    assertThat(symlinkMetadata).isEqualTo(targetMetadata);
    assertThat(DigestUtils.getCacheStats().hitCount()).isEqualTo(1);
  }
}
