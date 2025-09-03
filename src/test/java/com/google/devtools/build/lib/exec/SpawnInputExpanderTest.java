// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.FilesetOutputTree;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SpawnInputExpander}. */
@RunWith(JUnit4.class)
public final class SpawnInputExpanderTest {
  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final Path execRoot = fs.getPath("/root");
  private final ArtifactRoot rootDir = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out");

  private SpawnInputExpander expander = new SpawnInputExpander();
  private final Map<PathFragment, ActionInput> inputMap = new HashMap<>();

  @Test
  public void testRunfilesSingleFile() throws Exception {
    Artifact artifact =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))),
            fs.getPath("/root/dir/file"));
    Runfiles runfiles = new Runfiles.Builder("workspace").addArtifact(artifact).build();
    RunfilesTree runfilesTree =
        AnalysisTestUtil.createRunfilesTree(PathFragment.create("runfiles"), runfiles);

    expander.addSingleRunfilesTreeToInputs(
        runfilesTree,
        inputMap,
        new FakeActionInputFileCache(),
        PathMapper.NOOP,
        PathFragment.EMPTY_FRAGMENT);

    assertThat(inputMap)
        .containsExactly(PathFragment.create("runfiles/workspace/dir/file"), artifact);
  }

  @Test
  public void testRunfilesWithFileset() throws Exception {
    Artifact fileset = createFilesetArtifact("foo/biz/fs_out");
    Runfiles runfiles = new Runfiles.Builder("workspace").addArtifact(fileset).build();
    RunfilesTree runfilesTree =
        AnalysisTestUtil.createRunfilesTree(PathFragment.create("runfiles"), runfiles);
    FilesetOutputSymlink link = filesetSymlink("zizz", "xyz/zizz");
    FilesetOutputTree filesetOutputTree = FilesetOutputTree.create(ImmutableList.of(link));

    FakeActionInputFileCache fakeActionInputFileCache = new FakeActionInputFileCache();
    fakeActionInputFileCache.putFileset(fileset, filesetOutputTree);
    expander.addSingleRunfilesTreeToInputs(
        runfilesTree,
        inputMap,
        fakeActionInputFileCache,
        PathMapper.NOOP,
        PathFragment.EMPTY_FRAGMENT);

    assertThat(inputMap)
        .containsExactly(
            PathFragment.create("runfiles/workspace/foo/biz/fs_out/zizz"), link.target());
  }

  @Test
  public void testRunfilesDirectoryNonStrict() throws Exception {
    Artifact artifact =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))),
            fs.getPath("/root/dir/file"));
    Runfiles runfiles = new Runfiles.Builder("workspace").addArtifact(artifact).build();
    RunfilesTree runfilesTree =
        AnalysisTestUtil.createRunfilesTree(PathFragment.create("runfiles"), runfiles);

    expander.addSingleRunfilesTreeToInputs(
        runfilesTree,
        inputMap,
        new FakeActionInputFileCache(),
        PathMapper.NOOP,
        PathFragment.EMPTY_FRAGMENT);
    assertThat(inputMap)
        .containsExactly(PathFragment.create("runfiles/workspace/dir/file"), artifact);
  }

  @Test
  public void testRunfilesTwoFiles() throws Exception {
    Artifact artifact1 =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))),
            fs.getPath("/root/dir/file"));
    Artifact artifact2 =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))),
            fs.getPath("/root/dir/baz"));
    Runfiles runfiles =
        new Runfiles.Builder("workspace").addArtifact(artifact1).addArtifact(artifact2).build();
    RunfilesTree runfilesTree =
        AnalysisTestUtil.createRunfilesTree(PathFragment.create("runfiles"), runfiles);

    expander.addSingleRunfilesTreeToInputs(
        runfilesTree,
        inputMap,
        new FakeActionInputFileCache(),
        PathMapper.NOOP,
        PathFragment.EMPTY_FRAGMENT);
    assertThat(inputMap)
        .containsExactly(
            PathFragment.create("runfiles/workspace/dir/file"), artifact1,
            PathFragment.create("runfiles/workspace/dir/baz"), artifact2);
  }

  @Test
  public void testRunfilesTwoFiles_pathMapped() throws Exception {
    Artifact artifact1 =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))),
            fs.getPath("/root/dir/file"));
    Artifact artifact2 =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))),
            fs.getPath("/root/dir/baz"));
    Runfiles runfiles =
        new Runfiles.Builder("workspace").addArtifact(artifact1).addArtifact(artifact2).build();
    RunfilesTree runfilesTree =
        AnalysisTestUtil.createRunfilesTree(
            PathFragment.create("bazel-out/k8-opt/bin/foo.runfiles"), runfiles);

    expander.addSingleRunfilesTreeToInputs(
        runfilesTree,
        inputMap,
        new FakeActionInputFileCache(),
        execPath -> PathFragment.create(execPath.getPathString().replace("k8-opt/", "")),
        PathFragment.EMPTY_FRAGMENT);

    assertThat(inputMap)
        .containsExactly(
            PathFragment.create("bazel-out/bin/foo.runfiles/workspace/dir/file"),
            artifact1,
            PathFragment.create("bazel-out/bin/foo.runfiles/workspace/dir/baz"),
            artifact2);
  }

  @Test
  public void testRunfilesSymlink() throws Exception {
    Artifact artifact =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))),
            fs.getPath("/root/dir/file"));
    Runfiles runfiles =
        new Runfiles.Builder("workspace")
            .addSymlink(PathFragment.create("symlink"), artifact)
            .build();
    RunfilesTree runfilesTree =
        AnalysisTestUtil.createRunfilesTree(PathFragment.create("runfiles"), runfiles);

    expander.addSingleRunfilesTreeToInputs(
        runfilesTree,
        inputMap,
        new FakeActionInputFileCache(),
        PathMapper.NOOP,
        PathFragment.EMPTY_FRAGMENT);

    assertThat(inputMap)
        .containsExactly(PathFragment.create("runfiles/workspace/symlink"), artifact);
  }

  @Test
  public void testRunfilesRootSymlink() throws Exception {
    Artifact artifact =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))),
            fs.getPath("/root/dir/file"));
    Runfiles runfiles =
        new Runfiles.Builder("workspace")
            .addRootSymlink(PathFragment.create("symlink"), artifact)
            .build();
    RunfilesTree runfilesTree =
        AnalysisTestUtil.createRunfilesTree(PathFragment.create("runfiles"), runfiles);

    expander.addSingleRunfilesTreeToInputs(
        runfilesTree,
        inputMap,
        new FakeActionInputFileCache(),
        PathMapper.NOOP,
        PathFragment.EMPTY_FRAGMENT);

    assertThat(inputMap)
        .containsExactly(
            PathFragment.create("runfiles/symlink"),
            artifact,
            // If there's no other entry, Runfiles adds an empty file in the workspace to make sure
            // the directory gets created.
            PathFragment.create("runfiles/workspace/.runfile"),
            VirtualActionInput.EMPTY_MARKER);
  }

  @Test
  public void testRunfilesWithTreeArtifacts() throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("treeArtifact");
    TreeFileArtifact file1 = TreeFileArtifact.createTreeOutput(treeArtifact, "file1");
    TreeFileArtifact file2 = TreeFileArtifact.createTreeOutput(treeArtifact, "file2");
    FileSystemUtils.writeContentAsLatin1(file1.getPath(), "foo");
    FileSystemUtils.writeContentAsLatin1(file2.getPath(), "bar");

    TreeArtifactValue treeArtifactValue =
        TreeArtifactValue.newBuilder(treeArtifact)
            .putChild(file1, FileArtifactValue.createForTesting(file1.getPath()))
            .putChild(file2, FileArtifactValue.createForTesting(file2.getPath()))
            .build();

    FakeActionInputFileCache fakeActionInputFileCache = new FakeActionInputFileCache();
    fakeActionInputFileCache.putTreeArtifact(treeArtifact, treeArtifactValue);

    Runfiles runfiles = new Runfiles.Builder("workspace").addArtifact(treeArtifact).build();
    RunfilesTree runfilesTree =
        AnalysisTestUtil.createRunfilesTree(PathFragment.create("runfiles"), runfiles);

    expander.addSingleRunfilesTreeToInputs(
        runfilesTree,
        inputMap,
        fakeActionInputFileCache,
        PathMapper.NOOP,
        PathFragment.EMPTY_FRAGMENT);

    assertThat(inputMap)
        .containsExactly(
            PathFragment.create("runfiles/workspace/treeArtifact/file1"), file1,
            PathFragment.create("runfiles/workspace/treeArtifact/file2"), file2);
  }

  @Test
  public void testRunfilesWithTreeArtifacts_pathMapped() throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("treeArtifact");
    TreeFileArtifact file1 = TreeFileArtifact.createTreeOutput(treeArtifact, "file1");
    TreeFileArtifact file2 = TreeFileArtifact.createTreeOutput(treeArtifact, "file2");
    FileSystemUtils.writeContentAsLatin1(file1.getPath(), "foo");
    FileSystemUtils.writeContentAsLatin1(file2.getPath(), "bar");

    TreeArtifactValue treeArtifactValue =
        TreeArtifactValue.newBuilder(treeArtifact)
            .putChild(file1, FileArtifactValue.createForTesting(file1.getPath()))
            .putChild(file2, FileArtifactValue.createForTesting(file2.getPath()))
            .build();

    FakeActionInputFileCache fakeActionInputFileCache = new FakeActionInputFileCache();
    fakeActionInputFileCache.putTreeArtifact(treeArtifact, treeArtifactValue);

    Runfiles runfiles = new Runfiles.Builder("workspace").addArtifact(treeArtifact).build();
    RunfilesTree runfilesTree =
        AnalysisTestUtil.createRunfilesTree(
            PathFragment.create("bazel-out/k8-opt/bin/foo.runfiles"), runfiles);

    PathMapper pathMapper =
        execPath -> {
          // Replace the config segment "k8-opt" in "bazel-bin/k8-opt/bin" with a hash of the full
          // path to verify that the new paths are constructed by appending the child paths to the
          // mapped parent path, not by mapping the child paths directly.
          PathFragment runfilesPath = execPath.subFragment(3);
          String runfilesPathHash =
              DigestHashFunction.SHA256
                  .getHashFunction()
                  .hashString(runfilesPath.getPathString(), UTF_8)
                  .toString();
          return execPath
              .subFragment(0, 1)
              .getRelative(runfilesPathHash.substring(0, 8))
              .getRelative(execPath.subFragment(2));
        };

    expander.addSingleRunfilesTreeToInputs(
        runfilesTree, inputMap, fakeActionInputFileCache, pathMapper, PathFragment.EMPTY_FRAGMENT);

    assertThat(inputMap)
        .containsExactly(
            PathFragment.create("bazel-out/2c26b46b/bin/foo.runfiles/workspace/treeArtifact/file1"),
            file1,
            PathFragment.create("bazel-out/2c26b46b/bin/foo.runfiles/workspace/treeArtifact/file2"),
            file2);
  }

  @Test
  public void testRunfilesWithArchivedTreeArtifacts() throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("treeArtifact");
    ArchivedTreeArtifact archivedTreeArtifact = ArchivedTreeArtifact.createForTree(treeArtifact);
    TreeArtifactValue treeArtifactValue =
        TreeArtifactValue.newBuilder(treeArtifact)
            .setArchivedRepresentation(archivedTreeArtifact, FileArtifactValue.MISSING_FILE_MARKER)
            .build();

    FakeActionInputFileCache fakeActionInputFileCache = new FakeActionInputFileCache();
    fakeActionInputFileCache.putTreeArtifact(treeArtifact, treeArtifactValue);

    Runfiles runfiles = new Runfiles.Builder("workspace").addArtifact(treeArtifact).build();
    RunfilesTree runfilesTree =
        AnalysisTestUtil.createRunfilesTree(PathFragment.create("runfiles"), runfiles);

    expander = new SpawnInputExpander(/* expandArchivedTreeArtifacts= */ false);
    expander.addSingleRunfilesTreeToInputs(
        runfilesTree,
        inputMap,
        fakeActionInputFileCache,
        PathMapper.NOOP,
        PathFragment.EMPTY_FRAGMENT);

    assertThat(inputMap)
        .containsExactly(
            PathFragment.create("runfiles/workspace/treeArtifact"), archivedTreeArtifact);
  }

  @Test
  public void testRunfilesWithTreeArtifactsInSymlinks() throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("treeArtifact");
    TreeFileArtifact file1 = TreeFileArtifact.createTreeOutput(treeArtifact, "file1");
    TreeFileArtifact file2 = TreeFileArtifact.createTreeOutput(treeArtifact, "file2");
    FileSystemUtils.writeContentAsLatin1(file1.getPath(), "foo");
    FileSystemUtils.writeContentAsLatin1(file2.getPath(), "bar");
    TreeArtifactValue treeArtifactValue =
        TreeArtifactValue.newBuilder(treeArtifact)
            .putChild(file1, FileArtifactValue.createForTesting(file1.getPath()))
            .putChild(file2, FileArtifactValue.createForTesting(file2.getPath()))
            .build();

    FakeActionInputFileCache fakeActionInputFileCache = new FakeActionInputFileCache();
    fakeActionInputFileCache.putTreeArtifact(treeArtifact, treeArtifactValue);

    Runfiles runfiles =
        new Runfiles.Builder("workspace")
            .addSymlink(PathFragment.create("symlink"), treeArtifact)
            .build();
    RunfilesTree runfilesTree =
        AnalysisTestUtil.createRunfilesTree(PathFragment.create("runfiles"), runfiles);

    expander.addSingleRunfilesTreeToInputs(
        runfilesTree,
        inputMap,
        fakeActionInputFileCache,
        PathMapper.NOOP,
        PathFragment.EMPTY_FRAGMENT);

    assertThat(inputMap)
        .containsExactly(
            PathFragment.create("runfiles/workspace/symlink/file1"), file1,
            PathFragment.create("runfiles/workspace/symlink/file2"), file2);
  }

  @Test
  public void testTreeArtifactsInInputs() throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("treeArtifact");
    TreeFileArtifact file1 = TreeFileArtifact.createTreeOutput(treeArtifact, "file1");
    TreeFileArtifact file2 = TreeFileArtifact.createTreeOutput(treeArtifact, "file2");
    FileSystemUtils.writeContentAsLatin1(file1.getPath(), "foo");
    FileSystemUtils.writeContentAsLatin1(file2.getPath(), "bar");

    TreeArtifactValue treeArtifactValue =
        TreeArtifactValue.newBuilder(treeArtifact)
            .putChild(file1, FileArtifactValue.createForTesting(file1.getPath()))
            .putChild(file2, FileArtifactValue.createForTesting(file2.getPath()))
            .build();

    FakeActionInputFileCache fakeActionInputFileCache = new FakeActionInputFileCache();
    fakeActionInputFileCache.putTreeArtifact(treeArtifact, treeArtifactValue);

    Spawn spawn = new SpawnBuilder("/bin/echo", "Hello World").withInput(treeArtifact).build();
    Map<PathFragment, ActionInput> inputMappings =
        expander.getInputMapping(spawn, fakeActionInputFileCache, PathFragment.EMPTY_FRAGMENT);

    assertThat(inputMappings).hasSize(2);
    assertThat(inputMappings).containsEntry(PathFragment.create("out/treeArtifact/file1"), file1);
    assertThat(inputMappings).containsEntry(PathFragment.create("out/treeArtifact/file2"), file2);
  }

  private SpecialArtifact createTreeArtifact(String relPath) throws IOException {
    SpecialArtifact treeArtifact = createSpecialArtifact(relPath, SpecialArtifactType.TREE);
    treeArtifact.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
    return treeArtifact;
  }

  private SpecialArtifact createFilesetArtifact(String relPath) throws IOException {
    return createSpecialArtifact(relPath, SpecialArtifactType.FILESET);
  }

  private SpecialArtifact createSpecialArtifact(String relPath, SpecialArtifactType type)
      throws IOException {
    String outputSegment = "out";
    Path outputDir = execRoot.getRelative(outputSegment);
    Path outputPath = outputDir.getRelative(relPath);
    outputPath.createDirectoryAndParents();
    ArtifactRoot derivedRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, outputSegment);
    return SpecialArtifact.create(
        derivedRoot,
        derivedRoot.getExecPath().getRelative(derivedRoot.getRoot().relativize(outputPath)),
        ActionsTestUtil.NULL_ARTIFACT_OWNER,
        type);
  }

  @Test
  public void testEmptyManifest() {
    ImmutableMap<Artifact, FilesetOutputTree> filesetMappings =
        ImmutableMap.of(createFileset("out"), FilesetOutputTree.EMPTY);

    SpawnInputExpander.addFilesetManifests(filesetMappings, inputMap, PathFragment.EMPTY_FRAGMENT);

    assertThat(inputMap).isEmpty();
  }

  @Test
  public void fileset() {
    FilesetOutputSymlink link1 = filesetSymlink("foo/bar", "dir/file1");
    FilesetOutputSymlink link2 = filesetSymlink("foo/baz", "dir/file2");
    Artifact fileset = createFileset("out");
    ImmutableMap<Artifact, FilesetOutputTree> filesetMappings =
        ImmutableMap.of(fileset, FilesetOutputTree.create(ImmutableList.of(link1, link2)));

    SpawnInputExpander.addFilesetManifests(filesetMappings, inputMap, PathFragment.EMPTY_FRAGMENT);

    assertThat(inputMap)
        .containsExactly(
            PathFragment.create("out/foo/bar"), link1.target(),
            PathFragment.create("out/foo/baz"), link2.target());
  }

  private FilesetOutputSymlink filesetSymlink(String from, String to) {
    return new FilesetOutputSymlink(
        PathFragment.create(from),
        ActionsTestUtil.createArtifact(rootDir, to),
        FileArtifactValue.createForNormalFile(new byte[] {1}, null, 1));
  }

  private SpecialArtifact createFileset(String execPath) {
    return SpecialArtifact.create(
        rootDir,
        PathFragment.create(execPath),
        ActionsTestUtil.NULL_ARTIFACT_OWNER,
        SpecialArtifactType.FILESET);
  }
}
