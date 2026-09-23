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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.RunfileSymlinksMode;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link RunfilesTreeUpdater}. */
@RunWith(JUnit4.class)
public final class RunfilesTreeUpdaterTest {
  private static final String WORKSPACE_NAME = "_main";

  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final Path execRoot = fs.getPath("/execroot");
  private final ArtifactRoot outputRoot =
      ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "bin");
  private final PathFragment runfilesDirExecPath = PathFragment.create("bin/foo/bin.runfiles");
  private final Path runfilesDir = execRoot.getRelative(runfilesDirExecPath);
  private final Path inputManifest =
      execRoot.getRelative(RunfilesSupport.inputManifestExecPath(runfilesDirExecPath));
  private final Path outputManifest =
      execRoot.getRelative(RunfilesSupport.outputManifestExecPath(runfilesDirExecPath));

  private Artifact data;
  private Artifact other;

  @Before
  public void setUp() throws Exception {
    data = ActionsTestUtil.createArtifact(outputRoot, "foo/data.txt");
    other = ActionsTestUtil.createArtifact(outputRoot, "foo/other.txt");
    inputManifest.getParentDirectory().createDirectoryAndParents();
  }

  /** Writes the input manifest for a runfiles tree containing only the given runfile. */
  private RunfilesTree runfilesTree(Artifact runfile) throws IOException {
    PathFragment runfilesPath =
        PathFragment.create(WORKSPACE_NAME).getRelative(runfile.getRootRelativePath());
    FileSystemUtils.writeContent(
        inputManifest, UTF_8, String.format("%s %s\n", runfilesPath, runfile.getPath()));
    RunfilesTree tree = mock(RunfilesTree.class);
    when(tree.getExecPath()).thenReturn(runfilesDirExecPath);
    when(tree.getMapping()).thenReturn(ImmutableSortedMap.of(runfilesPath, runfile));
    when(tree.getSymlinksMode()).thenReturn(RunfileSymlinksMode.CREATE);
    when(tree.isBuildRunfileLinks()).thenReturn(true);
    when(tree.getWorkspaceName()).thenReturn(WORKSPACE_NAME);
    return tree;
  }

  private RunfilesTreeUpdater newUpdater(boolean runfilesTreesCreatedLazily) {
    RunfilesTreeUpdater updater = new RunfilesTreeUpdater(execRoot, SyscallCache.NO_CACHE);
    if (runfilesTreesCreatedLazily) {
      updater.setMaterializeBuiltRunfilesTrees();
    }
    return updater;
  }

  private Path runfilePath(Artifact runfile) {
    return runfilesDir.getRelative(WORKSPACE_NAME).getRelative(runfile.getRootRelativePath());
  }

  @Test
  public void updateRunfiles_createsSymlinksAndCopiesManifest() throws Exception {
    RunfilesTree tree = runfilesTree(data);
    RunfilesTreeUpdater updater = newUpdater(/* runfilesTreesCreatedLazily= */ true);
    assertThat(updater.isUpToDate(tree)).isFalse();

    updater.updateRunfiles(ImmutableList.of(tree));

    assertThat(runfilePath(data).readSymbolicLink()).isEqualTo(data.getPath().asFragment());
    assertThat(outputManifest.isSymbolicLink()).isFalse();
    assertThat(FileSystemUtils.readContent(outputManifest))
        .isEqualTo(FileSystemUtils.readContent(inputManifest));
    assertThat(updater.isUpToDate(tree)).isTrue();
  }

  @Test
  public void updateRunfiles_upToDate_leavesTreeAlone() throws Exception {
    RunfilesTree tree = runfilesTree(data);
    newUpdater(/* runfilesTreesCreatedLazily= */ true).updateRunfiles(ImmutableList.of(tree));
    // Syncing the tree would remove this file, which isn't part of the runfiles.
    Path canary = runfilesDir.getRelative(WORKSPACE_NAME).getRelative("canary");
    FileSystemUtils.createEmptyFile(canary);

    // A fresh updater, as created for every command, has no memory of the previous update.
    RunfilesTreeUpdater updater = newUpdater(/* runfilesTreesCreatedLazily= */ true);
    assertThat(updater.isUpToDate(tree)).isTrue();
    updater.updateRunfiles(ImmutableList.of(tree));

    assertThat(canary.exists(Symlinks.NOFOLLOW)).isTrue();
    assertThat(runfilePath(data).readSymbolicLink()).isEqualTo(data.getPath().asFragment());
  }

  @Test
  public void updateRunfiles_manifestChanged_syncsTree() throws Exception {
    newUpdater(/* runfilesTreesCreatedLazily= */ true)
        .updateRunfiles(ImmutableList.of(runfilesTree(data)));
    Path canary = runfilesDir.getRelative(WORKSPACE_NAME).getRelative("canary");
    FileSystemUtils.createEmptyFile(canary);

    // Rewrites the input manifest.
    RunfilesTree tree = runfilesTree(other);
    RunfilesTreeUpdater updater = newUpdater(/* runfilesTreesCreatedLazily= */ true);
    assertThat(updater.isUpToDate(tree)).isFalse();
    updater.updateRunfiles(ImmutableList.of(tree));

    assertThat(canary.exists(Symlinks.NOFOLLOW)).isFalse();
    assertThat(runfilePath(data).exists(Symlinks.NOFOLLOW)).isFalse();
    assertThat(runfilePath(other).readSymbolicLink()).isEqualTo(other.getPath().asFragment());
    assertThat(FileSystemUtils.readContent(outputManifest))
        .isEqualTo(FileSystemUtils.readContent(inputManifest));
    assertThat(updater.isUpToDate(tree)).isTrue();
  }

  @Test
  public void updateRunfiles_minimalTreeWithLinkedManifest_isNotUpToDate() throws Exception {
    RunfilesTree tree = runfilesTree(data);
    // Simulate the minimal runfiles directory created by SymlinkTreeAction when the symlinks are
    // created lazily (or with --noenable_runfiles).
    runfilesDir.createDirectoryAndParents();
    new SymlinkTreeHelper(inputManifest, outputManifest, runfilesDir, WORKSPACE_NAME)
        .createMinimalRunfilesDirectory();
    assertThat(outputManifest.isSymbolicLink()).isTrue();

    RunfilesTreeUpdater updater = newUpdater(/* runfilesTreesCreatedLazily= */ true);
    assertThat(updater.isUpToDate(tree)).isFalse();
    updater.updateRunfiles(ImmutableList.of(tree));

    assertThat(runfilePath(data).readSymbolicLink()).isEqualTo(data.getPath().asFragment());
    assertThat(outputManifest.isSymbolicLink()).isFalse();
    assertThat(updater.isUpToDate(tree)).isTrue();
  }

  @Test
  public void updateRunfiles_buildRunfileLinks_skippedUnlessCreatedLazily() throws Exception {
    RunfilesTree tree = runfilesTree(data);

    newUpdater(/* runfilesTreesCreatedLazily= */ false).updateRunfiles(ImmutableList.of(tree));

    assertThat(runfilesDir.exists()).isFalse();
  }
}
