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
import static com.google.devtools.build.lib.testutil.TestConstants.WORKSPACE_NAME;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link SymlinkTreeHelper}. */
@RunWith(TestParameterInjector.class)
public final class SymlinkTreeHelperTest {

  private enum TreeType {
    RUNFILES,
    FILESET
  }

  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final Path execRoot = fs.getPath("/execroot");
  private final ArtifactRoot outputRoot =
      ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out");

  @Before
  public void setUp() throws Exception {
    outputRoot.getRoot().asPath().createDirectoryAndParents();
  }

  @Test
  public void processFilesetLinks() {
    Artifact target1 = ActionsTestUtil.createArtifact(outputRoot, "target1");
    Artifact target2 = ActionsTestUtil.createArtifact(outputRoot, "target2");
    FileArtifactValue metadata =
        FileArtifactValue.createForNormalFile(new byte[] {1, 2, 3, 4}, null, 10);
    FilesetOutputSymlink link1 =
        new FilesetOutputSymlink(PathFragment.create("from1"), target1, metadata);
    FilesetOutputSymlink link2 =
        new FilesetOutputSymlink(PathFragment.create("from2"), target2, metadata);

    Map<PathFragment, PathFragment> symlinks =
        SymlinkTreeHelper.processFilesetLinks(ImmutableList.of(link1, link2), "workspace");
    assertThat(symlinks)
        .containsExactly(
            PathFragment.create("workspace/from1"),
            target1.getPath().asFragment(),
            PathFragment.create("workspace/from2"),
            target2.getPath().asFragment());
  }

  @Test
  public void createSymlinks(@TestParameter TreeType treeType, @TestParameter boolean replace)
      throws Exception {
    Path treeRoot = execRoot.getRelative("foo.runfiles");
    Path inputManifestPath = execRoot.getRelative("foo.runfiles_manifest");
    Path outputManifestPath = execRoot.getRelative("foo.runfiles/MANIFEST");
    SymlinkTreeHelper helper =
        new SymlinkTreeHelper(inputManifestPath, outputManifestPath, treeRoot, WORKSPACE_NAME);

    Artifact file = ActionsTestUtil.createArtifact(outputRoot, "file");
    Artifact symlink = ActionsTestUtil.createUnresolvedSymlinkArtifact(outputRoot, "symlink");

    FileSystemUtils.writeContent(file.getPath(), UTF_8, "content");
    FileSystemUtils.ensureSymbolicLink(symlink.getPath(), "/path/to/target");

    Path treeWorkspace = treeRoot.getRelative(WORKSPACE_NAME);
    Path treeEmpty = treeWorkspace.getRelative("empty");
    Path treeFile = treeWorkspace.getRelative("file");
    Path treeSymlink = treeWorkspace.getRelative("symlink");
    Path treeMissing = treeWorkspace.getRelative("missing");

    if (replace) {
      treeEmpty.createDirectoryAndParents();
      treeFile.createDirectoryAndParents();
      treeSymlink.createDirectoryAndParents();
      treeMissing.createDirectoryAndParents();
      treeWorkspace.chmod(000);
    }

    switch (treeType) {
      case RUNFILES -> {
        HashMap<PathFragment, Artifact> symlinkMap = new HashMap<>();
        symlinkMap.put(PathFragment.create(WORKSPACE_NAME + "/empty"), null);
        symlinkMap.put(PathFragment.create(WORKSPACE_NAME + "/file"), file);
        symlinkMap.put(PathFragment.create(WORKSPACE_NAME + "/symlink"), symlink);

        helper.createRunfilesSymlinks(symlinkMap);
      }
      case FILESET -> {
        HashMap<PathFragment, PathFragment> symlinkMap = new HashMap<>();
        symlinkMap.put(PathFragment.create(WORKSPACE_NAME + "/empty"), null);
        symlinkMap.put(PathFragment.create(WORKSPACE_NAME + "/file"), file.getPath().asFragment());
        symlinkMap.put(
            PathFragment.create(WORKSPACE_NAME + "/symlink"),
            PathFragment.create("/path/to/target"));

        helper.createFilesetSymlinks(symlinkMap);
      }
    }

    assertThat(treeRoot.isDirectory()).isTrue();
    assertThat(treeWorkspace.isDirectory()).isTrue();
    assertThat(treeEmpty.isFile()).isTrue();
    assertThat(FileSystemUtils.readContent(treeEmpty)).isEmpty();
    assertThat(treeFile.isSymbolicLink()).isTrue();
    assertThat(treeFile.readSymbolicLink()).isEqualTo(file.getPath().asFragment());
    assertThat(treeSymlink.isSymbolicLink()).isTrue();
    assertThat(treeSymlink.readSymbolicLink()).isEqualTo(PathFragment.create("/path/to/target"));
    assertThat(treeMissing.exists()).isFalse();
  }
}
