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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link SymlinkForest}.
 */
@RunWith(JUnit4.class)
public class SymlinkForestTest {
  private FileSystem fileSystem;

  @Before
  public final void initializeFileSystem() {
    ManualClock clock = new ManualClock();
    fileSystem = new InMemoryFileSystem(clock);
  }

  @Test
  public void testDeleteTreesBelowNotPrefixed() throws IOException {
    /*
     * Build a directory tree that looks like:
     *   top-dir/
     *     file-1
     *     file-2
     *     a-dir/
     *       file-3
     *       inner-dir/
     *         link-1 => file-4
     *         dir-link => b-dir
     *   file-4
     */
    Path topDir = fileSystem.getPath("/top-dir");
    Path file1 = fileSystem.getPath("/top-dir/file-1");
    Path file2 = fileSystem.getPath("/top-dir/file-2");
    Path aDir = fileSystem.getPath("/top-dir/a-dir");
    Path bDir = fileSystem.getPath("/top-dir/b-dir");
    Path file3 = fileSystem.getPath("/top-dir/a-dir/file-3");
    Path innerDir = fileSystem.getPath("/top-dir/a-dir/inner-dir");
    Path link1 = fileSystem.getPath("/top-dir/a-dir/inner-dir/link-1");
    Path dirLink = fileSystem.getPath("/top-dir/a-dir/inner-dir/dir-link");
    Path file4 = fileSystem.getPath("/file-4");
    Path file5 = fileSystem.getPath("/top-dir/b-dir/file-5");

    topDir.createDirectory();
    FileSystemUtils.createEmptyFile(file1);
    FileSystemUtils.createEmptyFile(file2);
    aDir.createDirectory();
    bDir.createDirectory();
    FileSystemUtils.createEmptyFile(file3);
    innerDir.createDirectory();
    link1.createSymbolicLink(file4);  // simple symlink
    dirLink.createSymbolicLink(bDir);
    FileSystemUtils.createEmptyFile(file4);
    FileSystemUtils.createEmptyFile(file5);

    SymlinkForest.deleteTreesBelowNotPrefixed(topDir, "file-");
    assertThat(file1.exists()).isTrue();
    assertThat(file2.exists()).isTrue();
    assertThat(aDir.exists()).isFalse();
  }

  // Create package for external repo
  private PackageIdentifier createExternalPkg(Root root, String repo, String pkg)
      throws IOException, LabelSyntaxException {
    if (root != null) {
      Path repoRoot = root.getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME).getRelative(repo);
      repoRoot.getRelative(pkg).createDirectoryAndParents();
      FileSystemUtils.createEmptyFile(repoRoot.getRelative(pkg).getChild("file"));
    }
    return PackageIdentifier.create(RepositoryName.create("@" + repo), PathFragment.create(pkg));
  }

  // Create package for main repo
  private PackageIdentifier createMainPkg(Root repoRoot, String pkg)
      throws IOException, LabelSyntaxException {
    if (repoRoot != null) {
      repoRoot.getRelative(pkg).createDirectoryAndParents();
      FileSystemUtils.createEmptyFile(repoRoot.getRelative(pkg).getChild("file"));
    }
    return PackageIdentifier.create(RepositoryName.create("@"), PathFragment.create(pkg));
  }

  private void assertLinksTo(Path fromRoot, Root toRoot, String relpart) throws IOException {
    assertLinksTo(fromRoot.getRelative(relpart), toRoot.getRelative(relpart));
  }

  private void assertLinksTo(Path fromRoot, Path toRoot) throws IOException {
    assertWithMessage("stat: " + fromRoot.stat()).that(fromRoot.isSymbolicLink()).isTrue();
    assertThat(fromRoot.readSymbolicLink()).isEqualTo(toRoot.asFragment());
  }

  @Test
  public void testPlantSymlinkForest() throws Exception {
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    mainRepo.asPath().createDirectoryAndParents();
    linkRoot.createDirectoryAndParents();

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createMainPkg(mainRepo, "dir_main"), mainRepo)
            .put(createMainPkg(mainRepo, "dir_lib/pkg"), mainRepo)
            .put(createMainPkg(mainRepo, ""), mainRepo)
            .put(createExternalPkg(outputBase, "X", "dir_x/pkg"), outputBase)
            .put(createExternalPkg(outputBase, "Y", "dir_y/pkg"), outputBase)
            .put(createExternalPkg(outputBase, "Z", "dir_z/pkg"), outputBase)
            .build();

    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME).plantSymlinkForest();

    assertLinksTo(linkRoot, mainRepo, "dir_main");
    assertLinksTo(linkRoot, mainRepo, "dir_lib");
    // This file will be a copy on Windows, so just check if it exists instead of if it's a symlink.
    assertThat(linkRoot.getChild("file").exists()).isTrue();
    assertLinksTo(linkRoot, outputBase, LabelConstants.EXTERNAL_PATH_PREFIX + "/X");
    assertLinksTo(linkRoot, outputBase, LabelConstants.EXTERNAL_PATH_PREFIX + "/Y");
    assertLinksTo(linkRoot, outputBase, LabelConstants.EXTERNAL_PATH_PREFIX + "/Z");
  }

  @Test
  public void testPlantSymlinkForestForMainRepo() throws Exception {
    // For the main repo, plantSymlinkForest function should only link all files and dirs under
    // main repo root that're presented in packageRootMap.
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    linkRoot.createDirectoryAndParents();
    mainRepo.asPath().createDirectoryAndParents();
    mainRepo.getRelative("dir4").createDirectoryAndParents();
    FileSystemUtils.createEmptyFile(mainRepo.getRelative("file"));

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createMainPkg(mainRepo, "dir1/pkg/foo"), mainRepo)
            .put(createMainPkg(mainRepo, "dir2/pkg"), mainRepo)
            .put(createMainPkg(mainRepo, "dir3"), mainRepo)
            .put(createExternalPkg(outputBase, "X", "dir_x/pkg"), outputBase)
            .build();

    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME).plantSymlinkForest();

    assertLinksTo(linkRoot, mainRepo, "dir1");
    assertLinksTo(linkRoot, mainRepo, "dir2");
    assertLinksTo(linkRoot, mainRepo, "dir3");
    // dir4 and the file under main repo root should not be linked
    // because they are not presented in packageRootMap.
    assertThat(linkRoot.getChild("dir4").exists()).isFalse();
    assertThat(linkRoot.getChild("file").exists()).isFalse();
    assertLinksTo(linkRoot, outputBase, LabelConstants.EXTERNAL_PATH_PREFIX + "/X");
  }

  @Test
  public void testExternalPackage() throws Exception {
    Path linkRoot = fileSystem.getPath("/linkRoot");
    linkRoot.createDirectoryAndParents();

    Root root = Root.fromPath(fileSystem.getPath("/src"));
    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            // Virtual root, shouldn't actually be linked in.
            .put(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER, root)
            .build();

    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME).plantSymlinkForest();
    assertThat(linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX).exists()).isFalse();
  }
}
