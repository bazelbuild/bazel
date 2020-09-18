// Copyright 2019 The Bazel Authors. All rights reserved.
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
//
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link SymlinkForest}. */
@RunWith(JUnit4.class)
public class SymlinkForestTest {
  private FileSystem fileSystem;

  private Path topDir;
  private Path file1;
  private Path file2;
  private Path aDir;

  // The execution root.
  private Path linkRoot;

  @Before
  public final void initializeFileSystem() throws Exception {
    ManualClock clock = new ManualClock();
    fileSystem = new InMemoryFileSystem(clock, DigestHashFunction.SHA256);
    linkRoot = fileSystem.getPath("/linkRoot");
    linkRoot.createDirectoryAndParents();
  }

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
  private void createTestDirectoryTree() throws IOException {
    topDir = fileSystem.getPath("/top-dir");
    file1 = fileSystem.getPath("/top-dir/file-1");
    file2 = fileSystem.getPath("/top-dir/file-2");
    aDir = fileSystem.getPath("/top-dir/a-dir");
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
    link1.createSymbolicLink(file4); // simple symlink
    dirLink.createSymbolicLink(bDir);
    FileSystemUtils.createEmptyFile(file4);
    FileSystemUtils.createEmptyFile(file5);
  }

  private static PathFragment longestPathPrefix(String path, String... prefixStrs) {
    ImmutableSet.Builder<PackageIdentifier> prefixes = ImmutableSet.builder();
    for (String prefix : prefixStrs) {
      prefixes.add(PackageIdentifier.createInMainRepo(prefix));
    }
    PackageIdentifier longest =
        SymlinkForest.longestPathPrefix(PackageIdentifier.createInMainRepo(path), prefixes.build());
    return longest != null ? longest.getPackageFragment() : null;
  }

  @Test
  public void testLongestPathPrefix() {
    PathFragment a = PathFragment.create("A");
    assertThat(longestPathPrefix("A/b", "A", "B")).isEqualTo(a); // simple parent
    assertThat(longestPathPrefix("A", "A", "B")).isEqualTo(a); // self
    assertThat(longestPathPrefix("A/B/c", "A", "A/B"))
        .isEqualTo(a.getRelative("B")); // want longest
    assertThat(longestPathPrefix("C/b", "A", "B")).isNull(); // not found in other parents
    assertThat(longestPathPrefix("A", "A/B", "B")).isNull(); // not found in child
    assertThat(longestPathPrefix("A/B/C/d/e/f.h", "A/B/C", "B/C/d"))
        .isEqualTo(a.getRelative("B/C"));
    assertThat(longestPathPrefix("A/f.h", "", "B/C/d")).isEqualTo(PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void testDeleteTreesBelowNotPrefixed() throws IOException {
    createTestDirectoryTree();
    new SymlinkForest(ImmutableMap.of(), topDir, "").deleteTreesBelowNotPrefixed(topDir, "file-");
    assertThat(file1.exists()).isTrue();
    assertThat(file2.exists()).isTrue();
    assertThat(aDir.exists()).isFalse();
  }

  // Create same package under two different roots
  private static PackageIdentifier createPkg(Root rootA, Root rootB, String pkg)
      throws IOException {
    if (rootA != null) {
      rootA.getRelative(pkg).createDirectoryAndParents();
      FileSystemUtils.createEmptyFile(rootA.getRelative(pkg).getChild("file"));
    }
    if (rootB != null) {
      rootB.getRelative(pkg).createDirectoryAndParents();
      FileSystemUtils.createEmptyFile(rootB.getRelative(pkg).getChild("file"));
    }
    return PackageIdentifier.createInMainRepo(pkg);
  }

  // Create package for external repo
  private static PackageIdentifier createExternalPkg(Root root, String repo, String pkg)
      throws IOException, LabelSyntaxException {
    if (root != null) {
      Path repoRoot = root.getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME).getRelative(repo);
      repoRoot.getRelative(pkg).createDirectoryAndParents();
      FileSystemUtils.createEmptyFile(repoRoot.getRelative(pkg).getChild("file"));
    }
    return PackageIdentifier.create(RepositoryName.create("@" + repo), PathFragment.create(pkg));
  }

  // Create package for main repo
  private static PackageIdentifier createMainPkg(Root repoRoot, String pkg)
      throws IOException, LabelSyntaxException {
    if (repoRoot != null) {
      repoRoot.getRelative(pkg).createDirectoryAndParents();
      FileSystemUtils.createEmptyFile(repoRoot.getRelative(pkg).getChild("file"));
    }
    return PackageIdentifier.create(RepositoryName.create("@"), PathFragment.create(pkg));
  }

  private static void assertLinksTo(Path fromRoot, Root toRoot, String relpart) throws IOException {
    assertLinksTo(fromRoot.getRelative(relpart), toRoot.getRelative(relpart));
  }

  private static void assertLinksTo(Path fromRoot, Path toRoot) throws IOException {
    assertWithMessage("stat: " + fromRoot.stat()).that(fromRoot.isSymbolicLink()).isTrue();
    assertThat(fromRoot.readSymbolicLink()).isEqualTo(toRoot.asFragment());
  }

  private static void assertIsDir(Path root, String relpart) {
    assertThat(root.getRelative(relpart).isDirectory(Symlinks.NOFOLLOW)).isTrue();
  }

  @Test
  public void testPlantLinkForestWithMultiplePackagePath() throws Exception {
    Root rootA = Root.fromPath(fileSystem.getPath("/A"));
    Root rootB = Root.fromPath(fileSystem.getPath("/B"));

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createPkg(rootA, rootB, "pkgA"), rootA)
            .put(createPkg(rootA, rootB, "dir1/pkgA"), rootA)
            .put(createPkg(rootA, rootB, "dir1/pkgB"), rootB)
            .put(createPkg(rootA, rootB, "dir2/pkg"), rootA)
            .put(createPkg(rootA, rootB, "dir2/pkg/pkg"), rootB)
            .put(createPkg(rootA, rootB, "pkgB"), rootB)
            .put(createPkg(rootA, rootB, "pkgB/dir/pkg"), rootA)
            .put(createPkg(rootA, rootB, "pkgB/pkg"), rootA)
            .put(createPkg(rootA, rootB, "pkgB/pkg/pkg"), rootA)
            .build();
    createPkg(rootA, rootB, "pkgB/dir"); // create a file in there

    Path linkRoot = fileSystem.getPath("/linkRoot");
    linkRoot.createDirectoryAndParents();
    ImmutableList<Path> plantedSymlinks =
        new SymlinkForest(
                packageRootMap,
                linkRoot,
                TestConstants.PRODUCT_NAME,
                ImmutableSortedSet.of(),
                false)
            .plantSymlinkForest();

    assertLinksTo(linkRoot, rootA, "pkgA");
    assertIsDir(linkRoot, "dir1");
    assertLinksTo(linkRoot, rootA, "dir1/pkgA");
    assertLinksTo(linkRoot, rootB, "dir1/pkgB");
    assertIsDir(linkRoot, "dir2");
    assertIsDir(linkRoot, "dir2/pkg");
    assertLinksTo(linkRoot, rootA, "dir2/pkg/file");
    assertLinksTo(linkRoot, rootB, "dir2/pkg/pkg");
    assertIsDir(linkRoot, "pkgB");
    assertIsDir(linkRoot, "pkgB/dir");
    assertLinksTo(linkRoot, rootB, "pkgB/dir/file");
    assertLinksTo(linkRoot, rootA, "pkgB/dir/pkg");
    assertLinksTo(linkRoot, rootA, "pkgB/pkg");
    assertThat(plantedSymlinks)
        .containsExactly(
            linkRoot.getRelative("pkgA"),
            linkRoot.getRelative("dir1/pkgA"),
            linkRoot.getRelative("dir1/pkgB"),
            linkRoot.getRelative("dir2/pkg/file"),
            linkRoot.getRelative("dir2/pkg/pkg"),
            linkRoot.getRelative("pkgB/file"),
            linkRoot.getRelative("pkgB/dir/file"),
            linkRoot.getRelative("pkgB/dir/pkg"),
            linkRoot.getRelative("pkgB/pkg"));
  }

  @Test
  public void testTopLevelPackage() throws Exception {
    Root rootX = Root.fromPath(fileSystem.getPath("/X"));
    Root rootY = Root.fromPath(fileSystem.getPath("/Y"));
    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createPkg(rootX, rootY, ""), rootX)
            .put(createPkg(rootX, rootY, "foo"), rootX)
            .build();

    ImmutableList<Path> plantedSymlinks =
        new SymlinkForest(
                packageRootMap,
                linkRoot,
                TestConstants.PRODUCT_NAME,
                ImmutableSortedSet.of(),
                false)
            .plantSymlinkForest();
    assertLinksTo(linkRoot, rootX, "file");
    assertThat(plantedSymlinks)
        .containsExactly(linkRoot.getRelative("file"), linkRoot.getRelative("foo"));
  }

  @Test
  public void testPlantSymlinkForest() throws Exception {
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Root externalSourceRoot =
        Root.fromPath(outputBase.asPath().getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    mainRepo.asPath().createDirectoryAndParents();
    linkRoot.createDirectoryAndParents();

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createMainPkg(mainRepo, "dir_main"), mainRepo)
            .put(createMainPkg(mainRepo, "dir_lib/pkg"), mainRepo)
            .put(createMainPkg(mainRepo, ""), mainRepo)
            // Remote repo without top-level package.
            .put(createExternalPkg(outputBase, "X", "dir_x/pkg"), externalSourceRoot)
            // Remote repo with and without top-level package.
            .put(createExternalPkg(outputBase, "Y", ""), externalSourceRoot)
            .put(createExternalPkg(outputBase, "Y", "dir_y/pkg"), externalSourceRoot)
            // Only top-level pkg.
            .put(createExternalPkg(outputBase, "Z", ""), externalSourceRoot)
            .build();

    ImmutableList<Path> plantedSymlinks =
        new SymlinkForest(
                packageRootMap,
                linkRoot,
                TestConstants.PRODUCT_NAME,
                ImmutableSortedSet.of(),
                false)
            .plantSymlinkForest();

    assertLinksTo(linkRoot, mainRepo, "dir_main");
    assertLinksTo(linkRoot, mainRepo, "dir_lib");
    assertLinksTo(linkRoot, mainRepo, "file");
    assertLinksTo(linkRoot, outputBase, LabelConstants.EXTERNAL_PATH_PREFIX + "/X");
    assertLinksTo(linkRoot, outputBase, LabelConstants.EXTERNAL_PATH_PREFIX + "/Y");
    assertLinksTo(linkRoot, outputBase, LabelConstants.EXTERNAL_PATH_PREFIX + "/Z");
    assertThat(
            linkRoot
                .getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME)
                .getRelative("Y/file")
                .exists())
        .isTrue();
    assertThat(
            linkRoot
                .getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME)
                .getRelative("Z/file")
                .exists())
        .isTrue();
    assertThat(plantedSymlinks)
        .containsExactly(
            linkRoot.getRelative("dir_main"),
            linkRoot.getRelative("dir_lib"),
            linkRoot.getRelative("file"),
            linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/X"),
            linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/Y"),
            linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/Z"));
  }

  @Test
  public void test_withSiblingRepoLayout_plantSymlinkForest() throws Exception {
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Root externalSourceRoot =
        Root.fromPath(outputBase.asPath().getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    mainRepo.asPath().createDirectoryAndParents();
    linkRoot.createDirectoryAndParents();

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createMainPkg(mainRepo, "dir_main"), mainRepo)
            .put(createMainPkg(mainRepo, "dir_lib/pkg"), mainRepo)
            .put(createMainPkg(mainRepo, ""), mainRepo)
            // Remote repo without top-level package.
            .put(createExternalPkg(outputBase, "X", "dir_x/pkg"), externalSourceRoot)
            // Remote repo with and without top-level package.
            .put(createExternalPkg(outputBase, "Y", ""), externalSourceRoot)
            .put(createExternalPkg(outputBase, "Y", "dir_y/pkg"), externalSourceRoot)
            // Only top-level pkg.
            .put(createExternalPkg(outputBase, "Z", ""), externalSourceRoot)
            .build();

    ImmutableList<Path> plantedSymlinks =
        new SymlinkForest(
                packageRootMap, linkRoot, TestConstants.PRODUCT_NAME, ImmutableSortedSet.of(), true)
            .plantSymlinkForest();

    // Expected sibling repository layout (X, Y and Z are siblings of ws_name):
    //
    // .
    // ├── execroot
    // │   ├── ws_name { ... }
    // │   ├── X -> external/X
    // │   ├── Y -> external/Y
    // │   └── Z -> external/Z
    // └── external
    //     ├── X
    //     ├── Y
    //     └── Z

    assertLinksTo(linkRoot, mainRepo, "dir_main");
    assertLinksTo(linkRoot, mainRepo, "dir_lib");
    assertLinksTo(linkRoot, mainRepo, "file");
    assertLinksTo(
        linkRoot.getParentDirectory().getRelative("X"),
        outputBase.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/X"));
    assertLinksTo(
        linkRoot.getParentDirectory().getRelative("Y"),
        outputBase.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/Y"));
    assertLinksTo(
        linkRoot.getParentDirectory().getRelative("Z"),
        outputBase.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/Z"));
    assertThat(linkRoot.getParentDirectory().getRelative("Y/file").exists()).isTrue();
    assertThat(linkRoot.getParentDirectory().getRelative("Z/file").exists()).isTrue();
    assertThat(plantedSymlinks)
        .containsExactly(
            linkRoot.getRelative("dir_main"),
            linkRoot.getRelative("dir_lib"),
            linkRoot.getRelative("file"),
            linkRoot.getParentDirectory().getRelative("X"),
            linkRoot.getParentDirectory().getRelative("Y"),
            linkRoot.getParentDirectory().getRelative("Z"));
  }

  @Test
  public void testPlantSymlinkForestForMainRepo() throws Exception {
    // For the main repo, plantSymlinkForest function should only link all files and dirs under
    // main repo root that're presented in packageRootMap.
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Root externalSourceRoot =
        Root.fromPath(outputBase.asPath().getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME));
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
            .put(createExternalPkg(outputBase, "X", "dir_x/pkg"), externalSourceRoot)
            .build();

    ImmutableList<Path> plantedSymlinks =
        new SymlinkForest(
                packageRootMap,
                linkRoot,
                TestConstants.PRODUCT_NAME,
                ImmutableSortedSet.of(),
                false)
            .plantSymlinkForest();

    assertLinksTo(linkRoot, mainRepo, "dir1");
    assertLinksTo(linkRoot, mainRepo, "dir2");
    assertLinksTo(linkRoot, mainRepo, "dir3");
    // dir4 and the file under main repo root should not be linked
    // because they are not presented in packageRootMap.
    assertThat(linkRoot.getChild("dir4").exists()).isFalse();
    assertThat(linkRoot.getChild("file").exists()).isFalse();
    assertLinksTo(linkRoot, outputBase, LabelConstants.EXTERNAL_PATH_PREFIX + "/X");
    assertThat(plantedSymlinks)
        .containsExactly(
            linkRoot.getRelative("dir1"),
            linkRoot.getRelative("dir2"),
            linkRoot.getRelative("dir3"),
            linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/X"));
  }

  @Test
  public void test_withSubdirRepoLayout_testExternalDirInMainRepoIsIgnored1() throws Exception {
    // Test external/ is ignored even when packages like "//external/foo" is specified.
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Root externalSourceRoot =
        Root.fromPath(outputBase.asPath().getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    linkRoot.createDirectoryAndParents();
    mainRepo.asPath().createDirectoryAndParents();

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createMainPkg(mainRepo, "dir1/pkg/foo"), mainRepo)
            .put(createMainPkg(mainRepo, "dir2/pkg"), mainRepo)
            .put(createMainPkg(mainRepo, "dir3"), mainRepo)
            // external/ should not be linked even we have "//external/foo" package
            .put(createMainPkg(mainRepo, "external/foo"), mainRepo)
            .put(createExternalPkg(outputBase, "X", "dir_x/pkg"), externalSourceRoot)
            .build();

    ImmutableList<Path> plantedSymlinks =
        new SymlinkForest(
                packageRootMap,
                linkRoot,
                TestConstants.PRODUCT_NAME,
                ImmutableSortedSet.of(),
                false)
            .plantSymlinkForest();

    assertLinksTo(linkRoot, mainRepo, "dir1");
    assertLinksTo(linkRoot, mainRepo, "dir2");
    assertLinksTo(linkRoot, mainRepo, "dir3");
    assertLinksTo(linkRoot, outputBase, LabelConstants.EXTERNAL_PATH_PREFIX + "/X");
    assertThat(outputBase.getRelative("external/foo").exists()).isFalse();
    assertThat(plantedSymlinks)
        .containsExactly(
            linkRoot.getRelative("dir1"),
            linkRoot.getRelative("dir2"),
            linkRoot.getRelative("dir3"),
            linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/X"));
  }

  @Test
  public void test_withSubDirRepoLayout_testExternalDirInMainRepoIsIgnored2() throws Exception {
    // Test external/ is ignored when root package "//:" is specified.
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Root externalSourceRoot =
        Root.fromPath(outputBase.asPath().getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    linkRoot.createDirectoryAndParents();
    mainRepo.asPath().createDirectoryAndParents();
    mainRepo.getRelative("dir3").createDirectoryAndParents();
    mainRepo.getRelative("external/foo").createDirectoryAndParents();

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createMainPkg(mainRepo, "dir1/pkg/foo"), mainRepo)
            .put(createMainPkg(mainRepo, "dir2/pkg"), mainRepo)
            // Empty package will cause every top-level files to be linked, except external/
            .put(createMainPkg(mainRepo, ""), mainRepo)
            .put(createExternalPkg(outputBase, "X", "dir_x/pkg"), externalSourceRoot)
            .build();

    ImmutableList<Path> plantedSymlinks =
        new SymlinkForest(
                packageRootMap,
                linkRoot,
                TestConstants.PRODUCT_NAME,
                ImmutableSortedSet.of(),
                false)
            .plantSymlinkForest();

    assertLinksTo(linkRoot, mainRepo, "dir1");
    assertLinksTo(linkRoot, mainRepo, "dir2");
    assertLinksTo(linkRoot, mainRepo, "dir3");
    assertLinksTo(linkRoot, outputBase, LabelConstants.EXTERNAL_PATH_PREFIX + "/X");
    assertThat(outputBase.getRelative("external/foo").exists()).isFalse();
    assertThat(plantedSymlinks)
        .containsExactly(
            linkRoot.getRelative("dir1"),
            linkRoot.getRelative("dir2"),
            linkRoot.getRelative("dir3"),
            linkRoot.getRelative("file"),
            linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/X"));
  }

  @Test
  public void test_withSiblingRepoLayout_testExternalDirInMainRepoExists() throws Exception {
    // Test external/ is ignored even when packages like "//external/foo" is specified.
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Root externalSourceRoot =
        Root.fromPath(outputBase.asPath().getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    linkRoot.createDirectoryAndParents();
    mainRepo.asPath().createDirectoryAndParents();

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createMainPkg(mainRepo, "dir1/pkg/foo"), mainRepo)
            .put(createMainPkg(mainRepo, "dir2/pkg"), mainRepo)
            .put(createMainPkg(mainRepo, "dir3"), mainRepo)
            // external/ should not be linked even we have "//external/foo" package
            .put(createMainPkg(mainRepo, "external/foo"), mainRepo)
            .put(createExternalPkg(outputBase, "X", "dir_x/pkg"), externalSourceRoot)
            .build();

    ImmutableList<Path> plantedSymlinks =
        new SymlinkForest(
                packageRootMap, linkRoot, TestConstants.PRODUCT_NAME, ImmutableSortedSet.of(), true)
            .plantSymlinkForest();

    // Expected output base layout with sibling repositories in the execroot where
    // ws_name and X are siblings:
    //
    // /ob
    // ├── execroot
    // │   ├── ws_name
    // │   │   ├── dir1
    // │   │   │   └── pkg
    // │   │   │       └── foo -> /my_repo/dir1/pkg/foo
    // │   │   ├── dir2
    // │   │   │   └── pkg -> /my_repo/dir2/pkg
    // │   │   ├── dir3 -> /my_repo/dir3
    // │   │   └── external -> /my_repo/external
    // │   └── X -> /ob/external/X
    // └── external
    //     └── X

    assertLinksTo(linkRoot, mainRepo, "dir1");
    assertLinksTo(linkRoot, mainRepo, "dir2");
    assertLinksTo(linkRoot, mainRepo, "dir3");

    assertThat(
            outputBase.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX).getRelative("X").exists())
        .isTrue();
    assertThat(outputBase.getRelative("execroot/X").exists()).isTrue();
    assertLinksTo(
        linkRoot.getParentDirectory().getRelative("X"), // Sibling of the main repo.
        outputBase.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX).getRelative("X"));

    assertThat(linkRoot.getRelative("external/foo").exists()).isTrue();

    assertThat(plantedSymlinks)
        .containsExactly(
            linkRoot.getRelative("dir1"),
            linkRoot.getRelative("dir2"),
            linkRoot.getRelative("dir3"),
            linkRoot.getRelative("external"), // Symlinked to the main repo's top level external dir
            linkRoot.getParentDirectory().getRelative("X")); // Symlinked to /ob/external/X
  }

  @Test
  public void test_withSiblingRepoLayoutAndRootPackageInRoots_testExternalDirInMainRepoExists()
      throws Exception {
    // Test external/ is ignored when root package "//:" is specified.
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Root externalSourceRoot =
        Root.fromPath(outputBase.asPath().getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    linkRoot.createDirectoryAndParents();
    mainRepo.asPath().createDirectoryAndParents();

    mainRepo.getRelative("external/foo").createDirectoryAndParents();

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createMainPkg(mainRepo, ""), mainRepo)
            .put(createExternalPkg(outputBase, "X", "dir_x/pkg"), externalSourceRoot)
            .build();

    ImmutableList<Path> plantedSymlinks =
        new SymlinkForest(
                packageRootMap, linkRoot, TestConstants.PRODUCT_NAME, ImmutableSortedSet.of(), true)
            .plantSymlinkForest();

    // Expected output base layout with sibling repositories in the execroot where
    // ws_name and X are siblings:
    //
    // /ob
    // ├── execroot
    // │   ├── ws_name
    // │   │   └── external -> /my_repo/external
    // │   └── X -> /ob/external/X
    // └── external
    //     └── X

    assertThat(
            outputBase.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX).getRelative("X").exists())
        .isTrue();
    assertThat(outputBase.getRelative("execroot/X").exists()).isTrue();
    assertLinksTo(
        linkRoot.getParentDirectory().getRelative("X"), // Sibling of the main repo.
        outputBase.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX).getRelative("X"));

    assertThat(linkRoot.getRelative("external/foo").exists()).isTrue();

    assertThat(plantedSymlinks)
        .containsExactly(
            linkRoot.getParentDirectory().getRelative("X"),
            linkRoot.getRelative("file"), // created by createMainPkg test setup
            linkRoot.getRelative("external") // symlink to main repo's top level external directory
            );
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

    ImmutableList<Path> plantedSymlinks =
        new SymlinkForest(
                packageRootMap,
                linkRoot,
                TestConstants.PRODUCT_NAME,
                ImmutableSortedSet.of(),
                false)
            .plantSymlinkForest();
    assertThat(linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX).exists()).isFalse();
    assertThat(plantedSymlinks).isEmpty();
  }

  @Test
  public void testNotSymlinkedDirectoriesInExecRootAllInMainRepo() throws Exception {
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Root externalSourceRoot =
        Root.fromPath(outputBase.asPath().getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    linkRoot.createDirectoryAndParents();
    mainRepo.asPath().createDirectoryAndParents();
    mainRepo.getRelative("dir3").createDirectoryAndParents();
    mainRepo.getRelative("build").createDirectoryAndParents();

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createMainPkg(mainRepo, "dir1/pkg/foo"), mainRepo)
            .put(createMainPkg(mainRepo, "dir2/pkg"), mainRepo)
            // Empty package will cause every top-level files to be linked, except external/
            .put(createMainPkg(mainRepo, ""), mainRepo)
            .put(createExternalPkg(outputBase, "X", "dir_x/pkg"), externalSourceRoot)
            .build();

    ImmutableList<Path> plantedSymlinks =
        new SymlinkForest(
                packageRootMap,
                linkRoot,
                TestConstants.PRODUCT_NAME,
                ImmutableSortedSet.of("build"),
                false)
            .plantSymlinkForest();

    assertLinksTo(linkRoot, mainRepo, "dir1");
    assertLinksTo(linkRoot, mainRepo, "dir2");
    assertLinksTo(linkRoot, mainRepo, "dir3");
    assertLinksTo(linkRoot, outputBase, LabelConstants.EXTERNAL_PATH_PREFIX + "/X");
    assertThat(linkRoot.getChild("build").exists()).isFalse();
    assertThat(plantedSymlinks)
        .containsExactly(
            linkRoot.getRelative("file"),
            linkRoot.getRelative("dir1"),
            linkRoot.getRelative("dir2"),
            linkRoot.getRelative("dir3"),
            linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/X"));
  }

  @Test
  public void testNotSymlinkedDirectoriesNotDeletedBetweenCommands() throws Exception {
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    linkRoot.createDirectoryAndParents();
    mainRepo.asPath().createDirectoryAndParents();
    mainRepo.getRelative("build").createDirectoryAndParents();

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createMainPkg(mainRepo, "dir1/pkg"), mainRepo)
            // Empty package will cause every top-level files to be linked, except external/
            .put(createMainPkg(mainRepo, ""), mainRepo)
            .build();

    SymlinkForest symlinkForest =
        new SymlinkForest(
            packageRootMap,
            linkRoot,
            TestConstants.PRODUCT_NAME,
            ImmutableSortedSet.of("build"),
            false);
    symlinkForest.plantSymlinkForest();

    assertLinksTo(linkRoot, mainRepo, "dir1");
    assertThat(linkRoot.getChild("build").exists()).isFalse();

    // Create some file in 'build' directory under exec root.
    Path notSymlinkedDir = linkRoot.getChild("build");
    notSymlinkedDir.createDirectoryAndParents();

    byte[] bytes = "text".getBytes(StandardCharsets.ISO_8859_1);
    Path childPath = notSymlinkedDir.getChild("child.txt");
    FileSystemUtils.writeContent(childPath, bytes);

    symlinkForest.plantSymlinkForest();

    assertLinksTo(linkRoot, mainRepo, "dir1");
    // Exists because it was explicitly created.
    assertThat(linkRoot.getChild("build").exists()).isTrue();
    // The presence of the manually added file indicates that SymlinkForest did not delete
    // the directory it's in.
    assertThat(childPath.exists()).isTrue();
    assertThat(FileSystemUtils.readContent(childPath, StandardCharsets.ISO_8859_1))
        .isEqualTo("text");
  }

  @Test
  public void testNotSymlinkedDirectoriesInExecRootPartialMainRepo1() throws Exception {
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Root externalSourceRoot =
        Root.fromPath(outputBase.asPath().getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    linkRoot.createDirectoryAndParents();
    mainRepo.asPath().createDirectoryAndParents();
    mainRepo.getRelative("dir3").createDirectoryAndParents();
    mainRepo.getRelative("build").createDirectoryAndParents();

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createMainPkg(mainRepo, "dir1/pkg/foo"), mainRepo)
            .put(createMainPkg(mainRepo, "dir2/pkg"), mainRepo)
            .put(createExternalPkg(outputBase, "X", "dir_x/pkg"), externalSourceRoot)
            .build();

    ImmutableList<Path> plantedSymlinks =
        new SymlinkForest(
                packageRootMap,
                linkRoot,
                TestConstants.PRODUCT_NAME,
                ImmutableSortedSet.of("build"),
                false)
            .plantSymlinkForest();

    assertLinksTo(linkRoot, mainRepo, "dir1");
    assertLinksTo(linkRoot, mainRepo, "dir2");
    assertLinksTo(linkRoot, outputBase, LabelConstants.EXTERNAL_PATH_PREFIX + "/X");
    assertThat(linkRoot.getChild("build").exists()).isFalse();
    // Not part of the package roots.
    assertThat(linkRoot.getChild("dir3").exists()).isFalse();
    assertThat(plantedSymlinks)
        .containsExactly(
            linkRoot.getRelative("dir1"),
            linkRoot.getRelative("dir2"),
            linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/X"));
  }

  @Test
  public void testNotSymlinkedDirectoriesInExecRootPartialMainRepo2() throws Exception {
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    linkRoot.createDirectoryAndParents();
    mainRepo.asPath().createDirectoryAndParents();

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.of(createMainPkg(mainRepo, "build"), mainRepo);

    AbruptExitException exception =
        assertThrows(
            AbruptExitException.class,
            () ->
                new SymlinkForest(
                        packageRootMap,
                        linkRoot,
                        TestConstants.PRODUCT_NAME,
                        ImmutableSortedSet.of("build"),
                        false)
                    .plantSymlinkForest());
    assertThat(exception)
        .hasMessageThat()
        .isEqualTo(
            "Directories specified with toplevel_output_directories should be "
                + "ignored and can not be used as sources.");
  }

  @Test
  public void testNotSymlinkedDirectoriesInExecRootMultiplePackageRoots() throws Exception {
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root mainRepo = Root.fromPath(fileSystem.getPath("/my_repo"));
    Root otherRepo = Root.fromPath(fileSystem.getPath("/other_repo"));
    Path linkRoot = outputBase.getRelative("execroot/ws_name");

    mainRepo.getRelative("build").createDirectoryAndParents();

    linkRoot.createDirectoryAndParents();
    mainRepo.asPath().createDirectoryAndParents();
    otherRepo.asPath().createDirectoryAndParents();

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            .put(createMainPkg(mainRepo, "dir1"), mainRepo)
            .put(createMainPkg(otherRepo, "dir2"), otherRepo)
            .build();

    AbruptExitException exception =
        assertThrows(
            AbruptExitException.class,
            () ->
                new SymlinkForest(
                        packageRootMap,
                        linkRoot,
                        TestConstants.PRODUCT_NAME,
                        ImmutableSortedSet.of("build"),
                        false)
                    .plantSymlinkForest());
    assertThat(exception)
        .hasMessageThat()
        .isEqualTo(
            "toplevel_output_directories is not supported together "
                + "with --package_path option.");
  }
}
