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
import static com.google.devtools.build.lib.vfs.FileSystemUtils.createDirectoryAndParents;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
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
import com.google.devtools.build.lib.vfs.Symlinks;
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

  private Path topDir;
  private Path file1;
  private Path file2;
  private Path aDir;

  // The execution root.
  private Path linkRoot;

  @Before
  public final void initializeFileSystem() throws Exception {
    ManualClock clock = new ManualClock();
    fileSystem = new InMemoryFileSystem(clock);
    linkRoot = fileSystem.getPath("/linkRoot");
    createDirectoryAndParents(linkRoot);
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
    link1.createSymbolicLink(file4);  // simple symlink
    dirLink.createSymbolicLink(bDir);
    FileSystemUtils.createEmptyFile(file4);
    FileSystemUtils.createEmptyFile(file5);
  }

  private static PathFragment longestPathPrefix(String path, String... prefixStrs) {
    ImmutableSet.Builder<PackageIdentifier> prefixes = ImmutableSet.builder();
    for (String prefix : prefixStrs) {
      prefixes.add(PackageIdentifier.createInMainRepo(prefix));
    }
    PackageIdentifier longest = SymlinkForest.longestPathPrefix(
        PackageIdentifier.createInMainRepo(path), prefixes.build());
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
    SymlinkForest.deleteTreesBelowNotPrefixed(topDir, new String[]{"file-"});
    assertThat(file1.exists()).isTrue();
    assertThat(file2.exists()).isTrue();
    assertThat(aDir.exists()).isFalse();
  }

  private PackageIdentifier createPkg(Root rootA, Root rootB, String pkg) throws IOException {
    if (rootA != null) {
      createDirectoryAndParents(rootA.getRelative(pkg));
      FileSystemUtils.createEmptyFile(rootA.getRelative(pkg).getChild("file"));
    }
    if (rootB != null) {
      createDirectoryAndParents(rootB.getRelative(pkg));
      FileSystemUtils.createEmptyFile(rootB.getRelative(pkg).getChild("file"));
    }
    return PackageIdentifier.createInMainRepo(pkg);
  }

  private PackageIdentifier createPkg(Root root, String repo, String pkg)
      throws IOException, LabelSyntaxException {
    if (root != null) {
      Path repoRoot = root.getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME).getRelative(repo);
      createDirectoryAndParents(repoRoot.getRelative(pkg));
      FileSystemUtils.createEmptyFile(repoRoot.getRelative(pkg).getChild("file"));
    }
    return PackageIdentifier.create(RepositoryName.create("@" + repo), PathFragment.create(pkg));
  }

  private void assertLinksTo(Path fromRoot, Root toRoot, String relpart) throws IOException {
    assertLinksTo(fromRoot.getRelative(relpart), toRoot.getRelative(relpart));
  }

  private void assertLinksTo(Path fromRoot, Path toRoot) throws IOException {
    assertWithMessage("stat: " + fromRoot.stat()).that(fromRoot.isSymbolicLink()).isTrue();
    assertThat(fromRoot.readSymbolicLink()).isEqualTo(toRoot.asFragment());
  }

  private void assertIsDir(Path root, String relpart) {
    assertThat(root.getRelative(relpart).isDirectory(Symlinks.NOFOLLOW)).isTrue();
  }

  @Test
  public void testPlantLinkForest() throws IOException {
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
    createPkg(rootA, rootB, "pkgB/dir");  // create a file in there

    Path linkRoot = fileSystem.getPath("/linkRoot");
    createDirectoryAndParents(linkRoot);
    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME, "wsname")
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

    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME, "wsname")
        .plantSymlinkForest();
    assertLinksTo(linkRoot, rootX, "file");
  }

  @Test
  public void testRemotePackage() throws Exception {
    Root outputBase = Root.fromPath(fileSystem.getPath("/ob"));
    Root rootY = Root.fromPath(outputBase.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX).getRelative("y"));
    Root rootZ = Root.fromPath(outputBase.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX).getRelative("z"));
    Root rootW = Root.fromPath(outputBase.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX).getRelative("w"));
    createDirectoryAndParents(rootY.asPath());
    FileSystemUtils.createEmptyFile(rootY.getRelative("file"));

    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            // Remote repo without top-level package.
            .put(createPkg(outputBase, "y", "w"), outputBase)
            // Remote repo with and without top-level package.
            .put(createPkg(outputBase, "z", ""), outputBase)
            .put(createPkg(outputBase, "z", "a/b/c"), outputBase)
            // Only top-level pkg.
            .put(createPkg(outputBase, "w", ""), outputBase)
            .build();

    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME, "wsname")
        .plantSymlinkForest();
    assertThat(linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/y/file").exists()).isFalse();
    assertLinksTo(
        linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/y/w"), rootY.getRelative("w"));
    assertLinksTo(
        linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/z/file"), rootZ.getRelative("file"));
    assertLinksTo(
        linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/z/a"), rootZ.getRelative("a"));
    assertLinksTo(
        linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX + "/w/file"),
        rootW.getRelative("file"));
  }

  @Test
  public void testExternalPackage() throws Exception {
    Root root = Root.fromPath(fileSystem.getPath("/src"));
    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            // Virtual root, shouldn't actually be linked in.
            .put(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER, root)
            .build();

    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME, "wsname")
        .plantSymlinkForest();
    assertThat(linkRoot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX).exists()).isFalse();
  }

  @Test
  public void testWorkspaceName() throws Exception {
    Root root = Root.fromPath(fileSystem.getPath("/src"));
    ImmutableMap<PackageIdentifier, Root> packageRootMap =
        ImmutableMap.<PackageIdentifier, Root>builder()
            // Remote repo without top-level package.
            .put(createPkg(root, "y", "w"), root)
            .build();

    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME, "wsname")
        .plantSymlinkForest();
    assertThat(linkRoot.getRelative("../wsname").exists()).isTrue();
  }

  @Test
  public void testExecrootVersionChanges() throws Exception {
    ImmutableMap<PackageIdentifier, Root> packageRootMap = ImmutableMap.of();
    linkRoot.getRelative("wsname").createDirectory();
    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME, "wsname")
        .plantSymlinkForest();
    assertThat(linkRoot.getRelative("../wsname").isSymbolicLink()).isTrue();
  }
}
