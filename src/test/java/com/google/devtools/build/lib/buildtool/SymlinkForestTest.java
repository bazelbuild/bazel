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
import static com.google.devtools.build.lib.vfs.FileSystemUtils.createDirectoryAndParents;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
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
    PathFragment a = new PathFragment("A");
    assertEquals(a, longestPathPrefix("A/b", "A", "B")); // simple parent
    assertEquals(a, longestPathPrefix("A", "A", "B")); // self
    assertEquals(a.getRelative("B"), longestPathPrefix("A/B/c", "A", "A/B"));  // want longest
    assertNull(longestPathPrefix("C/b", "A", "B"));  // not found in other parents
    assertNull(longestPathPrefix("A", "A/B", "B"));  // not found in child
    assertEquals(a.getRelative("B/C"), longestPathPrefix("A/B/C/d/e/f.h", "A/B/C", "B/C/d"));
    assertEquals(PathFragment.EMPTY_FRAGMENT, longestPathPrefix("A/f.h", "", "B/C/d"));
  }

  @Test
  public void testDeleteTreesBelowNotPrefixed() throws IOException {
    createTestDirectoryTree();
    SymlinkForest.deleteTreesBelowNotPrefixed(topDir, new String[]{"file-"});
    assertTrue(file1.exists());
    assertTrue(file2.exists());
    assertFalse(aDir.exists());
  }

  private PackageIdentifier createPkg(Path rootA, Path rootB, String pkg) throws IOException {
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

  private PackageIdentifier createPkg(Path root, String repo, String pkg)
      throws IOException, LabelSyntaxException {
    if (root != null) {
      Path repoRoot = root.getRelative(Label.EXTERNAL_PACKAGE_NAME).getRelative(repo);
      createDirectoryAndParents(repoRoot.getRelative(pkg));
      FileSystemUtils.createEmptyFile(repoRoot.getRelative(pkg).getChild("file"));
    }
    return PackageIdentifier.create(RepositoryName.create("@" + repo), new PathFragment(pkg));
  }

  private void assertLinksTo(Path fromRoot, Path toRoot, String relpart) throws IOException {
    assertLinksTo(fromRoot.getRelative(relpart), toRoot.getRelative(relpart));
  }

  private void assertLinksTo(Path fromRoot, Path toRoot) throws IOException {
    assertTrue("stat: " + fromRoot.stat(), fromRoot.isSymbolicLink());
    assertEquals(toRoot.asFragment(), fromRoot.readSymbolicLink());
  }

  private void assertIsDir(Path root, String relpart) {
    assertTrue(root.getRelative(relpart).isDirectory(Symlinks.NOFOLLOW));
  }

  @Test
  public void testPlantLinkForest() throws IOException {
    Path rootA = fileSystem.getPath("/A");
    Path rootB = fileSystem.getPath("/B");

    ImmutableMap<PackageIdentifier, Path> packageRootMap =
        ImmutableMap.<PackageIdentifier, Path>builder()
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
    Path rootX = fileSystem.getPath("/X");
    Path rootY = fileSystem.getPath("/Y");
    ImmutableMap<PackageIdentifier, Path> packageRootMap =
        ImmutableMap.<PackageIdentifier, Path>builder()
            .put(createPkg(rootX, rootY, ""), rootX)
            .put(createPkg(rootX, rootY, "foo"), rootX)
            .build();

    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME, "wsname")
        .plantSymlinkForest();
    assertLinksTo(linkRoot, rootX, "file");
  }

  @Test
  public void testRemotePackage() throws Exception {
    Path outputBase = fileSystem.getPath("/ob");
    Path rootY = outputBase.getRelative(Label.EXTERNAL_PATH_PREFIX).getRelative("y");
    Path rootZ = outputBase.getRelative(Label.EXTERNAL_PATH_PREFIX).getRelative("z");
    Path rootW = outputBase.getRelative(Label.EXTERNAL_PATH_PREFIX).getRelative("w");
    createDirectoryAndParents(rootY);
    FileSystemUtils.createEmptyFile(rootY.getRelative("file"));

    ImmutableMap<PackageIdentifier, Path> packageRootMap =
        ImmutableMap.<PackageIdentifier, Path>builder()
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
    assertFalse(linkRoot.getRelative(Label.EXTERNAL_PATH_PREFIX + "/y/file").exists());
    assertLinksTo(
        linkRoot.getRelative(Label.EXTERNAL_PATH_PREFIX + "/y/w"), rootY.getRelative("w"));
    assertLinksTo(
        linkRoot.getRelative(Label.EXTERNAL_PATH_PREFIX + "/z/file"), rootZ.getRelative("file"));
    assertLinksTo(
        linkRoot.getRelative(Label.EXTERNAL_PATH_PREFIX + "/z/a"), rootZ.getRelative("a"));
    assertLinksTo(
        linkRoot.getRelative(Label.EXTERNAL_PATH_PREFIX + "/w/file"),
        rootW.getRelative("file"));
  }

  @Test
  public void testExternalPackage() throws Exception {
    Path root = fileSystem.getPath("/src");
    ImmutableMap<PackageIdentifier, Path> packageRootMap =
        ImmutableMap.<PackageIdentifier, Path>builder()
            // Virtual root, shouldn't actually be linked in.
            .put(Label.EXTERNAL_PACKAGE_IDENTIFIER, root)
            .build();

    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME, "wsname")
        .plantSymlinkForest();
    assertThat(linkRoot.getRelative(Label.EXTERNAL_PATH_PREFIX).exists()).isFalse();
  }

  @Test
  public void testWorkspaceName() throws Exception {
    Path root = fileSystem.getPath("/src");
    ImmutableMap<PackageIdentifier, Path> packageRootMap =
        ImmutableMap.<PackageIdentifier, Path>builder()
            // Remote repo without top-level package.
            .put(createPkg(root, "y", "w"), root)
            .build();

    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME, "wsname")
        .plantSymlinkForest();
    assertThat(linkRoot.getRelative("../wsname").exists()).isTrue();
  }
}
