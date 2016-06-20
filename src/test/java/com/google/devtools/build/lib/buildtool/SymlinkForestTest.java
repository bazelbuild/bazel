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

import static com.google.devtools.build.lib.vfs.FileSystemUtils.createDirectoryAndParents;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

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

  @Before
  public final void initializeFileSystem() throws Exception  {
    ManualClock clock = new ManualClock();
    fileSystem = new InMemoryFileSystem(clock);
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

  private static String longestPathPrefixStr(String path, String... prefixStrs) {
    Set<PathFragment> prefixes = new HashSet<>();
    for (String prefix : prefixStrs) {
      prefixes.add(new PathFragment(prefix));
    }
    PathFragment longest = SymlinkForest.longestPathPrefix(new PathFragment(path), prefixes);
    return longest != null ? longest.getPathString() : null;
  }

  @Test
  public void testLongestPathPrefix() {
    assertEquals("A", longestPathPrefixStr("A/b", "A", "B")); // simple parent
    assertEquals("A", longestPathPrefixStr("A", "A", "B")); // self
    assertEquals("A/B", longestPathPrefixStr("A/B/c", "A", "A/B"));  // want longest
    assertNull(longestPathPrefixStr("C/b", "A", "B"));  // not found in other parents
    assertNull(longestPathPrefixStr("A", "A/B", "B"));  // not found in child
    assertEquals("A/B/C", longestPathPrefixStr("A/B/C/d/e/f.h", "A/B/C", "B/C/d"));
    assertEquals("", longestPathPrefixStr("A/f.h", "", "B/C/d"));
  }

  @Test
  public void testDeleteTreesBelowNotPrefixed() throws IOException {
    createTestDirectoryTree();
    SymlinkForest.deleteTreesBelowNotPrefixed(topDir, new String[] { "file-"});
    assertTrue(file1.exists());
    assertTrue(file2.exists());
    assertFalse(aDir.exists());
  }

  private PathFragment createPkg(Path rootA, Path rootB, String pkg) throws IOException {
    if (rootA != null) {
      createDirectoryAndParents(rootA.getRelative(pkg));
      FileSystemUtils.createEmptyFile(rootA.getRelative(pkg).getChild("file"));
    }
    if (rootB != null) {
      createDirectoryAndParents(rootB.getRelative(pkg));
      FileSystemUtils.createEmptyFile(rootB.getRelative(pkg).getChild("file"));
    }
    return new PathFragment(pkg);
  }

  private void assertLinksTo(Path fromRoot, Path toRoot, String relpart) throws IOException {
    assertTrue(fromRoot.getRelative(relpart).isSymbolicLink());
    assertEquals(toRoot.getRelative(relpart).asFragment(),
        fromRoot.getRelative(relpart).readSymbolicLink());
  }

  private void assertIsDir(Path root, String relpart) {
    assertTrue(root.getRelative(relpart).isDirectory(Symlinks.NOFOLLOW));
  }

  @Test
  public void testPlantLinkForest() throws IOException {
    Path rootA = fileSystem.getPath("/A");
    Path rootB = fileSystem.getPath("/B");

    ImmutableMap<PathFragment, Path> packageRootMap = ImmutableMap.<PathFragment, Path>builder()
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
    new SymlinkForest(packageRootMap, linkRoot, TestConstants.PRODUCT_NAME).plantSymlinkForest();

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
}
