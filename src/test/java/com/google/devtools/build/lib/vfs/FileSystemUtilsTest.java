// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.appendWithoutExtension;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.commonAncestor;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.copyFile;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.copyTool;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.createDirectoryAndParents;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.deleteTree;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.deleteTreesBelowNotPrefixed;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.longestPathPrefix;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.plantLinkForest;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.relativePath;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.removeExtension;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.touchFile;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.traverseTree;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 * This class tests the file system utilities.
 */
@RunWith(JUnit4.class)
public class FileSystemUtilsTest {
  private ManualClock clock;
  private FileSystem fileSystem;
  private Path workingDir;

  @Before
  public final void initializeFileSystem() throws Exception  {
    clock = new ManualClock();
    fileSystem = new InMemoryFileSystem(clock);
    workingDir = fileSystem.getPath("/workingDir");
  }

  Path topDir;
  Path file1;
  Path file2;
  Path aDir;
  Path bDir;
  Path file3;
  Path innerDir;
  Path link1;
  Path dirLink;
  Path file4;
  Path file5;

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
    bDir = fileSystem.getPath("/top-dir/b-dir");
    file3 = fileSystem.getPath("/top-dir/a-dir/file-3");
    innerDir = fileSystem.getPath("/top-dir/a-dir/inner-dir");
    link1 = fileSystem.getPath("/top-dir/a-dir/inner-dir/link-1");
    dirLink = fileSystem.getPath("/top-dir/a-dir/inner-dir/dir-link");
    file4 = fileSystem.getPath("/file-4");
    file5 = fileSystem.getPath("/top-dir/b-dir/file-5");

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

  private void checkTestDirectoryTreesBelow(Path toPath) throws IOException {
    Path copiedFile1 = toPath.getChild("file-1");
    assertTrue(copiedFile1.exists());
    assertTrue(copiedFile1.isFile());

    Path copiedFile2 = toPath.getChild("file-2");
    assertTrue(copiedFile2.exists());
    assertTrue(copiedFile2.isFile());

    Path copiedADir = toPath.getChild("a-dir");
    assertTrue(copiedADir.exists());
    assertTrue(copiedADir.isDirectory());
    Collection<Path> aDirEntries = copiedADir.getDirectoryEntries();
    assertThat(aDirEntries).hasSize(2);

    Path copiedFile3 = copiedADir.getChild("file-3");
    assertTrue(copiedFile3.exists());
    assertTrue(copiedFile3.isFile());

    Path copiedInnerDir = copiedADir.getChild("inner-dir");
    assertTrue(copiedInnerDir.exists());
    assertTrue(copiedInnerDir.isDirectory());

    Path copiedLink1 = copiedInnerDir.getChild("link-1");
    assertTrue(copiedLink1.exists());
    assertFalse(copiedLink1.isSymbolicLink());

    Path copiedDirLink = copiedInnerDir.getChild("dir-link");
    assertTrue(copiedDirLink.exists());
    assertTrue(copiedDirLink.isDirectory());
    assertTrue(copiedDirLink.getChild("file-5").exists());
  }

  // tests

  @Test
  public void testChangeModtime() throws IOException {
    Path file = fileSystem.getPath("/my-file");
    try {
      BlazeTestUtils.changeModtime(file);
      fail();
    } catch (FileNotFoundException e) {
      /* ok */
    }
    FileSystemUtils.createEmptyFile(file);
    long prevMtime = file.getLastModifiedTime();
    BlazeTestUtils.changeModtime(file);
    assertFalse(prevMtime == file.getLastModifiedTime());
  }

  @Test
  public void testCommonAncestor() {
    assertEquals(topDir, commonAncestor(topDir, topDir));
    assertEquals(topDir, commonAncestor(file1, file3));
    assertEquals(topDir, commonAncestor(file1, dirLink));
  }

  @Test
  public void testRelativePath() throws IOException {
    createTestDirectoryTree();
    assertEquals("file-1", relativePath(topDir, file1).getPathString());
    assertEquals(".", relativePath(topDir, topDir).getPathString());
    assertEquals("a-dir/inner-dir/dir-link", relativePath(topDir, dirLink).getPathString());
    assertEquals("../file-4", relativePath(topDir, file4).getPathString());
    assertEquals("../../../file-4", relativePath(innerDir, file4).getPathString());
  }

  private String longestPathPrefixStr(String path, String... prefixStrs) {
    Set<PathFragment> prefixes = new HashSet<>();
    for (String prefix : prefixStrs) {
      prefixes.add(new PathFragment(prefix));
    }
    PathFragment longest = longestPathPrefix(new PathFragment(path), prefixes);
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
  }

  @Test
  public void testRemoveExtension_Strings() throws Exception {
    assertEquals("foo", removeExtension("foo.c"));
    assertEquals("a/foo", removeExtension("a/foo.c"));
    assertEquals("a.b/foo", removeExtension("a.b/foo"));
    assertEquals("foo", removeExtension("foo"));
    assertEquals("foo", removeExtension("foo."));
  }

  @Test
  public void testRemoveExtension_Paths() throws Exception {
    assertPath("/foo", removeExtension(fileSystem.getPath("/foo.c")));
    assertPath("/a/foo", removeExtension(fileSystem.getPath("/a/foo.c")));
    assertPath("/a.b/foo", removeExtension(fileSystem.getPath("/a.b/foo")));
    assertPath("/foo", removeExtension(fileSystem.getPath("/foo")));
    assertPath("/foo", removeExtension(fileSystem.getPath("/foo.")));
  }

  private static void assertPath(String expected, PathFragment actual) {
    assertEquals(expected, actual.getPathString());
  }

  private static void assertPath(String expected, Path actual) {
    assertEquals(expected, actual.getPathString());
  }

  @Test
  public void testReplaceExtension_Path() throws Exception {
    assertPath("/foo/bar.baz",
               FileSystemUtils.replaceExtension(fileSystem.getPath("/foo/bar"), ".baz"));
    assertPath("/foo/bar.baz",
               FileSystemUtils.replaceExtension(fileSystem.getPath("/foo/bar.cc"), ".baz"));
    assertPath("/foo.baz", FileSystemUtils.replaceExtension(fileSystem.getPath("/foo/"), ".baz"));
    assertPath("/foo.baz",
               FileSystemUtils.replaceExtension(fileSystem.getPath("/foo.cc/"), ".baz"));
    assertPath("/foo.baz", FileSystemUtils.replaceExtension(fileSystem.getPath("/foo"), ".baz"));
    assertPath("/foo.baz", FileSystemUtils.replaceExtension(fileSystem.getPath("/foo.cc"), ".baz"));
    assertPath("/.baz", FileSystemUtils.replaceExtension(fileSystem.getPath("/.cc"), ".baz"));
    assertNull(FileSystemUtils.replaceExtension(fileSystem.getPath("/"), ".baz"));
  }

  @Test
  public void testReplaceExtension_PathFragment() throws Exception {
    assertPath("foo/bar.baz",
               FileSystemUtils.replaceExtension(new PathFragment("foo/bar"), ".baz"));
    assertPath("foo/bar.baz",
               FileSystemUtils.replaceExtension(new PathFragment("foo/bar.cc"), ".baz"));
    assertPath("/foo/bar.baz",
               FileSystemUtils.replaceExtension(new PathFragment("/foo/bar"), ".baz"));
    assertPath("/foo/bar.baz",
               FileSystemUtils.replaceExtension(new PathFragment("/foo/bar.cc"), ".baz"));
    assertPath("foo.baz", FileSystemUtils.replaceExtension(new PathFragment("foo/"), ".baz"));
    assertPath("foo.baz", FileSystemUtils.replaceExtension(new PathFragment("foo.cc/"), ".baz"));
    assertPath("/foo.baz", FileSystemUtils.replaceExtension(new PathFragment("/foo/"), ".baz"));
    assertPath("/foo.baz",
               FileSystemUtils.replaceExtension(new PathFragment("/foo.cc/"), ".baz"));
    assertPath("foo.baz", FileSystemUtils.replaceExtension(new PathFragment("foo"), ".baz"));
    assertPath("foo.baz", FileSystemUtils.replaceExtension(new PathFragment("foo.cc"), ".baz"));
    assertPath("/foo.baz", FileSystemUtils.replaceExtension(new PathFragment("/foo"), ".baz"));
    assertPath("/foo.baz",
               FileSystemUtils.replaceExtension(new PathFragment("/foo.cc"), ".baz"));
    assertPath(".baz", FileSystemUtils.replaceExtension(new PathFragment(".cc"), ".baz"));
    assertNull(FileSystemUtils.replaceExtension(new PathFragment("/"), ".baz"));
    assertNull(FileSystemUtils.replaceExtension(new PathFragment(""), ".baz"));
    assertPath("foo/bar.baz",
        FileSystemUtils.replaceExtension(new PathFragment("foo/bar.pony"), ".baz", ".pony"));
    assertPath("foo/bar.baz",
        FileSystemUtils.replaceExtension(new PathFragment("foo/bar"), ".baz", ""));
    assertNull(FileSystemUtils.replaceExtension(new PathFragment(""), ".baz", ".pony"));
    assertNull(
        FileSystemUtils.replaceExtension(new PathFragment("foo/bar.pony"), ".baz", ".unicorn"));
  }

  @Test
  public void testAppendWithoutExtension() throws Exception {
    assertPath("libfoo-src.jar",
        appendWithoutExtension(new PathFragment("libfoo.jar"), "-src"));
    assertPath("foo/libfoo-src.jar",
        appendWithoutExtension(new PathFragment("foo/libfoo.jar"), "-src"));
    assertPath("java/com/google/foo/libfoo-src.jar",
        appendWithoutExtension(new PathFragment("java/com/google/foo/libfoo.jar"), "-src"));
    assertPath("libfoo.bar-src.jar",
        appendWithoutExtension(new PathFragment("libfoo.bar.jar"), "-src"));
    assertPath("libfoo-src",
        appendWithoutExtension(new PathFragment("libfoo"), "-src"));
    assertPath("libfoo-src.jar",
        appendWithoutExtension(new PathFragment("libfoo.jar/"), "-src"));
    assertPath("libfoo.src.jar",
        appendWithoutExtension(new PathFragment("libfoo.jar"), ".src"));
    assertNull(appendWithoutExtension(new PathFragment("/"), "-src"));
    assertNull(appendWithoutExtension(new PathFragment(""), "-src"));
  }

  @Test
  public void testReplaceSegments() {
    assertPath(
        "poo/bar/baz.cc",
        FileSystemUtils.replaceSegments(new PathFragment("foo/bar/baz.cc"), "foo", "poo", true));
    assertPath(
        "poo/poo/baz.cc",
        FileSystemUtils.replaceSegments(new PathFragment("foo/foo/baz.cc"), "foo", "poo", true));
    assertPath(
        "poo/foo/baz.cc",
        FileSystemUtils.replaceSegments(new PathFragment("foo/foo/baz.cc"), "foo", "poo", false));
    assertPath(
        "foo/bar/baz.cc",
        FileSystemUtils.replaceSegments(new PathFragment("foo/bar/baz.cc"), "boo", "poo", true));
  }

  @Test
  public void testGetWorkingDirectory() {
    String userDir = System.getProperty("user.dir");

    assertEquals(FileSystemUtils.getWorkingDirectory(fileSystem),
        fileSystem.getPath(System.getProperty("user.dir", "/")));

    System.setProperty("user.dir", "/blah/blah/blah");
    assertEquals(FileSystemUtils.getWorkingDirectory(fileSystem),
        fileSystem.getPath("/blah/blah/blah"));

    System.setProperty("user.dir", userDir);
  }

  @Test
  public void testResolveRelativeToFilesystemWorkingDir() {
    PathFragment relativePath = new PathFragment("relative/path");
    assertEquals(workingDir.getRelative(relativePath),
                 workingDir.getRelative(relativePath));

    PathFragment absolutePath = new PathFragment("/absolute/path");
    assertEquals(fileSystem.getPath(absolutePath),
                 workingDir.getRelative(absolutePath));
  }

  @Test
  public void testTouchFileCreatesFile() throws IOException {
    createTestDirectoryTree();
    Path nonExistingFile = fileSystem.getPath("/previously-non-existing");
    assertFalse(nonExistingFile.exists());
    touchFile(nonExistingFile);

    assertTrue(nonExistingFile.exists());
  }

  @Test
  public void testTouchFileAdjustsFileTime() throws IOException {
    createTestDirectoryTree();
    Path testFile = file4;
    long oldTime = testFile.getLastModifiedTime();
    testFile.setLastModifiedTime(42);
    touchFile(testFile);

    assertThat(testFile.getLastModifiedTime()).isAtLeast(oldTime);
  }

  @Test
  public void testCopyFile() throws IOException {
    createTestDirectoryTree();
    Path originalFile = file1;
    byte[] content = new byte[] { 'a', 'b', 'c', 23, 42 };
    FileSystemUtils.writeContent(originalFile, content);

    Path copyTarget = file2;

    copyFile(originalFile, copyTarget);

    assertTrue(Arrays.equals(content, FileSystemUtils.readContent(copyTarget)));
  }

  @Test
  public void testReadContentWithLimit() throws IOException {
    createTestDirectoryTree();
    String str = "this is a test of readContentWithLimit method";
    FileSystemUtils.writeContent(file1, StandardCharsets.ISO_8859_1, str);
    assertEquals(readStringFromFile(file1, 0), "");
    assertEquals(readStringFromFile(file1, 10), str.substring(0, 10));
    assertEquals(readStringFromFile(file1, 1000000), str);
  }

  private String readStringFromFile(Path file, int limit) throws IOException {
    byte[] bytes = FileSystemUtils.readContentWithLimit(file, limit);
    return new String(bytes, StandardCharsets.ISO_8859_1);
  }

  @Test
  public void testAppend() throws IOException {
    createTestDirectoryTree();
    FileSystemUtils.writeIsoLatin1(file1, "nobody says ");
    FileSystemUtils.writeIsoLatin1(file1, "mary had");
    FileSystemUtils.appendIsoLatin1(file1, "a little lamb");
    assertEquals(
        "mary had\na little lamb\n",
        new String(FileSystemUtils.readContentAsLatin1(file1)));
  }

  @Test
  public void testCopyFileAttributes() throws IOException {
    createTestDirectoryTree();
    Path originalFile = file1;
    byte[] content = new byte[] { 'a', 'b', 'c', 23, 42 };
    FileSystemUtils.writeContent(originalFile, content);
    file1.setLastModifiedTime(12345L);
    file1.setWritable(false);
    file1.setExecutable(false);

    Path copyTarget = file2;
    copyFile(originalFile, copyTarget);

    assertEquals(12345L, file2.getLastModifiedTime());
    assertFalse(file2.isExecutable());
    assertFalse(file2.isWritable());

    file1.setWritable(true);
    file1.setExecutable(true);

    copyFile(originalFile, copyTarget);

    assertEquals(12345L, file2.getLastModifiedTime());
    assertTrue(file2.isExecutable());
    assertTrue(file2.isWritable());

  }

  @Test
  public void testCopyFileThrowsExceptionIfTargetCantBeDeleted() throws IOException {
    createTestDirectoryTree();
    Path originalFile = file1;
    byte[] content = new byte[] { 'a', 'b', 'c', 23, 42 };
    FileSystemUtils.writeContent(originalFile, content);

    try {
      copyFile(originalFile, aDir);
      fail();
    } catch (IOException ex) {
      assertThat(ex).hasMessage(
          "error copying file: couldn't delete destination: " + aDir + " (Directory not empty)");
    }
  }

  @Test
  public void testCopyTool() throws IOException {
    createTestDirectoryTree();
    Path originalFile = file1;
    byte[] content = new byte[] { 'a', 'b', 'c', 23, 42 };
    FileSystemUtils.writeContent(originalFile, content);

    Path copyTarget = copyTool(topDir.getRelative("file-1"), aDir.getRelative("file-1"));

    assertTrue(Arrays.equals(content, FileSystemUtils.readContent(copyTarget)));
    assertEquals(file1.isWritable(), copyTarget.isWritable());
    assertEquals(file1.isExecutable(), copyTarget.isExecutable());
    assertEquals(file1.getLastModifiedTime(), copyTarget.getLastModifiedTime());
  }

  @Test
  public void testCopyTreesBelow() throws IOException {
    createTestDirectoryTree();
    Path toPath = fileSystem.getPath("/copy-here");
    toPath.createDirectory();

    FileSystemUtils.copyTreesBelow(topDir, toPath);
    checkTestDirectoryTreesBelow(toPath);
  }

  @Test
  public void testCopyTreesBelowWithOverriding() throws IOException {
    createTestDirectoryTree();
    Path toPath = fileSystem.getPath("/copy-here");
    toPath.createDirectory();
    toPath.getChild("file-2");

    FileSystemUtils.copyTreesBelow(topDir, toPath);
    checkTestDirectoryTreesBelow(toPath);
  }

  @Test
  public void testCopyTreesBelowToSubtree() throws IOException {
    createTestDirectoryTree();
    try {
      FileSystemUtils.copyTreesBelow(topDir, aDir);
      fail("Should not be able to copy a directory to a subdir");
    } catch (IllegalArgumentException expected) {
      assertThat(expected).hasMessage("/top-dir/a-dir is a subdirectory of /top-dir");
    }
  }

  @Test
  public void testCopyFileAsDirectoryTree() throws IOException {
    createTestDirectoryTree();
    try {
      FileSystemUtils.copyTreesBelow(file1, aDir);
      fail("Should not be able to copy a file with copyDirectory method");
    } catch (IOException expected) {
      assertThat(expected).hasMessage("/top-dir/file-1 (Not a directory)");
    }
  }

  @Test
  public void testCopyTreesBelowToFile() throws IOException {
    createTestDirectoryTree();
    Path copyDir = fileSystem.getPath("/my-dir");
    Path copySubDir = fileSystem.getPath("/my-dir/subdir");
    FileSystemUtils.createDirectoryAndParents(copySubDir);
    try {
      FileSystemUtils.copyTreesBelow(copyDir, file4);
      fail("Should not be able to copy a directory to a file");
    } catch (IOException expected) {
      assertThat(expected).hasMessage("/file-4 (Not a directory)");
    }
  }

  @Test
  public void testCopyTreesBelowFromUnexistingDir() throws IOException {
    createTestDirectoryTree();

    try {
      Path unexistingDir = fileSystem.getPath("/unexisting-dir");
      FileSystemUtils.copyTreesBelow(unexistingDir, aDir);
      fail("Should not be able to copy from an unexisting path");
    } catch (FileNotFoundException expected) {
      assertThat(expected).hasMessage("/unexisting-dir (No such file or directory)");
    }
  }

  @Test
  public void testTraverseTree() throws IOException {
    createTestDirectoryTree();

    Collection<Path> paths = traverseTree(topDir, new Predicate<Path>() {
      @Override
      public boolean apply(Path p) {
        return !p.getPathString().contains("a-dir");
      }
    });
    assertThat(paths).containsExactly(file1, file2, bDir, file5);
  }

  @Test
  public void testTraverseTreeDeep() throws IOException {
    createTestDirectoryTree();

    Collection<Path> paths = traverseTree(topDir,
        Predicates.alwaysTrue());
    assertThat(paths).containsExactly(aDir,
        file3,
        innerDir,
        link1,
        file1,
        file2,
        dirLink,
        bDir,
        file5);
  }

  @Test
  public void testTraverseTreeLinkDir() throws IOException {
    // Use a new little tree for this test:
    //  top-dir/
    //    dir-link2 => linked-dir
    //  linked-dir/
    //    file
    topDir = fileSystem.getPath("/top-dir");
    Path dirLink2 = fileSystem.getPath("/top-dir/dir-link2");
    Path linkedDir = fileSystem.getPath("/linked-dir");
    Path linkedDirFile = fileSystem.getPath("/top-dir/dir-link2/file");

    topDir.createDirectory();
    linkedDir.createDirectory();
    dirLink2.createSymbolicLink(linkedDir);  // simple symlink
    FileSystemUtils.createEmptyFile(linkedDirFile);  // created through the link

    // traverseTree doesn't follow links:
    Collection<Path> paths = traverseTree(topDir, Predicates.alwaysTrue());
    assertThat(paths).containsExactly(dirLink2);

    paths = traverseTree(linkedDir, Predicates.alwaysTrue());
    assertThat(paths).containsExactly(fileSystem.getPath("/linked-dir/file"));
  }

  @Test
  public void testDeleteTreeCommandDeletesTree() throws IOException {
    createTestDirectoryTree();
    Path toDelete = topDir;
    deleteTree(toDelete);

    assertTrue(file4.exists());
    assertFalse(topDir.exists());
    assertFalse(file1.exists());
    assertFalse(file2.exists());
    assertFalse(aDir.exists());
    assertFalse(file3.exists());
  }

  @Test
  public void testDeleteTreeCommandsDeletesUnreadableDirectories() throws IOException {
    createTestDirectoryTree();
    Path toDelete = topDir;

    try {
      aDir.setReadable(false);
    } catch (UnsupportedOperationException e) {
      // For file systems that do not support setting readable attribute to
      // false, this test is simply skipped.

      return;
    }

    deleteTree(toDelete);
    assertFalse(topDir.exists());
    assertFalse(aDir.exists());

  }

  @Test
  public void testDeleteTreeCommandDoesNotFollowLinksOut() throws IOException {
    createTestDirectoryTree();
    Path toDelete = topDir;
    Path outboundLink = fileSystem.getPath("/top-dir/outbound-link");
    outboundLink.createSymbolicLink(file4);

    deleteTree(toDelete);

    assertTrue(file4.exists());
    assertFalse(topDir.exists());
    assertFalse(file1.exists());
    assertFalse(file2.exists());
    assertFalse(aDir.exists());
    assertFalse(file3.exists());
  }

  @Test
  public void testDeleteTreesBelowNotPrefixed() throws IOException {
    createTestDirectoryTree();
    deleteTreesBelowNotPrefixed(topDir, new String[] { "file-"});
    assertTrue(file1.exists());
    assertTrue(file2.exists());
    assertFalse(aDir.exists());
  }

  @Test
  public void testCreateDirectories() throws IOException {
    Path mainPath = fileSystem.getPath("/some/where/deep/in/the/hierarchy");
    assertTrue(createDirectoryAndParents(mainPath));
    assertTrue(mainPath.exists());
    assertFalse(createDirectoryAndParents(mainPath));
  }

  @Test
  public void testCreateDirectoriesWhenAncestorIsFile() throws IOException {
    Path somewhereDeepIn = fileSystem.getPath("/somewhere/deep/in");
    assertTrue(createDirectoryAndParents(somewhereDeepIn.getParentDirectory()));
    FileSystemUtils.createEmptyFile(somewhereDeepIn);
    Path theHierarchy = somewhereDeepIn.getChild("the-hierarchy");
    try {
      createDirectoryAndParents(theHierarchy);
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage("/somewhere/deep/in (Not a directory)");
    }
  }

  @Test
  public void testCreateDirectoriesWhenSymlinkToDir() throws IOException {
    Path somewhereDeepIn = fileSystem.getPath("/somewhere/deep/in");
    assertTrue(createDirectoryAndParents(somewhereDeepIn));
    Path realDir = fileSystem.getPath("/real/dir");
    assertTrue(createDirectoryAndParents(realDir));

    Path theHierarchy = somewhereDeepIn.getChild("the-hierarchy");
    theHierarchy.createSymbolicLink(realDir);

    assertFalse(createDirectoryAndParents(theHierarchy));
  }

  @Test
  public void testCreateDirectoriesWhenSymlinkEmbedded() throws IOException {
    Path somewhereDeepIn = fileSystem.getPath("/somewhere/deep/in");
    assertTrue(createDirectoryAndParents(somewhereDeepIn));
    Path realDir = fileSystem.getPath("/real/dir");
    assertTrue(createDirectoryAndParents(realDir));

    Path the = somewhereDeepIn.getChild("the");
    the.createSymbolicLink(realDir);

    Path theHierarchy = somewhereDeepIn.getChild("hierarchy");
    assertTrue(createDirectoryAndParents(theHierarchy));
  }

  PathFragment createPkg(Path rootA, Path rootB, String pkg) throws IOException {
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

  void assertLinksTo(Path fromRoot, Path toRoot, String relpart) throws IOException {
    assertTrue(fromRoot.getRelative(relpart).isSymbolicLink());
    assertEquals(toRoot.getRelative(relpart).asFragment(),
                 fromRoot.getRelative(relpart).readSymbolicLink());
  }

  void assertIsDir(Path root, String relpart) {
    assertTrue(root.getRelative(relpart).isDirectory(Symlinks.NOFOLLOW));
  }

  void dumpTree(Path root, PrintStream out) throws IOException {
    out.println("\n" + root);
    for (Path p : FileSystemUtils.traverseTree(root, Predicates.alwaysTrue())) {
      if (p.isDirectory(Symlinks.NOFOLLOW)) {
        out.println("  " + p + "/");
      } else if (p.isSymbolicLink()) {
        out.println("  " + p + " => " + p.readSymbolicLink());
      } else {
        out.println("  " + p + " [" + p.resolveSymbolicLinks() + "]");
      }
    }
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

    //dumpTree(rootA, System.err);
    //dumpTree(rootB, System.err);

    Path linkRoot = fileSystem.getPath("/linkRoot");
    createDirectoryAndParents(linkRoot);
    plantLinkForest(packageRootMap, linkRoot);

    //dumpTree(linkRoot, System.err);

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
  public void testWriteIsoLatin1() throws Exception {
    Path file = fileSystem.getPath("/does/not/exist/yet.txt");
    FileSystemUtils.writeIsoLatin1(file, "Line 1", "Line 2", "Line 3");
    String expected = "Line 1\nLine 2\nLine 3\n";
    String actual = new String(FileSystemUtils.readContentAsLatin1(file));
    assertEquals(expected, actual);
  }

  @Test
  public void testWriteLinesAs() throws Exception {
    Path file = fileSystem.getPath("/does/not/exist/yet.txt");
    FileSystemUtils.writeLinesAs(file, UTF_8, "\u00F6"); // an oe umlaut
    byte[] expected = new byte[] {(byte) 0xC3, (byte) 0xB6, 0x0A};//"\u00F6\n";
    byte[] actual = FileSystemUtils.readContent(file);
    assertArrayEquals(expected, actual);
  }

  @Test
  public void testGetFileSystem() throws Exception {
    Path mountTable = fileSystem.getPath("/proc/mounts");
    FileSystemUtils.writeIsoLatin1(mountTable,
        "/dev/sda1 / ext2 blah 0 0",
        "/dev/mapper/_dev_sda6 /usr/local/google ext3 blah 0 0",
        "devshm /dev/shm tmpfs blah 0 0",
        "/dev/fuse /fuse/mnt fuse blah 0 0",
        "mtvhome22.nfs:/vol/mtvhome22/johndoe /home/johndoe nfs blah 0 0",
        "/dev/foo /foo dummy_foo blah 0 0",
        "/dev/foobar /foobar dummy_foobar blah 0 0",
        "proc proc proc rw,noexec,nosuid,nodev 0 0");
    Path path = fileSystem.getPath("/usr/local/google/_blaze");
    FileSystemUtils.createDirectoryAndParents(path);
    assertEquals("ext3", FileSystemUtils.getFileSystem(path));

    // Should match the root "/"
    path = fileSystem.getPath("/usr/local/tmp");
    FileSystemUtils.createDirectoryAndParents(path);
    assertEquals("ext2", FileSystemUtils.getFileSystem(path));

    // Make sure we don't consider /foobar matches /foo
    path = fileSystem.getPath("/foo");
    FileSystemUtils.createDirectoryAndParents(path);
    assertEquals("dummy_foo", FileSystemUtils.getFileSystem(path));
    path = fileSystem.getPath("/foobar");
    FileSystemUtils.createDirectoryAndParents(path);
    assertEquals("dummy_foobar", FileSystemUtils.getFileSystem(path));

    path = fileSystem.getPath("/dev/shm/blaze");
    FileSystemUtils.createDirectoryAndParents(path);
    assertEquals("tmpfs", FileSystemUtils.getFileSystem(path));

    Path fusePath = fileSystem.getPath("/fuse/mnt/tmp");
    FileSystemUtils.createDirectoryAndParents(fusePath);
    assertEquals("fuse", FileSystemUtils.getFileSystem(fusePath));

    // Create a symlink and make sure it gives the file system of the symlink target.
    path = fileSystem.getPath("/usr/local/google/_blaze/out");
    path.createSymbolicLink(fusePath);
    assertEquals("fuse", FileSystemUtils.getFileSystem(path));

    // Non existent path should return "unknown"
    path = fileSystem.getPath("/does/not/exist");
    assertEquals("unknown", FileSystemUtils.getFileSystem(path));
  }

  @Test
  public void testStartsWithAnySuccess() throws Exception {
    PathFragment a = new PathFragment("a");
    assertTrue(FileSystemUtils.startsWithAny(a,
        Arrays.asList(new PathFragment("b"), new PathFragment("a"))));
  }

  @Test
  public void testStartsWithAnyNotFound() throws Exception {
    PathFragment a = new PathFragment("a");
    assertFalse(FileSystemUtils.startsWithAny(a,
        Arrays.asList(new PathFragment("b"), new PathFragment("c"))));
  }

  @Test
  public void testIterateLines() throws Exception {
    Path file = fileSystem.getPath("/test.txt");
    FileSystemUtils.writeContent(file, ISO_8859_1, "a\nb");
    assertEquals(Arrays.asList("a", "b"),
        Lists.newArrayList(FileSystemUtils.iterateLinesAsLatin1(file)));

    FileSystemUtils.writeContent(file, ISO_8859_1, "a\rb");
    assertEquals(Arrays.asList("a", "b"),
        Lists.newArrayList(FileSystemUtils.iterateLinesAsLatin1(file)));

    FileSystemUtils.writeContent(file, ISO_8859_1, "a\r\nb");
    assertEquals(Arrays.asList("a", "b"),
        Lists.newArrayList(FileSystemUtils.iterateLinesAsLatin1(file)));
  }

  @Test
  public void testEnsureSymbolicLinkDoesNotMakeUnnecessaryChanges() throws Exception {
    PathFragment target = new PathFragment("/b");
    Path file = fileSystem.getPath("/a");
    file.createSymbolicLink(target);
    long prevTimeMillis = clock.currentTimeMillis();
    clock.advanceMillis(1000);
    FileSystemUtils.ensureSymbolicLink(file, target);
    long timestamp = file.getLastModifiedTime(Symlinks.NOFOLLOW);
    assertEquals(prevTimeMillis, timestamp);
  }
}
