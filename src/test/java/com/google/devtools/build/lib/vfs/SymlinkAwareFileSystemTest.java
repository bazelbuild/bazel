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
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.vfs.FileSystem.NotASymlinkException;

import org.junit.Before;
import org.junit.Test;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;

/**
 * This class handles the generic tests that any filesystem must pass.
 *
 * <p>Each filesystem-test should inherit from this class, thereby obtaining
 * all the tests.
 */
public abstract class SymlinkAwareFileSystemTest extends FileSystemTest {

  protected Path xLinkToFile;
  protected Path xLinkToLinkToFile;
  protected Path xLinkToDirectory;
  protected Path xDanglingLink;

  @Before
  public final void createSymbolicLinks() throws Exception  {
    // % ls -lR
    // -rw-rw-r-- xFile
    // drwxrwxr-x xNonEmptyDirectory
    // -rw-rw-r-- xNonEmptyDirectory/foo
    // drwxrwxr-x xEmptyDirectory
    // lrwxrwxr-x xLinkToFile -> xFile
    // lrwxrwxr-x xLinkToDirectory -> xEmptyDirectory
    // lrwxrwxr-x xLinkToLinkToFile -> xLinkToFile
    // lrwxrwxr-x xDanglingLink -> xNothing

    xLinkToFile = absolutize("xLinkToFile");
    xLinkToLinkToFile = absolutize("xLinkToLinkToFile");
    xLinkToDirectory = absolutize("xLinkToDirectory");
    xDanglingLink = absolutize("xDanglingLink");

    createSymbolicLink(xLinkToFile, xFile);
    createSymbolicLink(xLinkToLinkToFile, xLinkToFile);
    createSymbolicLink(xLinkToDirectory, xEmptyDirectory);
    createSymbolicLink(xDanglingLink, xNothing);
  }

  @Test
  public void testCreateLinkToFile() throws IOException {
    Path newPath = xEmptyDirectory.getChild("new-file");
    FileSystemUtils.createEmptyFile(newPath);

    Path linkPath = xEmptyDirectory.getChild("some-link");

    createSymbolicLink(linkPath, newPath);

    assertTrue(linkPath.isSymbolicLink());

    assertTrue(linkPath.isFile());
    assertFalse(linkPath.isFile(Symlinks.NOFOLLOW));
    assertTrue(linkPath.isFile(Symlinks.FOLLOW));

    assertFalse(linkPath.isDirectory());
    assertFalse(linkPath.isDirectory(Symlinks.NOFOLLOW));
    assertFalse(linkPath.isDirectory(Symlinks.FOLLOW));

    if (supportsSymlinks) {
      assertEquals(newPath.toString().length(), linkPath.getFileSize(Symlinks.NOFOLLOW));
      assertEquals(newPath.getFileSize(Symlinks.NOFOLLOW), linkPath.getFileSize());
    }
    assertThat(linkPath.getParentDirectory().getDirectoryEntries()).hasSize(2);
    assertThat(linkPath.getParentDirectory().getDirectoryEntries()).containsExactly(newPath,
        linkPath);
  }

  @Test
  public void testCreateLinkToDirectory() throws IOException {
    Path newPath = xEmptyDirectory.getChild("new-file");
    newPath.createDirectory();

    Path linkPath = xEmptyDirectory.getChild("some-link");

    createSymbolicLink(linkPath, newPath);

    assertTrue(linkPath.isSymbolicLink());
    assertFalse(linkPath.isFile());
    assertTrue(linkPath.isDirectory());
    assertThat(linkPath.getParentDirectory().getDirectoryEntries()).hasSize(2);
    assertThat(linkPath.getParentDirectory().
      getDirectoryEntries()).containsExactly(newPath, linkPath);
  }

  @Test
  public void testFileCanonicalPath() throws IOException {
    Path newPath = absolutize("new-file");
    FileSystemUtils.createEmptyFile(newPath);
    newPath = newPath.resolveSymbolicLinks();

    Path link1 = absolutize("some-link");
    Path link2 = absolutize("some-link2");

    createSymbolicLink(link1, newPath);
    createSymbolicLink(link2, link1);

    assertCanonicalPathsMatch(newPath, link1, link2);
  }

  @Test
  public void testDirectoryCanonicalPath() throws IOException {
    Path newPath = absolutize("new-folder");
    newPath.createDirectory();
    newPath = newPath.resolveSymbolicLinks();

    Path newFile = newPath.getChild("file");
    FileSystemUtils.createEmptyFile(newFile);

    Path link1 = absolutize("some-link");
    Path link2 = absolutize("some-link2");

    createSymbolicLink(link1, newPath);
    createSymbolicLink(link2, link1);

    Path linkFile1 = link1.getChild("file");
    Path linkFile2 = link2.getChild("file");

    assertCanonicalPathsMatch(newFile, linkFile1, linkFile2);
  }

  private void assertCanonicalPathsMatch(Path newPath, Path link1, Path link2)
      throws IOException {
    assertEquals(newPath, link1.resolveSymbolicLinks());
    assertEquals(newPath, link2.resolveSymbolicLinks());
  }

  //
  //  createDirectory
  //

  @Test
  public void testCreateDirectoryWhereDanglingSymlinkAlreadyExists() {
    try {
      xDanglingLink.createDirectory();
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage(xDanglingLink + " (File exists)");
    }
    assertTrue(xDanglingLink.isSymbolicLink()); // still a symbolic link
    assertFalse(xDanglingLink.isDirectory(Symlinks.FOLLOW)); // link still dangles
  }

  @Test
  public void testCreateDirectoryWhereSymlinkAlreadyExists() {
    try {
      xLinkToDirectory.createDirectory();
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage(xLinkToDirectory + " (File exists)");
    }
    assertTrue(xLinkToDirectory.isSymbolicLink()); // still a symbolic link
    assertTrue(xLinkToDirectory.isDirectory(Symlinks.FOLLOW)); // link still points to dir
  }

  //  createSymbolicLink(PathFragment)

  @Test
  public void testCreateSymbolicLinkFromFragment() throws IOException {
    String[] linkTargets = {
      "foo",
      "foo/bar",
      ".",
      "..",
      "../foo",
      "../../foo",
      "../../../../../../../../../../../../../../../../../../../../../foo",
      "/foo",
      "/foo/bar",
      "/..",
      "/foo/../bar",
    };
    Path linkPath = absolutize("link");
    for (String linkTarget : linkTargets) {
      PathFragment relative = PathFragment.create(linkTarget);
      linkPath.delete();
      createSymbolicLink(linkPath, relative);
      if (supportsSymlinks) {
        assertEquals(linkTarget.length(), linkPath.getFileSize(Symlinks.NOFOLLOW));
        assertEquals(relative, linkPath.readSymbolicLink());
      }
    }
  }

  @Test
  public void testLinkToRootResolvesCorrectly() throws IOException {
    Path rootPath = testFS.getPath("/");

    try {
      rootPath.getChild("testDir").createDirectory();
    } catch (IOException e) {
      // Do nothing. This is a real FS, and we don't have permission.
    }

    Path linkPath = absolutize("link");
    createSymbolicLink(linkPath, rootPath);

    // resolveSymbolicLinks requires an existing path:
    try {
      linkPath.getRelative("test").resolveSymbolicLinks();
      fail();
    } catch (FileNotFoundException e) { /* ok */ }

    // The path may not be a symlink, neither on Darwin nor on Linux.
    String nonLinkEntry = null;
    for (Path p : testFS.getDirectoryEntries(rootPath)) {
      if (!p.isSymbolicLink() && p.isDirectory()) {
        nonLinkEntry = p.getBaseName();
        break;
      }
    }
    
    assertNotNull(nonLinkEntry);
    Path rootChild = testFS.getPath("/" + nonLinkEntry);
    assertEquals(rootChild, linkPath.getRelative(nonLinkEntry).resolveSymbolicLinks());
  }

  @Test
  public void testLinkToFragmentContainingLinkResolvesCorrectly() throws IOException {
    Path link1 = absolutize("link1");
    PathFragment link1target = PathFragment.create("link2/foo");
    Path link2 = absolutize("link2");
    Path link2target = xNonEmptyDirectory;

    createSymbolicLink(link1, link1target); // ln -s link2/foo link1
    createSymbolicLink(link2, link2target); // ln -s xNonEmptyDirectory link2
    // link1 --> xNonEmptyDirectory/foo
    assertEquals(link1.resolveSymbolicLinks(), link2target.getRelative("foo"));
  }

  //
  //  readSymbolicLink / resolveSymbolicLinks
  //

  @Test
  public void testRecursiveSymbolicLink() throws IOException {
    Path link = absolutize("recursive-link");
    createSymbolicLink(link, link);

    if (supportsSymlinks) {
      try {
        link.resolveSymbolicLinks();
        fail();
      } catch (IOException e) {
        assertThat(e).hasMessage(link + " (Too many levels of symbolic links)");
      }
    }
  }

  @Test
  public void testMutuallyRecursiveSymbolicLinks() throws IOException {
    Path link1 = absolutize("link1");
    Path link2 = absolutize("link2");
    createSymbolicLink(link2, link1);
    createSymbolicLink(link1, link2);

    if (supportsSymlinks) {
      try {
        link1.resolveSymbolicLinks();
        fail();
      } catch (IOException e) {
        assertThat(e).hasMessage(link1 + " (Too many levels of symbolic links)");
      }
    }
  }

  @Test
  public void testResolveSymbolicLinksENOENT() {
    if (supportsSymlinks) {
      try {
        xDanglingLink.resolveSymbolicLinks();
        fail();
      } catch (IOException e) {
        assertThat(e).hasMessage(xNothing + " (No such file or directory)");
      }
    }
  }

  @Test
  public void testResolveSymbolicLinksENOTDIR() throws IOException {
    if (supportsSymlinks) {
      Path badLinkTarget = xFile.getChild("bad"); // parent is not a directory!
      Path badLink = absolutize("badLink");
      createSymbolicLink(badLink, badLinkTarget);
      try {
        badLink.resolveSymbolicLinks();
        fail();
      } catch (IOException e) {
        // ok.  Ideally we would assert "(Not a directory)" in the error
        // message, but that would require yet another stat in the
        // implementation.
      }
    }
  }

  @Test
  public void testResolveSymbolicLinksWithUplevelRefs() throws IOException {
    if (supportsSymlinks) {
      // Create a series of links that refer to xFile as ./xFile,
      // ./../foo/xFile, ./../../bar/foo/xFile, etc.  They should all resolve
      // to xFile.
      Path ancestor = xFile;
      String prefix = "./";
      while ((ancestor = ancestor.getParentDirectory()) != null) {
        xLinkToFile.delete();
        createSymbolicLink(xLinkToFile, PathFragment.create(prefix + xFile.relativeTo(ancestor)));
        assertEquals(xFile, xLinkToFile.resolveSymbolicLinks());

        prefix += "../";
      }
    }
  }

  @Test
  public void testReadSymbolicLink() throws IOException {
    if (supportsSymlinks) {
      assertEquals(xNothing.toString(),
                   xDanglingLink.readSymbolicLink().toString());
    }

    assertEquals(xFile.toString(),
                 xLinkToFile.readSymbolicLink().toString());

    assertEquals(xEmptyDirectory.toString(),
                 xLinkToDirectory.readSymbolicLink().toString());

    try {
      xFile.readSymbolicLink(); // not a link
      fail();
    } catch (NotASymlinkException e) {
      assertThat(e).hasMessage(xFile.toString());
    }

    try {
      xNothing.readSymbolicLink(); // nothing there
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage(xNothing + " (No such file or directory)");
    }
  }

  @Test
  public void testCannotCreateSymbolicLinkWithReadOnlyParent()
      throws IOException {
    xEmptyDirectory.setWritable(false);
    Path xChildOfReadonlyDir = xEmptyDirectory.getChild("x");
    if (supportsSymlinks) {
      try {
        xChildOfReadonlyDir.createSymbolicLink(xNothing);
        fail();
      } catch (IOException e) {
        assertThat(e).hasMessage(xChildOfReadonlyDir + " (Permission denied)");
      }
    }
  }

  //
  // createSymbolicLink
  //

  @Test
  public void testCanCreateDanglingLink() throws IOException {
    Path newPath = absolutize("non-existing-dir/new-file");
    Path someLink = absolutize("dangling-link");
    createSymbolicLink(someLink, newPath);
    assertTrue(someLink.isSymbolicLink());
    assertTrue(someLink.exists(Symlinks.NOFOLLOW)); // the link itself exists
    assertFalse(someLink.exists()); // ...but the referent doesn't
    if (supportsSymlinks) {
      try {
        someLink.resolveSymbolicLinks();
      } catch (FileNotFoundException e) {
        assertThat(e).hasMessage(newPath.getParentDirectory() + " (No such file or directory)");
      }
    }
  }

  @Test
  public void testCannotCreateSymbolicLinkWithoutParent() throws IOException {
    Path xChildOfMissingDir = xNothing.getChild("x");
    if (supportsSymlinks) {
      try {
        xChildOfMissingDir.createSymbolicLink(xFile);
        fail();
      } catch (FileNotFoundException e) {
        assertThat(e.getMessage()).endsWith(" (No such file or directory)");
      }
    }
  }

  @Test
  public void testCreateSymbolicLinkWhereNothingExists() throws IOException {
    createSymbolicLink(xNothing, xFile);
    assertTrue(xNothing.isSymbolicLink());
  }

  @Test
  public void testCreateSymbolicLinkWhereDirectoryAlreadyExists() {
    try {
      createSymbolicLink(xEmptyDirectory, xFile);
      fail();
    } catch (IOException e) { // => couldn't be created
      assertThat(e).hasMessage(xEmptyDirectory + " (File exists)");
    }
    assertTrue(xEmptyDirectory.isDirectory(Symlinks.NOFOLLOW));
  }

  @Test
  public void testCreateSymbolicLinkWhereFileAlreadyExists() {
    try {
      createSymbolicLink(xFile, xEmptyDirectory);
      fail();
    } catch (IOException e) { // => couldn't be created
      assertThat(e).hasMessage(xFile + " (File exists)");
    }
    assertTrue(xFile.isFile(Symlinks.NOFOLLOW));
  }

  @Test
  public void testCreateSymbolicLinkWhereDanglingSymlinkAlreadyExists() {
    try {
      createSymbolicLink(xDanglingLink, xFile);
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage(xDanglingLink + " (File exists)");
    }
    assertTrue(xDanglingLink.isSymbolicLink()); // still a symbolic link
    assertFalse(xDanglingLink.isDirectory()); // link still dangles
  }

  @Test
  public void testCreateSymbolicLinkWhereSymlinkAlreadyExists() {
    try {
      createSymbolicLink(xLinkToDirectory, xNothing);
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage(xLinkToDirectory + " (File exists)");
    }
    assertTrue(xLinkToDirectory.isSymbolicLink()); // still a symbolic link
    assertTrue(xLinkToDirectory.isDirectory()); // link still points to dir
  }

  @Test
  public void testDeleteLink() throws IOException {
    Path newPath = xEmptyDirectory.getChild("new-file");
    Path someLink = xEmptyDirectory.getChild("a-link");
    FileSystemUtils.createEmptyFile(newPath);
    createSymbolicLink(someLink, newPath);

    assertEquals(xEmptyDirectory.getDirectoryEntries().size(), 2);

    assertTrue(someLink.delete());
    assertEquals(xEmptyDirectory.getDirectoryEntries().size(), 1);

    assertThat(xEmptyDirectory.getDirectoryEntries()).containsExactly(newPath);
  }

  // Testing the links
  @Test
  public void testLinkFollowedToDirectory() throws IOException {
    Path theDirectory = absolutize("foo/");
    assertTrue(theDirectory.createDirectory());
    Path newPath1 = absolutize("foo/new-file-1");
    Path newPath2 = absolutize("foo/new-file-2");
    Path newPath3 = absolutize("foo/new-file-3");

    FileSystemUtils.createEmptyFile(newPath1);
    FileSystemUtils.createEmptyFile(newPath2);
    FileSystemUtils.createEmptyFile(newPath3);

    Path linkPath = absolutize("link");
    createSymbolicLink(linkPath, theDirectory);

    Path resultPath1 = absolutize("link/new-file-1");
    Path resultPath2 = absolutize("link/new-file-2");
    Path resultPath3 = absolutize("link/new-file-3");
    assertThat(linkPath.getDirectoryEntries()).containsExactly(resultPath1, resultPath2,
        resultPath3);
  }

  @Test
  public void testDanglingLinkIsNoFile() throws IOException {
    Path newPath1 = absolutize("new-file-1");
    Path newPath2 = absolutize("new-file-2");
    FileSystemUtils.createEmptyFile(newPath1);
    assertTrue(newPath2.createDirectory());

    Path linkPath1 = absolutize("link1");
    Path linkPath2 = absolutize("link2");
    createSymbolicLink(linkPath1, newPath1);
    createSymbolicLink(linkPath2, newPath2);

    newPath1.delete();
    newPath2.delete();

    assertFalse(linkPath1.isFile());
    assertFalse(linkPath2.isDirectory());
  }

  @Test
  public void testWriteOnLinkChangesFile() throws IOException {
    Path testFile = absolutize("test-file");
    FileSystemUtils.createEmptyFile(testFile);
    String testData = "abc19";

    Path testLink = absolutize("a-link");
    createSymbolicLink(testLink, testFile);

    FileSystemUtils.writeContentAsLatin1(testLink, testData);
    String resultData =
      new String(FileSystemUtils.readContentAsLatin1(testFile));

    assertEquals(testData,resultData);
  }

  //
  // Symlink tests:
  //

  @Test
  public void testExistsWithSymlinks() throws IOException {
    Path a = absolutize("a");
    Path b = absolutize("b");
    FileSystemUtils.createEmptyFile(b);
    createSymbolicLink(a, b);  // ln -sf "b" "a"
    assertTrue(a.exists()); // = exists(FOLLOW)
    assertTrue(b.exists()); // = exists(FOLLOW)
    assertTrue(a.exists(Symlinks.FOLLOW));
    assertTrue(b.exists(Symlinks.FOLLOW));
    assertTrue(a.exists(Symlinks.NOFOLLOW));
    assertTrue(b.exists(Symlinks.NOFOLLOW));
    b.delete(); // "a" is now a dangling link
    assertFalse(a.exists()); // = exists(FOLLOW)
    assertFalse(b.exists()); // = exists(FOLLOW)
    assertFalse(a.exists(Symlinks.FOLLOW));
    assertFalse(b.exists(Symlinks.FOLLOW));

    assertTrue(a.exists(Symlinks.NOFOLLOW)); // symlink still exists
    assertFalse(b.exists(Symlinks.NOFOLLOW));
  }

  @Test
  public void testIsDirectoryWithSymlinks() throws IOException {
    Path a = absolutize("a");
    Path b = absolutize("b");
    b.createDirectory();
    createSymbolicLink(a, b);  // ln -sf "b" "a"
    assertTrue(a.isDirectory()); // = isDirectory(FOLLOW)
    assertTrue(b.isDirectory()); // = isDirectory(FOLLOW)
    assertTrue(a.isDirectory(Symlinks.FOLLOW));
    assertTrue(b.isDirectory(Symlinks.FOLLOW));
    assertFalse(a.isDirectory(Symlinks.NOFOLLOW)); // it's a link!
    assertTrue(b.isDirectory(Symlinks.NOFOLLOW));
    b.delete(); // "a" is now a dangling link
    assertFalse(a.isDirectory()); // = isDirectory(FOLLOW)
    assertFalse(b.isDirectory()); // = isDirectory(FOLLOW)
    assertFalse(a.isDirectory(Symlinks.FOLLOW));
    assertFalse(b.isDirectory(Symlinks.FOLLOW));
    assertFalse(a.isDirectory(Symlinks.NOFOLLOW));
    assertFalse(b.isDirectory(Symlinks.NOFOLLOW));
  }

  @Test
  public void testIsFileWithSymlinks() throws IOException {
    Path a = absolutize("a");
    Path b = absolutize("b");
    FileSystemUtils.createEmptyFile(b);
    createSymbolicLink(a, b);  // ln -sf "b" "a"
    assertTrue(a.isFile()); // = isFile(FOLLOW)
    assertTrue(b.isFile()); // = isFile(FOLLOW)
    assertTrue(a.isFile(Symlinks.FOLLOW));
    assertTrue(b.isFile(Symlinks.FOLLOW));
    assertFalse(a.isFile(Symlinks.NOFOLLOW)); // it's a link!
    assertTrue(b.isFile(Symlinks.NOFOLLOW));
    b.delete(); // "a" is now a dangling link
    assertFalse(a.isFile()); // = isFile()
    assertFalse(b.isFile()); // = isFile()
    assertFalse(a.isFile());
    assertFalse(b.isFile());
    assertFalse(a.isFile(Symlinks.NOFOLLOW));
    assertFalse(b.isFile(Symlinks.NOFOLLOW));
  }

  @Test
  public void testGetDirectoryEntriesOnLinkToDirectory() throws Exception {
    Path fooAlias = xNothing.getChild("foo");
    createSymbolicLink(xNothing, xNonEmptyDirectory);
    Collection<Path> dirents = xNothing.getDirectoryEntries();
    assertThat(dirents).containsExactly(fooAlias);
  }

  @Test
  public void testFilesOfLinkedDirectories() throws Exception {
    Path child = xEmptyDirectory.getChild("child");
    Path aliasToChild = xLinkToDirectory.getChild("child");

    assertFalse(aliasToChild.exists());
    FileSystemUtils.createEmptyFile(child);
    assertTrue(aliasToChild.exists());
    assertTrue(aliasToChild.isFile());
    assertFalse(aliasToChild.isDirectory());

    validateLinkedReferenceObeysReadOnly(child, aliasToChild);
    validateLinkedReferenceObeysExecutable(child, aliasToChild);
  }

  @Test
  public void testDirectoriesOfLinkedDirectories() throws Exception {
    Path childDir = xEmptyDirectory.getChild("childDir");
    Path linkToChildDir = xLinkToDirectory.getChild("childDir");

    assertFalse(linkToChildDir.exists());
    childDir.createDirectory();
    assertTrue(linkToChildDir.exists());
    assertTrue(linkToChildDir.isDirectory());
    assertFalse(linkToChildDir.isFile());

    validateLinkedReferenceObeysReadOnly(childDir, linkToChildDir);
    validateLinkedReferenceObeysExecutable(childDir, linkToChildDir);
  }

  @Test
  public void testDirectoriesOfLinkedDirectoriesOfLinkedDirectories() throws Exception {
    Path childDir = xEmptyDirectory.getChild("childDir");
    Path linkToLinkToDirectory = absolutize("xLinkToLinkToDirectory");
    createSymbolicLink(linkToLinkToDirectory, xLinkToDirectory);
    Path linkToChildDir = linkToLinkToDirectory.getChild("childDir");

    assertFalse(linkToChildDir.exists());
    childDir.createDirectory();
    assertTrue(linkToChildDir.exists());
    assertTrue(linkToChildDir.isDirectory());
    assertFalse(linkToChildDir.isFile());

    validateLinkedReferenceObeysReadOnly(childDir, linkToChildDir);
    validateLinkedReferenceObeysExecutable(childDir, linkToChildDir);
  }

  private void validateLinkedReferenceObeysReadOnly(Path path, Path link) throws IOException {
    path.setWritable(false);
    assertFalse(path.isWritable());
    assertFalse(link.isWritable());
    path.setWritable(true);
    assertTrue(path.isWritable());
    assertTrue(link.isWritable());
    path.setWritable(false);
    assertFalse(path.isWritable());
    assertFalse(link.isWritable());
  }

  private void validateLinkedReferenceObeysExecutable(Path path, Path link) throws IOException {
    path.setExecutable(true);
    assertTrue(path.isExecutable());
    assertTrue(link.isExecutable());
    path.setExecutable(false);
    assertFalse(path.isExecutable());
    assertFalse(link.isExecutable());
    path.setExecutable(true);
    assertTrue(path.isExecutable());
    assertTrue(link.isExecutable());
  }

  @Test
  public void testReadingFileFromLinkedDirectory() throws Exception {
    Path linkedTo = absolutize("linkedTo");
    linkedTo.createDirectory();
    Path child = linkedTo.getChild("child");
    FileSystemUtils.createEmptyFile(child);

    byte[] outputData = "This is a test".getBytes();
    FileSystemUtils.writeContent(child, outputData);

    Path link = absolutize("link");
    createSymbolicLink(link, linkedTo);
    Path linkedChild = link.getChild("child");
    byte[] inputData = FileSystemUtils.readContent(linkedChild);
    assertArrayEquals(outputData, inputData);
  }

  @Test
  public void testCreatingFileInLinkedDirectory() throws Exception {
    Path linkedTo = absolutize("linkedTo");
    linkedTo.createDirectory();
    Path child = linkedTo.getChild("child");

    Path link = absolutize("link");
    createSymbolicLink(link, linkedTo);
    Path linkedChild = link.getChild("child");
    byte[] outputData = "This is a test".getBytes();
    FileSystemUtils.writeContent(linkedChild, outputData);

    byte[] inputData = FileSystemUtils.readContent(child);
    assertArrayEquals(outputData, inputData);
  }
}
