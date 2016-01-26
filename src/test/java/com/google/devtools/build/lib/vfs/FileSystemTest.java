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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * This class handles the generic tests that any filesystem must pass.
 *
 * <p>Each filesystem-test should inherit from this class, thereby obtaining
 * all the tests.
 */
public abstract class FileSystemTest {

  private long savedTime;
  protected FileSystem testFS;
  protected boolean supportsSymlinks;
  protected Path workingDir;

  // Some useful examples of various kinds of files (mnemonic: "x" = "eXample")
  protected Path xNothing;
  protected Path xLink;
  protected Path xFile;
  protected Path xNonEmptyDirectory;
  protected Path xNonEmptyDirectoryFoo;
  protected Path xEmptyDirectory;

  @Before
  public final void createDirectories() throws Exception  {
    executeBeforeCreatingDirectories();
    testFS = getFreshFileSystem();
    workingDir = testFS.getPath(getTestTmpDir());
    cleanUpWorkingDirectory(workingDir);
    supportsSymlinks = testFS.supportsSymbolicLinksNatively();

    // % ls -lR
    // -rw-rw-r-- xFile
    // drwxrwxr-x xNonEmptyDirectory
    // -rw-rw-r-- xNonEmptyDirectory/foo
    // drwxrwxr-x xEmptyDirectory

    xNothing = absolutize("xNothing");
    xLink = absolutize("xLink");
    xFile = absolutize("xFile");
    xNonEmptyDirectory = absolutize("xNonEmptyDirectory");
    xNonEmptyDirectoryFoo = xNonEmptyDirectory.getChild("foo");
    xEmptyDirectory = absolutize("xEmptyDirectory");

    FileSystemUtils.createEmptyFile(xFile);
    xNonEmptyDirectory.createDirectory();
    FileSystemUtils.createEmptyFile(xNonEmptyDirectoryFoo);
    xEmptyDirectory.createDirectory();
  }

  protected void executeBeforeCreatingDirectories() throws Exception {
    // This method exists because LazyDigestFileSystemTest requires some code to be run before
    // createDirectories().
  }

  @After
  public final void destroyFileSystem() throws Exception  {
    destroyFileSystem(testFS);
  }

  /**
   * Returns an instance of the file system to test.
   */
  protected abstract FileSystem getFreshFileSystem() throws IOException;

  protected boolean isSymbolicLink(File file) {
    return com.google.devtools.build.lib.unix.FilesystemUtils.isSymbolicLink(file);
  }

  protected void setWritable(File file) throws IOException {
    com.google.devtools.build.lib.unix.FilesystemUtils.setWritable(file);
  }

  protected void setExecutable(File file) throws IOException {
    com.google.devtools.build.lib.unix.FilesystemUtils.setExecutable(file);
  }

  private static final Pattern STAT_SUBDIR_ERROR = Pattern.compile("(.*) \\(Not a directory\\)");

  // Test that file is not present, using statIfFound. Base implementation throws an exception, but
  // subclasses may override statIfFound to return null, in which case their tests should override
  // this method.
  @SuppressWarnings("unused") // Subclasses may throw.
  protected void expectNotFound(Path path) throws IOException {
    try {
      assertNull(path.statIfFound());
    } catch (IOException e) {
      // May be because of a non-directory path component. Parse exception to check this.
      Matcher matcher = STAT_SUBDIR_ERROR.matcher(e.getMessage());
      if (!matcher.matches() || !path.getPathString().startsWith(matcher.group(1))) {
        // Throw if this doesn't match what an ENOTDIR error looks like.
        throw e;
      }
    }
  }

  /**
   * Removes all stuff from the test filesystem.
   */
  protected void destroyFileSystem(FileSystem fileSystem) throws IOException {
    Preconditions.checkArgument(fileSystem.equals(workingDir.getFileSystem()));
    cleanUpWorkingDirectory(workingDir);
  }

  /**
   * Cleans up the working directory by removing everything.
   */
  protected void cleanUpWorkingDirectory(Path workingPath)
      throws IOException {
    if (workingPath.exists()) {
      removeEntireDirectory(workingPath.getPathFile()); // uses java.io.File!
    }
    FileSystemUtils.createDirectoryAndParents(workingPath);
  }

  /**
   * This function removes an entire directory and all of its contents.
   * Much like rm -rf directoryToRemove
   */
  protected void removeEntireDirectory(File directoryToRemove)
      throws IOException {
    // make sure that we do not remove anything outside the test directory
    Path testDirPath = testFS.getPath(getTestTmpDir());
    if (!testFS.getPath(directoryToRemove.getAbsolutePath()).startsWith(testDirPath)) {
      throw new IOException("trying to remove files outside of the testdata directory");
    }
    // Some tests set the directories read-only and/or non-executable, so
    // override that:
    setWritable(directoryToRemove);
    setExecutable(directoryToRemove);

    File[] files = directoryToRemove.listFiles();
    if (files != null) {
      for (File currentFile : files) {
        boolean isSymbolicLink = isSymbolicLink(currentFile);
        if (!isSymbolicLink && currentFile.isDirectory()) {
          removeEntireDirectory(currentFile);
        } else {
          if (!isSymbolicLink) {
            setWritable(currentFile);
          }
          if (!currentFile.delete()) {
            throw new IOException("Failed to delete '" + currentFile + "'");
          }
        }
      }
    }
    if (!directoryToRemove.delete()) {
      throw new IOException("Failed to delete '" + directoryToRemove + "'");
    }
  }

  /**
   * Returns the directory to use as the FileSystem's working directory.
   * Canonicalized to make tests hermetic against symbolic links in TEST_TMPDIR.
   */
  protected final String getTestTmpDir() throws IOException {
    return new File(TestUtils.tmpDir()).getCanonicalPath() + "/testdir";
  }

  /**
   * Indirection to create links so we can test FileSystems that do not support
   * link creation.  For example, JavaFileSystemTest overrides this method
   * and creates the link with an alternate FileSystem.
   */
  protected void createSymbolicLink(Path link, Path target) throws IOException {
    createSymbolicLink(link, target.asFragment());
  }

  /**
   * Indirection to create links so we can test FileSystems that do not support
   * link creation.  For example, JavaFileSystemTest overrides this method
   * and creates the link with an alternate FileSystem.
   */
  protected void createSymbolicLink(Path link, PathFragment target) throws IOException {
    link.createSymbolicLink(target);
  }

  /**
   * Indirection to setReadOnly(false) on FileSystems that do not
   * support setReadOnly(false).  For example, JavaFileSystemTest overrides this
   * method and makes the Path writable with an alternate FileSystem.
   */
  protected void makeWritable(Path target) throws IOException {
    target.setWritable(true);
  }

  /**
   * Indirection to {@link Path#setExecutable(boolean)} on FileSystems that do
   * not support setExecutable.  For example, JavaFileSystemTest overrides this
   * method and makes the Path executable with an alternate FileSystem.
   */
  protected void setExecutable(Path target, boolean mode) throws IOException {
    target.setExecutable(mode);
  }

  // TODO(bazel-team): (2011) Put in a setLastModifiedTime into the various objects
  // and clobber the current time of the object we're currently handling.
  // Otherwise testing the thing might get a little hard, depending on the clock.
  void storeReferenceTime(long timeToMark) {
    savedTime = timeToMark;
  }

  boolean isLaterThanreferenceTime(long testTime) {
    return (savedTime <= testTime);
  }

  protected Path absolutize(String relativePathName) {
    return workingDir.getRelative(relativePathName);
  }

  // Here the tests begin.

  @Test
  public void testIsFileForNonexistingPath() {
    Path nonExistingPath = testFS.getPath("/something/strange");
    assertFalse(nonExistingPath.isFile());
  }

  @Test
  public void testIsDirectoryForNonexistingPath() {
    Path nonExistingPath = testFS.getPath("/something/strange");
    assertFalse(nonExistingPath.isDirectory());
  }

  @Test
  public void testIsLinkForNonexistingPath() {
    Path nonExistingPath = testFS.getPath("/something/strange");
    assertFalse(nonExistingPath.isSymbolicLink());
  }

  @Test
  public void testExistsForNonexistingPath() throws Exception {
    Path nonExistingPath = testFS.getPath("/something/strange");
    assertFalse(nonExistingPath.exists());
    expectNotFound(nonExistingPath);
  }

  @Test
  public void testBadPermissionsThrowsExceptionOnStatIfFound() throws Exception {
    Path inaccessible = absolutize("inaccessible");
    inaccessible.createDirectory();
    Path child = inaccessible.getChild("child");
    FileSystemUtils.createEmptyFile(child);
    inaccessible.setExecutable(false);
    assertFalse(child.exists());
    try {
      child.statIfFound();
      fail();
    } catch (IOException expected) {
      // Expected.
    }
  }

  @Test
  public void testStatIfFoundReturnsNullForChildOfNonDir() throws Exception {
    Path foo = absolutize("foo");
    foo.createDirectory();
    Path nonDir = foo.getRelative("bar");
    FileSystemUtils.createEmptyFile(nonDir);
    assertNull(nonDir.getRelative("file").statIfFound());
  }

  // The following tests check the handling of the current working directory.
  @Test
  public void testCreatePathRelativeToWorkingDirectory() {
    Path relativeCreatedPath = absolutize("some-file");
    Path expectedResult = workingDir.getRelative(new PathFragment("some-file"));

    assertEquals(expectedResult, relativeCreatedPath);
  }

  // The following tests check the handling of the root directory
  @Test
  public void testRootIsDirectory() {
    Path rootPath = testFS.getPath("/");
    assertTrue(rootPath.isDirectory());
  }

  @Test
  public void testRootHasNoParent() {
    Path rootPath = testFS.getPath("/");
    assertNull(rootPath.getParentDirectory());
  }

  // The following functions test the creation of files/links/directories.
  @Test
  public void testFileExists() throws Exception {
    Path someFile = absolutize("some-file");
    FileSystemUtils.createEmptyFile(someFile);
    assertTrue(someFile.exists());
    assertNotNull(someFile.statIfFound());
  }

  @Test
  public void testFileIsFile() throws Exception {
    Path someFile = absolutize("some-file");
    FileSystemUtils.createEmptyFile(someFile);
    assertTrue(someFile.isFile());
  }

  @Test
  public void testFileIsNotDirectory() throws Exception {
    Path someFile = absolutize("some-file");
    FileSystemUtils.createEmptyFile(someFile);
    assertFalse(someFile.isDirectory());
  }

  @Test
  public void testFileIsNotSymbolicLink() throws Exception {
    Path someFile = absolutize("some-file");
    FileSystemUtils.createEmptyFile(someFile);
    assertFalse(someFile.isSymbolicLink());
  }

  @Test
  public void testDirectoryExists() throws Exception {
    Path someDirectory = absolutize("some-dir");
    someDirectory.createDirectory();
    assertTrue(someDirectory.exists());
    assertNotNull(someDirectory.statIfFound());
  }

  @Test
  public void testDirectoryIsDirectory() throws Exception {
    Path someDirectory = absolutize("some-dir");
    someDirectory.createDirectory();
    assertTrue(someDirectory.isDirectory());
  }

  @Test
  public void testDirectoryIsNotFile() throws Exception {
    Path someDirectory = absolutize("some-dir");
    someDirectory.createDirectory();
    assertFalse(someDirectory.isFile());
  }

  @Test
  public void testDirectoryIsNotSymbolicLink() throws Exception {
    Path someDirectory = absolutize("some-dir");
    someDirectory.createDirectory();
    assertFalse(someDirectory.isSymbolicLink());
  }

  @Test
  public void testSymbolicFileLinkExists() throws Exception {
    if (supportsSymlinks) {
      Path someLink = absolutize("some-link");
      someLink.createSymbolicLink(xFile);
      assertTrue(someLink.exists());
      assertNotNull(someLink.statIfFound());
    }
  }

  @Test
  public void testSymbolicFileLinkIsSymbolicLink() throws Exception {
    if (supportsSymlinks) {
      Path someLink = absolutize("some-link");
      someLink.createSymbolicLink(xFile);
      assertTrue(someLink.isSymbolicLink());
    }
  }

  @Test
  public void testSymbolicFileLinkIsFile() throws Exception {
    if (supportsSymlinks) {
      Path someLink = absolutize("some-link");
      someLink.createSymbolicLink(xFile);
      assertTrue(someLink.isFile());
    }
  }

  @Test
  public void testSymbolicFileLinkIsNotDirectory() throws Exception {
    if (supportsSymlinks) {
      Path someLink = absolutize("some-link");
      someLink.createSymbolicLink(xFile);
      assertFalse(someLink.isDirectory());
    }
  }

  @Test
  public void testSymbolicDirLinkExists() throws Exception {
    if (supportsSymlinks) {
      Path someLink = absolutize("some-link");
      someLink.createSymbolicLink(xEmptyDirectory);
      assertTrue(someLink.exists());
      assertNotNull(someLink.statIfFound());
    }
  }

  @Test
  public void testSymbolicDirLinkIsSymbolicLink() throws Exception {
    if (supportsSymlinks) {
      Path someLink = absolutize("some-link");
      someLink.createSymbolicLink(xEmptyDirectory);
      assertTrue(someLink.isSymbolicLink());
    }
  }

  @Test
  public void testSymbolicDirLinkIsDirectory() throws Exception {
    if (supportsSymlinks) {
      Path someLink = absolutize("some-link");
      someLink.createSymbolicLink(xEmptyDirectory);
      assertTrue(someLink.isDirectory());
    }
  }

  @Test
  public void testSymbolicDirLinkIsNotFile() throws Exception {
    if (supportsSymlinks) {
      Path someLink = absolutize("some-link");
      someLink.createSymbolicLink(xEmptyDirectory);
      assertFalse(someLink.isFile());
    }
  }

  @Test
  public void testChildOfNonDirectory() throws Exception {
    Path somePath = absolutize("file-name");
    FileSystemUtils.createEmptyFile(somePath);
    Path childOfNonDir = somePath.getChild("child");
    assertFalse(childOfNonDir.exists());
    expectNotFound(childOfNonDir);
  }

  @Test
  public void testCreateDirectoryIsEmpty() throws Exception {
    Path newPath = xEmptyDirectory.getChild("new-dir");
    newPath.createDirectory();
    assertEquals(newPath.getDirectoryEntries().size(), 0);
  }

  @Test
  public void testCreateDirectoryIsOnlyChildInParent() throws Exception {
    Path newPath = xEmptyDirectory.getChild("new-dir");
    newPath.createDirectory();
    assertThat(newPath.getParentDirectory().getDirectoryEntries()).hasSize(1);
    assertThat(newPath.getParentDirectory().getDirectoryEntries()).containsExactly(newPath);
  }

  @Test
  public void testCreateDirectories() throws Exception {
    Path newPath = absolutize("new-dir/sub/directory");
    assertTrue(FileSystemUtils.createDirectoryAndParents(newPath));
  }

  @Test
  public void testCreateDirectoriesIsDirectory() throws Exception {
    Path newPath = absolutize("new-dir/sub/directory");
    FileSystemUtils.createDirectoryAndParents(newPath);
    assertTrue(newPath.isDirectory());
  }

  @Test
  public void testCreateDirectoriesIsNotFile() throws Exception {
    Path newPath = absolutize("new-dir/sub/directory");
    FileSystemUtils.createDirectoryAndParents(newPath);
    assertFalse(newPath.isFile());
  }

  @Test
  public void testCreateDirectoriesIsNotSymbolicLink() throws Exception {
    Path newPath = absolutize("new-dir/sub/directory");
    FileSystemUtils.createDirectoryAndParents(newPath);
    assertFalse(newPath.isSymbolicLink());
  }

  @Test
  public void testCreateDirectoriesIsEmpty() throws Exception {
    Path newPath = absolutize("new-dir/sub/directory");
    FileSystemUtils.createDirectoryAndParents(newPath);
    assertEquals(newPath.getDirectoryEntries().size(), 0);
  }

  @Test
  public void testCreateDirectoriesIsOnlyChildInParent() throws Exception {
    Path newPath = absolutize("new-dir/sub/directory");
    FileSystemUtils.createDirectoryAndParents(newPath);
    assertThat(newPath.getParentDirectory().getDirectoryEntries()).hasSize(1);
    assertThat(newPath.getParentDirectory().getDirectoryEntries()).containsExactly(newPath);
  }

  @Test
  public void testCreateEmptyFileIsEmpty() throws Exception {
    Path newPath = xEmptyDirectory.getChild("new-file");
    FileSystemUtils.createEmptyFile(newPath);

    assertEquals(newPath.getFileSize(), 0);
  }

  @Test
  public void testCreateFileIsOnlyChildInParent() throws Exception {
    Path newPath = xEmptyDirectory.getChild("new-file");
    FileSystemUtils.createEmptyFile(newPath);
    assertThat(newPath.getParentDirectory().getDirectoryEntries()).hasSize(1);
    assertThat(newPath.getParentDirectory().getDirectoryEntries()).containsExactly(newPath);
  }

  // The following functions test the behavior if errors occur during the
  // creation of files/links/directories.
  @Test
  public void testCreateDirectoryWhereDirectoryAlreadyExists() throws Exception {
    assertFalse(xEmptyDirectory.createDirectory());
  }

  @Test
  public void testCreateDirectoryWhereFileAlreadyExists() {
    try {
      xFile.createDirectory();
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage(xFile + " (File exists)");
    }
  }

  @Test
  public void testCannotCreateDirectoryWithoutExistingParent() throws Exception {
    Path newPath = testFS.getPath("/deep/new-dir");
    try {
      newPath.createDirectory();
      fail();
    } catch (FileNotFoundException e) {
      assertThat(e.getMessage()).endsWith(" (No such file or directory)");
    }
  }

  @Test
  public void testCannotCreateDirectoryWithReadOnlyParent() throws Exception {
    xEmptyDirectory.setWritable(false);
    Path xChildOfReadonlyDir = xEmptyDirectory.getChild("x");
    try {
      xChildOfReadonlyDir.createDirectory();
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage(xChildOfReadonlyDir + " (Permission denied)");
    }
  }

  @Test
  public void testCannotCreateFileWithoutExistingParent() throws Exception {
    Path newPath = testFS.getPath("/non-existing-dir/new-file");
    try {
      FileSystemUtils.createEmptyFile(newPath);
      fail();
    } catch (FileNotFoundException e) {
      assertThat(e.getMessage()).endsWith(" (No such file or directory)");
    }
  }

  @Test
  public void testCannotCreateFileWithReadOnlyParent() throws Exception {
    xEmptyDirectory.setWritable(false);
    Path xChildOfReadonlyDir = xEmptyDirectory.getChild("x");
    try {
      FileSystemUtils.createEmptyFile(xChildOfReadonlyDir);
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage(xChildOfReadonlyDir + " (Permission denied)");
    }
  }

  @Test
  public void testCannotCreateFileWithinFile() throws Exception {
    Path newFilePath = absolutize("some-file");
    FileSystemUtils.createEmptyFile(newFilePath);
    Path wrongPath = absolutize("some-file/new-file");
    try {
      FileSystemUtils.createEmptyFile(wrongPath);
      fail();
    } catch (IOException e) {
      assertThat(e.getMessage()).endsWith(" (Not a directory)");
    }
  }

  @Test
  public void testCannotCreateDirectoryWithinFile() throws Exception {
    Path newFilePath = absolutize("some-file");
    FileSystemUtils.createEmptyFile(newFilePath);
    Path wrongPath = absolutize("some-file/new-file");
    try {
      wrongPath.createDirectory();
      fail();
    } catch (IOException e) {
      assertThat(e.getMessage()).endsWith(" (Not a directory)");
    }
  }

  // Test directory contents
  @Test
  public void testCreateMultipleChildren() throws Exception {
    Path theDirectory = absolutize("foo/");
    theDirectory.createDirectory();
    Path newPath1 = absolutize("foo/new-file-1");
    Path newPath2 = absolutize("foo/new-file-2");
    Path newPath3 = absolutize("foo/new-file-3");

    FileSystemUtils.createEmptyFile(newPath1);
    FileSystemUtils.createEmptyFile(newPath2);
    FileSystemUtils.createEmptyFile(newPath3);

    assertThat(theDirectory.getDirectoryEntries()).containsExactly(newPath1, newPath2, newPath3);
  }

  @Test
  public void testGetDirectoryEntriesThrowsExceptionWhenRunOnFile() throws Exception {
    try {
      xFile.getDirectoryEntries();
      fail("No Exception thrown.");
    } catch (IOException ex) {
      if (ex instanceof FileNotFoundException) {
        fail("The method should throw an object of class IOException.");
      }
      assertThat(ex).hasMessage(xFile + " (Not a directory)");
    }
  }

  @Test
  public void testGetDirectoryEntriesThrowsExceptionForNonexistingPath() {
    Path somePath = testFS.getPath("/non-existing-path");
    try {
      somePath.getDirectoryEntries();
      fail("FileNotFoundException not thrown.");
    } catch (Exception x) {
      assertThat(x).hasMessage(somePath + " (No such file or directory)");
    }
  }

  // Test the removal of items
  @Test
  public void testDeleteDirectory() throws Exception {
    assertTrue(xEmptyDirectory.delete());
  }

  @Test
  public void testDeleteDirectoryIsNotDirectory() throws Exception {
    xEmptyDirectory.delete();
    assertFalse(xEmptyDirectory.isDirectory());
  }

  @Test
  public void testDeleteDirectoryParentSize() throws Exception {
    int parentSize = workingDir.getDirectoryEntries().size();
    xEmptyDirectory.delete();
    assertEquals(workingDir.getDirectoryEntries().size(), parentSize - 1);
  }

  @Test
  public void testDeleteFile() throws Exception {
    assertTrue(xFile.delete());
  }

  @Test
  public void testDeleteFileIsNotFile() throws Exception {
    xFile.delete();
    assertFalse(xEmptyDirectory.isFile());
  }

  @Test
  public void testDeleteFileParentSize() throws Exception {
    int parentSize = workingDir.getDirectoryEntries().size();
    xFile.delete();
    assertEquals(workingDir.getDirectoryEntries().size(), parentSize - 1);
  }

  @Test
  public void testDeleteRemovesCorrectFile() throws Exception {
    Path newPath1 = xEmptyDirectory.getChild("new-file-1");
    Path newPath2 = xEmptyDirectory.getChild("new-file-2");
    Path newPath3 = xEmptyDirectory.getChild("new-file-3");

    FileSystemUtils.createEmptyFile(newPath1);
    FileSystemUtils.createEmptyFile(newPath2);
    FileSystemUtils.createEmptyFile(newPath3);

    assertTrue(newPath2.delete());
    assertThat(xEmptyDirectory.getDirectoryEntries()).containsExactly(newPath1, newPath3);
  }

  @Test
  public void testDeleteNonExistingDir() throws Exception {
    Path path = xEmptyDirectory.getRelative("non-existing-dir");
    assertFalse(path.delete());
  }

  @Test
  public void testDeleteNotADirectoryPath() throws Exception {
    Path path = xFile.getChild("new-file");
    assertFalse(path.delete());
  }

  // Here we test the situations where delete should throw exceptions.
  @Test
  public void testDeleteNonEmptyDirectoryThrowsException() throws Exception {
    try {
      xNonEmptyDirectory.delete();
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage(xNonEmptyDirectory + " (Directory not empty)");
    }
  }

  @Test
  public void testDeleteNonEmptyDirectoryNotDeletedDirectory() throws Exception {
    try {
      xNonEmptyDirectory.delete();
      fail();
    } catch (IOException e) {
      // Expected
    }

    assertTrue(xNonEmptyDirectory.isDirectory());
  }

  @Test
  public void testDeleteNonEmptyDirectoryNotDeletedFile() throws Exception {
    try {
      xNonEmptyDirectory.delete();
      fail();
    } catch (IOException e) {
      // Expected
    }

    assertTrue(xNonEmptyDirectoryFoo.isFile());
  }

  @Test
  public void testCannotRemoveRoot() {
    Path rootDirectory = testFS.getRootDirectory();
    try {
      rootDirectory.delete();
      fail();
    } catch (IOException e) {
      String msg = e.getMessage();
      assertTrue(String.format("got %s want EBUSY or ENOTEMPTY", msg),
          msg.endsWith(" (Directory not empty)")
          || msg.endsWith(" (Device or resource busy)")
          || msg.endsWith(" (Is a directory)"));  // Happens on OS X.
    }
  }

  // Test the date functions
  @Test
  public void testCreateFileChangesTimeOfDirectory() throws Exception {
    storeReferenceTime(workingDir.getLastModifiedTime());
    Path newPath = absolutize("new-file");
    FileSystemUtils.createEmptyFile(newPath);
    assertTrue(isLaterThanreferenceTime(workingDir.getLastModifiedTime()));
  }

  @Test
  public void testRemoveFileChangesTimeOfDirectory() throws Exception {
    Path newPath = absolutize("new-file");
    FileSystemUtils.createEmptyFile(newPath);
    storeReferenceTime(workingDir.getLastModifiedTime());
    newPath.delete();
    assertTrue(isLaterThanreferenceTime(workingDir.getLastModifiedTime()));
  }

  // This test is a little bit strange, as we cannot test the progression
  // of the time directly. As the Java time and the OS time are slightly different.
  // Therefore, we first create an unrelated file to get a notion
  // of the current OS time and use that as a baseline.
  @Test
  public void testCreateFileTimestamp() throws Exception {
    Path syncFile = absolutize("sync-file");
    FileSystemUtils.createEmptyFile(syncFile);

    Path newFile = absolutize("new-file");
    storeReferenceTime(syncFile.getLastModifiedTime());
    FileSystemUtils.createEmptyFile(newFile);
    assertTrue(isLaterThanreferenceTime(newFile.getLastModifiedTime()));
  }

  @Test
  public void testCreateDirectoryTimestamp() throws Exception {
    Path syncFile = absolutize("sync-file");
    FileSystemUtils.createEmptyFile(syncFile);

    Path newPath = absolutize("new-dir");
    storeReferenceTime(syncFile.getLastModifiedTime());
    assertTrue(newPath.createDirectory());
    assertTrue(isLaterThanreferenceTime(newPath.getLastModifiedTime()));
  }

  @Test
  public void testWriteChangesModifiedTime() throws Exception {
    storeReferenceTime(xFile.getLastModifiedTime());
    FileSystemUtils.writeContentAsLatin1(xFile, "abc19");
    assertTrue(isLaterThanreferenceTime(xFile.getLastModifiedTime()));
  }

  @Test
  public void testGetLastModifiedTimeThrowsExceptionForNonexistingPath() throws Exception {
    Path newPath = testFS.getPath("/non-existing-dir");
    try {
      newPath.getLastModifiedTime();
      fail("FileNotFoundException not thrown!");
    } catch (FileNotFoundException x) {
      assertThat(x).hasMessage(newPath + " (No such file or directory)");
    }
  }

  // Test file size
  @Test
  public void testFileSizeThrowsExceptionForNonexistingPath() throws Exception {
    Path newPath = testFS.getPath("/non-existing-file");
    try {
      newPath.getFileSize();
      fail("FileNotFoundException not thrown.");
    } catch (FileNotFoundException e) {
      assertThat(e).hasMessage(newPath + " (No such file or directory)");
    }
  }

  @Test
  public void testFileSizeAfterWrite() throws Exception {
    String testData = "abc19";

    FileSystemUtils.writeContentAsLatin1(xFile, testData);
    assertEquals(testData.length(), xFile.getFileSize());
  }

  // Testing the input/output routines
  @Test
  public void testFileWriteAndReadAsLatin1() throws Exception {
    String testData = "abc19";

    FileSystemUtils.writeContentAsLatin1(xFile, testData);
    String resultData = new String(FileSystemUtils.readContentAsLatin1(xFile));

    assertEquals(testData,resultData);
  }

  @Test
  public void testInputAndOutputStreamEOF() throws Exception {
    try (OutputStream outStream = xFile.getOutputStream()) {
      outStream.write(1);
    }

    try (InputStream inStream = xFile.getInputStream()) {
      inStream.read();
    assertEquals(-1, inStream.read());
    }
  }

  @Test
  public void testInputAndOutputStream() throws Exception {
    try (OutputStream outStream = xFile.getOutputStream()) {
      for (int i = 33; i < 126; i++) {
        outStream.write(i);
      }
    }

    try (InputStream inStream = xFile.getInputStream()) {
      for (int i = 33; i < 126; i++) {
        int readValue = inStream.read();
        assertEquals(i, readValue);
      }
    }
  }

  @Test
  public void testInputAndOutputStreamAppend() throws Exception {
    try (OutputStream outStream = xFile.getOutputStream()) {
      for (int i = 33; i < 126; i++) {
        outStream.write(i);
      }
    }

    try (OutputStream appendOut = xFile.getOutputStream(true)) {
      for (int i = 126; i < 155; i++) {
        appendOut.write(i);
      }
    }

    try (InputStream inStream = xFile.getInputStream()) {
      for (int i = 33; i < 155; i++) {
        int readValue = inStream.read();
        assertEquals(i, readValue);
      }
    }
  }

  @Test
  public void testInputAndOutputStreamNoAppend() throws Exception {
    try (OutputStream outStream = xFile.getOutputStream()) {
      outStream.write(1);
    }

    try (OutputStream noAppendOut = xFile.getOutputStream(false)) {
    }

    try (InputStream inStream = xFile.getInputStream()) {
      assertEquals(-1, inStream.read());
    }
  }

  @Test
  public void testGetOutputStreamCreatesFile() throws Exception {
    Path newFile = absolutize("does_not_exist_yet.txt");

    try (OutputStream out = newFile.getOutputStream()) {
      out.write(42);
    }

    assertTrue(newFile.isFile());
  }

  @Test
  public void testOutputStreamThrowExceptionOnDirectory() throws Exception {
    try {
      xEmptyDirectory.getOutputStream();
      fail("The Exception was not thrown!");
    } catch (IOException ex) {
      assertThat(ex).hasMessage(xEmptyDirectory + " (Is a directory)");
    }
  }

  @Test
  public void testInputStreamThrowExceptionOnDirectory() throws Exception {
    try {
      xEmptyDirectory.getInputStream();
      fail("The Exception was not thrown!");
    } catch (IOException ex) {
      assertThat(ex).hasMessage(xEmptyDirectory + " (Is a directory)");
    }
  }

  // Test renaming
  @Test
  public void testCanRenameToUnusedName() throws Exception {
    xFile.renameTo(xNothing);
    assertFalse(xFile.exists());
    assertTrue(xNothing.isFile());
  }

  @Test
  public void testCanRenameFileToExistingFile() throws Exception {
    Path otherFile = absolutize("otherFile");
    FileSystemUtils.createEmptyFile(otherFile);
    xFile.renameTo(otherFile); // succeeds
    assertFalse(xFile.exists());
    assertTrue(otherFile.isFile());
  }

  @Test
  public void testCanRenameDirToExistingEmptyDir() throws Exception {
    xNonEmptyDirectory.renameTo(xEmptyDirectory); // succeeds
    assertFalse(xNonEmptyDirectory.exists());
    assertTrue(xEmptyDirectory.isDirectory());
    assertThat(xEmptyDirectory.getDirectoryEntries()).isNotEmpty();
  }

  @Test
  public void testCantRenameDirToExistingNonEmptyDir() throws Exception {
    try {
      xEmptyDirectory.renameTo(xNonEmptyDirectory);
      fail();
    } catch (IOException e) {
      assertThat(e.getMessage()).endsWith(" (Directory not empty)");
    }
  }

  @Test
  public void testCantRenameDirToExistingNonEmptyDirNothingChanged() throws Exception {
    try {
      xEmptyDirectory.renameTo(xNonEmptyDirectory);
      fail();
    } catch (IOException e) {
      // Expected
    }

    assertTrue(xNonEmptyDirectory.isDirectory());
    assertTrue(xEmptyDirectory.isDirectory());
    assertThat(xEmptyDirectory.getDirectoryEntries()).isEmpty();
    assertThat(xNonEmptyDirectory.getDirectoryEntries()).isNotEmpty();
  }

  @Test
  public void testCantRenameDirToExistingFile() {
    try {
      xEmptyDirectory.renameTo(xFile);
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage(xEmptyDirectory + " -> " + xFile + " (Not a directory)");
    }
  }

  @Test
  public void testCantRenameDirToExistingFileNothingChanged() {
    try {
      xEmptyDirectory.renameTo(xFile);
      fail();
    } catch (IOException e) {
      // Expected
    }

    assertTrue(xEmptyDirectory.isDirectory());
    assertTrue(xFile.isFile());
  }

  @Test
  public void testCantRenameFileToExistingDir() {
    try {
      xFile.renameTo(xEmptyDirectory);
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage(xFile + " -> " + xEmptyDirectory + " (Is a directory)");
    }
  }

  @Test
  public void testCantRenameFileToExistingDirNothingChanged() {
    try {
      xFile.renameTo(xEmptyDirectory);
      fail();
    } catch (IOException e) {
      // Expected
    }

    assertTrue(xEmptyDirectory.isDirectory());
    assertTrue(xFile.isFile());
  }

  @Test
  public void testMoveOnNonExistingFileThrowsException() throws Exception {
    Path nonExistingPath = absolutize("non-existing");
    Path targetPath = absolutize("does-not-matter");
    try {
      nonExistingPath.renameTo(targetPath);
      fail();
    } catch (FileNotFoundException e) {
      assertThat(e.getMessage()).endsWith(" (No such file or directory)");
    }
  }

  // Test the Paths
  @Test
  public void testGetPathOnlyAcceptsAbsolutePath() {
    try {
      testFS.getPath("not-absolute");
      fail("The expected Exception was not thrown.");
    } catch (IllegalArgumentException ex) {
      assertThat(ex).hasMessage("not-absolute (not an absolute path)");
    }
  }

  @Test
  public void testGetPathOnlyAcceptsAbsolutePathFragment() {
    try {
      testFS.getPath(new PathFragment("not-absolute"));
      fail("The expected Exception was not thrown.");
    } catch (IllegalArgumentException ex) {
      assertThat(ex).hasMessage("not-absolute (not an absolute path)");
    }
  }

  // Test the access permissions
  @Test
  public void testNewFilesAreWritable() throws Exception {
    assertTrue(xFile.isWritable());
  }

  @Test
  public void testNewFilesAreReadable() throws Exception {
    assertTrue(xFile.isReadable());
  }

  @Test
  public void testNewDirsAreWritable() throws Exception {
    assertTrue(xEmptyDirectory.isWritable());
  }

  @Test
  public void testNewDirsAreReadable() throws Exception {
    assertTrue(xEmptyDirectory.isReadable());
  }

  @Test
  public void testNewDirsAreExecutable() throws Exception {
    assertTrue(xEmptyDirectory.isExecutable());
  }

  @Test
  public void testCannotGetExecutableOnNonexistingFile() throws Exception {
    try {
      xNothing.isExecutable();
      fail("No exception thrown.");
    } catch (FileNotFoundException ex) {
      assertThat(ex).hasMessage(xNothing + " (No such file or directory)");
    }
  }

  @Test
  public void testCannotSetExecutableOnNonexistingFile() throws Exception {
    try {
      xNothing.setExecutable(true);
      fail("No exception thrown.");
    } catch (FileNotFoundException ex) {
      assertThat(ex).hasMessage(xNothing + " (No such file or directory)");
    }
  }

  @Test
  public void testCannotGetWritableOnNonexistingFile() throws Exception {
    try {
      xNothing.isWritable();
      fail("No exception thrown.");
    } catch (FileNotFoundException ex) {
      assertThat(ex).hasMessage(xNothing + " (No such file or directory)");
    }
  }

  @Test
  public void testCannotSetWritableOnNonexistingFile() throws Exception {
    try {
      xNothing.setWritable(false);
      fail("No exception thrown.");
    } catch (FileNotFoundException ex) {
      assertThat(ex).hasMessage(xNothing + " (No such file or directory)");
    }
  }

  @Test
  public void testSetReadableOnFile() throws Exception {
    xFile.setReadable(false);
    assertFalse(xFile.isReadable());
    xFile.setReadable(true);
    assertTrue(xFile.isReadable());
  }

  @Test
  public void testSetWritableOnFile() throws Exception {
    xFile.setWritable(false);
    assertFalse(xFile.isWritable());
    xFile.setWritable(true);
    assertTrue(xFile.isWritable());
  }

  @Test
  public void testSetExecutableOnFile() throws Exception {
    xFile.setExecutable(true);
    assertTrue(xFile.isExecutable());
    xFile.setExecutable(false);
    assertFalse(xFile.isExecutable());
  }

  @Test
  public void testSetExecutableOnDirectory() throws Exception {
    setExecutable(xNonEmptyDirectory, false);

    try {
      // We can't map names->inodes in a non-executable directory:
      xNonEmptyDirectoryFoo.isWritable(); // i.e. stat
      fail();
    } catch (IOException e) {
      assertThat(e.getMessage()).endsWith(" (Permission denied)");
    }
  }

  @Test
  public void testWritingToReadOnlyFileThrowsException() throws Exception {
    xFile.setWritable(false);
    try {
      FileSystemUtils.writeContent(xFile, "hello, world!".getBytes());
      fail("No exception thrown.");
    } catch (IOException e) {
      assertThat(e).hasMessage(xFile + " (Permission denied)");
    }
  }

  @Test
  public void testReadingFromUnreadableFileThrowsException() throws Exception {
    FileSystemUtils.writeContent(xFile, "hello, world!".getBytes());
    xFile.setReadable(false);
    try {
      FileSystemUtils.readContent(xFile);
      fail("No exception thrown.");
    } catch (IOException e) {
      assertThat(e).hasMessage(xFile + " (Permission denied)");
    }
  }

  @Test
  public void testCannotCreateFileInReadOnlyDirectory() throws Exception {
    Path xNonEmptyDirectoryBar = xNonEmptyDirectory.getChild("bar");
    xNonEmptyDirectory.setWritable(false);

    try {
      FileSystemUtils.createEmptyFile(xNonEmptyDirectoryBar);
      fail("No exception thrown.");
    } catch (IOException e) {
      assertThat(e).hasMessage(xNonEmptyDirectoryBar + " (Permission denied)");
    }
  }

  @Test
  public void testCannotCreateDirectoryInReadOnlyDirectory() throws Exception {
    Path xNonEmptyDirectoryBar = xNonEmptyDirectory.getChild("bar");
    xNonEmptyDirectory.setWritable(false);

    try {
      xNonEmptyDirectoryBar.createDirectory();
      fail("No exception thrown.");
    } catch (IOException e) {
      assertThat(e).hasMessage(xNonEmptyDirectoryBar + " (Permission denied)");
    }
  }

  @Test
  public void testCannotMoveIntoReadOnlyDirectory() throws Exception {
    Path xNonEmptyDirectoryBar = xNonEmptyDirectory.getChild("bar");
    xNonEmptyDirectory.setWritable(false);

    try {
      xFile.renameTo(xNonEmptyDirectoryBar);
      fail("No exception thrown.");
    } catch (IOException e) {
      assertThat(e.getMessage()).endsWith(" (Permission denied)");
    }
  }

  @Test
  public void testCannotMoveFromReadOnlyDirectory() throws Exception {
    xNonEmptyDirectory.setWritable(false);

    try {
      xNonEmptyDirectoryFoo.renameTo(xNothing);
      fail("No exception thrown.");
    } catch (IOException e) {
      assertThat(e.getMessage()).endsWith(" (Permission denied)");
    }
  }

  @Test
  public void testCannotDeleteInReadOnlyDirectory() throws Exception {
    xNonEmptyDirectory.setWritable(false);

    try {
      xNonEmptyDirectoryFoo.delete();
      fail("No exception thrown.");
    } catch (IOException e) {
      assertThat(e).hasMessage(xNonEmptyDirectoryFoo + " (Permission denied)");
    }
  }

  @Test
  public void testCannotCreatSymbolicLinkInReadOnlyDirectory() throws Exception {
    Path xNonEmptyDirectoryBar = xNonEmptyDirectory.getChild("bar");
    xNonEmptyDirectory.setWritable(false);

    if (supportsSymlinks) {
      try {
        createSymbolicLink(xNonEmptyDirectoryBar, xNonEmptyDirectoryFoo);
        fail("No exception thrown.");
      } catch (IOException e) {
        assertThat(e).hasMessage(xNonEmptyDirectoryBar + " (Permission denied)");
      }
    }
  }

  @Test
  public void testGetMD5DigestForEmptyFile() throws Exception {
    Fingerprint fp = new Fingerprint();
    fp.addBytes(new byte[0]);
    assertEquals(BaseEncoding.base16().lowerCase().encode(xFile.getMD5Digest()),
        fp.hexDigestAndReset());
  }

  @Test
  public void testGetMD5Digest() throws Exception {
    byte[] buffer = new byte[500000];
    for (int i = 0; i < buffer.length; ++i) {
      buffer[i] = 1;
    }
    FileSystemUtils.writeContent(xFile, buffer);
    Fingerprint fp = new Fingerprint();
    fp.addBytes(buffer);
    assertEquals(BaseEncoding.base16().lowerCase().encode(xFile.getMD5Digest()),
        fp.hexDigestAndReset());
  }

  @Test
  public void testStatFailsFastOnNonExistingFiles() throws Exception {
    try {
      xNothing.stat();
      fail("Expected IOException");
    } catch(IOException e) {
      // Do nothing.
    }
  }

  @Test
  public void testStatNullableFailsFastOnNonExistingFiles() throws Exception {
    assertNull(xNothing.statNullable());
  }

  @Test
  public void testResolveSymlinks() throws Exception {
    if (supportsSymlinks) {
      createSymbolicLink(xLink, xFile);
      FileSystemUtils.createEmptyFile(xFile);
      assertEquals(xFile.asFragment(), testFS.resolveOneLink(xLink));
      assertEquals(xFile, xLink.resolveSymbolicLinks());
    }
  }

  @Test
  public void testResolveDanglingSymlinks() throws Exception {
    if (supportsSymlinks) {
      createSymbolicLink(xLink, xNothing);
      assertEquals(xNothing.asFragment(), testFS.resolveOneLink(xLink));
      try {
        xLink.resolveSymbolicLinks();
        fail();
      } catch (IOException expected) {
      }
    }
  }

  @Test
  public void testResolveNonSymlinks() throws Exception {
    if (supportsSymlinks) {
      assertNull(testFS.resolveOneLink(xFile));
      assertEquals(xFile, xFile.resolveSymbolicLinks());
    }
  }

}
