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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.unix.FileStatus;
import com.google.devtools.build.lib.unix.NativePosixFiles;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.FileAlreadyExistsException;
import java.util.Collection;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

/**
 * This class handles the generic tests that any filesystem must pass.
 *
 * <p>Each filesystem-test should inherit from this class, thereby obtaining
 * all the tests.
 */
@RunWith(Parameterized.class)
public abstract class FileSystemTest {

  private long savedTime;
  protected FileSystem testFS;
  protected Path workingDir;

  // Some useful examples of various kinds of files (mnemonic: "x" = "eXample")
  protected Path xNothing;
  protected Path xLink;
  protected Path xFile;
  protected Path xNonEmptyDirectory;
  protected Path xNonEmptyDirectoryFoo;
  protected Path xEmptyDirectory;

  @Parameters(name = "{index}: digestHashFunction={0}")
  public static Collection<DigestHashFunction[]> hashFunctions() {
    // TODO(b/112537387): Remove the array-ification and return Collection<DigestHashFunction>. This
    // is possible in Junit4.12, but 4.11 requires the array. Bazel 0.18 will have Junit4.12, so
    // this can change then.
    return DigestHashFunction.getPossibleHashFunctions()
        .stream()
        .map(dhf -> new DigestHashFunction[] {dhf})
        .collect(ImmutableList.toImmutableList());
  }

  @Parameter public DigestHashFunction digestHashFunction;

  @Before
  public final void createDirectories() throws Exception  {
    testFS = getFreshFileSystem(digestHashFunction);
    workingDir = testFS.getPath(getTestTmpDir());
    cleanUpWorkingDirectory(workingDir);

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

  @After
  public final void destroyFileSystem() throws Exception  {
    destroyFileSystem(testFS);
  }

  /**
   * Returns an instance of the file system to test.
   */
  protected abstract FileSystem getFreshFileSystem(DigestHashFunction digestHashFunction)
      throws IOException;

  protected boolean isSymbolicLink(File file) throws IOException {
    return NativePosixFiles.lstat(file.getPath()).isSymbolicLink();
  }

  private static final Pattern STAT_SUBDIR_ERROR = Pattern.compile("(.*) \\(Not a directory\\)");

  // Test that file is not present, using statIfFound. Base implementation throws an exception, but
  // subclasses may override statIfFound to return null, in which case their tests should override
  // this method.
  @SuppressWarnings("unused") // Subclasses may throw.
  protected void expectNotFound(Path path) throws IOException {
    try {
      assertThat(path.statIfFound()).isNull();
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
    workingPath.createDirectoryAndParents();
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
    NativePosixFiles.chmod(
        directoryToRemove.getPath(),
        NativePosixFiles.lstat(directoryToRemove.getPath()).getPermissions()
            | FileStatus.S_IWUSR
            | FileStatus.S_IXUSR);

    File[] files = directoryToRemove.listFiles();
    if (files != null) {
      for (File currentFile : files) {
        boolean isSymbolicLink = isSymbolicLink(currentFile);
        if (!isSymbolicLink && currentFile.isDirectory()) {
          removeEntireDirectory(currentFile);
        } else {
          if (!isSymbolicLink) {
            NativePosixFiles.chmod(
                currentFile.getPath(),
                NativePosixFiles.lstat(currentFile.getPath()).getPermissions()
                    | FileStatus.S_IWUSR);
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
    assertThat(nonExistingPath.isFile()).isFalse();
  }

  @Test
  public void testIsDirectoryForNonexistingPath() {
    Path nonExistingPath = testFS.getPath("/something/strange");
    assertThat(nonExistingPath.isDirectory()).isFalse();
  }

  @Test
  public void testIsLinkForNonexistingPath() {
    Path nonExistingPath = testFS.getPath("/something/strange");
    assertThat(nonExistingPath.isSymbolicLink()).isFalse();
  }

  @Test
  public void testExistsForNonexistingPath() throws Exception {
    Path nonExistingPath = testFS.getPath("/something/strange");
    assertThat(nonExistingPath.exists()).isFalse();
    expectNotFound(nonExistingPath);
  }

  @Test
  public void testBadPermissionsThrowsExceptionOnStatIfFound() throws Exception {
    Path inaccessible = absolutize("inaccessible");
    inaccessible.createDirectory();
    Path child = inaccessible.getChild("child");
    FileSystemUtils.createEmptyFile(child);
    inaccessible.setExecutable(false);
    assertThat(child.exists()).isFalse();
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
    assertThat(nonDir.getRelative("file").statIfFound()).isNull();
  }

  // The following tests check the handling of the current working directory.
  @Test
  public void testCreatePathRelativeToWorkingDirectory() {
    Path relativeCreatedPath = absolutize("some-file");
    Path expectedResult = workingDir.getRelative(PathFragment.create("some-file"));

    assertThat(relativeCreatedPath).isEqualTo(expectedResult);
  }

  // The following tests check the handling of the root directory
  @Test
  public void testRootIsDirectory() {
    Path rootPath = testFS.getPath("/");
    assertThat(rootPath.isDirectory()).isTrue();
  }

  @Test
  public void testRootHasNoParent() {
    Path rootPath = testFS.getPath("/");
    assertThat(rootPath.getParentDirectory()).isNull();
  }

  // The following functions test the creation of files/links/directories.
  @Test
  public void testFileExists() throws Exception {
    Path someFile = absolutize("some-file");
    FileSystemUtils.createEmptyFile(someFile);
    assertThat(someFile.exists()).isTrue();
    assertThat(someFile.statIfFound()).isNotNull();
  }

  @Test
  public void testFileIsFile() throws Exception {
    Path someFile = absolutize("some-file");
    FileSystemUtils.createEmptyFile(someFile);
    assertThat(someFile.isFile()).isTrue();
  }

  @Test
  public void testFileIsNotDirectory() throws Exception {
    Path someFile = absolutize("some-file");
    FileSystemUtils.createEmptyFile(someFile);
    assertThat(someFile.isDirectory()).isFalse();
  }

  @Test
  public void testFileIsNotSymbolicLink() throws Exception {
    Path someFile = absolutize("some-file");
    FileSystemUtils.createEmptyFile(someFile);
    assertThat(someFile.isSymbolicLink()).isFalse();
  }

  @Test
  public void testDirectoryExists() throws Exception {
    Path someDirectory = absolutize("some-dir");
    someDirectory.createDirectory();
    assertThat(someDirectory.exists()).isTrue();
    assertThat(someDirectory.statIfFound()).isNotNull();
  }

  @Test
  public void testDirectoryIsDirectory() throws Exception {
    Path someDirectory = absolutize("some-dir");
    someDirectory.createDirectory();
    assertThat(someDirectory.isDirectory()).isTrue();
  }

  @Test
  public void testDirectoryIsNotFile() throws Exception {
    Path someDirectory = absolutize("some-dir");
    someDirectory.createDirectory();
    assertThat(someDirectory.isFile()).isFalse();
  }

  @Test
  public void testDirectoryIsNotSymbolicLink() throws Exception {
    Path someDirectory = absolutize("some-dir");
    someDirectory.createDirectory();
    assertThat(someDirectory.isSymbolicLink()).isFalse();
  }

  @Test
  public void testSymbolicFileLinkExists() throws Exception {
    Path someLink = absolutize("some-link");
    if (testFS.supportsSymbolicLinksNatively(someLink)) {
      someLink.createSymbolicLink(xFile);
      assertThat(someLink.exists()).isTrue();
      assertThat(someLink.statIfFound()).isNotNull();
    }
  }

  @Test
  public void testSymbolicFileLinkIsSymbolicLink() throws Exception {
    Path someLink = absolutize("some-link");
    if (testFS.supportsSymbolicLinksNatively(someLink)) {
      someLink.createSymbolicLink(xFile);
      assertThat(someLink.isSymbolicLink()).isTrue();
    }
  }

  @Test
  public void testSymbolicFileLinkIsFile() throws Exception {
    Path someLink = absolutize("some-link");
    if (testFS.supportsSymbolicLinksNatively(someLink)) {
      someLink.createSymbolicLink(xFile);
      assertThat(someLink.isFile()).isTrue();
    }
  }

  @Test
  public void testSymbolicFileLinkIsNotDirectory() throws Exception {
    Path someLink = absolutize("some-link");
    if (testFS.supportsSymbolicLinksNatively(someLink)) {
      someLink.createSymbolicLink(xFile);
      assertThat(someLink.isDirectory()).isFalse();
    }
  }

  @Test
  public void testSymbolicDirLinkExists() throws Exception {
    Path someLink = absolutize("some-link");
    if (testFS.supportsSymbolicLinksNatively(someLink)) {
      someLink.createSymbolicLink(xEmptyDirectory);
      assertThat(someLink.exists()).isTrue();
      assertThat(someLink.statIfFound()).isNotNull();
    }
  }

  @Test
  public void testSymbolicDirLinkIsSymbolicLink() throws Exception {
    Path someLink = absolutize("some-link");
    if (testFS.supportsSymbolicLinksNatively(someLink)) {
      someLink.createSymbolicLink(xEmptyDirectory);
      assertThat(someLink.isSymbolicLink()).isTrue();
    }
  }

  @Test
  public void testSymbolicDirLinkIsDirectory() throws Exception {
    Path someLink = absolutize("some-link");
    if (testFS.supportsSymbolicLinksNatively(someLink)) {
      someLink.createSymbolicLink(xEmptyDirectory);
      assertThat(someLink.isDirectory()).isTrue();
    }
  }

  @Test
  public void testSymbolicDirLinkIsNotFile() throws Exception {
    Path someLink = absolutize("some-link");
    if (testFS.supportsSymbolicLinksNatively(someLink)) {
      someLink.createSymbolicLink(xEmptyDirectory);
      assertThat(someLink.isFile()).isFalse();
    }
  }

  @Test
  public void testChildOfNonDirectory() throws Exception {
    Path somePath = absolutize("file-name");
    FileSystemUtils.createEmptyFile(somePath);
    Path childOfNonDir = somePath.getChild("child");
    assertThat(childOfNonDir.exists()).isFalse();
    expectNotFound(childOfNonDir);
  }

  @Test
  public void testCreateDirectoryIsEmpty() throws Exception {
    Path newPath = xEmptyDirectory.getChild("new-dir");
    newPath.createDirectory();
    assertThat(newPath.getDirectoryEntries()).isEmpty();
  }

  @Test
  public void testCreateDirectoryIsOnlyChildInParent() throws Exception {
    Path newPath = xEmptyDirectory.getChild("new-dir");
    newPath.createDirectory();
    assertThat(newPath.getParentDirectory().getDirectoryEntries()).hasSize(1);
    assertThat(newPath.getParentDirectory().getDirectoryEntries()).containsExactly(newPath);
  }

  @Test
  public void testCreateDirectoryAndParents() throws Exception {
    Path newPath = absolutize("new-dir/sub/directory");
    newPath.createDirectoryAndParents();
    assertThat(newPath.isDirectory()).isTrue();
  }

  @Test
  public void testCreateDirectoryAndParentsCreatesEmptyDirectory() throws Exception {
    Path newPath = absolutize("new-dir/sub/directory");
    newPath.createDirectoryAndParents();
    assertThat(newPath.getDirectoryEntries()).isEmpty();
  }

  @Test
  public void testCreateDirectoryAndParentsIsOnlyChildInParent() throws Exception {
    Path newPath = absolutize("new-dir/sub/directory");
    newPath.createDirectoryAndParents();
    assertThat(newPath.getParentDirectory().getDirectoryEntries()).hasSize(1);
    assertThat(newPath.getParentDirectory().getDirectoryEntries()).containsExactly(newPath);
  }

  @Test
  public void testCreateDirectoryAndParentsWhenAlreadyExistsSucceeds() throws Exception {
    Path newPath = absolutize("new-dir");
    newPath.createDirectory();
    newPath.createDirectoryAndParents();
    assertThat(newPath.isDirectory()).isTrue();
  }

  @Test
  public void testCreateDirectoryAndParentsWhenAncestorIsFile() throws IOException {
    Path path = absolutize("somewhere/deep/in");
    path.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.createEmptyFile(path);
    Path theHierarchy = path.getChild("the-hierarchy");
    MoreAsserts.assertThrows(IOException.class, theHierarchy::createDirectoryAndParents);
  }

  @Test
  public void testCreateDirectoryAndParentsWhenSymlinkToDir() throws IOException {
    Path somewhereDeepIn = absolutize("somewhere/deep/in");
    somewhereDeepIn.createDirectoryAndParents();
    Path realDir = absolutize("real/dir");
    realDir.createDirectoryAndParents();
    assertThat(realDir.isDirectory()).isTrue();
    Path theHierarchy = somewhereDeepIn.getChild("the-hierarchy");
    theHierarchy.createSymbolicLink(realDir);
    assertThat(theHierarchy.isDirectory()).isTrue();
    theHierarchy.createDirectoryAndParents();
  }

  @Test
  public void testCreateDirectoryAndParentsWhenSymlinkEmbedded() throws IOException {
    Path somewhereDeepIn = absolutize("somewhere/deep/in");
    somewhereDeepIn.createDirectoryAndParents();
    Path realDir = absolutize("real/dir");
    realDir.createDirectoryAndParents();
    Path the = somewhereDeepIn.getChild("the");
    the.createSymbolicLink(realDir);
    Path theHierarchy = somewhereDeepIn.getChild("hierarchy");
    theHierarchy.createDirectoryAndParents();
  }

  @Test
  public void testCreateDirectoryAtFileFails() throws Exception {
    Path newPath = absolutize("file");
    FileSystemUtils.createEmptyFile(newPath);
    MoreAsserts.assertThrows(IOException.class, newPath::createDirectoryAndParents);
  }

  @Test
  public void testCreateEmptyFileIsEmpty() throws Exception {
    Path newPath = xEmptyDirectory.getChild("new-file");
    FileSystemUtils.createEmptyFile(newPath);

    assertThat(newPath.getFileSize()).isEqualTo(0);
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
    assertThat(xEmptyDirectory.createDirectory()).isFalse();
  }

  @Test
  public void testCreateDirectoryWhereFileAlreadyExists() {
    try {
      xFile.createDirectory();
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessageThat().isEqualTo(xFile + " (File exists)");
    }
  }

  @Test
  public void testCannotCreateDirectoryWithoutExistingParent() throws Exception {
    Path newPath = testFS.getPath("/deep/new-dir");
    try {
      newPath.createDirectory();
      fail();
    } catch (FileNotFoundException e) {
      assertThat(e).hasMessageThat().endsWith(" (No such file or directory)");
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
      assertThat(e).hasMessageThat().isEqualTo(xChildOfReadonlyDir + " (Permission denied)");
    }
  }

  @Test
  public void testCannotCreateFileWithoutExistingParent() throws Exception {
    Path newPath = testFS.getPath("/non-existing-dir/new-file");
    try {
      FileSystemUtils.createEmptyFile(newPath);
      fail();
    } catch (FileNotFoundException e) {
      assertThat(e).hasMessageThat().endsWith(" (No such file or directory)");
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
      assertThat(e).hasMessageThat().isEqualTo(xChildOfReadonlyDir + " (Permission denied)");
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
      assertThat(e).hasMessageThat().endsWith(" (Not a directory)");
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
      assertThat(e).hasMessageThat().endsWith(" (Not a directory)");
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
      assertThat(ex).hasMessageThat().isEqualTo(xFile + " (Not a directory)");
    }
  }

  @Test
  public void testGetDirectoryEntriesThrowsExceptionForNonexistingPath() {
    Path somePath = testFS.getPath("/non-existing-path");
    try {
      somePath.getDirectoryEntries();
      fail("FileNotFoundException not thrown.");
    } catch (Exception x) {
      assertThat(x).hasMessageThat().isEqualTo(somePath + " (No such file or directory)");
    }
  }

  // Test the removal of items
  @Test
  public void testDeleteDirectory() throws Exception {
    assertThat(xEmptyDirectory.delete()).isTrue();
  }

  @Test
  public void testDeleteDirectoryIsNotDirectory() throws Exception {
    xEmptyDirectory.delete();
    assertThat(xEmptyDirectory.isDirectory()).isFalse();
  }

  @Test
  public void testDeleteDirectoryParentSize() throws Exception {
    int parentSize = workingDir.getDirectoryEntries().size();
    xEmptyDirectory.delete();
    assertThat(parentSize - 1).isEqualTo(workingDir.getDirectoryEntries().size());
  }

  @Test
  public void testDeleteFile() throws Exception {
    assertThat(xFile.delete()).isTrue();
  }

  @Test
  public void testDeleteFileIsNotFile() throws Exception {
    xFile.delete();
    assertThat(xEmptyDirectory.isFile()).isFalse();
  }

  @Test
  public void testDeleteFileParentSize() throws Exception {
    int parentSize = workingDir.getDirectoryEntries().size();
    xFile.delete();
    assertThat(parentSize - 1).isEqualTo(workingDir.getDirectoryEntries().size());
  }

  @Test
  public void testDeleteRemovesCorrectFile() throws Exception {
    Path newPath1 = xEmptyDirectory.getChild("new-file-1");
    Path newPath2 = xEmptyDirectory.getChild("new-file-2");
    Path newPath3 = xEmptyDirectory.getChild("new-file-3");

    FileSystemUtils.createEmptyFile(newPath1);
    FileSystemUtils.createEmptyFile(newPath2);
    FileSystemUtils.createEmptyFile(newPath3);

    assertThat(newPath2.delete()).isTrue();
    assertThat(xEmptyDirectory.getDirectoryEntries()).containsExactly(newPath1, newPath3);
  }

  @Test
  public void testDeleteNonExistingDir() throws Exception {
    Path path = xEmptyDirectory.getRelative("non-existing-dir");
    assertThat(path.delete()).isFalse();
  }

  @Test
  public void testDeleteNotADirectoryPath() throws Exception {
    Path path = xFile.getChild("new-file");
    assertThat(path.delete()).isFalse();
  }

  // Here we test the situations where delete should throw exceptions.
  @Test
  public void testDeleteNonEmptyDirectoryThrowsException() throws Exception {
    try {
      xNonEmptyDirectory.delete();
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessageThat().isEqualTo(xNonEmptyDirectory + " (Directory not empty)");
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

    assertThat(xNonEmptyDirectory.isDirectory()).isTrue();
  }

  @Test
  public void testDeleteNonEmptyDirectoryNotDeletedFile() throws Exception {
    try {
      xNonEmptyDirectory.delete();
      fail();
    } catch (IOException e) {
      // Expected
    }

    assertThat(xNonEmptyDirectoryFoo.isFile()).isTrue();
  }

  @Test
  public void testDeleteTreeDeletesContents() throws IOException {
    Path topDir = absolutize("top-dir");
    Path file1 = absolutize("top-dir/file-1");
    Path file2 = absolutize("top-dir/file-2");
    Path aDir = absolutize("top-dir/a-dir");
    Path file3 = absolutize("top-dir/a-dir/file-3");
    Path file4 = absolutize("file-4");

    topDir.createDirectory();
    FileSystemUtils.createEmptyFile(file1);
    FileSystemUtils.createEmptyFile(file2);
    aDir.createDirectory();
    FileSystemUtils.createEmptyFile(file3);
    FileSystemUtils.createEmptyFile(file4);

    topDir.deleteTree();
    assertThat(file4.exists()).isTrue();
    assertThat(topDir.exists()).isFalse();
    assertThat(file1.exists()).isFalse();
    assertThat(file2.exists()).isFalse();
    assertThat(aDir.exists()).isFalse();
    assertThat(file3.exists()).isFalse();
  }

  @Test
  public void testDeleteTreeDeletesUnreadableDirectories() throws IOException {
    Path topDir = absolutize("top-dir");
    Path aDir = absolutize("top-dir/a-dir");
    Path file1 = absolutize("top-dir/a-dir/file1");
    Path file2 = absolutize("top-dir/a-dir/file2");

    topDir.createDirectory();
    aDir.createDirectory();
    FileSystemUtils.createEmptyFile(file1);
    FileSystemUtils.createEmptyFile(file2);

    try {
      aDir.setReadable(false);
      aDir.setExecutable(false);
    } catch (UnsupportedOperationException e) {
      // Skip testing if the file system does not support clearing the needed attibutes.
      return;
    }

    topDir.deleteTree();
    assertThat(topDir.exists()).isFalse();
    assertThat(aDir.exists()).isFalse();
    assertThat(file1.exists()).isFalse();
    assertThat(file2.exists()).isFalse();
  }

  @Test
  public void testDeleteTreeDeletesUnwritableDirectories() throws IOException {
    Path topDir = absolutize("top-dir");
    Path aDir = absolutize("top-dir/a-dir");
    Path file1 = absolutize("top-dir/a-dir/file1");
    Path file2 = absolutize("top-dir/a-dir/file2");

    topDir.createDirectory();
    aDir.createDirectory();
    FileSystemUtils.createEmptyFile(file1);
    FileSystemUtils.createEmptyFile(file2);

    try {
      aDir.setWritable(false);
    } catch (UnsupportedOperationException e) {
      // Skip testing if the file system does not support clearing the needed attibutes.
      return;
    }

    topDir.deleteTree();
    assertThat(topDir.exists()).isFalse();
    assertThat(aDir.exists()).isFalse();
    assertThat(file1.exists()).isFalse();
    assertThat(file2.exists()).isFalse();
  }

  @Test
  public void testDeleteTreeDoesNotFollowInnerLinks() throws IOException {
    Path topDir = absolutize("top-dir");
    Path file = absolutize("file");
    Path outboundLink = absolutize("top-dir/outbound-link");

    topDir.createDirectory();
    FileSystemUtils.createEmptyFile(file);
    outboundLink.createSymbolicLink(file);

    topDir.deleteTree();
    assertThat(file.exists()).isTrue();
    assertThat(topDir.exists()).isFalse();
  }

  @Test
  public void testDeleteTreeDoesNotFollowTopLink() throws IOException {
    Path topDir = absolutize("top-dir");
    Path file = absolutize("file");

    FileSystemUtils.createEmptyFile(file);
    topDir.createSymbolicLink(file);

    topDir.deleteTree();
    assertThat(file.exists()).isTrue();
    assertThat(topDir.exists()).isFalse();
  }

  @Test
  public void testTreesDeletesBelowDeletesContentsOnly() throws IOException {
    Path topDir = absolutize("top-dir");
    Path file = absolutize("top-dir/file");
    Path subdir = absolutize("top-dir/subdir");

    topDir.createDirectory();
    FileSystemUtils.createEmptyFile(file);
    subdir.createDirectory();

    topDir.deleteTreesBelow();
    assertThat(topDir.exists()).isTrue();
    assertThat(file.exists()).isFalse();
    assertThat(subdir.exists()).isFalse();
  }

  @Test
  public void testTreesDeletesBelowIgnoresMissingTopDir() throws IOException {
    Path topDir = absolutize("top-dir");

    assertThat(topDir.exists()).isFalse();
    topDir.deleteTreesBelow(); // Expect no exception.
    assertThat(topDir.exists()).isFalse();
  }

  @Test
  public void testTreesDeletesBelowIgnoresNonDirectories() throws IOException {
    Path topFile = absolutize("top-file");

    FileSystemUtils.createEmptyFile(topFile);

    assertThat(topFile.exists()).isTrue();
    topFile.deleteTreesBelow(); // Expect no exception.
    assertThat(topFile.exists()).isTrue();
  }

  // Test the date functions
  @Test
  public void testCreateFileChangesTimeOfDirectory() throws Exception {
    storeReferenceTime(workingDir.getLastModifiedTime());
    Path newPath = absolutize("new-file");
    FileSystemUtils.createEmptyFile(newPath);
    assertThat(isLaterThanreferenceTime(workingDir.getLastModifiedTime())).isTrue();
  }

  @Test
  public void testRemoveFileChangesTimeOfDirectory() throws Exception {
    Path newPath = absolutize("new-file");
    FileSystemUtils.createEmptyFile(newPath);
    storeReferenceTime(workingDir.getLastModifiedTime());
    newPath.delete();
    assertThat(isLaterThanreferenceTime(workingDir.getLastModifiedTime())).isTrue();
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
    assertThat(isLaterThanreferenceTime(newFile.getLastModifiedTime())).isTrue();
  }

  @Test
  public void testCreateDirectoryTimestamp() throws Exception {
    Path syncFile = absolutize("sync-file");
    FileSystemUtils.createEmptyFile(syncFile);

    Path newPath = absolutize("new-dir");
    storeReferenceTime(syncFile.getLastModifiedTime());
    assertThat(newPath.createDirectory()).isTrue();
    assertThat(isLaterThanreferenceTime(newPath.getLastModifiedTime())).isTrue();
  }

  @Test
  public void testWriteChangesModifiedTime() throws Exception {
    storeReferenceTime(xFile.getLastModifiedTime());
    FileSystemUtils.writeContentAsLatin1(xFile, "abc19");
    assertThat(isLaterThanreferenceTime(xFile.getLastModifiedTime())).isTrue();
  }

  @Test
  public void testGetLastModifiedTimeThrowsExceptionForNonexistingPath() throws Exception {
    Path newPath = testFS.getPath("/non-existing-dir");
    try {
      newPath.getLastModifiedTime();
      fail("FileNotFoundException not thrown!");
    } catch (FileNotFoundException x) {
      assertThat(x).hasMessageThat().isEqualTo(newPath + " (No such file or directory)");
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
      assertThat(e).hasMessageThat().isEqualTo(newPath + " (No such file or directory)");
    }
  }

  @Test
  public void testFileSizeAfterWrite() throws Exception {
    String testData = "abc19";

    FileSystemUtils.writeContentAsLatin1(xFile, testData);
    assertThat(xFile.getFileSize()).isEqualTo(testData.length());
  }

  // Testing the input/output routines
  @Test
  public void testFileWriteAndReadAsLatin1() throws Exception {
    String testData = "abc19";

    FileSystemUtils.writeContentAsLatin1(xFile, testData);
    String resultData = new String(FileSystemUtils.readContentAsLatin1(xFile));

    assertThat(resultData).isEqualTo(testData);
  }

  @Test
  public void testInputAndOutputStreamEOF() throws Exception {
    try (OutputStream outStream = xFile.getOutputStream()) {
      outStream.write(1);
    }

    try (InputStream inStream = xFile.getInputStream()) {
      inStream.read();
      assertThat(inStream.read()).isEqualTo(-1);
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
        assertThat(readValue).isEqualTo(i);
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
        assertThat(readValue).isEqualTo(i);
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
      assertThat(inStream.read()).isEqualTo(-1);
    }
  }

  @Test
  public void testGetOutputStreamCreatesFile() throws Exception {
    Path newFile = absolutize("does_not_exist_yet.txt");

    try (OutputStream out = newFile.getOutputStream()) {
      out.write(42);
    }

    assertThat(newFile.isFile()).isTrue();
  }

  @Test
  public void testOutputStreamThrowExceptionOnDirectory() throws Exception {
    try {
      xEmptyDirectory.getOutputStream();
      fail("The Exception was not thrown!");
    } catch (IOException ex) {
      assertThat(ex).hasMessageThat().isEqualTo(xEmptyDirectory + " (Is a directory)");
    }
  }

  @Test
  public void testInputStreamThrowExceptionOnDirectory() throws Exception {
    try {
      xEmptyDirectory.getInputStream();
      fail("The Exception was not thrown!");
    } catch (IOException ex) {
      assertThat(ex).hasMessageThat().isEqualTo(xEmptyDirectory + " (Is a directory)");
    }
  }

  // Test renaming
  @Test
  public void testCanRenameToUnusedName() throws Exception {
    xFile.renameTo(xNothing);
    assertThat(xFile.exists()).isFalse();
    assertThat(xNothing.isFile()).isTrue();
  }

  @Test
  public void testCanRenameFileToExistingFile() throws Exception {
    Path otherFile = absolutize("otherFile");
    FileSystemUtils.createEmptyFile(otherFile);
    xFile.renameTo(otherFile); // succeeds
    assertThat(xFile.exists()).isFalse();
    assertThat(otherFile.isFile()).isTrue();
  }

  @Test
  public void testCanRenameDirToExistingEmptyDir() throws Exception {
    xNonEmptyDirectory.renameTo(xEmptyDirectory); // succeeds
    assertThat(xNonEmptyDirectory.exists()).isFalse();
    assertThat(xEmptyDirectory.isDirectory()).isTrue();
    assertThat(xEmptyDirectory.getDirectoryEntries()).isNotEmpty();
  }

  @Test
  public void testCantRenameDirToExistingNonEmptyDir() throws Exception {
    try {
      xEmptyDirectory.renameTo(xNonEmptyDirectory);
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessageThat().containsMatch("\\((File exists|Directory not empty)\\)$");
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

    assertThat(xNonEmptyDirectory.isDirectory()).isTrue();
    assertThat(xEmptyDirectory.isDirectory()).isTrue();
    assertThat(xEmptyDirectory.getDirectoryEntries()).isEmpty();
    assertThat(xNonEmptyDirectory.getDirectoryEntries()).isNotEmpty();
  }

  @Test
  public void testCantRenameDirToExistingFile() {
    try {
      xEmptyDirectory.renameTo(xFile);
      fail();
    } catch (IOException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(xEmptyDirectory + " -> " + xFile + " (Not a directory)");
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

    assertThat(xEmptyDirectory.isDirectory()).isTrue();
    assertThat(xFile.isFile()).isTrue();
  }

  @Test
  public void testCantRenameFileToExistingDir() {
    try {
      xFile.renameTo(xEmptyDirectory);
      fail();
    } catch (IOException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(xFile + " -> " + xEmptyDirectory + " (Is a directory)");
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

    assertThat(xEmptyDirectory.isDirectory()).isTrue();
    assertThat(xFile.isFile()).isTrue();
  }

  @Test
  public void testMoveOnNonExistingFileThrowsException() throws Exception {
    Path nonExistingPath = absolutize("non-existing");
    Path targetPath = absolutize("does-not-matter");
    try {
      nonExistingPath.renameTo(targetPath);
      fail();
    } catch (FileNotFoundException e) {
      assertThat(e).hasMessageThat().endsWith(" (No such file or directory)");
    }
  }

  // Test the Paths
  @Test
  public void testGetPathOnlyAcceptsAbsolutePath() {
    MoreAsserts.assertThrows(IllegalArgumentException.class, () -> testFS.getPath("not-absolute"));
  }

  @Test
  public void testGetPathOnlyAcceptsAbsolutePathFragment() {
    MoreAsserts.assertThrows(
        IllegalArgumentException.class, () -> testFS.getPath(PathFragment.create("not-absolute")));
  }

  // Test the access permissions
  @Test
  public void testNewFilesAreWritable() throws Exception {
    assertThat(xFile.isWritable()).isTrue();
  }

  @Test
  public void testNewFilesAreReadable() throws Exception {
    assertThat(xFile.isReadable()).isTrue();
  }

  @Test
  public void testNewDirsAreWritable() throws Exception {
    assertThat(xEmptyDirectory.isWritable()).isTrue();
  }

  @Test
  public void testNewDirsAreReadable() throws Exception {
    assertThat(xEmptyDirectory.isReadable()).isTrue();
  }

  @Test
  public void testNewDirsAreExecutable() throws Exception {
    assertThat(xEmptyDirectory.isExecutable()).isTrue();
  }

  @Test
  public void testCannotGetExecutableOnNonexistingFile() throws Exception {
    try {
      xNothing.isExecutable();
      fail("No exception thrown.");
    } catch (FileNotFoundException ex) {
      assertThat(ex).hasMessageThat().isEqualTo(xNothing + " (No such file or directory)");
    }
  }

  @Test
  public void testCannotSetExecutableOnNonexistingFile() throws Exception {
    try {
      xNothing.setExecutable(true);
      fail("No exception thrown.");
    } catch (FileNotFoundException ex) {
      assertThat(ex).hasMessageThat().isEqualTo(xNothing + " (No such file or directory)");
    }
  }

  @Test
  public void testCannotGetWritableOnNonexistingFile() throws Exception {
    try {
      xNothing.isWritable();
      fail("No exception thrown.");
    } catch (FileNotFoundException ex) {
      assertThat(ex).hasMessageThat().isEqualTo(xNothing + " (No such file or directory)");
    }
  }

  @Test
  public void testCannotSetWritableOnNonexistingFile() throws Exception {
    try {
      xNothing.setWritable(false);
      fail("No exception thrown.");
    } catch (FileNotFoundException ex) {
      assertThat(ex).hasMessageThat().isEqualTo(xNothing + " (No such file or directory)");
    }
  }

  @Test
  public void testSetReadableOnFile() throws Exception {
    xFile.setReadable(false);
    assertThat(xFile.isReadable()).isFalse();
    xFile.setReadable(true);
    assertThat(xFile.isReadable()).isTrue();
  }

  @Test
  public void testSetWritableOnFile() throws Exception {
    xFile.setWritable(false);
    assertThat(xFile.isWritable()).isFalse();
    xFile.setWritable(true);
    assertThat(xFile.isWritable()).isTrue();
  }

  @Test
  public void testSetExecutableOnFile() throws Exception {
    xFile.setExecutable(true);
    assertThat(xFile.isExecutable()).isTrue();
    xFile.setExecutable(false);
    assertThat(xFile.isExecutable()).isFalse();
  }

  @Test
  public void testSetExecutableOnDirectory() throws Exception {
    setExecutable(xNonEmptyDirectory, false);

    try {
      // We can't map names->inodes in a non-executable directory:
      xNonEmptyDirectoryFoo.isWritable(); // i.e. stat
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessageThat().endsWith(" (Permission denied)");
    }
  }

  @Test
  public void testWritingToReadOnlyFileThrowsException() throws Exception {
    xFile.setWritable(false);
    try {
      FileSystemUtils.writeContent(xFile, "hello, world!".getBytes(UTF_8));
      fail("No exception thrown.");
    } catch (IOException e) {
      assertThat(e).hasMessageThat().isEqualTo(xFile + " (Permission denied)");
    }
  }

  @Test
  public void testReadingFromUnreadableFileThrowsException() throws Exception {
    FileSystemUtils.writeContent(xFile, "hello, world!".getBytes(UTF_8));
    xFile.setReadable(false);
    try {
      FileSystemUtils.readContent(xFile);
      fail("No exception thrown.");
    } catch (IOException e) {
      assertThat(e).hasMessageThat().isEqualTo(xFile + " (Permission denied)");
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
      assertThat(e).hasMessageThat().isEqualTo(xNonEmptyDirectoryBar + " (Permission denied)");
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
      assertThat(e).hasMessageThat().isEqualTo(xNonEmptyDirectoryBar + " (Permission denied)");
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
      assertThat(e).hasMessageThat().endsWith(" (Permission denied)");
    }
  }

  @Test
  public void testCannotMoveFromReadOnlyDirectory() throws Exception {
    xNonEmptyDirectory.setWritable(false);

    try {
      xNonEmptyDirectoryFoo.renameTo(xNothing);
      fail("No exception thrown.");
    } catch (IOException e) {
      assertThat(e).hasMessageThat().endsWith(" (Permission denied)");
    }
  }

  @Test
  public void testCannotDeleteInReadOnlyDirectory() throws Exception {
    xNonEmptyDirectory.setWritable(false);

    try {
      xNonEmptyDirectoryFoo.delete();
      fail("No exception thrown.");
    } catch (IOException e) {
      assertThat(e).hasMessageThat().isEqualTo(xNonEmptyDirectoryFoo + " (Permission denied)");
    }
  }

  @Test
  public void testCannotCreatSymbolicLinkInReadOnlyDirectory() throws Exception {
    Path xNonEmptyDirectoryBar = xNonEmptyDirectory.getChild("bar");
    xNonEmptyDirectory.setWritable(false);

    if (testFS.supportsSymbolicLinksNatively(xNonEmptyDirectoryBar)) {
      try {
        createSymbolicLink(xNonEmptyDirectoryBar, xNonEmptyDirectoryFoo);
        fail("No exception thrown.");
      } catch (IOException e) {
        assertThat(e).hasMessageThat().isEqualTo(xNonEmptyDirectoryBar + " (Permission denied)");
      }
    }
  }

  @Test
  public void testGetDigestForEmptyFile() throws Exception {
    Fingerprint fp = new Fingerprint(digestHashFunction);
    fp.addBytes(new byte[0]);
    assertThat(fp.hexDigestAndReset())
        .isEqualTo(BaseEncoding.base16().lowerCase().encode(xFile.getDigest()));
  }

  @Test
  public void testGetDigest() throws Exception {
    byte[] buffer = new byte[500000];
    for (int i = 0; i < buffer.length; ++i) {
      buffer[i] = 1;
    }
    FileSystemUtils.writeContent(xFile, buffer);
    Fingerprint fp = new Fingerprint(digestHashFunction);
    fp.addBytes(buffer);
    assertThat(fp.hexDigestAndReset())
        .isEqualTo(BaseEncoding.base16().lowerCase().encode(xFile.getDigest()));
  }

  @Test
  public void testStatFailsFastOnNonExistingFiles() throws Exception {
    try {
      xNothing.stat();
      fail("Expected IOException");
    } catch (IOException e) {
      // Do nothing.
    }
  }

  @Test
  public void testStatNullableFailsFastOnNonExistingFiles() throws Exception {
    assertThat(xNothing.statNullable()).isNull();
  }

  @Test
  public void testResolveSymlinks() throws Exception {
    if (testFS.supportsSymbolicLinksNatively(xLink)) {
      createSymbolicLink(xLink, xFile);
      FileSystemUtils.createEmptyFile(xFile);
      assertThat(testFS.resolveOneLink(xLink)).isEqualTo(xFile.asFragment());
      assertThat(xLink.resolveSymbolicLinks()).isEqualTo(xFile);
    }
  }

  @Test
  public void testResolveDanglingSymlinks() throws Exception {
    if (testFS.supportsSymbolicLinksNatively(xLink)) {
      createSymbolicLink(xLink, xNothing);
      assertThat(testFS.resolveOneLink(xLink)).isEqualTo(xNothing.asFragment());
      try {
        xLink.resolveSymbolicLinks();
        fail();
      } catch (IOException expected) {
      }
    }
  }

  @Test
  public void testResolveNonSymlinks() throws Exception {
    if (testFS.supportsSymbolicLinksNatively(xFile)) {
      assertThat(testFS.resolveOneLink(xFile)).isNull();
      assertThat(xFile.resolveSymbolicLinks()).isEqualTo(xFile);
    }
  }

  @Test
  public void testCreateHardLink_Success() throws Exception {
    if (!testFS.supportsHardLinksNatively(xFile)) {
      return;
    }
    xFile.createHardLink(xLink);
    assertThat(xFile.exists()).isTrue();
    assertThat(xLink.exists()).isTrue();
    assertThat(xFile.isFile()).isTrue();
    assertThat(xLink.isFile()).isTrue();
    assertThat(isHardLinked(xFile, xLink)).isTrue();
  }

  @Test
  public void testCreateHardLink_NeitherOriginalNorLinkExists() throws Exception {
    if (!testFS.supportsHardLinksNatively(xFile)) {
      return;
    }

    /* Neither original file nor link file exists */
    xFile.delete();
    try {
      xFile.createHardLink(xLink);
      fail("expected FileNotFoundException: File \"xFile\" linked from \"xLink\" does not exist");
    } catch (FileNotFoundException expected) {
      assertThat(expected)
          .hasMessageThat()
          .isEqualTo("File \"xFile\" linked from \"xLink\" does not exist");
    }
    assertThat(xFile.exists()).isFalse();
    assertThat(xLink.exists()).isFalse();
  }

  @Test
  public void testCreateHardLink_OriginalDoesNotExistAndLinkExists() throws Exception {

    if (!testFS.supportsHardLinksNatively(xFile)) {
      return;
    }

    /* link file exists and original file does not exist */
    xFile.delete();
    FileSystemUtils.createEmptyFile(xLink);

    try {
      xFile.createHardLink(xLink);
      fail("expected FileNotFoundException: File \"xFile\" linked from \"xLink\" does not exist");
    } catch (FileNotFoundException expected) {
      assertThat(expected)
          .hasMessageThat()
          .isEqualTo("File \"xFile\" linked from \"xLink\" does not exist");
    }
    assertThat(xFile.exists()).isFalse();
    assertThat(xLink.exists()).isTrue();
  }

  @Test
  public void testCreateHardLink_BothOriginalAndLinkExist() throws Exception {

    if (!testFS.supportsHardLinksNatively(xFile)) {
      return;
    }
    /* Both original file and link file exist */
    FileSystemUtils.createEmptyFile(xLink);

    try {
      xFile.createHardLink(xLink);
      fail("expected FileAlreadyExistsException: New link file \"xLink\" already exists");
    } catch (FileAlreadyExistsException expected) {
      assertThat(expected).hasMessageThat().isEqualTo("New link file \"xLink\" already exists");
    }
    assertThat(xFile.exists()).isTrue();
    assertThat(xLink.exists()).isTrue();
    assertThat(isHardLinked(xFile, xLink)).isFalse();
  }

  protected boolean isHardLinked(Path a, Path b) throws IOException {
    return testFS.stat(a, false).getNodeId() == testFS.stat(b, false).getNodeId();
  }
}
