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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.unix.FileStatus;
import com.google.devtools.build.lib.unix.NativePosixFiles;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.util.Collection;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
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

  /** Recursively make directories readable/executable and files readable. */
  protected void makeTreeReadable(Path path) throws IOException {
    if (path.isDirectory(Symlinks.NOFOLLOW)) {
      path.setReadable(true);
      path.setExecutable(true);
      for (Path entry : path.getDirectoryEntries()) {
        makeTreeReadable(entry);
      }
    } else {
      path.setReadable(true);
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
    assertThrows(IOException.class, () -> child.statIfFound());
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
    assertThrows(IOException.class, theHierarchy::createDirectoryAndParents);
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
    assertThrows(IOException.class, newPath::createDirectoryAndParents);
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
    IOException e = assertThrows(IOException.class, () -> xFile.createDirectory());
    assertThat(e).hasMessageThat().isEqualTo(xFile + " (File exists)");
  }

  @Test
  public void testCannotCreateDirectoryWithoutExistingParent() throws Exception {
    Path newPath = testFS.getPath("/deep/new-dir");
    FileNotFoundException e =
        assertThrows(FileNotFoundException.class, () -> newPath.createDirectory());
    assertThat(e).hasMessageThat().endsWith(" (No such file or directory)");
  }

  @Test
  public void testCannotCreateDirectoryWithReadOnlyParent() throws Exception {
    xEmptyDirectory.setWritable(false);
    Path xChildOfReadonlyDir = xEmptyDirectory.getChild("x");
    IOException e = assertThrows(IOException.class, () -> xChildOfReadonlyDir.createDirectory());
    assertThat(e).hasMessageThat().isEqualTo(xChildOfReadonlyDir + " (Permission denied)");
  }

  @Test
  public void testCannotCreateFileWithoutExistingParent() throws Exception {
    Path newPath = testFS.getPath("/non-existing-dir/new-file");
    FileNotFoundException e =
        assertThrows(FileNotFoundException.class, () -> FileSystemUtils.createEmptyFile(newPath));
    assertThat(e).hasMessageThat().endsWith(" (No such file or directory)");
  }

  @Test
  public void testCannotCreateFileWithReadOnlyParent() throws Exception {
    xEmptyDirectory.setWritable(false);
    Path xChildOfReadonlyDir = xEmptyDirectory.getChild("x");
    IOException e =
        assertThrows(IOException.class, () -> FileSystemUtils.createEmptyFile(xChildOfReadonlyDir));
    assertThat(e).hasMessageThat().isEqualTo(xChildOfReadonlyDir + " (Permission denied)");
  }

  @Test
  public void testCannotCreateFileWithinFile() throws Exception {
    Path newFilePath = absolutize("some-file");
    FileSystemUtils.createEmptyFile(newFilePath);
    Path wrongPath = absolutize("some-file/new-file");
    IOException e =
        assertThrows(IOException.class, () -> FileSystemUtils.createEmptyFile(wrongPath));
    assertThat(e).hasMessageThat().endsWith(" (Not a directory)");
  }

  @Test
  public void testCannotCreateDirectoryWithinFile() throws Exception {
    Path newFilePath = absolutize("some-file");
    FileSystemUtils.createEmptyFile(newFilePath);
    Path wrongPath = absolutize("some-file/new-file");
    IOException e = assertThrows(IOException.class, () -> wrongPath.createDirectory());
    assertThat(e).hasMessageThat().endsWith(" (Not a directory)");
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
    IOException ex = assertThrows(IOException.class, () -> xFile.getDirectoryEntries());
    if (ex instanceof FileNotFoundException) {
        fail("The method should throw an object of class IOException.");
      }
    assertThat(ex).hasMessageThat().isEqualTo(xFile + " (Not a directory)");
  }

  @Test
  public void testGetDirectoryEntriesThrowsExceptionForNonexistingPath() {
    Path somePath = testFS.getPath("/non-existing-path");
    Exception x = assertThrows(Exception.class, () -> somePath.getDirectoryEntries());
    assertThat(x).hasMessageThat().isEqualTo(somePath + " (No such file or directory)");
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
    IOException e = assertThrows(IOException.class, () -> xNonEmptyDirectory.delete());
    assertThat(e).hasMessageThat().isEqualTo(xNonEmptyDirectory + " (Directory not empty)");
  }

  @Test
  public void testDeleteNonEmptyDirectoryNotDeletedDirectory() throws Exception {
    assertThrows(IOException.class, () -> xNonEmptyDirectory.delete());

    assertThat(xNonEmptyDirectory.isDirectory()).isTrue();
  }

  @Test
  public void testDeleteNonEmptyDirectoryNotDeletedFile() throws Exception {
    assertThrows(IOException.class, () -> xNonEmptyDirectory.delete());

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

  private static enum DeleteFunc {
    DELETE_TREE,
    DELETE_TREES_BELOW
  };

  private void doTestDeleteUnreadableDirectories(DeleteFunc deleteFunc) throws IOException {
    Path topDir = absolutize("top-dir");
    Path aDir = absolutize("top-dir/a-dir");
    Path file1 = absolutize("top-dir/a-dir/file1");
    Path file2 = absolutize("top-dir/a-dir/file2");
    Path bDir = absolutize("top-dir/b-dir");
    Path file3 = absolutize("top-dir/b-dir/file3");

    topDir.createDirectory();
    aDir.createDirectory();
    FileSystemUtils.createEmptyFile(file1);
    FileSystemUtils.createEmptyFile(file2);
    bDir.createDirectory();
    FileSystemUtils.createEmptyFile(file3);

    try {
      aDir.setReadable(false);
      bDir.setReadable(false);
      topDir.setReadable(false);
    } catch (UnsupportedOperationException e) {
      // Skip testing if the file system does not support clearing the needed attibutes.
      return;
    }

    switch (deleteFunc) {
      case DELETE_TREE:
        topDir.deleteTree();
        assertThat(topDir.exists()).isFalse();
        break;
      case DELETE_TREES_BELOW:
        topDir.deleteTreesBelow();
        makeTreeReadable(topDir);
        assertThat(topDir.exists()).isTrue();
        assertThat(FileSystemUtils.traverseTree(topDir, unused -> true)).isEmpty();
        break;
    }
  }

  @Test
  public void testDeleteTreeDeletesUnreadableDirectories() throws IOException {
    doTestDeleteUnreadableDirectories(DeleteFunc.DELETE_TREE);
  }

  @Test
  public void testDeleteTreesBelowDeletesUnreadableDirectories() throws IOException {
    doTestDeleteUnreadableDirectories(DeleteFunc.DELETE_TREES_BELOW);
  }

  private void doTestDeleteUnwritableDirectories(DeleteFunc deleteFunc) throws IOException {
    Path topDir = absolutize("top-dir");
    Path aDir = absolutize("top-dir/a-dir");
    Path file1 = absolutize("top-dir/a-dir/file1");
    Path file2 = absolutize("top-dir/a-dir/file2");
    Path bDir = absolutize("top-dir/b-dir");
    Path file3 = absolutize("top-dir/b-dir/file3");

    topDir.createDirectory();
    aDir.createDirectory();
    FileSystemUtils.createEmptyFile(file1);
    FileSystemUtils.createEmptyFile(file2);
    bDir.createDirectory();
    FileSystemUtils.createEmptyFile(file3);

    try {
      aDir.setWritable(false);
      bDir.setWritable(false);
      topDir.setWritable(false);
    } catch (UnsupportedOperationException e) {
      // Skip testing if the file system does not support clearing the needed attibutes.
      return;
    }

    switch (deleteFunc) {
      case DELETE_TREE:
        topDir.deleteTree();
        assertThat(topDir.exists()).isFalse();
        break;
      case DELETE_TREES_BELOW:
        topDir.deleteTreesBelow();
        makeTreeReadable(topDir);
        assertThat(topDir.exists()).isTrue();
        assertThat(FileSystemUtils.traverseTree(topDir, unused -> true)).isEmpty();
        break;
    }
  }

  @Test
  public void testDeleteTreeDeletesUnwritableDirectories() throws IOException {
    doTestDeleteUnwritableDirectories(DeleteFunc.DELETE_TREE);
  }

  @Test
  public void testDeleteTreesBelowDeletesUnwritableDirectories() throws IOException {
    doTestDeleteUnwritableDirectories(DeleteFunc.DELETE_TREES_BELOW);
  }

  private void doTestDeleteReadableUnexecutableDirectories(DeleteFunc deleteFunc)
      throws IOException {
    Path topDir = absolutize("top-dir");
    Path aDir = absolutize("top-dir/a-dir");
    Path file1 = absolutize("top-dir/a-dir/file1");
    Path file2 = absolutize("top-dir/a-dir/file2");
    Path bDir = absolutize("top-dir/b-dir");
    Path file3 = absolutize("top-dir/b-dir/file3");

    topDir.createDirectory();
    aDir.createDirectory();
    FileSystemUtils.createEmptyFile(file1);
    FileSystemUtils.createEmptyFile(file2);
    bDir.createDirectory();
    FileSystemUtils.createEmptyFile(file3);

    try {
      aDir.setExecutable(false);
      bDir.setExecutable(false);
      topDir.setExecutable(false);
    } catch (UnsupportedOperationException e) {
      // Skip testing if the file system does not support clearing the needed attibutes.
      return;
    }

    switch (deleteFunc) {
      case DELETE_TREE:
        topDir.deleteTree();
        assertThat(topDir.exists()).isFalse();
        break;
      case DELETE_TREES_BELOW:
        topDir.deleteTreesBelow();
        makeTreeReadable(topDir);
        assertThat(topDir.exists()).isTrue();
        assertThat(FileSystemUtils.traverseTree(topDir, unused -> true)).isEmpty();
        break;
    }
  }

  @Test
  public void testDeleteTreeDeletesReadableUnexecutableDirectories() throws IOException {
    doTestDeleteReadableUnexecutableDirectories(DeleteFunc.DELETE_TREE);
  }

  @Test
  public void testDeleteTreesBelowDeletesReadableUnexecutableDirectories() throws IOException {
    doTestDeleteReadableUnexecutableDirectories(DeleteFunc.DELETE_TREES_BELOW);
  }

  private void doTestDeleteUnreadableUnexecutableDirectories(DeleteFunc deleteFunc)
      throws IOException {
    Path topDir = absolutize("top-dir");
    Path aDir = absolutize("top-dir/a-dir");
    Path file1 = absolutize("top-dir/a-dir/file1");
    Path file2 = absolutize("top-dir/a-dir/file2");
    Path bDir = absolutize("top-dir/b-dir");
    Path file3 = absolutize("top-dir/b-dir/file3");

    topDir.createDirectory();
    aDir.createDirectory();
    FileSystemUtils.createEmptyFile(file1);
    FileSystemUtils.createEmptyFile(file2);
    bDir.createDirectory();
    FileSystemUtils.createEmptyFile(file3);

    try {
      aDir.setReadable(false);
      aDir.setExecutable(false);
      bDir.setReadable(false);
      bDir.setExecutable(false);
      topDir.setReadable(false);
      topDir.setExecutable(false);
    } catch (UnsupportedOperationException e) {
      // Skip testing if the file system does not support clearing the needed attibutes.
      return;
    }

    switch (deleteFunc) {
      case DELETE_TREE:
        topDir.deleteTree();
        assertThat(topDir.exists()).isFalse();
        break;
      case DELETE_TREES_BELOW:
        topDir.deleteTreesBelow();
        makeTreeReadable(topDir);
        assertThat(topDir.exists()).isTrue();
        assertThat(FileSystemUtils.traverseTree(topDir, unused -> true)).isEmpty();
        break;
    }
  }

  @Test
  public void testDeleteTreeDeletesUnreadableUnexecutableDirectories() throws IOException {
    doTestDeleteUnreadableUnexecutableDirectories(DeleteFunc.DELETE_TREE);
  }

  @Test
  public void testDeleteTreesBelowDeletesUnreadableUnexecutableDirectories() throws IOException {
    doTestDeleteUnreadableUnexecutableDirectories(DeleteFunc.DELETE_TREES_BELOW);
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
  public void testDeleteTreesBelowDeletesContentsOnly() throws IOException {
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
  public void testDeleteTreesBelowIgnoresMissingTopDir() throws IOException {
    Path topDir = absolutize("top-dir");

    assertThat(topDir.exists()).isFalse();
    topDir.deleteTreesBelow(); // Expect no exception.
    assertThat(topDir.exists()).isFalse();
  }

  @Test
  public void testDeleteTreesBelowIgnoresNonDirectories() throws IOException {
    Path topFile = absolutize("top-file");

    FileSystemUtils.createEmptyFile(topFile);

    assertThat(topFile.exists()).isTrue();
    topFile.deleteTreesBelow(); // Expect no exception.
    assertThat(topFile.exists()).isTrue();
  }

  /**
   * Executes {@link FileSystem#deleteTreesBelow} on {@code topDir} and tries to race its execution
   * by deleting {@code fileToDelete} concurrently.
   */
  private static void deleteTreesBelowRaceTest(Path topDir, Path fileToDelete) throws Exception {
    CountDownLatch latch = new CountDownLatch(2);
    AtomicBoolean wonRace = new AtomicBoolean(false);
    Thread t =
        new Thread(
            () -> {
              try {
                latch.countDown();
                latch.await();
                wonRace.compareAndSet(false, fileToDelete.delete());
              } catch (IOException | InterruptedException e) {
                // Don't care.
              }
            });
    t.start();
    try {
      try {
        latch.countDown();
        latch.await();
        topDir.deleteTreesBelow();
      } finally {
        t.join();
      }
      if (!wonRace.get()) {
        assertThat(topDir.exists()).isTrue();
      }
    } catch (IOException e) {
      if (wonRace.get()) {
        assertThat(e).hasMessageThat().contains(fileToDelete.toString());
        assertThat(e).hasMessageThat().contains("No such file");
      } else {
        throw e;
      }
    }
  }

  @Test
  public void testDeleteTreesBelowFailsGracefullyIfTreeGoesMissing() throws Exception {
    Path topDir = absolutize("maybe-missing-dir");
    for (int i = 0; i < 1000; i++) {
      topDir.createDirectory();
      deleteTreesBelowRaceTest(topDir, topDir);
    }
  }

  @Test
  public void testDeleteTreesBelowFailsGracefullyIfContentsGoMissing() throws Exception {
    Path topDir = absolutize("top-dir");
    Path file = absolutize("top-dir/maybe-missing-file");
    for (int i = 0; i < 1000; i++) {
      topDir.createDirectory();
      FileSystemUtils.createEmptyFile(file);
      deleteTreesBelowRaceTest(topDir, file);
    }
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
    FileNotFoundException x =
        assertThrows(FileNotFoundException.class, () -> newPath.getLastModifiedTime());
    assertThat(x).hasMessageThat().isEqualTo(newPath + " (No such file or directory)");
  }

  // Test file size
  @Test
  public void testFileSizeThrowsExceptionForNonexistingPath() throws Exception {
    Path newPath = testFS.getPath("/non-existing-file");
    FileNotFoundException e =
        assertThrows(FileNotFoundException.class, () -> newPath.getFileSize());
    assertThat(e).hasMessageThat().isEqualTo(newPath + " (No such file or directory)");
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
  public void testFileChannelEOF() throws Exception {
    try (OutputStream outStream = xFile.getOutputStream()) {
      outStream.write(new byte[] {1});
    }

    try (ReadableByteChannel channel = xFile.createReadableByteChannel()) {
      ByteBuffer buffer = ByteBuffer.allocate(2);
      int numRead = readFromChannel(channel, buffer, 1);
      assertThat(numRead).isEqualTo(1);
      assertThat(readFromChannel(channel, buffer, 1)).isEqualTo(-1);
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
  public void testFileChannel() throws Exception {
    byte[] bytes = "abcdefghijklmnoprstuvwxyz".getBytes(StandardCharsets.ISO_8859_1);
    try (OutputStream outStream = xFile.getOutputStream()) {
      outStream.write(bytes);
    }

    try (ReadableByteChannel channel = xFile.createReadableByteChannel()) {
      ByteBuffer buffer = ByteBuffer.allocate(bytes.length);
      int numRead = readFromChannel(channel, buffer, bytes.length);
      assertThat(numRead).isEqualTo(bytes.length);
      assertThat(buffer.hasArray()).isTrue();
      assertThat(buffer.array()).isEqualTo(bytes);
    }
  }

  private static int readFromChannel(ReadableByteChannel channel, ByteBuffer buffer, int expected)
      throws IOException {
    int numRead = 0;
    for (int i = 0; i < 100; i++) {
      int stepRead = channel.read(buffer);
      if (stepRead < 0) {
        return stepRead;
      }
      numRead += stepRead;
      if (numRead >= expected) {
        return numRead;
      }
    }
    throw new IOException("Can not read the specified number of bytes.");
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
    IOException ex = assertThrows(IOException.class, () -> xEmptyDirectory.getOutputStream());
    assertThat(ex).hasMessageThat().isEqualTo(xEmptyDirectory + " (Is a directory)");
  }

  @Test
  public void testInputStreamThrowExceptionOnDirectory() throws Exception {
    IOException ex = assertThrows(IOException.class, () -> xEmptyDirectory.getInputStream());
    assertThat(ex).hasMessageThat().isEqualTo(xEmptyDirectory + " (Is a directory)");
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
    IOException e =
        assertThrows(IOException.class, () -> xEmptyDirectory.renameTo(xNonEmptyDirectory));
    assertThat(e).hasMessageThat().containsMatch("\\((File exists|Directory not empty)\\)$");
  }

  @Test
  public void testCantRenameDirToExistingNonEmptyDirNothingChanged() throws Exception {
    assertThrows(IOException.class, () -> xEmptyDirectory.renameTo(xNonEmptyDirectory));

    assertThat(xNonEmptyDirectory.isDirectory()).isTrue();
    assertThat(xEmptyDirectory.isDirectory()).isTrue();
    assertThat(xEmptyDirectory.getDirectoryEntries()).isEmpty();
    assertThat(xNonEmptyDirectory.getDirectoryEntries()).isNotEmpty();
  }

  @Test
  public void testCantRenameDirToExistingFile() {
    IOException e = assertThrows(IOException.class, () -> xEmptyDirectory.renameTo(xFile));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(xEmptyDirectory + " -> " + xFile + " (Not a directory)");
  }

  @Test
  public void testCantRenameDirToExistingFileNothingChanged() {
    assertThrows(IOException.class, () -> xEmptyDirectory.renameTo(xFile));

    assertThat(xEmptyDirectory.isDirectory()).isTrue();
    assertThat(xFile.isFile()).isTrue();
  }

  @Test
  public void testCantRenameFileToExistingDir() {
    IOException e = assertThrows(IOException.class, () -> xFile.renameTo(xEmptyDirectory));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(xFile + " -> " + xEmptyDirectory + " (Is a directory)");
  }

  @Test
  public void testCantRenameFileToExistingDirNothingChanged() {
    assertThrows(IOException.class, () -> xFile.renameTo(xEmptyDirectory));

    assertThat(xEmptyDirectory.isDirectory()).isTrue();
    assertThat(xFile.isFile()).isTrue();
  }

  @Test
  public void testMoveOnNonExistingFileThrowsException() throws Exception {
    Path nonExistingPath = absolutize("non-existing");
    Path targetPath = absolutize("does-not-matter");
    FileNotFoundException e =
        assertThrows(FileNotFoundException.class, () -> nonExistingPath.renameTo(targetPath));
    assertThat(e).hasMessageThat().endsWith(" (No such file or directory)");
  }

  // Test the Paths
  @Test
  public void testGetPathOnlyAcceptsAbsolutePath() {
    assertThrows(IllegalArgumentException.class, () -> testFS.getPath("not-absolute"));
  }

  @Test
  public void testGetPathOnlyAcceptsAbsolutePathFragment() {
    assertThrows(
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
    FileNotFoundException ex =
        assertThrows(FileNotFoundException.class, () -> xNothing.isExecutable());
    assertThat(ex).hasMessageThat().isEqualTo(xNothing + " (No such file or directory)");
  }

  @Test
  public void testCannotSetExecutableOnNonexistingFile() throws Exception {
    FileNotFoundException ex =
        assertThrows(FileNotFoundException.class, () -> xNothing.setExecutable(true));
    assertThat(ex).hasMessageThat().isEqualTo(xNothing + " (No such file or directory)");
  }

  @Test
  public void testCannotGetWritableOnNonexistingFile() throws Exception {
    FileNotFoundException ex =
        assertThrows(FileNotFoundException.class, () -> xNothing.isWritable());
    assertThat(ex).hasMessageThat().isEqualTo(xNothing + " (No such file or directory)");
  }

  @Test
  public void testCannotSetWritableOnNonexistingFile() throws Exception {
    FileNotFoundException ex =
        assertThrows(FileNotFoundException.class, () -> xNothing.setWritable(false));
    assertThat(ex).hasMessageThat().isEqualTo(xNothing + " (No such file or directory)");
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

    IOException e = assertThrows(IOException.class, () -> xNonEmptyDirectoryFoo.isWritable());
    assertThat(e).hasMessageThat().endsWith(" (Permission denied)");
  }

  @Test
  public void testWritingToReadOnlyFileThrowsException() throws Exception {
    xFile.setWritable(false);
    IOException e =
        assertThrows(
            IOException.class,
            () -> FileSystemUtils.writeContent(xFile, "hello, world!".getBytes(UTF_8)));
    assertThat(e).hasMessageThat().isEqualTo(xFile + " (Permission denied)");
  }

  @Test
  public void testReadingFromUnreadableFileThrowsException() throws Exception {
    FileSystemUtils.writeContent(xFile, "hello, world!".getBytes(UTF_8));
    xFile.setReadable(false);
    IOException e = assertThrows(IOException.class, () -> FileSystemUtils.readContent(xFile));
    assertThat(e).hasMessageThat().isEqualTo(xFile + " (Permission denied)");
  }

  @Test
  public void testCannotCreateFileInReadOnlyDirectory() throws Exception {
    Path xNonEmptyDirectoryBar = xNonEmptyDirectory.getChild("bar");
    xNonEmptyDirectory.setWritable(false);

    IOException e =
        assertThrows(
            IOException.class, () -> FileSystemUtils.createEmptyFile(xNonEmptyDirectoryBar));
    assertThat(e).hasMessageThat().isEqualTo(xNonEmptyDirectoryBar + " (Permission denied)");
  }

  @Test
  public void testCannotCreateDirectoryInReadOnlyDirectory() throws Exception {
    Path xNonEmptyDirectoryBar = xNonEmptyDirectory.getChild("bar");
    xNonEmptyDirectory.setWritable(false);

    IOException e = assertThrows(IOException.class, () -> xNonEmptyDirectoryBar.createDirectory());
    assertThat(e).hasMessageThat().isEqualTo(xNonEmptyDirectoryBar + " (Permission denied)");
  }

  @Test
  public void testCannotMoveIntoReadOnlyDirectory() throws Exception {
    Path xNonEmptyDirectoryBar = xNonEmptyDirectory.getChild("bar");
    xNonEmptyDirectory.setWritable(false);

    IOException e = assertThrows(IOException.class, () -> xFile.renameTo(xNonEmptyDirectoryBar));
    assertThat(e).hasMessageThat().endsWith(" (Permission denied)");
  }

  @Test
  public void testCannotMoveFromReadOnlyDirectory() throws Exception {
    xNonEmptyDirectory.setWritable(false);

    IOException e = assertThrows(IOException.class, () -> xNonEmptyDirectoryFoo.renameTo(xNothing));
    assertThat(e).hasMessageThat().endsWith(" (Permission denied)");
  }

  @Test
  public void testCannotDeleteInReadOnlyDirectory() throws Exception {
    xNonEmptyDirectory.setWritable(false);

    IOException e = assertThrows(IOException.class, () -> xNonEmptyDirectoryFoo.delete());
    assertThat(e).hasMessageThat().isEqualTo(xNonEmptyDirectoryFoo + " (Permission denied)");
  }

  @Test
  public void testCannotCreatSymbolicLinkInReadOnlyDirectory() throws Exception {
    Path xNonEmptyDirectoryBar = xNonEmptyDirectory.getChild("bar");
    xNonEmptyDirectory.setWritable(false);

    if (testFS.supportsSymbolicLinksNatively(xNonEmptyDirectoryBar)) {
      IOException e =
          assertThrows(
              IOException.class,
              () -> createSymbolicLink(xNonEmptyDirectoryBar, xNonEmptyDirectoryFoo));
      assertThat(e).hasMessageThat().isEqualTo(xNonEmptyDirectoryBar + " (Permission denied)");
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
    assertThrows(IOException.class, () -> xNothing.stat());
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
      assertThrows(IOException.class, () -> xLink.resolveSymbolicLinks());
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
  public void testCreateHardLink_success() throws Exception {
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
  public void testCreateHardLink_neitherOriginalNorLinkExists() throws Exception {
    if (!testFS.supportsHardLinksNatively(xFile)) {
      return;
    }

    /* Neither original file nor link file exists */
    xFile.delete();
    FileNotFoundException expected =
        assertThrows(FileNotFoundException.class, () -> xFile.createHardLink(xLink));
    assertThat(expected)
        .hasMessageThat()
        .isEqualTo("File \"xFile\" linked from \"xLink\" does not exist");
    assertThat(xFile.exists()).isFalse();
    assertThat(xLink.exists()).isFalse();
  }

  @Test
  public void testCreateHardLink_originalDoesNotExistAndLinkExists() throws Exception {

    if (!testFS.supportsHardLinksNatively(xFile)) {
      return;
    }

    /* link file exists and original file does not exist */
    xFile.delete();
    FileSystemUtils.createEmptyFile(xLink);

    FileNotFoundException expected =
        assertThrows(FileNotFoundException.class, () -> xFile.createHardLink(xLink));
    assertThat(expected)
        .hasMessageThat()
        .isEqualTo("File \"xFile\" linked from \"xLink\" does not exist");
    assertThat(xFile.exists()).isFalse();
    assertThat(xLink.exists()).isTrue();
  }

  @Test
  public void testCreateHardLink_bothOriginalAndLinkExist() throws Exception {

    if (!testFS.supportsHardLinksNatively(xFile)) {
      return;
    }
    /* Both original file and link file exist */
    FileSystemUtils.createEmptyFile(xLink);

    FileAlreadyExistsException expected =
        assertThrows(FileAlreadyExistsException.class, () -> xFile.createHardLink(xLink));
    assertThat(expected).hasMessageThat().isEqualTo("New link file \"xLink\" already exists");
    assertThat(xFile.exists()).isTrue();
    assertThat(xLink.exists()).isTrue();
    assertThat(isHardLinked(xFile, xLink)).isFalse();
  }

  protected boolean isHardLinked(Path a, Path b) throws IOException {
    return testFS.stat(a, false).getNodeId() == testFS.stat(b, false).getNodeId();
  }
}
