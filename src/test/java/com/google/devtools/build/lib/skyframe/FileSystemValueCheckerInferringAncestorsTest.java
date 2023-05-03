// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.common.truth.Truth8.assertThat;
import static com.google.devtools.build.lib.actions.FileStateValue.DIRECTORY_FILE_STATE_NODE;
import static com.google.devtools.build.lib.actions.FileStateValue.NONEXISTENT_FILE_STATE_NODE;
import static com.google.devtools.build.lib.testing.common.DirectoryListingHelper.file;
import static com.google.devtools.build.lib.testing.common.DirectoryListingHelper.symlink;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.server.FailureDetails.DiffAwareness.Code;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.DelegateFileSystem;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import com.google.devtools.build.skyframe.ImmutableDiff;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link FileSystemValueCheckerInferringAncestors}. */
@RunWith(TestParameterInjector.class)
public final class FileSystemValueCheckerInferringAncestorsTest {
  private static final Delta DIRECTORY_FILE_STATE_NODE_DELTA =
      Delta.justNew(DIRECTORY_FILE_STATE_NODE);
  private static final Delta NONEXISTENT_FILE_STATE_NODE_DELTA =
      Delta.justNew(NONEXISTENT_FILE_STATE_NODE);
  private final Scratch scratch = new Scratch();
  private final List<String> statedPaths = new ArrayList<>();
  private final DefaultSyscallCache syscallCache = DefaultSyscallCache.newBuilder().build();
  private Root root;
  private Root untrackedRoot;
  private Exception throwOnStat;

  @TestParameter({"1", "16"})
  private int fsvcThreads;

  @Before
  public void createRoot() throws IOException {
    Path srcRootPath = scratch.dir("/src");
    PathFragment srcRoot = srcRootPath.asFragment();
    FileSystem trackingFileSystem =
        new DelegateFileSystem(scratch.getFileSystem()) {
          @Nullable
          @Override
          public synchronized FileStatus statIfFound(PathFragment path, boolean followSymlinks)
              throws IOException {
            if (throwOnStat != null) {
              Exception toThrow = throwOnStat;
              throwOnStat = null;
              Throwables.propagateIfPossible(toThrow, IOException.class);
              fail("Unexpected exception type");
            }
            statedPaths.add(path.relativeTo(srcRoot).toString());
            return super.statIfFound(path, followSymlinks);
          }
        };
    root = Root.fromPath(trackingFileSystem.getPath(srcRoot));
    scratch.setWorkingDir("/src");
    untrackedRoot = Root.fromPath(srcRootPath);
  }

  @After
  public void checkExceptionThrown() {
    assertThat(throwOnStat).isNull();
    syscallCache.clear();
  }

  @Test
  public void getDiffWithInferredAncestors_unknownFileChanged_returnsFileAndDirs()
      throws Exception {
    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(fileStateValueKey("foo/file")),
            fsvcThreads,
            syscallCache);

    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(
            fileStateValueKey(""),
            fileStateValueKey("foo"),
            fileStateValueKey("foo/file"),
            directoryListingStateValueKey(""),
            directoryListingStateValueKey("foo"));
    assertThat(diff.changedKeysWithNewValues()).isEmpty();
    assertThat(statedPaths).isEmpty();
  }

  @Test
  public void getDiffWithInferredAncestors_fileModified_returnsFileWithValues() throws Exception {
    scratch.file("file", "hello");
    FileStateKey key = fileStateValueKey("file");
    FileStateValue value = fileStateValue("file");
    scratch.overwriteFile("file", "there");

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(fileStateValueKey("file"), value),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(key),
            fsvcThreads,
            syscallCache);

    Delta newValue = fileStateValueDelta("file");
    assertThat(diff.changedKeysWithNewValues()).containsExactly(key, newValue);
    assertThat(diff.changedKeysWithoutNewValues()).isEmpty();
    assertThat(statedPaths).containsExactly("file");
  }

  @Test
  public void getDiffWithInferredAncestors_fileAdded_returnsFileAndDirListing() throws Exception {
    scratch.file("file");
    FileStateKey key = fileStateValueKey("file");

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                key, NONEXISTENT_FILE_STATE_NODE, fileStateValueKey(""), fileStateValue("")),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(key),
            fsvcThreads,
            syscallCache);

    Delta delta = fileStateValueDelta("file");
    assertThat(diff.changedKeysWithNewValues()).containsExactly(key, delta);
    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(directoryListingStateValueKey(""));
    assertThat(statedPaths).containsExactly("file");
  }

  @Test
  public void getDiffWithInferredAncestors_fileWithDirsAdded_returnsFileAndInjectsDirs()
      throws Exception {
    scratch.file("a/b/file");
    FileStateKey fileKey = fileStateValueKey("a/b/file");

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                fileStateValueKey(""),
                fileStateValue(""),
                fileStateValueKey("a"),
                NONEXISTENT_FILE_STATE_NODE,
                fileStateValueKey("a/b"),
                NONEXISTENT_FILE_STATE_NODE,
                fileKey,
                NONEXISTENT_FILE_STATE_NODE),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(fileKey),
            fsvcThreads,
            syscallCache);

    Delta delta = fileStateValueDelta("a/b/file");
    assertThat(diff.changedKeysWithNewValues())
        .containsExactly(
            fileKey,
            delta,
            fileStateValueKey("a"),
            DIRECTORY_FILE_STATE_NODE_DELTA,
            fileStateValueKey("a/b"),
            DIRECTORY_FILE_STATE_NODE_DELTA);
    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(
            directoryListingStateValueKey(""),
            directoryListingStateValueKey("a"),
            directoryListingStateValueKey("a/b"));
    assertThat(statedPaths).containsExactly("a/b/file");
  }

  @Test
  public void getDiffWithInferredAncestors_addedFileWithReportedDirs_returnsFileAndInjectsDirs()
      throws Exception {
    scratch.file("a/b/file");
    FileStateKey fileKey = fileStateValueKey("a/b/file");

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                fileStateValueKey(""),
                fileStateValue(""),
                fileStateValueKey("a"),
                NONEXISTENT_FILE_STATE_NODE,
                fileStateValueKey("a/b"),
                NONEXISTENT_FILE_STATE_NODE,
                fileKey,
                NONEXISTENT_FILE_STATE_NODE),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(fileKey, fileStateValueKey("a")),
            fsvcThreads,
            syscallCache);

    Delta newState = fileStateValueDelta("a/b/file");
    assertThat(diff.changedKeysWithNewValues())
        .containsExactly(
            fileKey,
            newState,
            fileStateValueKey("a"),
            DIRECTORY_FILE_STATE_NODE_DELTA,
            fileStateValueKey("a/b"),
            DIRECTORY_FILE_STATE_NODE_DELTA);
    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(
            directoryListingStateValueKey(""),
            directoryListingStateValueKey("a"),
            directoryListingStateValueKey("a/b"));
    assertThat(statedPaths).containsExactly("a/b/file");
  }

  /**
   * This is a degenerate case since we normally only know about a file if we checked all parents,
   * but that is theoretically possible with this API.
   */
  @Test
  public void getDiffWithInferredAncestors_fileWithUnknownDirsAdded_returnsFileAndDirs()
      throws Exception {
    scratch.file("a/b/c/d");

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                fileStateValueKey(""),
                fileStateValue(""),
                fileStateValueKey("a/b/c/d"),
                NONEXISTENT_FILE_STATE_NODE),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(fileStateValueKey("a/b/c/d")),
            fsvcThreads,
            syscallCache);

    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(
            fileStateValueKey("a"),
            fileStateValueKey("a/b"),
            fileStateValueKey("a/b/c"),
            directoryListingStateValueKey(""),
            directoryListingStateValueKey("a"),
            directoryListingStateValueKey("a/b"),
            directoryListingStateValueKey("a/b/c"));
    assertThat(diff.changedKeysWithNewValues())
        .containsExactly(fileStateValueKey("a/b/c/d"), fileStateValueDelta("a/b/c/d"));
    assertThat(statedPaths).containsExactly("a/b/c/d");
  }

  @Test
  public void getDiffWithInferredAncestors_addEmptyDir_returnsDirAndParentListing()
      throws Exception {
    scratch.dir("dir");
    FileStateKey key = fileStateValueKey("dir");
    Delta delta = fileStateValueDelta("dir");

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                key, NONEXISTENT_FILE_STATE_NODE, fileStateValueKey(""), fileStateValue("")),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(key),
            fsvcThreads,
            syscallCache);

    assertThat(diff.changedKeysWithNewValues()).containsExactly(key, delta);
    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(directoryListingStateValueKey(""));
    assertThat(statedPaths).containsExactly("dir");
  }

  @Test
  public void getDiffWithInferredAncestors_deleteFile_returnsFileParentListing() throws Exception {
    Path file = scratch.file("dir/file1");
    scratch.file("dir/file2");
    FileStateKey key = fileStateValueKey("dir/file1");
    FileStateValue oldValue = fileStateValue("dir/file1");
    file.delete();

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                key, oldValue, fileStateValueKey("dir"), fileStateValue("dir")),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(key),
            fsvcThreads,
            syscallCache);

    assertThat(diff.changedKeysWithNewValues())
        .containsExactly(key, NONEXISTENT_FILE_STATE_NODE_DELTA);
    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(directoryListingStateValueKey("dir"));
    assertThat(statedPaths).containsExactly("dir/file1", "dir");
  }

  @Test
  public void getDiffWithInferredAncestors_deleteFileFromDirWithListing_skipsDirStat()
      throws Exception {
    Path file1 = scratch.file("dir/file1");
    FileStateKey key = fileStateValueKey("dir/file1");
    FileStateValue oldValue = fileStateValue("dir/file1");
    file1.delete();

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                key, oldValue, fileStateValueKey("dir"), fileStateValue("dir")),
            /*graphDoneValues=*/ ImmutableMap.of(
                directoryListingStateValueKey("dir"),
                directoryListingStateValue(file("file1"), file("file2"))),
            /*modifiedKeys=*/ ImmutableSet.of(key),
            fsvcThreads,
            syscallCache);

    assertThat(diff.changedKeysWithNewValues())
        .containsExactly(key, NONEXISTENT_FILE_STATE_NODE_DELTA);
    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(directoryListingStateValueKey("dir"));
    assertThat(statedPaths).containsExactly("dir/file1");
  }

  @Test
  public void getDiffWithInferredAncestors_deleteLastFileFromDir_ignoresInvalidatedListing()
      throws Exception {
    Path file1 = scratch.file("dir/file1");
    FileStateKey key = fileStateValueKey("dir/file1");
    FileStateValue oldValue = fileStateValue("dir/file1");
    file1.delete();

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                key,
                oldValue,
                fileStateValueKey("dir"),
                fileStateValue("dir"),
                directoryListingStateValueKey("dir"),
                directoryListingStateValue(file("file1"), file("file2"))),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(key),
            fsvcThreads,
            syscallCache);

    assertThat(diff.changedKeysWithNewValues())
        .containsExactly(key, NONEXISTENT_FILE_STATE_NODE_DELTA);
    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(directoryListingStateValueKey("dir"));
    assertThat(statedPaths).containsExactly("dir/file1", "dir");
  }

  @Test
  public void getDiffWithInferredAncestors_modifyAllUnknownEntriesInDirWithListing_skipsDir()
      throws Exception {
    Path file = scratch.file("dir/file");
    file.getParentDirectory()
        .getRelative("symlink")
        .createSymbolicLink(PathFragment.create("file"));
    FileStateKey fileKey = fileStateValueKey("dir/file");
    FileStateKey symlinkKey = fileStateValueKey("dir/symlink");

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(fileStateValueKey("dir"), fileStateValue("dir")),
            /*graphDoneValues=*/ ImmutableMap.of(
                directoryListingStateValueKey("dir"),
                directoryListingStateValue(file("file"), symlink("symlink"))),
            /*modifiedKeys=*/ ImmutableSet.of(fileKey, symlinkKey),
            fsvcThreads,
            syscallCache);

    assertThat(diff.changedKeysWithNewValues())
        .containsExactly(
            fileKey,
            fileStateValueDelta("dir/file"),
            symlinkKey,
            fileStateValueDelta("dir/symlink"));
    assertThat(diff.changedKeysWithoutNewValues()).isEmpty();
    assertThat(statedPaths).containsExactly("dir/file", "dir/symlink");
  }

  @Test
  public void getDiffWithInferredAncestors_replaceUnknownEntriesInDirWithListing_skipsSiblingStat()
      throws Exception {
    scratch.dir("dir/file1");
    scratch.dir("dir/file2");
    FileStateKey file1Key = fileStateValueKey("dir/file1");
    FileStateKey file2Key = fileStateValueKey("dir/file2");
    DirectoryListingStateValue.Key dirKey = directoryListingStateValueKey("dir");

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(fileStateValueKey("dir"), fileStateValue("dir")),
            /*graphDoneValues=*/ ImmutableMap.of(
                dirKey, directoryListingStateValue(file("file1"), file("file2"))),
            /*modifiedKeys=*/ ImmutableSet.of(file1Key, file2Key),
            fsvcThreads,
            syscallCache);

    assertIsSubsetOf(
        diff.changedKeysWithNewValues().entrySet(),
        Maps.immutableEntry(file1Key, fileStateValueDelta("dir/file1")),
        Maps.immutableEntry(file2Key, fileStateValueDelta("dir/file2")));
    assertThat(diff.changedKeysWithoutNewValues()).contains(dirKey);
    assertThat(
            Streams.concat(
                diff.changedKeysWithoutNewValues().stream(),
                diff.changedKeysWithNewValues().keySet().stream()))
        .containsExactly(file1Key, file2Key, dirKey);
    assertThat(statedPaths).isNotEmpty();
    assertIsSubsetOf(statedPaths, "dir/file1", "dir/file2");
    if (fsvcThreads == 1) {
      // In case of single-threaded execution, we know that once we check dir/file1 or dir/file2, we
      // will be able to skip stat on the other one.
      assertThat(diff.changedKeysWithNewValues()).hasSize(1);
      assertThat(diff.changedKeysWithoutNewValues()).hasSize(2);
      assertThat(statedPaths).hasSize(1);
    }
  }

  @Test
  public void getDiffWithInferredAncestors_deleteAllFilesFromDir_returnsFilesAndDirListing()
      throws Exception {
    Path file1 = scratch.file("dir/file1");
    Path file2 = scratch.file("dir/file2");
    Path file3 = scratch.file("dir/file3");
    FileStateKey key1 = fileStateValueKey("dir/file1");
    FileStateValue oldValue1 = fileStateValue("dir/file1");
    FileStateKey key2 = fileStateValueKey("dir/file2");
    FileStateValue oldValue2 = fileStateValue("dir/file2");
    FileStateKey key3 = fileStateValueKey("dir/file3");
    FileStateValue oldValue3 = fileStateValue("dir/file3");
    file1.delete();
    file2.delete();
    file3.delete();

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                key1,
                oldValue1,
                key2,
                oldValue2,
                key3,
                oldValue3,
                fileStateValueKey("dir"),
                fileStateValue("dir")),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(key1, key2, key3),
            fsvcThreads,
            syscallCache);

    assertThat(diff.changedKeysWithNewValues())
        .containsExactly(
            key1, NONEXISTENT_FILE_STATE_NODE_DELTA,
            key2, NONEXISTENT_FILE_STATE_NODE_DELTA,
            key3, NONEXISTENT_FILE_STATE_NODE_DELTA);
    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(directoryListingStateValueKey("dir"));
    assertThat(statedPaths).containsExactly("dir", "dir/file1", "dir/file2", "dir/file3");
  }

  @Test
  public void getDiffWithInferredAncestors_deleteFileWithDirs_returnsFileAndDirs()
      throws Exception {
    scratch.file("a/b/c/file");
    FileStateKey abKey = fileStateValueKey("a/b");
    FileStateValue abValue = fileStateValue("a/b");
    FileStateKey abcKey = fileStateValueKey("a/b/c");
    FileStateValue abcValue = fileStateValue("a/b/c");
    FileStateKey abcFileKey = fileStateValueKey("a/b/c/file");
    FileStateValue abcFileValue = fileStateValue("a/b/c/file");
    scratch.dir("a/b").deleteTree();

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                fileStateValueKey("a"),
                fileStateValue("a"),
                abKey,
                abValue,
                abcKey,
                abcValue,
                abcFileKey,
                abcFileValue),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(abcFileKey),
            fsvcThreads,
            syscallCache);

    assertThat(diff.changedKeysWithNewValues())
        .containsExactly(
            abKey, NONEXISTENT_FILE_STATE_NODE_DELTA,
            abcKey, NONEXISTENT_FILE_STATE_NODE_DELTA,
            abcFileKey, NONEXISTENT_FILE_STATE_NODE_DELTA);
    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(
            directoryListingStateValueKey("a"),
            directoryListingStateValueKey("a/b"),
            directoryListingStateValueKey("a/b/c"));
    assertThat(statedPaths).containsExactly("a", "a/b", "a/b/c", "a/b/c/file");
  }

  @Test
  public void getDiffWithInferredAncestors_deleteFileWithReportedDirs_returnsFileAndDirListings()
      throws Exception {
    scratch.file("a/b/c/file");
    FileStateKey abKey = fileStateValueKey("a/b");
    FileStateValue abValue = fileStateValue("a/b");
    FileStateKey abcKey = fileStateValueKey("a/b/c");
    FileStateValue abcValue = fileStateValue("a/b/c");
    FileStateKey abcFileKey = fileStateValueKey("a/b/c/file");
    FileStateValue abcFileValue = fileStateValue("a/b/c/file");
    scratch.dir("a/b").deleteTree();

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                fileStateValueKey("a"),
                fileStateValue("a"),
                abKey,
                abValue,
                abcKey,
                abcValue,
                abcFileKey,
                abcFileValue),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(abcFileKey, abKey),
            fsvcThreads,
            syscallCache);

    assertThat(diff.changedKeysWithNewValues())
        .containsExactly(
            abKey, NONEXISTENT_FILE_STATE_NODE_DELTA,
            abcKey, NONEXISTENT_FILE_STATE_NODE_DELTA,
            abcFileKey, NONEXISTENT_FILE_STATE_NODE_DELTA);
    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(
            directoryListingStateValueKey("a"),
            directoryListingStateValueKey("a/b"),
            directoryListingStateValueKey("a/b/c"));
    assertThat(statedPaths).containsExactly("a", "a/b", "a/b/c", "a/b/c/file");
  }

  @Test
  public void getDiffWithInferredAncestors_deleteFile_infersDirFromModifiedSibling()
      throws Exception {
    Path file1 = scratch.file("dir/file1");
    scratch.file("dir/file2", "1");
    FileStateKey file1Key = fileStateValueKey("dir/file1");
    FileStateValue file1Value = fileStateValue("dir/file1");
    FileStateKey file2Key = fileStateValueKey("dir/file2");
    FileStateValue file2Value = fileStateValue("dir/file2");
    file1.delete();
    scratch.overwriteFile("dir/file2", "12");

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                fileStateValueKey("dir"),
                fileStateValue("dir"),
                file1Key,
                file1Value,
                file2Key,
                file2Value),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(file1Key, file2Key, fileStateValueKey("dir")),
            fsvcThreads,
            syscallCache);

    Delta file2NewValue = fileStateValueDelta("dir/file2");
    assertThat(diff.changedKeysWithNewValues())
        .containsExactly(file1Key, NONEXISTENT_FILE_STATE_NODE_DELTA, file2Key, file2NewValue);
    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(directoryListingStateValueKey("dir"));
    assertThat(statedPaths).containsExactly("dir/file1", "dir/file2");
  }

  @Test
  public void getDiffWithInferredAncestors_deleteDirReportDirOnly_returnsDir() throws Exception {
    Path file1 = scratch.file("dir/file1");
    scratch.file("dir/file2");
    FileStateKey file1Key = fileStateValueKey("dir/file1");
    FileStateValue file1Value = fileStateValue("dir/file1");
    FileStateKey file2Key = fileStateValueKey("dir/file2");
    FileStateValue file2Value = fileStateValue("dir/file2");
    FileStateKey dirKey = fileStateValueKey("dir");
    FileStateValue dirValue = fileStateValue("dir");
    file1.getParentDirectory().deleteTree();

    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                file1Key,
                file1Value,
                file2Key,
                file2Value,
                dirKey,
                dirValue,
                fileStateValueKey(""),
                fileStateValue("")),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(dirKey),
            fsvcThreads,
            syscallCache);

    assertThat(diff.changedKeysWithNewValues())
        .containsExactly(dirKey, NONEXISTENT_FILE_STATE_NODE_DELTA);
    assertThat(diff.changedKeysWithoutNewValues())
        .containsExactly(directoryListingStateValueKey(""));
    assertThat(statedPaths).containsExactly("dir", "");
  }

  @Test
  public void getDiffWithInferredAncestors_phantomChangeForNonexistentEntry_returnsEmptyDiff()
      throws Exception {
    ImmutableDiff diff =
        FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
            /*tsgm=*/ null,
            /*graphValues=*/ ImmutableMap.of(
                fileStateValueKey("file"), NONEXISTENT_FILE_STATE_NODE),
            /*graphDoneValues=*/ ImmutableMap.of(),
            /*modifiedKeys=*/ ImmutableSet.of(fileStateValueKey("file")),
            fsvcThreads,
            syscallCache);

    assertThat(diff.changedKeysWithoutNewValues()).isEmpty();
    assertThat(diff.changedKeysWithNewValues()).isEmpty();
    assertThat(statedPaths).containsExactly("file");
  }

  @Test
  public void getDiffWithInferredAncestors_statFails_fails() {
    throwOnStat = new IOException("oh no");
    ImmutableMap<SkyKey, SkyValue> graphValues =
        ImmutableMap.of(fileStateValueKey("file"), NONEXISTENT_FILE_STATE_NODE);
    ImmutableSet<FileStateKey> modifiedKeys = ImmutableSet.of(fileStateValueKey("file"));

    AbruptExitException e =
        assertThrows(
            AbruptExitException.class,
            () ->
                FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
                    /*tsgm=*/ null,
                    graphValues,
                    /*graphDoneValues=*/ ImmutableMap.of(),
                    modifiedKeys,
                    fsvcThreads,
                    syscallCache));

    assertThat(e.getDetailedExitCode().getFailureDetail().hasDiffAwareness()).isTrue();
    assertThat(e.getDetailedExitCode().getFailureDetail().getDiffAwareness().getCode())
        .isEqualTo(Code.DIFF_STAT_FAILED);
    assertThat(e).hasMessageThat().isEqualTo("Failed to stat: '/src/file' while computing diff");
  }

  @Test
  public void getDiffWithInferredAncestors_statCrashes_fails() {
    throwOnStat = new RuntimeException("oh no");
    ImmutableMap<SkyKey, SkyValue> graphValues =
        ImmutableMap.of(fileStateValueKey("file"), NONEXISTENT_FILE_STATE_NODE);
    ImmutableSet<FileStateKey> modifiedKeys = ImmutableSet.of(fileStateValueKey("file"));

    assertThrows(
        IllegalStateException.class,
        () ->
            FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
                /*tsgm=*/ null,
                graphValues,
                /*graphDoneValues=*/ ImmutableMap.of(),
                modifiedKeys,
                fsvcThreads,
                syscallCache));
  }

  private static <T> void assertIsSubsetOf(Iterable<T> list, T... elements) {
    ImmutableSet<T> set = ImmutableSet.copyOf(elements);
    assertWithMessage("%s has elements from outside of %s", list, set)
        .that(set)
        .containsAtLeastElementsIn(list);
  }

  private FileStateKey fileStateValueKey(String relativePath) {
    return FileStateValue.key(
        RootedPath.toRootedPath(root, root.asPath().getRelative(relativePath)));
  }

  private DirectoryListingStateValue.Key directoryListingStateValueKey(String relativePath) {
    return DirectoryListingStateValue.key(
        RootedPath.toRootedPath(root, root.asPath().getRelative(relativePath)));
  }

  private static DirectoryListingStateValue directoryListingStateValue(Dirent... dirents) {
    return DirectoryListingStateValue.create(ImmutableList.copyOf(dirents));
  }

  private FileStateValue fileStateValue(String relativePath) throws IOException {
    return FileStateValue.create(
        RootedPath.toRootedPath(
            untrackedRoot, untrackedRoot.asPath().asFragment().getRelative(relativePath)),
        SyscallCache.NO_CACHE,
        /*tsgm=*/ null);
  }

  private Delta fileStateValueDelta(String relativePath) throws IOException {
    return Delta.justNew(fileStateValue(relativePath));
  }
}
