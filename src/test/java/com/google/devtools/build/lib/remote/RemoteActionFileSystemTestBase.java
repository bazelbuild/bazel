// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.junit.Test;

public abstract class RemoteActionFileSystemTestBase {
  protected abstract FileSystem createActionFileSystem(
      ActionInputMap inputs, Iterable<Artifact> outputs) throws IOException;

  protected FileSystem createActionFileSystem() throws IOException {
    ActionInputMap inputs = new ActionInputMap(0);
    return createActionFileSystem(inputs);
  }

  protected FileSystem createActionFileSystem(ActionInputMap inputs) throws IOException {
    return createActionFileSystem(inputs, ImmutableList.of());
  }

  protected abstract FileSystem getLocalFileSystem(FileSystem actionFs);

  protected abstract FileSystem getRemoteFileSystem(FileSystem actionFs);

  protected abstract PathFragment getOutputPath(String outputRootRelativePath);

  protected abstract void writeLocalFile(FileSystem actionFs, PathFragment path, String content)
      throws IOException;

  protected abstract void injectRemoteFile(FileSystem actionFs, PathFragment path, String content)
      throws IOException;

  @Test
  public void exists_fileDoesNotExist_returnsFalse() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    PathFragment path = getOutputPath("file");

    boolean exists = actionFs.exists(path);

    assertThat(exists).isFalse();
  }

  @Test
  public void exists_localFile_returnsTrue() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    PathFragment path = getOutputPath("file");

    writeLocalFile(actionFs, path, "local contents");

    assertThat(actionFs.exists(path)).isTrue();
  }

  @Test
  public void exists_remoteFile_returnsTrue() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    PathFragment path = getOutputPath("file");

    injectRemoteFile(actionFs, path, "remote contents");

    assertThat(actionFs.exists(path)).isTrue();
  }

  @Test
  public void exists_localAndRemoteFile_returnsTrue() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    PathFragment path = getOutputPath("file");

    writeLocalFile(actionFs, path, "local contents");
    injectRemoteFile(actionFs, path, "remote contents");

    assertThat(actionFs.exists(path)).isTrue();
  }

  @Test
  public void delete_fileDoesNotExist_returnsFalse() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    PathFragment path = getOutputPath("file");

    boolean success = actionFs.getPath(path).delete();

    assertThat(success).isFalse();
  }

  @Test
  public void delete_localFile_succeeds() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    PathFragment path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local contents");
    assertThat(getLocalFileSystem(actionFs).exists(path)).isTrue();

    boolean success = actionFs.getPath(path).delete();

    assertThat(success).isTrue();
    assertThat(getLocalFileSystem(actionFs).exists(path)).isFalse();
  }

  @Test
  public void delete_remoteFile_succeeds() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    PathFragment path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote contents");
    assertThat(getRemoteFileSystem(actionFs).exists(path)).isTrue();

    boolean success = actionFs.getPath(path).delete();

    assertThat(success).isTrue();
    assertThat(actionFs.exists(path)).isFalse();
    assertThat(getRemoteFileSystem(actionFs).exists(path)).isFalse();
  }

  @Test
  public void delete_localAndRemoteFile_succeeds() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    PathFragment path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local contents");
    injectRemoteFile(actionFs, path, "remote contents");
    assertThat(getLocalFileSystem(actionFs).exists(path)).isTrue();
    assertThat(getRemoteFileSystem(actionFs).exists(path)).isTrue();

    boolean success = actionFs.getPath(path).delete();

    assertThat(success).isTrue();
    assertThat(actionFs.exists(path)).isFalse();
    assertThat(getLocalFileSystem(actionFs).exists(path)).isFalse();
    assertThat(getRemoteFileSystem(actionFs).exists(path)).isFalse();
  }

  @Test
  public void renameTo_fileDoesNotExist_throwError() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    PathFragment path = getOutputPath("file");
    PathFragment newPath = getOutputPath("file-new");

    assertThrows(FileNotFoundException.class, () -> actionFs.renameTo(path, newPath));
  }

  @Test
  public void renameTo_onlyRemoteFile_renameRemoteFile() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    PathFragment path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    PathFragment newPath = getOutputPath("file-new");

    actionFs.renameTo(path, newPath);

    assertThat(actionFs.exists(path)).isFalse();
    assertThat(actionFs.exists(newPath)).isTrue();
    assertThat(getRemoteFileSystem(actionFs).exists(path)).isFalse();
    assertThat(getRemoteFileSystem(actionFs).exists(newPath)).isTrue();
  }

  @Test
  public void renameTo_localAndRemoteFile_renameBoth() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    PathFragment path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");
    PathFragment newPath = getOutputPath("file-new");

    actionFs.renameTo(path, newPath);

    assertThat(actionFs.exists(path)).isFalse();
    assertThat(actionFs.exists(newPath)).isTrue();
    assertThat(getRemoteFileSystem(actionFs).exists(path)).isFalse();
    assertThat(getRemoteFileSystem(actionFs).exists(newPath)).isTrue();
    assertThat(getLocalFileSystem(actionFs).exists(path)).isFalse();
    assertThat(getLocalFileSystem(actionFs).exists(newPath)).isTrue();
  }

  @Test
  public void createDirectoryAndParents_createLocallyAndRemotely() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    PathFragment path = getOutputPath("dir");

    actionFs.createDirectoryAndParents(path);

    assertThat(getRemoteFileSystem(actionFs).getPath(path).isDirectory()).isTrue();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isDirectory()).isTrue();
  }

  @Test
  public void createDirectory_createLocallyAndRemotely() throws Exception {
    FileSystem actionFs = createActionFileSystem();
    actionFs.createDirectoryAndParents(getOutputPath("parent"));
    PathFragment path = getOutputPath("parent/dir");

    actionFs.createDirectory(path);

    assertThat(getRemoteFileSystem(actionFs).getPath(path).isDirectory()).isTrue();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isDirectory()).isTrue();
  }

  interface DirectoryEntriesProvider {
    ImmutableList<String> getDirectoryEntries(Path path) throws IOException;
  }

  private void readdirNonEmptyLocalDirectoryReadFromLocal(
      DirectoryEntriesProvider directoryEntriesProvider) throws IOException {
    FileSystem actionFs = createActionFileSystem();
    PathFragment dir = getOutputPath("parent/dir");
    actionFs.getPath(dir).createDirectoryAndParents();
    writeLocalFile(actionFs, dir.getChild("file1"), "content1");
    writeLocalFile(actionFs, dir.getChild("file2"), "content2");

    ImmutableList<String> entries =
        directoryEntriesProvider.getDirectoryEntries(actionFs.getPath(dir));

    assertThat(entries).containsExactly("file1", "file2");
  }

  private void readdirNonEmptyInMemoryDirectoryReadFromMemory(
      DirectoryEntriesProvider directoryEntriesProvider) throws IOException {
    FileSystem actionFs = createActionFileSystem();
    PathFragment dir = getOutputPath("parent/dir");
    actionFs.getPath(dir).createDirectoryAndParents();
    injectRemoteFile(actionFs, dir.getChild("file1"), "content1");
    injectRemoteFile(actionFs, dir.getChild("file2"), "content2");

    ImmutableList<String> entries =
        directoryEntriesProvider.getDirectoryEntries(actionFs.getPath(dir));

    assertThat(entries).containsExactly("file1", "file2");
  }

  private void readdirNonEmptyLocalAndInMemoryDirectoryCombineThem(
      DirectoryEntriesProvider directoryEntriesProvider) throws IOException {
    FileSystem actionFs = createActionFileSystem();
    PathFragment dir = getOutputPath("parent/dir");
    actionFs.getPath(dir).createDirectoryAndParents();
    writeLocalFile(actionFs, dir.getChild("file1"), "content1");
    writeLocalFile(actionFs, dir.getChild("file2"), "content2");
    injectRemoteFile(actionFs, dir.getChild("file2"), "content2inmemory");
    injectRemoteFile(actionFs, dir.getChild("file3"), "content3");

    ImmutableList<String> entries =
        directoryEntriesProvider.getDirectoryEntries(actionFs.getPath(dir));

    assertThat(entries).containsExactly("file1", "file2", "file3");
  }

  private void readdirNothingThereThrowsFileNotFound(
      DirectoryEntriesProvider directoryEntriesProvider) throws IOException {
    FileSystem actionFs = createActionFileSystem();
    PathFragment dir = getOutputPath("parent/dir");

    assertThrows(
        FileNotFoundException.class,
        () -> directoryEntriesProvider.getDirectoryEntries(actionFs.getPath(dir)));
  }

  @Test
  public void readdir_nonEmptyLocalDirectory_readFromLocal() throws IOException {
    readdirNonEmptyLocalDirectoryReadFromLocal(
        path ->
            path.readdir(Symlinks.FOLLOW).stream().map(Dirent::getName).collect(toImmutableList()));
  }

  @Test
  public void readdir_nonEmptyInMemoryDirectory_readFromMemory() throws IOException {
    readdirNonEmptyInMemoryDirectoryReadFromMemory(
        path ->
            path.readdir(Symlinks.FOLLOW).stream().map(Dirent::getName).collect(toImmutableList()));
  }

  @Test
  public void readdir_nonEmptyLocalAndInMemoryDirectory_combineThem() throws IOException {
    readdirNonEmptyLocalAndInMemoryDirectoryCombineThem(
        path ->
            path.readdir(Symlinks.FOLLOW).stream().map(Dirent::getName).collect(toImmutableList()));
  }

  @Test
  public void readdir_nothingThere_throwsFileNotFound() throws IOException {
    readdirNothingThereThrowsFileNotFound(
        path ->
            path.readdir(Symlinks.FOLLOW).stream().map(Dirent::getName).collect(toImmutableList()));
  }

  @Test
  public void getDirectoryEntries_nonEmptyLocalDirectory_readFromLocal() throws IOException {
    readdirNonEmptyLocalDirectoryReadFromLocal(
        path ->
            path.getDirectoryEntries().stream().map(Path::getBaseName).collect(toImmutableList()));
  }

  @Test
  public void getDirectoryEntries_nonEmptyInMemoryDirectory_readFromMemory() throws IOException {
    readdirNonEmptyInMemoryDirectoryReadFromMemory(
        path ->
            path.getDirectoryEntries().stream().map(Path::getBaseName).collect(toImmutableList()));
  }

  @Test
  public void getDirectoryEntries_nonEmptyLocalAndInMemoryDirectory_combineThem()
      throws IOException {
    readdirNonEmptyLocalAndInMemoryDirectoryCombineThem(
        path ->
            path.getDirectoryEntries().stream().map(Path::getBaseName).collect(toImmutableList()));
  }

  @Test
  public void getDirectoryEntries_nothingThere_throwsFileNotFound() throws IOException {
    readdirNothingThereThrowsFileNotFound(
        path ->
            path.getDirectoryEntries().stream().map(Path::getBaseName).collect(toImmutableList()));
  }
}
