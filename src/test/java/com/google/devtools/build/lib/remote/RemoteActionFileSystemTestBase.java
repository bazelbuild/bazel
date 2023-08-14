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
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.StaticInputMetadataProvider;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.junit.Test;

public abstract class RemoteActionFileSystemTestBase {

  protected abstract FileSystem createActionFileSystem(
      ActionInputMap inputs, Iterable<Artifact> outputs, InputMetadataProvider fileCache)
      throws IOException;

  protected FileSystem createActionFileSystem(ActionInputMap inputs, Iterable<Artifact> outputs)
      throws IOException {
    return createActionFileSystem(inputs, outputs, StaticInputMetadataProvider.empty());
  }

  protected FileSystem createActionFileSystem(ActionInputMap inputs) throws IOException {
    return createActionFileSystem(inputs, ImmutableList.of());
  }

  protected FileSystem createActionFileSystem() throws IOException {
    return createActionFileSystem(new ActionInputMap(0));
  }

  protected abstract FileSystem getLocalFileSystem(FileSystem actionFs);

  protected abstract FileSystem getRemoteFileSystem(FileSystem actionFs);

  protected abstract PathFragment getOutputPath(String outputRootRelativePath);

  protected abstract void writeLocalFile(FileSystem actionFs, PathFragment path, String content)
      throws IOException;

  @CanIgnoreReturnValue
  protected abstract FileArtifactValue injectRemoteFile(
      FileSystem actionFs, PathFragment path, String content) throws IOException;

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

  @Test
  public void isReadable_fileDoesNotExist_throwError() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");

    assertThrows(FileNotFoundException.class, () -> actionFs.getPath(path).isReadable());
  }

  @Test
  public void isReadable_onlyRemoteFile_returnsTrue() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");

    var readable = actionFs.getPath(path).isReadable();

    assertThat(readable).isTrue();
  }

  @Test
  public void isReadable_onlyRemoteDirectory_returnsTrue() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("dir");
    getRemoteFileSystem(actionFs).createDirectoryAndParents(path);

    var readable = actionFs.getPath(path).isReadable();

    assertThat(readable).isTrue();
  }

  @Test
  public void isReadable_localReadableFile_returnsTrue() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local-content");

    var readable = actionFs.getPath(path).isReadable();

    assertThat(readable).isTrue();
  }

  @Test
  public void isReadable_localNonReadableFile_returnsFalse() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local-content");
    getLocalFileSystem(actionFs).getPath(path).setReadable(false);

    var readable = actionFs.getPath(path).isReadable();

    assertThat(readable).isFalse();
  }

  @Test
  public void isReadable_localReadableFileAndRemoteFile_returnsTrue() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");

    var readable = actionFs.getPath(path).isReadable();

    assertThat(readable).isTrue();
  }

  @Test
  public void isReadable_localNonReadableFileAndRemoteFile_returnsFalse() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");
    getLocalFileSystem(actionFs).getPath(path).setReadable(false);

    var readable = actionFs.getPath(path).isReadable();

    assertThat(readable).isFalse();
  }

  @Test
  public void isWritable_fileDoesNotExist_throwError() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");

    assertThrows(FileNotFoundException.class, () -> actionFs.getPath(path).isWritable());
  }

  @Test
  public void isWritable_onlyRemoteFile_returnsTrue() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");

    var writable = actionFs.getPath(path).isWritable();

    assertThat(writable).isTrue();
  }

  @Test
  public void isWritable_onlyRemoteDirectory_returnsTrue() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("dir");
    getRemoteFileSystem(actionFs).createDirectoryAndParents(path);

    var writable = actionFs.getPath(path).isWritable();

    assertThat(writable).isTrue();
  }

  @Test
  public void isWritable_localWritableFile_returnsTrue() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local-content");

    var writable = actionFs.getPath(path).isWritable();

    assertThat(writable).isTrue();
  }

  @Test
  public void isWritable_localNonWritableFile_returnsFalse() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local-content");
    getLocalFileSystem(actionFs).getPath(path).setWritable(false);

    var writable = actionFs.getPath(path).isWritable();

    assertThat(writable).isFalse();
  }

  @Test
  public void isWritable_localWritableFileAndRemoteFile_returnsTrue() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");

    var writable = actionFs.getPath(path).isWritable();

    assertThat(writable).isTrue();
  }

  @Test
  public void isWritable_localNonWritableFileAndRemoteFile_returnsFalse() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");
    getLocalFileSystem(actionFs).getPath(path).setWritable(false);

    var writable = actionFs.getPath(path).isWritable();

    assertThat(writable).isFalse();
  }

  @Test
  public void isWritable_localNonWritableDirectoryAndRemoteDirectory_returnsFalse()
      throws Exception {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("dir");
    getRemoteFileSystem(actionFs).getPath(path).createDirectoryAndParents();
    getLocalFileSystem(actionFs).getPath(path).createDirectoryAndParents();
    getLocalFileSystem(actionFs).getPath(path).setWritable(false);

    boolean writable = actionFs.getPath(path).isWritable();

    assertThat(writable).isFalse();
  }

  @Test
  public void isExecutable_fileDoesNotExist_throwError() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");

    assertThrows(FileNotFoundException.class, () -> actionFs.getPath(path).isExecutable());
  }

  @Test
  public void isExecutable_onlyRemoteFile_returnsTrue() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");

    var executable = actionFs.getPath(path).isExecutable();

    assertThat(executable).isTrue();
  }

  @Test
  public void isExecutable_onlyRemoteDirecotry_returnsTrue() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("dir");
    getRemoteFileSystem(actionFs).createDirectoryAndParents(path);

    var executable = actionFs.getPath(path).isExecutable();

    assertThat(executable).isTrue();
  }

  @Test
  public void isExecutable_localExecutableFile_returnsTrue() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local-content");
    getLocalFileSystem(actionFs).getPath(path).setExecutable(true);

    var executable = actionFs.getPath(path).isExecutable();

    assertThat(executable).isTrue();
  }

  @Test
  public void isExecutable_localNonExecutableFile_returnsFalse() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local-content");

    var executable = actionFs.getPath(path).isExecutable();

    assertThat(executable).isFalse();
  }

  @Test
  public void isExecutable_localExecutableFileAndRemoteFile_returnsTrue() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");
    getLocalFileSystem(actionFs).getPath(path).setExecutable(true);

    var executable = actionFs.getPath(path).isExecutable();

    assertThat(executable).isTrue();
  }

  @Test
  public void isExecutable_localNonExecutableFileAndRemoteFile_returnsFalse() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");

    var executable = actionFs.getPath(path).isExecutable();

    assertThat(executable).isFalse();
  }

  @Test
  public void setReadable_fileDoesNotExist_throwError() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");

    assertThrows(FileNotFoundException.class, () -> actionFs.getPath(path).setReadable(false));
  }

  @Test
  public void setReadable_onlyRemoteFile_remainsReadable() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");

    actionFs.getPath(path).setReadable(false);

    assertThat(actionFs.getPath(path).isReadable()).isTrue();
  }

  @Test
  public void setReadable_onlyRemoteDirecotry_remainsReadable() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("dir");
    getRemoteFileSystem(actionFs).createDirectoryAndParents(path);

    actionFs.getPath(path).setReadable(false);

    assertThat(actionFs.getPath(path).isReadable()).isTrue();
  }

  @Test
  public void setReadable_localFile_change() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local-content");
    assertThat(actionFs.getPath(path).isReadable()).isTrue();

    actionFs.getPath(path).setReadable(false);

    assertThat(actionFs.getPath(path).isReadable()).isFalse();
  }

  @Test
  public void setReadable_localFileAndRemoteFile_changeLocal() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");
    assertThat(actionFs.getPath(path).isReadable()).isTrue();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isReadable()).isTrue();

    actionFs.getPath(path).setReadable(false);

    assertThat(actionFs.getPath(path).isReadable()).isFalse();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isReadable()).isFalse();
  }

  @Test
  public void setWritable_fileDoesNotExist_throwError() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");

    assertThrows(FileNotFoundException.class, () -> actionFs.getPath(path).setWritable(false));
  }

  @Test
  public void setWritable_onlyRemoteFile_remainsWritable() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");

    actionFs.getPath(path).setWritable(false);

    assertThat(actionFs.getPath(path).isWritable()).isTrue();
  }

  @Test
  public void setWritable_onlyRemoteDirecotry_remainsWritable() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("dir");
    getRemoteFileSystem(actionFs).createDirectoryAndParents(path);

    actionFs.getPath(path).setWritable(false);

    assertThat(actionFs.getPath(path).isWritable()).isTrue();
  }

  @Test
  public void setWritable_localFile_change() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local-content");
    assertThat(actionFs.getPath(path).isWritable()).isTrue();

    actionFs.getPath(path).setWritable(false);

    assertThat(actionFs.getPath(path).isWritable()).isFalse();
  }

  @Test
  public void setWritable_localFileAndRemoteFile_changeLocal() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");
    assertThat(actionFs.getPath(path).isWritable()).isTrue();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isWritable()).isTrue();

    actionFs.getPath(path).setWritable(false);

    assertThat(actionFs.getPath(path).isWritable()).isFalse();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isWritable()).isFalse();
  }

  @Test
  public void setExecutable_fileDoesNotExist_throwError() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");

    assertThrows(FileNotFoundException.class, () -> actionFs.getPath(path).setExecutable(false));
  }

  @Test
  public void setExecutable_onlyRemoteFile_remainsExecutable() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");

    actionFs.getPath(path).setExecutable(false);

    assertThat(actionFs.getPath(path).isExecutable()).isTrue();
  }

  @Test
  public void setExecutable_onlyRemoteDirecotry_remainsExecutable() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("dir");
    getRemoteFileSystem(actionFs).createDirectoryAndParents(path);

    actionFs.getPath(path).setExecutable(false);

    assertThat(actionFs.getPath(path).isExecutable()).isTrue();
  }

  @Test
  public void setExecutable_localFile_change() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local-content");
    assertThat(actionFs.getPath(path).isExecutable()).isFalse();

    actionFs.getPath(path).setExecutable(true);

    assertThat(actionFs.getPath(path).isExecutable()).isTrue();
  }

  @Test
  public void setExecutable_localFileAndRemoteFile_changeLocal() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");
    assertThat(actionFs.getPath(path).isExecutable()).isFalse();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isExecutable()).isFalse();

    actionFs.getPath(path).setExecutable(true);

    assertThat(actionFs.getPath(path).isExecutable()).isTrue();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isExecutable()).isTrue();
  }

  @Test
  public void chmod_fileDoesNotExist_throwError() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");

    assertThrows(FileNotFoundException.class, () -> actionFs.getPath(path).chmod(000));
  }

  @Test
  public void chmod_onlyRemoteFile_remainsSame() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    assertThat(actionFs.getPath(path).isReadable()).isTrue();
    assertThat(actionFs.getPath(path).isWritable()).isTrue();
    assertThat(actionFs.getPath(path).isExecutable()).isTrue();

    actionFs.getPath(path).chmod(000);

    assertThat(actionFs.getPath(path).isReadable()).isTrue();
    assertThat(actionFs.getPath(path).isWritable()).isTrue();
    assertThat(actionFs.getPath(path).isExecutable()).isTrue();
  }

  @Test
  public void chmod_onlyRemoteDirectory_remainsSame() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("dir");
    getRemoteFileSystem(actionFs).createDirectoryAndParents(path);
    assertThat(actionFs.getPath(path).isReadable()).isTrue();
    assertThat(actionFs.getPath(path).isWritable()).isTrue();
    assertThat(actionFs.getPath(path).isExecutable()).isTrue();

    actionFs.getPath(path).chmod(000);

    assertThat(actionFs.getPath(path).isReadable()).isTrue();
    assertThat(actionFs.getPath(path).isWritable()).isTrue();
    assertThat(actionFs.getPath(path).isExecutable()).isTrue();
  }

  @Test
  public void chmod_localFile_change() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local-content");
    assertThat(actionFs.getPath(path).isReadable()).isTrue();
    assertThat(actionFs.getPath(path).isWritable()).isTrue();
    assertThat(actionFs.getPath(path).isExecutable()).isFalse();

    actionFs.getPath(path).chmod(0111);

    assertThat(actionFs.getPath(path).isReadable()).isFalse();
    assertThat(actionFs.getPath(path).isWritable()).isFalse();
    assertThat(actionFs.getPath(path).isExecutable()).isTrue();
  }

  @Test
  public void chmod_localFileAndRemoteFile_changeLocal() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");
    assertThat(actionFs.getPath(path).isReadable()).isTrue();
    assertThat(actionFs.getPath(path).isWritable()).isTrue();
    assertThat(actionFs.getPath(path).isExecutable()).isFalse();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isReadable()).isTrue();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isWritable()).isTrue();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isExecutable()).isFalse();

    actionFs.getPath(path).chmod(0111);

    assertThat(actionFs.getPath(path).isReadable()).isFalse();
    assertThat(actionFs.getPath(path).isWritable()).isFalse();
    assertThat(actionFs.getPath(path).isExecutable()).isTrue();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isReadable()).isFalse();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isWritable()).isFalse();
    assertThat(getLocalFileSystem(actionFs).getPath(path).isExecutable()).isTrue();
  }

  @Test
  public void getLastModifiedTime_fileDoesNotExist_throwError() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");

    assertThrows(FileNotFoundException.class, () -> actionFs.getPath(path).getLastModifiedTime());
  }

  @Test
  public void getLastModifiedTime_onlyRemoteFile_returnRemote() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");

    var mtime = actionFs.getPath(path).getLastModifiedTime();

    assertThat(mtime).isEqualTo(getRemoteFileSystem(actionFs).getPath(path).getLastModifiedTime());
  }

  @Test
  public void getLastModifiedTime_onlyLocalFile_returnLocal() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local-content");

    var mtime = actionFs.getPath(path).getLastModifiedTime();

    assertThat(mtime).isEqualTo(getLocalFileSystem(actionFs).getPath(path).getLastModifiedTime());
  }

  @Test
  public void getLastModifiedTime_localAndRemoteFile_returnRemote() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");

    var mtime = actionFs.getPath(path).getLastModifiedTime();

    assertThat(mtime).isEqualTo(getRemoteFileSystem(actionFs).getPath(path).getLastModifiedTime());
  }

  @Test
  public void setLastModifiedTime_fileDoesNotExist_throwError() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");

    assertThrows(FileNotFoundException.class, () -> actionFs.getPath(path).setLastModifiedTime(0));
  }

  @Test
  public void setLastModifiedTime_onlyRemoteFile_successfullySet() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    assertThat(actionFs.getPath(path).getLastModifiedTime()).isNotEqualTo(0);

    actionFs.getPath(path).setLastModifiedTime(0);

    assertThat(actionFs.getPath(path).getLastModifiedTime()).isEqualTo(0);
  }

  @Test
  public void setLastModifiedTime_onlyLocalFile_successfullySet() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    writeLocalFile(actionFs, path, "local-content");
    assertThat(actionFs.getPath(path).getLastModifiedTime()).isNotEqualTo(0);

    actionFs.getPath(path).setLastModifiedTime(0);

    assertThat(actionFs.getPath(path).getLastModifiedTime()).isEqualTo(0);
  }

  @Test
  public void setLastModifiedTime_localAndRemoteFile_changeBoth() throws IOException {
    var actionFs = createActionFileSystem();
    var path = getOutputPath("file");
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");
    assertThat(getLocalFileSystem(actionFs).getPath(path).getLastModifiedTime()).isNotEqualTo(0);
    assertThat(getRemoteFileSystem(actionFs).getPath(path).getLastModifiedTime()).isNotEqualTo(0);

    actionFs.getPath(path).setLastModifiedTime(0);

    assertThat(getLocalFileSystem(actionFs).getPath(path).getLastModifiedTime()).isEqualTo(0);
    assertThat(getRemoteFileSystem(actionFs).getPath(path).getLastModifiedTime()).isEqualTo(0);
  }
}
