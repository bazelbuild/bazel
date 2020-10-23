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

package com.google.devtools.build.lib.remote;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.vfs.DelegateFileSystem;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.channels.ReadableByteChannel;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * This is a basic implementation and incomplete implementation of an action file system that's been
 * tuned to what native (non-spawn) actions in Bazel currently use.
 *
 * <p>The implementation mostly delegates to the local file system except for the case where an
 * action input is a remotely stored action output. Most notably {@link #getInputStream(Path)} and
 * {@link #createSymbolicLink(Path, PathFragment)}.
 *
 * <p>This implementation only supports creating local action outputs.
 */
class RemoteActionFileSystem extends DelegateFileSystem {

  private final Path execRoot;
  private final Path outputBase;
  private final ActionInputMap inputArtifactData;
  private final RemoteActionInputFetcher inputFetcher;

  RemoteActionFileSystem(
      FileSystem localDelegate,
      PathFragment execRootFragment,
      String relativeOutputPath,
      ActionInputMap inputArtifactData,
      RemoteActionInputFetcher inputFetcher) {
    super(localDelegate);
    this.execRoot = getPath(Preconditions.checkNotNull(execRootFragment, "execRootFragment"));
    this.outputBase =
        execRoot.getRelative(Preconditions.checkNotNull(relativeOutputPath, "relativeOutputPath"));
    this.inputArtifactData = Preconditions.checkNotNull(inputArtifactData, "inputArtifactData");
    this.inputFetcher = Preconditions.checkNotNull(inputFetcher, "inputFetcher");
  }

  /** Returns true if {@code path} is a file that's stored remotely. */
  boolean isRemote(Path path) {
    return getRemoteInputMetadata(path) != null;
  }

  @Override
  public String getFileSystemType(Path path) {
    return "remoteActionFS";
  }

  @Override
  public boolean delete(Path path) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m == null) {
      return super.delete(path);
    }
    return true;
  }

  @Override
  protected InputStream getInputStream(Path path) throws IOException {
    downloadFileIfRemote(path);
    return super.getInputStream(path);
  }

  @Override
  protected ReadableByteChannel createReadableByteChannel(Path path) throws IOException {
    downloadFileIfRemote(path);
    return super.createReadableByteChannel(path);
  }

  @Override
  public void setLastModifiedTime(Path path, long newTime) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m == null) {
      super.setLastModifiedTime(path, newTime);
    }
  }

  @Override
  protected byte[] getFastDigest(Path path) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m != null) {
      return m.getDigest();
    }
    return super.getFastDigest(path);
  }

  @Override
  protected byte[] getDigest(Path path) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m != null) {
      return m.getDigest();
    }
    return super.getDigest(path);
  }

  // -------------------- File Permissions --------------------

  @Override
  protected boolean isReadable(Path path) throws IOException {
    FileArtifactValue m = getRemoteInputMetadata(path);
    return m != null || super.isReadable(path);
  }

  @Override
  protected boolean isWritable(Path path) throws IOException {
    FileArtifactValue m = getRemoteInputMetadata(path);
    return m != null || super.isWritable(path);
  }

  @Override
  protected boolean isExecutable(Path path) throws IOException {
    FileArtifactValue m = getRemoteInputMetadata(path);
    return m != null || super.isExecutable(path);
  }

  @Override
  protected void setReadable(Path path, boolean readable) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m == null) {
      super.setReadable(path, readable);
    }
  }

  @Override
  public void setWritable(Path path, boolean writable) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m == null) {
      super.setWritable(path, writable);
    }
  }

  @Override
  protected void setExecutable(Path path, boolean executable) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m == null) {
      super.setExecutable(path, executable);
    }
  }

  @Override
  protected void chmod(Path path, int mode) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m == null) {
      super.chmod(path, mode);
    }
  }

  // -------------------- Symlinks --------------------

  @Override
  protected PathFragment readSymbolicLink(Path path) throws IOException {
    FileArtifactValue m = getRemoteInputMetadata(path);
    if (m != null) {
      // We don't support symlinks as remote action outputs.
      throw new IOException(path + " is not a symbolic link");
    }
    return super.readSymbolicLink(path);
  }

  @Override
  protected void createSymbolicLink(Path linkPath, PathFragment targetFragment) throws IOException {
    /*
     * TODO(buchgr): Optimize the case where we are creating a symlink to a remote output. This does
     * add a non-trivial amount of complications though (as symlinks tend to do).
     */
    downloadFileIfRemote(execRoot.getRelative(targetFragment));
    super.createSymbolicLink(linkPath, targetFragment);
  }

  // -------------------- Implementations based on stat() --------------------

  @Override
  protected long getLastModifiedTime(Path path, boolean followSymlinks) throws IOException {
    FileStatus stat = stat(path, followSymlinks);
    return stat.getLastModifiedTime();
  }

  @Override
  protected long getFileSize(Path path, boolean followSymlinks) throws IOException {
    FileStatus stat = stat(path, followSymlinks);
    return stat.getSize();
  }

  @Override
  protected boolean isFile(Path path, boolean followSymlinks) {
    FileStatus stat = statNullable(path, followSymlinks);
    return stat != null && stat.isFile();
  }

  @Override
  protected boolean isSymbolicLink(Path path) {
    FileStatus stat = statNullable(path, /* followSymlinks= */ false);
    return stat != null && stat.isSymbolicLink();
  }

  @Override
  protected boolean isDirectory(Path path, boolean followSymlinks) {
    FileStatus stat = statNullable(path, followSymlinks);
    return stat != null && stat.isDirectory();
  }

  @Override
  protected boolean isSpecialFile(Path path, boolean followSymlinks) {
    FileStatus stat = statNullable(path, followSymlinks);
    return stat != null && stat.isDirectory();
  }

  @Override
  protected boolean exists(Path path, boolean followSymlinks) {
    try {
      return statIfFound(path, followSymlinks) != null;
    } catch (IOException e) {
      return false;
    }
  }

  @Override
  public boolean exists(Path path) {
    return exists(path, /* followSymlinks= */ true);
  }

  @Nullable
  @Override
  protected FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
    try {
      return stat(path, followSymlinks);
    } catch (FileNotFoundException e) {
      return null;
    }
  }

  @Nullable
  @Override
  protected FileStatus statNullable(Path path, boolean followSymlinks) {
    try {
      return stat(path, followSymlinks);
    } catch (IOException e) {
      return null;
    }
  }

  @Override
  protected FileStatus stat(Path path, boolean followSymlinks) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m != null) {
      return statFromRemoteMetadata(m);
    }
    return super.stat(path, followSymlinks);
  }

  private static FileStatus statFromRemoteMetadata(RemoteFileArtifactValue m) {
    return new FileStatus() {
      @Override
      public boolean isFile() {
        return m.getType().isFile();
      }

      @Override
      public boolean isDirectory() {
        return m.getType().isDirectory();
      }

      @Override
      public boolean isSymbolicLink() {
        return m.getType().isSymlink();
      }

      @Override
      public boolean isSpecialFile() {
        return m.getType().isSpecialFile();
      }

      @Override
      public long getSize() {
        return m.getSize();
      }

      @Override
      public long getLastModifiedTime() {
        return m.getModifiedTime();
      }

      @Override
      public long getLastChangeTime() {
        return m.getModifiedTime();
      }

      @Override
      public long getNodeId() {
        throw new UnsupportedOperationException("Cannot get node id for " + m);
      }
    };
  }

  // -------------------- Implementation Helpers --------------------

  private String execPathString(Path path) {
    return path.relativeTo(execRoot).getPathString();
  }

  @Nullable
  private RemoteFileArtifactValue getRemoteInputMetadata(Path path) {
    if (!path.startsWith(outputBase)) {
      return null;
    }
    return getRemoteInputMetadata(execPathString(path));
  }

  @Nullable
  private RemoteFileArtifactValue getRemoteInputMetadata(String execPathString) {
    FileArtifactValue m = inputArtifactData.getMetadata(execPathString);
    if (m != null && m.isRemote()) {
      return (RemoteFileArtifactValue) m;
    }
    return null;
  }

  private void downloadFileIfRemote(Path path) throws IOException {
    FileArtifactValue m = getRemoteInputMetadata(path);
    if (m != null) {
      try {
        inputFetcher.downloadFile(toDelegatePath(path), m);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException(
            String.format("Received interrupt while fetching file '%s'", path), e);
      }
    }
  }

  /*
   * -------------------- TODO(buchgr): Not yet implemented --------------------
   *
   * The below methods have not (yet) been properly implemented due to time constraints mostly and
   * with little risk as they currently don't seem to be used by native actions in Bazel. However,
   * before making the --experimental_remote_download_outputs flag non-experimental we should make
   * sure to fully implement this file system.
   */

  @Override
  protected Collection<String> getDirectoryEntries(Path path) throws IOException {
    return super.getDirectoryEntries(path);
  }

  @Override
  protected void createFSDependentHardLink(Path linkPath, Path originalPath) throws IOException {
    super.createFSDependentHardLink(linkPath, originalPath);
  }

  @Override
  protected Collection<Dirent> readdir(Path path, boolean followSymlinks) throws IOException {
    return super.readdir(path, followSymlinks);
  }

  @Override
  protected void createHardLink(Path linkPath, Path originalPath) throws IOException {
    super.createHardLink(linkPath, originalPath);
  }
}
