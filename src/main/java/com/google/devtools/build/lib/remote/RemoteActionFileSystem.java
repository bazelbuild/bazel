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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.collect.Streams.stream;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.DelegateFileSystem;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.FileInfo;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryContentInfo;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.ReadableByteChannel;
import java.util.Collection;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * This is a basic implementation and incomplete implementation of an action file system that's been
 * tuned to what native (non-spawn) actions in Bazel currently use.
 *
 * <p>The implementation mostly delegates to the local file system except for the case where an
 * action input is a remotely stored action output. Most notably {@link
 * #getInputStream(PathFragment)} and {@link #createSymbolicLink(PathFragment, PathFragment)}.
 *
 * <p>This implementation only supports creating local action outputs.
 */
public class RemoteActionFileSystem extends DelegateFileSystem {

  private final PathFragment execRoot;
  private final PathFragment outputBase;
  private final ActionInputMap inputArtifactData;
  private final ImmutableMap<PathFragment, Artifact> outputMapping;
  private final RemoteActionInputFetcher inputFetcher;
  private final RemoteInMemoryFileSystem remoteOutputTree;

  @Nullable private MetadataInjector metadataInjector = null;

  RemoteActionFileSystem(
      FileSystem localDelegate,
      PathFragment execRootFragment,
      String relativeOutputPath,
      ActionInputMap inputArtifactData,
      Iterable<Artifact> outputArtifacts,
      RemoteActionInputFetcher inputFetcher) {
    super(localDelegate);
    this.execRoot = checkNotNull(execRootFragment, "execRootFragment");
    this.outputBase = execRoot.getRelative(checkNotNull(relativeOutputPath, "relativeOutputPath"));
    this.inputArtifactData = checkNotNull(inputArtifactData, "inputArtifactData");
    this.outputMapping =
        stream(outputArtifacts).collect(toImmutableMap(Artifact::getExecPath, a -> a));
    this.inputFetcher = checkNotNull(inputFetcher, "inputFetcher");
    this.remoteOutputTree = new RemoteInMemoryFileSystem(getDigestFunction());
  }

  /** Returns true if {@code path} is a file that's stored remotely. */
  boolean isRemote(Path path) {
    return getRemoteInputMetadata(path.asFragment()) != null;
  }

  public void updateContext(MetadataInjector metadataInjector) {
    this.metadataInjector = metadataInjector;
  }

  void injectRemoteFile(PathFragment path, byte[] digest, long size, String actionId)
      throws IOException {
    if (!path.startsWith(outputBase)) {
      return;
    }
    remoteOutputTree.injectRemoteFile(path, digest, size, actionId);
  }

  void flush() throws IOException {
    checkNotNull(metadataInjector, "metadataInjector is null");

    for (Map.Entry<PathFragment, Artifact> entry : outputMapping.entrySet()) {
      PathFragment execPath = entry.getKey();
      PathFragment path = execRoot.getRelative(execPath);
      Artifact output = entry.getValue();
      if (output.isTreeArtifact()) {
        if (remoteOutputTree.exists(path)) {
          SpecialArtifact parent = (SpecialArtifact) output;
          TreeArtifactValue.Builder tree = TreeArtifactValue.newBuilder(parent);

          // TODO: Check directory content on the local fs to support mixed tree.
          TreeArtifactValue.visitTree(
              remoteOutputTree.getPath(path),
              (parentRelativePath, type) -> {
                if (type == Dirent.Type.DIRECTORY) {
                  return;
                }
                RemoteFileInfo remoteFile =
                    remoteOutputTree.getRemoteFileInfo(
                        path.getRelative(parentRelativePath), /* followSymlinks= */ true);
                if (remoteFile != null) {
                  TreeFileArtifact child =
                      TreeFileArtifact.createTreeOutput(parent, parentRelativePath);
                  tree.putChild(child, createRemoteMetadata(remoteFile));
                }
              });

          metadataInjector.injectTree(parent, tree.build());
        }
      } else {
        RemoteFileInfo remoteFile =
            remoteOutputTree.getRemoteFileInfo(path, /* followSymlinks= */ true);
        if (remoteFile != null) {
          metadataInjector.injectFile(output, createRemoteMetadata(remoteFile));
        }
      }
    }
  }

  private RemoteFileArtifactValue createRemoteMetadata(RemoteFileInfo remoteFile) {
    return RemoteFileArtifactValue.create(
        remoteFile.getFastDigest(),
        remoteFile.getSize(),
        /* locationIndex= */ 1,
        remoteFile.getActionId());
  }

  @Override
  public String getFileSystemType(PathFragment path) {
    return "remoteActionFS";
  }

  @Override
  protected boolean delete(PathFragment path) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m == null) {
      return super.delete(path);
    }
    return remoteOutputTree.getPath(path).delete();
  }

  @Override
  protected InputStream getInputStream(PathFragment path) throws IOException {
    downloadFileIfRemote(path);
    return super.getInputStream(path);
  }

  @Override
  protected ReadableByteChannel createReadableByteChannel(PathFragment path) throws IOException {
    downloadFileIfRemote(path);
    return super.createReadableByteChannel(path);
  }

  @Override
  public void setLastModifiedTime(PathFragment path, long newTime) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m == null) {
      super.setLastModifiedTime(path, newTime);
    }
    remoteOutputTree.setLastModifiedTime(path, newTime);
  }

  @Override
  protected byte[] getFastDigest(PathFragment path) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m != null) {
      return m.getDigest();
    }
    return super.getFastDigest(path);
  }

  @Override
  protected byte[] getDigest(PathFragment path) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m != null) {
      return m.getDigest();
    }
    return super.getDigest(path);
  }

  // -------------------- File Permissions --------------------

  @Override
  protected boolean isReadable(PathFragment path) throws IOException {
    FileArtifactValue m = getRemoteInputMetadata(path);
    return m != null || super.isReadable(path);
  }

  @Override
  protected boolean isWritable(PathFragment path) throws IOException {
    FileArtifactValue m = getRemoteInputMetadata(path);
    return m != null || super.isWritable(path);
  }

  @Override
  protected boolean isExecutable(PathFragment path) throws IOException {
    FileArtifactValue m = getRemoteInputMetadata(path);
    return m != null || super.isExecutable(path);
  }

  @Override
  protected void setReadable(PathFragment path, boolean readable) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m == null) {
      super.setReadable(path, readable);
    }
  }

  @Override
  public void setWritable(PathFragment path, boolean writable) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m == null) {
      super.setWritable(path, writable);
    }
  }

  @Override
  protected void setExecutable(PathFragment path, boolean executable) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m == null) {
      super.setExecutable(path, executable);
    }
  }

  @Override
  protected void chmod(PathFragment path, int mode) throws IOException {
    RemoteFileArtifactValue m = getRemoteInputMetadata(path);
    if (m == null) {
      super.chmod(path, mode);
    }
  }

  // -------------------- Symlinks --------------------

  @Override
  protected PathFragment readSymbolicLink(PathFragment path) throws IOException {
    FileArtifactValue m = getRemoteInputMetadata(path);
    if (m != null) {
      // We don't support symlinks as remote action outputs.
      throw new IOException(path + " is not a symbolic link");
    }
    return super.readSymbolicLink(path);
  }

  @Override
  protected void createSymbolicLink(PathFragment linkPath, PathFragment targetFragment)
      throws IOException {
    /*
     * TODO(buchgr): Optimize the case where we are creating a symlink to a remote output. This does
     * add a non-trivial amount of complications though (as symlinks tend to do).
     */
    downloadFileIfRemote(execRoot.getRelative(targetFragment));
    super.createSymbolicLink(linkPath, targetFragment);
  }

  // -------------------- Implementations based on stat() --------------------

  @Override
  protected long getLastModifiedTime(PathFragment path, boolean followSymlinks) throws IOException {
    FileStatus stat = stat(path, followSymlinks);
    return stat.getLastModifiedTime();
  }

  @Override
  protected long getFileSize(PathFragment path, boolean followSymlinks) throws IOException {
    FileStatus stat = stat(path, followSymlinks);
    return stat.getSize();
  }

  @Override
  protected boolean isFile(PathFragment path, boolean followSymlinks) {
    FileStatus stat = statNullable(path, followSymlinks);
    return stat != null && stat.isFile();
  }

  @Override
  protected boolean isSymbolicLink(PathFragment path) {
    FileStatus stat = statNullable(path, /* followSymlinks= */ false);
    return stat != null && stat.isSymbolicLink();
  }

  @Override
  protected boolean isDirectory(PathFragment path, boolean followSymlinks) {
    FileStatus stat = statNullable(path, followSymlinks);
    return stat != null && stat.isDirectory();
  }

  @Override
  protected boolean isSpecialFile(PathFragment path, boolean followSymlinks) {
    FileStatus stat = statNullable(path, followSymlinks);
    return stat != null && stat.isDirectory();
  }

  @Override
  protected boolean exists(PathFragment path, boolean followSymlinks) {
    try {
      return statIfFound(path, followSymlinks) != null;
    } catch (IOException e) {
      return false;
    }
  }

  @Override
  public boolean exists(PathFragment path) {
    return exists(path, /* followSymlinks= */ true);
  }

  @Nullable
  @Override
  protected FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
    try {
      return stat(path, followSymlinks);
    } catch (FileNotFoundException e) {
      return null;
    }
  }

  @Nullable
  @Override
  protected FileStatus statNullable(PathFragment path, boolean followSymlinks) {
    try {
      return stat(path, followSymlinks);
    } catch (IOException e) {
      return null;
    }
  }

  @Override
  protected FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
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

  @Nullable
  private RemoteFileArtifactValue getRemoteInputMetadata(PathFragment path) {
    if (!path.startsWith(outputBase)) {
      return null;
    }
    PathFragment execPath = path.relativeTo(execRoot);
    FileArtifactValue m = inputArtifactData.getMetadata(execPath);
    if (m != null && m.isRemote()) {
      return (RemoteFileArtifactValue) m;
    }

    RemoteFileInfo remoteFile =
        remoteOutputTree.getRemoteFileInfo(path, /* followSymlinks= */ true);
    if (remoteFile != null) {
      return createRemoteMetadata(remoteFile);
    }

    return null;
  }

  private void downloadFileIfRemote(PathFragment path) throws IOException {
    FileArtifactValue m = getRemoteInputMetadata(path);
    if (m != null) {
      try {
        inputFetcher.downloadFile(delegateFs.getPath(path), m);
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
  protected Collection<String> getDirectoryEntries(PathFragment path) throws IOException {
    return super.getDirectoryEntries(path);
  }

  @Override
  protected void createFSDependentHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    super.createFSDependentHardLink(linkPath, originalPath);
  }

  @Override
  protected Collection<Dirent> readdir(PathFragment path, boolean followSymlinks)
      throws IOException {
    return super.readdir(path, followSymlinks);
  }

  @Override
  protected void createHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    super.createHardLink(linkPath, originalPath);
  }

  static class RemoteInMemoryFileSystem extends InMemoryFileSystem {

    public RemoteInMemoryFileSystem(DigestHashFunction hashFunction) {
      super(hashFunction);
    }

    @Override
    protected synchronized OutputStream getOutputStream(PathFragment path, boolean append)
        throws IOException {
      // To get an output stream from remote file, we need to first stage it.
      throw new IllegalStateException("Shouldn't be called directly");
    }

    @Override
    protected FileInfo newFile(Clock clock, PathFragment path) {
      return new RemoteFileInfo(clock);
    }

    void injectRemoteFile(PathFragment path, byte[] digest, long size, String actionId)
        throws IOException {
      createDirectoryAndParents(path.getParentDirectory());
      InMemoryContentInfo node = getOrCreateWritableInode(path);
      // If a node was already existed and is not a remote file node (i.e. directory or symlink node
      // ), throw an error.
      if (!(node instanceof RemoteFileInfo)) {
        throw new IOException("Could not inject into " + node);
      }

      RemoteFileInfo remoteFileInfo = (RemoteFileInfo) node;
      remoteFileInfo.set(digest, size, actionId);
    }

    @Nullable
    RemoteFileInfo getRemoteFileInfo(PathFragment path, boolean followSymlinks) {
      InMemoryContentInfo node = inodeStatErrno(path, followSymlinks).inode();
      if (!(node instanceof RemoteFileInfo)) {
        return null;
      }
      return (RemoteFileInfo) node;
    }
  }

  static class RemoteFileInfo extends FileInfo {

    private byte[] digest;
    private long size;
    private String actionId;

    RemoteFileInfo(Clock clock) {
      super(clock);
    }

    private void set(byte[] digest, long size, String actionId) {
      this.digest = digest;
      this.size = size;
      this.actionId = actionId;
    }

    @Override
    public OutputStream getOutputStream(boolean append) throws IOException {
      throw new IllegalStateException("Shouldn't be called directly");
    }

    @Override
    public InputStream getInputStream() throws IOException {
      throw new IllegalStateException("Shouldn't be called directly");
    }

    @Override
    public byte[] getxattr(String name) throws IOException {
      throw new IllegalStateException("Shouldn't be called directly");
    }

    @Override
    public byte[] getFastDigest() {
      return digest;
    }

    @Override
    public long getSize() {
      return size;
    }

    public String getActionId() {
      return actionId;
    }
  }
}
