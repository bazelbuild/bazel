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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.collect.Streams.stream;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Priority;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.UnresolvedSymlinkArtifactValue;
import com.google.devtools.build.lib.actions.FileStatusWithMetadata;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.AbstractFileSystemWithCustomStat;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.FileInfo;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryContentInfo;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.SeekableByteChannel;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * This is a basic implementation and incomplete implementation of an action file system that's been
 * tuned to what internal (non-spawn) actions in Bazel currently use.
 *
 * <p>The implementation mostly delegates to the local file system except for the case where an
 * action input is a remotely stored action output. Most notably {@link
 * #getInputStream(PathFragment)} and {@link #createSymbolicLink(PathFragment, PathFragment)}.
 *
 * <p>This implementation only supports creating local action outputs.
 */
public class RemoteActionFileSystem extends AbstractFileSystemWithCustomStat {
  private final PathFragment execRoot;
  private final PathFragment outputBase;
  private final InputMetadataProvider fileCache;
  private final ActionInputMap inputArtifactData;
  private final TreeArtifactDirectoryCache inputTreeArtifactDirectoryCache;
  private final ImmutableMap<PathFragment, Artifact> outputMapping;
  private final RemoteActionInputFetcher inputFetcher;
  private final FileSystem localFs;
  private final RemoteInMemoryFileSystem remoteOutputTree;

  @Nullable private ActionExecutionMetadata action = null;

  /** Describes how to handle symlinks when calling {@link #statUnchecked}. */
  private enum FollowMode {
    /** Dereference the entire path. This is equivalent to {@link Symlinks.FOLLOW}. */
    FOLLOW_ALL,
    /** Dereference the parent path. This is equivalent to {@link Symlinks.NOFOLLOW}. */
    FOLLOW_PARENT,
    /** Do not dereference. This is only used internally to resolve symlinks efficiently. */
    FOLLOW_NONE
  };

  private static final FileStatus DIRECTORY_FILE_STATUS =
      new FileStatus() {
        @Override
        public boolean isFile() {
          return false;
        }

        @Override
        public boolean isDirectory() {
          return true;
        }

        @Override
        public boolean isSymbolicLink() {
          return false;
        }

        @Override
        public boolean isSpecialFile() {
          return false;
        }

        @Override
        public long getSize() {
          return 0;
        }

        @Override
        public long getLastModifiedTime() {
          throw new UnsupportedOperationException();
        }

        @Override
        public long getLastChangeTime() {
          throw new UnsupportedOperationException();
        }

        @Override
        public long getNodeId() {
          throw new UnsupportedOperationException();
        }
      };

  /**
   * Caches the contents of intermediate subdirectories of tree artifact inputs, to speed up {@link
   * #stat} and {@link #readdir} operations. Note that actions are not expected to modify their
   * inputs.
   *
   * <p>Safe for concurrent access.
   */
  private class TreeArtifactDirectoryCache {
    private final Set<SpecialArtifact> cachedTrees = new HashSet<>();
    private final HashMap<PathFragment, HashSet<Dirent>> dirToEntries = new HashMap<>();

    @Nullable
    public synchronized Collection<Dirent> get(PathFragment execPath) {
      ensureCached(execPath);
      return dirToEntries.get(execPath);
    }

    private void ensureCached(PathFragment execPath) {
      TreeArtifactValue treeMetadata = inputArtifactData.getTreeMetadataForPrefix(execPath);
      if (treeMetadata == null || treeMetadata.getChildren().isEmpty()) {
        return;
      }
      SpecialArtifact parent = Iterables.getFirst(treeMetadata.getChildren(), null).getParent();
      if (!cachedTrees.contains(parent)) {
        insertTree(treeMetadata);
        cachedTrees.add(parent);
      }
    }

    private void insertTree(TreeArtifactValue treeMetadata) {
      for (TreeFileArtifact child : treeMetadata.getChildren()) {
        insertChild(child);
      }
    }

    private void insertChild(TreeFileArtifact child) {
      PathFragment treeRoot = child.getParent().getExecPath();
      PathFragment path = child.getExecPath();

      while (!path.equals(treeRoot)) {
        PathFragment parentPath = path.getParentDirectory();
        String name = path.getBaseName();
        Dirent.Type type =
            path.equals(child.getExecPath()) ? Dirent.Type.FILE : Dirent.Type.DIRECTORY;

        HashSet<Dirent> entries =
            dirToEntries.computeIfAbsent(parentPath, unused -> new HashSet<>());

        if (!entries.add(new Dirent(name, type))) {
          // Avoid wasted work on common prefixes.
          break;
        }

        path = parentPath;
      }
    }
  }

  public RemoteActionFileSystem(
      FileSystem localFs,
      PathFragment execRootFragment,
      String relativeOutputPath,
      ActionInputMap inputArtifactData,
      Iterable<Artifact> outputArtifacts,
      InputMetadataProvider fileCache,
      RemoteActionInputFetcher inputFetcher) {
    super(localFs.getDigestFunction());
    this.execRoot = checkNotNull(execRootFragment, "execRootFragment");
    this.outputBase = execRoot.getRelative(checkNotNull(relativeOutputPath, "relativeOutputPath"));
    this.inputArtifactData = checkNotNull(inputArtifactData, "inputArtifactData");
    this.inputTreeArtifactDirectoryCache = new TreeArtifactDirectoryCache();
    this.outputMapping =
        stream(outputArtifacts).collect(toImmutableMap(Artifact::getExecPath, a -> a));
    this.fileCache = checkNotNull(fileCache, "fileCache");
    this.inputFetcher = checkNotNull(inputFetcher, "inputFetcher");
    this.localFs = checkNotNull(localFs, "localFs");
    this.remoteOutputTree = new RemoteInMemoryFileSystem(getDigestFunction());
  }

  @Override
  public boolean supportsModifications(PathFragment path) {
    return localFs.supportsModifications(path);
  }

  @Override
  public boolean supportsSymbolicLinksNatively(PathFragment path) {
    return localFs.supportsSymbolicLinksNatively(path);
  }

  @Override
  public boolean supportsHardLinksNatively(PathFragment path) {
    return localFs.supportsHardLinksNatively(path);
  }

  @Override
  public boolean isFilePathCaseSensitive() {
    return localFs.isFilePathCaseSensitive();
  }

  @VisibleForTesting
  protected RemoteInMemoryFileSystem getRemoteOutputTree() {
    return remoteOutputTree;
  }

  @VisibleForTesting
  protected FileSystem getLocalFileSystem() {
    return localFs;
  }

  /** Returns true if {@code path} is a file that's stored remotely. */
  boolean isRemote(Path path) {
    return isRemote(path.asFragment());
  }

  private boolean isRemote(PathFragment path) {
    var status = statInMemory(path, FollowMode.FOLLOW_ALL);
    return (status instanceof FileStatusWithMetadata)
        && ((FileStatusWithMetadata) status).getMetadata().isRemote();
  }

  public void updateContext(ActionExecutionMetadata action) {
    this.action = action;
  }

  void injectRemoteFile(PathFragment path, byte[] digest, long size, long expireAtEpochMilli)
      throws IOException {
    if (!isOutput(path)) {
      return;
    }
    var metadata =
        RemoteFileArtifactValue.create(digest, size, /* locationIndex= */ 1, expireAtEpochMilli);
    remoteOutputTree.injectFile(path, metadata);
  }

  @Override
  public String getFileSystemType(PathFragment path) {
    return "remoteActionFS";
  }

  @Override
  protected boolean delete(PathFragment path) throws IOException {
    boolean deleted = localFs.getPath(path).delete();
    if (isOutput(path)) {
      deleted = remoteOutputTree.getPath(path).delete() || deleted;
    }
    return deleted;
  }

  @Override
  protected InputStream getInputStream(PathFragment path) throws IOException {
    downloadFileIfRemote(path);
    // TODO(tjgq): Consider only falling back to the local filesystem for source (non-output) files.
    // See getMetadata() for why this isn't currently possible.
    return localFs.getPath(path).getInputStream();
  }

  @Override
  protected OutputStream getOutputStream(PathFragment path, boolean append, boolean internal)
      throws IOException {
    return localFs.getPath(path).getOutputStream(append, internal);
  }

  @Override
  protected ReadableByteChannel createReadableByteChannel(PathFragment path) throws IOException {
    downloadFileIfRemote(path);
    return localFs.getPath(path).createReadableByteChannel();
  }

  @Override
  protected SeekableByteChannel createReadWriteByteChannel(PathFragment path) throws IOException {
    return localFs.getPath(path).createReadWriteByteChannel();
  }

  @Override
  public void setLastModifiedTime(PathFragment path, long newTime) throws IOException {
    FileNotFoundException remoteException = null;
    try {
      // We can't set mtime for a remote file, set mtime of in-memory file node instead.
      remoteOutputTree.setLastModifiedTime(path, newTime);
    } catch (FileNotFoundException e) {
      remoteException = e;
    }

    FileNotFoundException localException = null;
    try {
      localFs.getPath(path).setLastModifiedTime(newTime);
    } catch (FileNotFoundException e) {
      localException = e;
    }

    if (remoteException == null || localException == null) {
      return;
    }

    localException.addSuppressed(remoteException);
    throw localException;
  }

  @Override
  public byte[] getxattr(PathFragment path, String name, boolean followSymlinks)
      throws IOException {
    return localFs
        .getPath(path)
        .getxattr(name, followSymlinks ? Symlinks.FOLLOW : Symlinks.NOFOLLOW);
  }

  @Override
  protected byte[] getFastDigest(PathFragment path) throws IOException {
    var stat = statInMemory(path, FollowMode.FOLLOW_ALL);
    if (stat instanceof FileStatusWithDigest) {
      return ((FileStatusWithDigest) stat).getDigest();
    }
    return localFs.getPath(path).getFastDigest();
  }

  @Override
  protected byte[] getDigest(PathFragment path) throws IOException {
    var status = statInMemory(path, FollowMode.FOLLOW_ALL);
    if (status instanceof FileStatusWithDigest) {
      return ((FileStatusWithDigest) status).getDigest();
    }
    return localFs.getPath(path).getDigest();
  }

  // -------------------- File Permissions --------------------
  // Remote files are always readable, writable and executable since we can't control their
  // permissions.

  @Override
  protected boolean isReadable(PathFragment path) throws IOException {
    path = resolveSymbolicLinks(path).asFragment();
    try {
      return localFs.getPath(path).isReadable();
    } catch (FileNotFoundException e) {
      return true;
    }
  }

  @Override
  protected boolean isWritable(PathFragment path) throws IOException {
    path = resolveSymbolicLinks(path).asFragment();
    try {
      return localFs.getPath(path).isWritable();
    } catch (FileNotFoundException e) {
      return true;
    }
  }

  @Override
  protected boolean isExecutable(PathFragment path) throws IOException {
    path = resolveSymbolicLinks(path).asFragment();
    try {
      return localFs.getPath(path).isExecutable();
    } catch (FileNotFoundException e) {
      return true;
    }
  }

  @Override
  protected void setReadable(PathFragment path, boolean readable) throws IOException {
    path = resolveSymbolicLinks(path).asFragment();
    try {
      localFs.getPath(path).setReadable(readable);
    } catch (FileNotFoundException e) {
      // Intentionally ignored.
    }
  }

  @Override
  public void setWritable(PathFragment path, boolean writable) throws IOException {
    path = resolveSymbolicLinks(path).asFragment();
    try {
      localFs.getPath(path).setWritable(writable);
    } catch (FileNotFoundException e) {
      // Intentionally ignored.
    }
  }

  @Override
  protected void setExecutable(PathFragment path, boolean executable) throws IOException {
    path = resolveSymbolicLinks(path).asFragment();
    try {
      localFs.getPath(path).setExecutable(executable);
    } catch (FileNotFoundException e) {
      // Intentionally ignored.
    }
  }

  @Override
  protected void chmod(PathFragment path, int mode) throws IOException {
    path = resolveSymbolicLinks(path).asFragment();
    try {
      localFs.getPath(path).chmod(mode);
    } catch (FileNotFoundException e) {
      // Intentionally ignored.
    }
  }

  // -------------------- Symlinks --------------------

  @Override
  protected PathFragment readSymbolicLink(PathFragment path) throws IOException {
    PathFragment parentPath = path.getParentDirectory();
    if (parentPath != null) {
      path = resolveSymbolicLinks(parentPath).asFragment().getChild(path.getBaseName());
    }

    if (path.startsWith(execRoot)) {
      var execPath = path.relativeTo(execRoot);
      var metadata = inputArtifactData.getMetadata(execPath);
      if (metadata instanceof UnresolvedSymlinkArtifactValue) {
        return PathFragment.create(((UnresolvedSymlinkArtifactValue) metadata).getSymlinkTarget());
      }
      if (metadata != null) {
        // Other input artifacts are never symlinks.
        throw new NotASymlinkException(path);
      }
      if (inputTreeArtifactDirectoryCache.get(execPath) != null) {
        // Tree artifacts never contain symlinks.
        throw new NotASymlinkException(path);
      }
    }

    if (isOutput(path)) {
      try {
        return remoteOutputTree.getPath(path).readSymbolicLink();
      } catch (FileNotFoundException e) {
        // Intentionally ignored.
      }
    }

    return localFs.getPath(path).readSymbolicLink();
  }

  @Override
  protected void createSymbolicLink(PathFragment linkPath, PathFragment targetFragment)
      throws IOException {
    PathFragment parentPath = linkPath.getParentDirectory();
    if (parentPath != null) {
      linkPath = resolveSymbolicLinks(parentPath).asFragment().getChild(linkPath.getBaseName());
    }

    if (isOutput(linkPath)) {
      remoteOutputTree.getPath(linkPath).createSymbolicLink(targetFragment);
    }

    localFs.getPath(linkPath).createSymbolicLink(targetFragment);
  }

  @Nullable
  @Override
  protected PathFragment resolveOneLink(PathFragment path) throws IOException {
    // The base implementation attempts to readSymbolicLink first and falls back to stat, but that
    // unnecessarily allocates a NotASymlinkException in the overwhelmingly likely non-symlink case.
    // It's more efficient to stat unconditionally.
    //
    // The parent path has already been canonicalized, so FOLLOW_NONE is effectively the same as
    // FOLLOW_PARENT, but much more efficient as it doesn't call stat recursively.
    var stat = statUnchecked(path, FollowMode.FOLLOW_NONE);
    return stat.isSymbolicLink() ? readSymbolicLink(path) : null;
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
  protected boolean exists(PathFragment path, boolean followSymlinks) {
    try {
      return statIfFound(path, followSymlinks) != null;
    } catch (IOException e) {
      return false;
    }
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
    return statUnchecked(path, followSymlinks ? FollowMode.FOLLOW_ALL : FollowMode.FOLLOW_PARENT);
  }

  @Nullable
  private FileStatus statUnchecked(PathFragment path, FollowMode followMode) throws IOException {
    if (followMode == FollowMode.FOLLOW_ALL) {
      path = resolveSymbolicLinks(path).asFragment();
    } else if (followMode == FollowMode.FOLLOW_PARENT) {
      PathFragment parent = path.getParentDirectory();
      if (parent != null) {
        path = resolveSymbolicLinks(parent).asFragment().getChild(path.getBaseName());
      }
    }

    var status = statInMemory(path, followMode);
    if (status != null) {
      return status;
    }
    // The path has already been canonicalized above.
    return localFs.getPath(path).stat(Symlinks.NOFOLLOW);
  }

  @Nullable
  private FileStatus statInMemory(PathFragment path, FollowMode followMode) {
    if (path.startsWith(execRoot)) {
      var execPath = path.relativeTo(execRoot);
      var metadata = inputArtifactData.getMetadata(execPath);
      if (metadata != null) {
        return statFromMetadata(metadata);
      }
      if (inputTreeArtifactDirectoryCache.get(execPath) != null) {
        return DIRECTORY_FILE_STATUS;
      }
    }

    return remoteOutputTree.statNullable(
        path, /* followSymlinks= */ followMode == FollowMode.FOLLOW_ALL);
  }

  private static FileStatusWithMetadata statFromMetadata(FileArtifactValue m) {
    return new FileStatusWithMetadata() {
      @Override
      public byte[] getDigest() {
        return m.getDigest();
      }

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

      @Override
      public FileArtifactValue getMetadata() {
        return m;
      }
    };
  }

  @Nullable
  @VisibleForTesting
  ActionInput getInput(String execPath) {
    ActionInput input = inputArtifactData.getInput(execPath);
    if (input != null) {
      return input;
    }
    input = outputMapping.get(PathFragment.create(execPath));
    if (input != null) {
      return input;
    }
    if (!isOutput(execRoot.getRelative(execPath))) {
      return fileCache.getInput(execPath);
    }
    return null;
  }

  @Nullable
  @VisibleForTesting
  FileArtifactValue getInputMetadata(ActionInput input) {
    PathFragment execPath = input.getExecPath();
    return inputArtifactData.getMetadata(execPath);
  }

  private void downloadFileIfRemote(PathFragment path) throws IOException {
    if (!isRemote(path)) {
      return;
    }
    PathFragment execPath = path.relativeTo(execRoot);
    try {
      ActionInput input = getInput(execPath.getPathString());
      if (input == null) {
        // For undeclared outputs, getInput returns null as there's no artifact associated with the
        // path. Therefore, we synthesize one here just so we're able to call prefetchFiles.
        input = ActionInputHelper.fromPath(execPath);
      }
      getFromFuture(
          inputFetcher.prefetchFiles(
              action, ImmutableList.of(input), this::getInputMetadata, Priority.CRITICAL));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException(String.format("Received interrupt while fetching file '%s'", path), e);
    }
  }

  private boolean isOutput(PathFragment path) {
    return path.startsWith(outputBase);
  }

  @Override
  public void renameTo(PathFragment sourcePath, PathFragment targetPath) throws IOException {
    checkArgument(isOutput(sourcePath), "sourcePath must be an output path");
    checkArgument(isOutput(targetPath), "targetPath must be an output path");

    FileNotFoundException remoteException = null;
    try {
      remoteOutputTree.renameTo(sourcePath, targetPath);
    } catch (FileNotFoundException e) {
      remoteException = e;
    }

    FileNotFoundException localException = null;
    try {
      localFs.renameTo(sourcePath, targetPath);
    } catch (FileNotFoundException e) {
      localException = e;
    }

    if (remoteException == null || localException == null) {
      return;
    }

    localException.addSuppressed(remoteException);
    throw localException;
  }

  @Override
  public void createDirectoryAndParents(PathFragment path) throws IOException {
    localFs.createDirectoryAndParents(path);
    if (isOutput(path)) {
      remoteOutputTree.createDirectoryAndParents(path);
    }
  }

  @CanIgnoreReturnValue
  @Override
  public boolean createDirectory(PathFragment path) throws IOException {
    boolean created = localFs.createDirectory(path);
    if (isOutput(path)) {
      created = remoteOutputTree.createDirectory(path) || created;
    }
    return created;
  }

  @Override
  protected ImmutableList<String> getDirectoryEntries(PathFragment path) throws IOException {
    HashSet<String> entries = new HashSet<>();

    boolean found = false;

    if (path.startsWith(execRoot)) {
      var execPath = path.relativeTo(execRoot);
      Collection<Dirent> treeEntries = inputTreeArtifactDirectoryCache.get(execPath);
      if (treeEntries != null) {
        for (var entry : treeEntries) {
          entries.add(entry.getName());
        }
        found = true;
      }
    }

    if (isOutput(path)) {
      try {
        remoteOutputTree.getPath(path).getDirectoryEntries().stream()
            .map(Path::getBaseName)
            .forEach(entries::add);
        found = true;
      } catch (FileNotFoundException ignored) {
        // Will be rethrown below if directory exists on neither side.
      }
    }

    try {
      localFs.getPath(path).getDirectoryEntries().stream()
          .map(Path::getBaseName)
          .forEach(entries::add);
    } catch (FileNotFoundException e) {
      if (!found) {
        throw e;
      }
    }

    // sort entries to get a deterministic order.
    return ImmutableList.sortedCopyOf(entries);
  }

  @Override
  protected Collection<Dirent> readdir(PathFragment path, boolean followSymlinks)
      throws IOException {
    HashMap<String, Dirent> entries = new HashMap<>();

    boolean found = false;

    path = resolveSymbolicLinks(path).asFragment();

    if (path.startsWith(execRoot)) {
      var execPath = path.relativeTo(execRoot);
      Collection<Dirent> treeEntries = inputTreeArtifactDirectoryCache.get(execPath);
      if (treeEntries != null) {
        for (var entry : treeEntries) {
          entries.put(entry.getName(), entry);
        }
        found = true;
      }
    }

    if (isOutput(path)) {
      try {
        for (var entry : remoteOutputTree.getPath(path).readdir(Symlinks.NOFOLLOW)) {
          entry = maybeFollowSymlinkForDirent(path, entry, followSymlinks);
          entries.put(entry.getName(), entry);
        }
        found = true;
      } catch (FileNotFoundException ignored) {
        // Will be rethrown below if directory exists on neither side.
      }
    }

    try {
      for (var entry : localFs.getPath(path).readdir(Symlinks.NOFOLLOW)) {
        entry = maybeFollowSymlinkForDirent(path, entry, followSymlinks);
        entries.put(entry.getName(), entry);
      }
    } catch (FileNotFoundException e) {
      if (!found) {
        throw e;
      }
    }

    // sort entries to get a deterministic order.
    return ImmutableList.sortedCopyOf(entries.values());
  }

  private Dirent maybeFollowSymlinkForDirent(
      PathFragment dirPath, Dirent entry, boolean followSymlinks) {
    if (!followSymlinks || !entry.getType().equals(Dirent.Type.SYMLINK)) {
      return entry;
    }
    PathFragment path = dirPath.getChild(entry.getName());
    FileStatus st = statNullable(path, /* followSymlinks= */ true);
    if (st == null) {
      return new Dirent(entry.getName(), Dirent.Type.UNKNOWN);
    }
    return new Dirent(entry.getName(), direntFromStat(st));
  }

  /*
   * -------------------- TODO(buchgr): Not yet implemented --------------------
   *
   * The below methods have not (yet) been properly implemented due to time constraints mostly and
   * with little risk as they currently don't seem to be used by internal actions in Bazel. However,
   * before making the --experimental_remote_download_outputs flag non-experimental we should make
   * sure to fully implement this file system.
   */

  @Override
  protected void createFSDependentHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    // Only called by the AbstractFileSystem#createHardLink base implementation, overridden below.
    throw new UnsupportedOperationException();
  }

  @Override
  protected void createHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    localFs.getPath(linkPath).createHardLink(getPath(originalPath));
  }

  static class RemoteInMemoryFileSystem extends InMemoryFileSystem {

    public RemoteInMemoryFileSystem(DigestHashFunction hashFunction) {
      super(hashFunction);
    }

    @Override
    protected synchronized OutputStream getOutputStream(
        PathFragment path, boolean append, boolean internal) throws IOException {
      // To get an output stream from remote file, we need to first stage it.
      throw new IllegalStateException("Shouldn't be called directly");
    }

    @Override
    protected FileInfo newFile(Clock clock, PathFragment path) {
      return new RemoteInMemoryFileInfo(clock);
    }

    protected void injectFile(PathFragment path, FileArtifactValue metadata) throws IOException {
      createDirectoryAndParents(path.getParentDirectory());
      InMemoryContentInfo node = getOrCreateWritableInode(path);
      // If a node was already existed and is not a remote file node (i.e. directory or symlink node
      // ), throw an error.
      if (!(node instanceof RemoteInMemoryFileInfo)) {
        throw new IOException("Could not inject into " + node);
      }

      RemoteInMemoryFileInfo remoteInMemoryFileInfo = (RemoteInMemoryFileInfo) node;
      remoteInMemoryFileInfo.set(metadata);
    }

    // Override for access within this class
    @Nullable
    @Override
    protected FileStatus statNullable(PathFragment path, boolean followSymlinks) {
      return super.statNullable(path, followSymlinks);
    }
  }

  static class RemoteInMemoryFileInfo extends FileInfo implements FileStatusWithMetadata {
    private FileArtifactValue metadata;

    RemoteInMemoryFileInfo(Clock clock) {
      super(clock);
    }

    private void set(FileArtifactValue metadata) {
      this.metadata = metadata;
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
      return metadata.getDigest();
    }

    @Override
    public byte[] getDigest() throws IOException {
      return metadata.getDigest();
    }

    @Override
    public long getSize() {
      return metadata.getSize();
    }

    @Override
    public FileArtifactValue getMetadata() {
      return metadata;
    }
  }
}
