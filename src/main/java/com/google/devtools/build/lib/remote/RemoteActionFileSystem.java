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
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Priority;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Reason;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStatusWithMetadata;
import com.google.devtools.build.lib.actions.ImportantOutputHandler.LostArtifacts;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.LostInputsActionExecutionException;
import com.google.devtools.build.lib.actions.LostInputsExecException;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.OverlayFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SymlinkTargetType;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.SeekableByteChannel;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * An action filesystem suitable for use when building with disk/remote caching or execution.
 *
 * <p>It acts as a union filesystem over three different sources:
 *
 * <ul>
 *   <li>The action input map, providing read-only in-memory access to the metadata (but not the
 *       contents) of the action's declared inputs.
 *   <li>The remote output tree, an in-memory filesystem providing read/write access to the metadata
 *       (but not the contents) of remotely stored files injected during action execution.
 *   <li>The local filesystem, providing read/write access to the metadata and contents of files
 *       residing on disk, including the inputs and outputs of local spawns.
 * </ul>
 *
 * <p>Generally speaking, file operations consult the underlying sources in that order and operate
 * on the first result found, although some (e.g. readdir) collate information from all sources.
 * The contents of remotely stored files are transparently downloaded when an operation requires
 * them.
 *
 * <p>Special care must be taken with operations that follow symlinks, as the symlink and its
 * target path may reside on different sources, with an arbitrary number of indirections in
 * between. This is required because some actions (notably SymlinkAction) may materialize an output
 * as a symlink to an input. This is handled by inheriting from {@link OverlayFileSystem}, whose
 * canonicalize-first machinery takes every source into account and only then invokes the
 * source-dispatching hooks implemented here.
 *
 * <p>The implementation assumes that an action never modifies its input paths, but may otherwise
 * modify any path in the output tree. Concurrent operations are supported as long as they don't
 * affect filesystem structure (i.e., create, move or delete paths). Otherwise, they might fail or
 * produce inconsistent results. No effort is made to detect irreconcilable differences between
 * sources, such as the same path existing in multiple underlying sources with different type or
 * contents.
 */
public class RemoteActionFileSystem extends OverlayFileSystem {
  private final PathFragment execRoot;
  private final PathFragment outputBase;
  private final InputMetadataProvider inputArtifactData;
  private final TreeArtifactDirectoryCache inputTreeArtifactDirectoryCache;
  private final RemoteActionInputFetcher inputFetcher;
  private final FileSystem localFs;
  private final RemoteInMemoryFileSystem remoteOutputTree;
  // Concurrent access is rare and most builds don't have lost inputs.
  private final List<LostArtifacts> lostInputs = Collections.synchronizedList(new ArrayList<>(0));

  @Nullable private ActionExecutionMetadata action = null;

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
   * FileSystem#stat} and {@link FileSystem#readdir} operations. Note that actions are not expected
   * to modify their inputs.
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
      TreeArtifactValue treeMetadata = inputArtifactData.getEnclosingTreeMetadata(execPath);
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
      InputMetadataProvider inputArtifactData,
      RemoteActionInputFetcher inputFetcher) {
    super(localFs.getDigestFunction());
    this.execRoot = checkNotNull(execRootFragment, "execRootFragment");
    this.outputBase = execRoot.getRelative(checkNotNull(relativeOutputPath, "relativeOutputPath"));
    this.inputArtifactData = checkNotNull(inputArtifactData, "inputArtifactData");
    this.inputTreeArtifactDirectoryCache = new TreeArtifactDirectoryCache();
    this.inputFetcher = checkNotNull(inputFetcher, "inputFetcher");
    this.localFs = checkNotNull(localFs, "localFs");
    this.remoteOutputTree = new RemoteInMemoryFileSystem(getDigestFunction());
  }

  @Override
  public FileSystem getHostFileSystem() {
    return localFs.getHostFileSystem();
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
  public boolean mayBeCaseOrNormalizationInsensitive() {
    return localFs.mayBeCaseOrNormalizationInsensitive()
        || remoteOutputTree.mayBeCaseOrNormalizationInsensitive();
  }

  @VisibleForTesting
  protected RemoteInMemoryFileSystem getRemoteOutputTree() {
    return remoteOutputTree;
  }

  @VisibleForTesting
  protected FileSystem getLocalFileSystem() {
    return localFs;
  }

  /** Returns whether a path is stored remotely. Follows symlinks. */
  boolean isRemote(Path path) throws IOException {
    return isRemote(path.asFragment());
  }

  private boolean isRemote(PathFragment path) throws IOException {
    // Files in the local filesystem are non-remote by definition, so stat only in-memory sources.
    FileStatus status;
    try {
      status = statInMemoryNofollow(canonicalize(path));
    } catch (FileNotFoundException e) {
      return false;
    }
    return status instanceof FileStatusWithMetadata fileStatusWithMetadata
        && fileStatusWithMetadata.getMetadata().isRemote();
  }

  public void updateContext(ActionExecutionMetadata action) {
    this.action = action;
  }

  void injectRemoteFile(
      PathFragment path, byte[] digest, long size, Instant expirationTime, boolean inMemoryOutput)
      throws IOException {
    if (!isOutput(path)) {
      return;
    }
    var metadata =
        FileArtifactValue.createForRemoteFileWithMaterializationData(
            digest, size, /* locationIndex= */ 1, expirationTime, inMemoryOutput);
    remoteOutputTree.injectFile(path, metadata);
  }

  @Override
  public String getFileSystemType(PathFragment path) {
    return "remoteActionFS";
  }

  @Override
  protected boolean deleteNofollow(PathFragment path) throws IOException {
    boolean deleted = localFs.getPath(path).delete();
    if (isOutput(path)) {
      deleted = remoteOutputTree.getPath(path).delete() || deleted;
    }

    return deleted;
  }

  @Override
  public InputStream getInputStream(PathFragment path) throws IOException {
    try {
      getFromFuture(downloadIfRemote(path));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException(String.format("Received interrupt while fetching file '%s'", path), e);
    } catch (BulkTransferException e) {
      var newlyLostInputs = e.getLostArtifacts(inputArtifactData::getInput);
      if (!newlyLostInputs.isEmpty()) {
        lostInputs.add(newlyLostInputs);
      }
      throw e;
    }
    return localFs.getPath(path).getInputStream();
  }

  /** Downloads the file at {@code path} if it is remote. */
  public ListenableFuture<Void> downloadIfRemote(PathFragment path) {
    try {
      if (!isRemote(path)) {
        return immediateVoidFuture();
      }
    } catch (IOException e) {
      return immediateFailedFuture(e);
    }
    PathFragment execPath = path.relativeTo(execRoot);
    ActionInput input = inputArtifactData.getInput(execPath);
    if (input == null) {
      // TODO(tjgq): Also look up the remote output tree.
      return immediateVoidFuture();
    }
    return inputFetcher.prefetchFiles(
        action,
        /* spawn= */ null,
        () -> ImmutableList.of(input),
        inputArtifactData,
        Priority.CRITICAL,
        Reason.INPUTS);
  }

  @Override
  public OutputStream getOutputStream(PathFragment path, boolean append, boolean internal)
      throws IOException {
    return localFs.getPath(path).getOutputStream(append, internal);
  }

  @Override
  public SeekableByteChannel createReadWriteByteChannel(PathFragment path) throws IOException {
    return localFs.getPath(path).createReadWriteByteChannel();
  }

  @Override
  protected void setLastModifiedTimeNofollow(PathFragment path, long newTime) throws IOException {
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
  @Nullable
  protected byte[] getFastDigestNofollow(PathFragment path) throws IOException {
    // Try to obtain a fast digest through a stat. This is only possible for in-memory files.
    var status = statInMemoryNofollow(path);
    if (status instanceof FileStatusWithDigest fileStatusWithDigest) {
      return fileStatusWithDigest.getDigest();
    }
    return localFs.getPath(path).getFastDigest();
  }

  @Override
  protected byte[] getDigestNofollow(PathFragment path) throws IOException {
    // Try to obtain a fast digest through a stat. This is only possible for in-memory files.
    var status = statInMemoryNofollow(path);
    if (status instanceof FileStatusWithDigest fileStatusWithDigest) {
      return fileStatusWithDigest.getDigest();
    }
    return localFs.getPath(path).getDigest();
  }

  @Override
  protected boolean isReadableNofollow(PathFragment path) throws IOException {
    try {
      return localFs.getPath(path).isReadable();
    } catch (FileNotFoundException e) {
      // Remote files are always readable since we can't control their permissions.
      return true;
    }
  }

  @Override
  protected boolean isWritableNofollow(PathFragment path) throws IOException {
    try {
      return localFs.getPath(path).isWritable();
    } catch (FileNotFoundException e) {
      // Remote files are always writable since we can't control their permissions.
      return true;
    }
  }

  @Override
  protected boolean isExecutableNofollow(PathFragment path) throws IOException {
    try {
      return localFs.getPath(path).isExecutable();
    } catch (FileNotFoundException e) {
      // Remote files are always executable since we can't control their permissions.
      return true;
    }
  }

  @Override
  protected void setReadableNofollow(PathFragment path, boolean readable) throws IOException {
    try {
      localFs.getPath(path).setReadable(readable);
    } catch (FileNotFoundException e) {
      // Intentionally ignored.
    }
  }

  @Override
  protected void setWritableNofollow(PathFragment path, boolean writable) throws IOException {
    try {
      localFs.getPath(path).setWritable(writable);
    } catch (FileNotFoundException e) {
      // Intentionally ignored.
    }
  }

  @Override
  protected void setExecutableNofollow(PathFragment path, boolean executable) throws IOException {
    try {
      localFs.getPath(path).setExecutable(executable);
    } catch (FileNotFoundException e) {
      // Intentionally ignored.
    }
  }

  @Override
  protected void chmodNofollow(PathFragment path, int mode) throws IOException {
    try {
      localFs.getPath(path).chmod(mode);
    } catch (FileNotFoundException e) {
      // Intentionally ignored.
    }
  }

  @Override
  protected PathFragment readSymlinkNofollow(PathFragment path) throws IOException {
    if (path.startsWith(execRoot)) {
      var execPath = path.relativeTo(execRoot);
      var actionInput = inputArtifactData.getInput(execPath);
      var metadata = actionInput != null ? inputArtifactData.getInputMetadata(actionInput) : null;
      if (metadata != null && metadata.getType().isSymlink()) {
        return PathFragment.create(metadata.getUnresolvedSymlinkTarget());
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
  protected void createSymbolicLinkNofollow(
      PathFragment linkPath, PathFragment targetFragment, SymlinkTargetType type)
      throws IOException {
    if (isOutput(linkPath)) {
      remoteOutputTree.getPath(linkPath).createSymbolicLink(targetFragment, type);
    }

    localFs.getPath(linkPath).createSymbolicLink(targetFragment, type);
  }

  @Override
  @Nullable
  protected FileStatus statNofollow(PathFragment path) throws IOException {
    FileStatus stat = statInMemoryNofollow(path);
    if (stat != null) {
      return stat;
    }

    return localFs.getPath(path).statIfFound(Symlinks.NOFOLLOW);
  }

  /**
   * Like {@link #statNofollow}, but only considers the in-memory sources (action input map and
   * remote output tree).
   */
  @Nullable
  private FileStatus statInMemoryNofollow(PathFragment path) throws IOException {
    if (path.startsWith(execRoot)) {
      var execPath = path.relativeTo(execRoot);
      var actionInput = inputArtifactData.getInput(execPath);
      var metadata = actionInput != null ? inputArtifactData.getInputMetadata(actionInput) : null;
      if (metadata != null) {
        return statFromMetadata(metadata);
      }
      if (inputTreeArtifactDirectoryCache.get(execPath) != null) {
        return DIRECTORY_FILE_STATUS;
      }
    }

    return remoteOutputTree.statIfFound(path, /* followSymlinks= */ false);
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
        try {
          return m.getModifiedTime();
        } catch (UnsupportedOperationException e) {
          // Not every FileArtifactValue supports getModifiedTime.
          return 0;
        }
      }

      @Override
      public long getLastChangeTime() {
        return getLastModifiedTime();
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

  private boolean isOutput(PathFragment path) {
    return path.startsWith(outputBase);
  }

  @Override
  protected void renameToNofollow(PathFragment srcPath, PathFragment dstPath) throws IOException {
    checkArgument(isOutput(srcPath), "srcPath must be an output path");
    checkArgument(isOutput(dstPath), "dstPath must be an output path");

    FileNotFoundException remoteException = null;
    try {
      remoteOutputTree.renameTo(srcPath, dstPath);
    } catch (FileNotFoundException e) {
      remoteException = e;
    }

    FileNotFoundException localException = null;
    try {
      localFs.renameTo(srcPath, dstPath);
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
  protected Collection<Dirent> readdirNofollow(PathFragment path) throws IOException {
    HashMap<String, Dirent> entries = new HashMap<>();
    boolean exists = false;

    if (path.startsWith(execRoot)) {
      var execPath = path.relativeTo(execRoot);
      Collection<Dirent> treeEntries = inputTreeArtifactDirectoryCache.get(execPath);
      if (treeEntries != null) {
        for (var entry : treeEntries) {
          entries.put(entry.getName(), entry);
        }
        exists = true;
      }
    }

    // Since actions are assumed not to modify their inputs, a directory belonging to an input tree
    // artifact cannot also contain an output, so we can safely skip the other sources.
    if (!exists) {
      if (isOutput(path)) {
        try {
          for (var entry : remoteOutputTree.getPath(path).readdir(Symlinks.NOFOLLOW)) {
            entries.put(entry.getName(), entry);
          }
          exists = true;
        } catch (FileNotFoundException ignored) {
          // Will be rethrown below if directory does not exist in any of the sources.
        }
      }

      try {
        for (var entry : localFs.getPath(path).readdir(Symlinks.NOFOLLOW)) {
          entries.put(entry.getName(), entry);
        }
        exists = true;
      } catch (FileNotFoundException ignored) {
        // Will be rethrown below if directory does not exist in any of the sources.
      }
    }

    if (!exists) {
      throw new FileNotFoundException(path.getPathString() + " (No such file or directory)");
    }

    return entries.values();
  }

  @Override
  public void createFSDependentHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    // Only called by the FileSystem#createHardLink base implementation, overridden below.
    throw new UnsupportedOperationException();
  }

  @Override
  public void createHardLink(PathFragment linkPath, PathFragment originalPath) throws IOException {
    localFs.getPath(linkPath).createHardLink(getPath(originalPath));
  }

  public void checkForLostInputs(Action action) throws LostInputsActionExecutionException {
    var mergedException =
        lostInputs.stream()
            .map(lostArtifacts -> new LostInputsExecException(lostArtifacts.byDigest()))
            .reduce(LostInputsExecException::combine);
    if (mergedException.isPresent()) {
      throw (LostInputsActionExecutionException)
          ActionExecutionException.fromExecException(mergedException.get(), action);
    }
  }
}
