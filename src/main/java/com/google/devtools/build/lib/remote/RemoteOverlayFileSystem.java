package com.google.devtools.build.lib.remote;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.devtools.build.lib.remote.util.Utils.waitForBulkTransfer;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.RemoteActionFileSystem.RemoteInMemoryFileInfo;
import com.google.devtools.build.lib.remote.RemoteActionFileSystem.RemoteInMemoryFileSystem;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext.CachePolicy;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.nio.channels.SeekableByteChannel;
import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

public final class RemoteOverlayFileSystem extends FileSystem {
  private static final CountDownLatch LATCH_AT_0 = new CountDownLatch(0);

  private final PathFragment externalDirectory;
  private final int externalDirectorySegmentCount;
  private final FileSystem nativeFs;
  private final RemoteExternalFileSystem externalFs;
  // The count of the latch represents the current materialization state:
  // * 0 - materialized to nativeFs
  // * 1 - in the process of being materialized
  // * 2 - materialization hasn't been started yet, only available through externalFs
  private final ConcurrentHashMap<String, CountDownLatch> remoteRepoMaterializationState =
      new ConcurrentHashMap<>();

  @Nullable private CombinedCache cache;
  @Nullable private RemoteActionExecutionContext remoteContext;
  @Nullable private Reporter reporter;

  public RemoteOverlayFileSystem(PathFragment externalDirectory, FileSystem nativeFs) {
    super(nativeFs.getDigestFunction());
    this.externalDirectory = externalDirectory;
    this.externalDirectorySegmentCount = externalDirectory.segmentCount();
    this.nativeFs = nativeFs;
    this.externalFs = new RemoteExternalFileSystem(nativeFs.getDigestFunction());
  }

  public void beforeCommand(
      CombinedCache cache, Reporter reporter, String buildRequestId, String commandId) {
    checkState(this.cache == null && this.reporter == null && this.remoteContext == null);
    this.cache = cache;
    this.reporter = reporter;
    var metadata =
        TracingMetadataUtils.buildMetadata(
            buildRequestId, commandId, "remote repository", /* actionExecutionMetadata= */ null);
    this.remoteContext =
        RemoteActionExecutionContext.create(metadata)
            .withReadCachePolicy(CachePolicy.ANY_CACHE)
            .withWriteCachePolicy(CachePolicy.ANY_CACHE);
  }

  public void afterCommand() {
    this.cache = null;
    this.reporter = null;
    this.remoteContext = null;
  }

  public FileSystem underlying() {
    return nativeFs;
  }

  public void injectRemoteRepo(RepositoryName repo, Tree remoteContents) throws IOException {
    var childMap =
        remoteContents.getChildrenList().stream()
            .collect(
                toImmutableMap(cache.digestUtil::compute, directory -> directory, (a, b) -> a));
    injectRecursively(
        externalFs, externalDirectory.getChild(repo.getName()), remoteContents.getRoot(), childMap);
    // Create the repo directory on disk so that readdir reflects the overlaid state of the external
    // directory.
    nativeFs.createDirectoryAndParents(externalDirectory.getChild(repo.getName()));
    remoteRepoMaterializationState.put(repo.getName(), new CountDownLatch(2));
  }

  private static void injectRecursively(
      RemoteExternalFileSystem fs,
      PathFragment path,
      Directory dir,
      ImmutableMap<Digest, Directory> childMap)
      throws IOException {
    fs.createDirectoryAndParents(path);
    for (var file : dir.getFilesList()) {
      fs.injectFile(
          path.getRelative(unicodeToInternal(file.getName())),
          FileArtifactValue.createForRemoteFile(
              DigestUtil.toBinaryDigest(file.getDigest()),
              file.getDigest().getSizeBytes(),
              /* locationIndex= */ 1));
    }
    for (var symlink : dir.getSymlinksList()) {
      fs.createSymbolicLinkTrusted(
          path.getRelative(unicodeToInternal(symlink.getName())),
          PathFragment.create(unicodeToInternal(symlink.getTarget())));
    }
    for (var subdirNode : dir.getDirectoriesList()) {
      var subdirPath = path.getRelative(unicodeToInternal(subdirNode.getName()));
      var subdir = childMap.get(subdirNode.getDigest());
      if (subdir == null) {
        throw new IOException(
            "Directory %s with digest %s not found in tree"
                .formatted(subdirPath, subdirNode.getDigest().getHash()));
      }
      injectRecursively(fs, subdirPath, subdir, childMap);
    }
  }

  public void ensureMaterialized(RepositoryName repo, Runnable report)
      throws IOException, InterruptedException {
    String repoName = repo.getName();
    // Fast path that avoids write locking and allocations.
    var state = remoteRepoMaterializationState.get(repoName);
    if (state == null || state.getCount() == 0) {
      // Never injected or already materialized.
      return;
    }
    var newLatch = new CountDownLatch(1);
    var materializationLatch =
        checkNotNull(
            remoteRepoMaterializationState.computeIfPresent(
                repoName,
                (key, currentValue) -> {
                  if (currentValue.getCount() == 2) {
                    // Claim this materialization.
                    return newLatch;
                  }
                  return currentValue;
                }));
    if (materializationLatch == newLatch) {
      report.run();
      var nativeRepoPath = nativeFs.getPath(externalDirectory.getChild(repoName));
      var remoteRepoPath = externalFs.getPath(externalDirectory.getChild(repoName));
      FileSystemUtils.copyTreesBelow(remoteRepoPath, nativeRepoPath);
      materializationLatch.countDown();
    } else {
      materializationLatch.await();
    }
  }

  private final class RemoteExternalFileSystem extends RemoteInMemoryFileSystem {

    RemoteExternalFileSystem(DigestHashFunction hashFunction) {
      super(hashFunction);
    }

    @Override
    public InputStream getInputStream(PathFragment path) throws IOException {
      var info = (RemoteInMemoryFileInfo) stat(path, /* followSymlinks= */ true);
      reporter.post(
          new ExtendedEventHandler.FetchProgress() {
            @Override
            public String getResourceIdentifier() {
              return path.relativeTo(externalDirectory).getPathString();
            }

            @Override
            public String getProgress() {
              return "(%s)".formatted(Utils.bytesCountToDisplayString(info.getSize()));
            }

            @Override
            public boolean isFinished() {
              return false;
            }
          });
      var contentFuture =
          cache.downloadBlob(
              remoteContext,
              path.getPathString(),
              /* execPath= */ null,
              DigestUtil.buildDigest(info.getMetadata().getDigest(), info.getSize()));
      try {
        waitForBulkTransfer(
            ImmutableList.of(contentFuture), /* cancelRemainingOnInterrupt= */ true);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new InterruptedIOException("interrupted while waiting for remote file transfer");
      }
      try {
        return new ByteArrayInputStream(contentFuture.get());
      } catch (InterruptedException | ExecutionException e) {
        throw new IllegalStateException("waitForBulkTransfer should have thrown", e);
      } finally {
        reporter.post(
            new ExtendedEventHandler.FetchProgress() {
              @Override
              public String getResourceIdentifier() {
                return path.relativeTo(externalDirectory).getPathString();
              }

              @Override
              public String getProgress() {
                return "";
              }

              @Override
              public boolean isFinished() {
                return true;
              }
            });
      }
    }

    /**
     * Used internally to circumvent the read-only nature of this file system for the case of
     * injecting remote repos.
     */
    private void createSymbolicLinkTrusted(PathFragment linkPath, PathFragment targetFragment)
        throws IOException {
      super.createSymbolicLink(linkPath, targetFragment);
    }

    @Override
    public byte[] getDigest(PathFragment path) throws IOException {
      var info = (RemoteInMemoryFileInfo) stat(path, /* followSymlinks= */ true);
      return info.getMetadata().getDigest();
    }

    @Override
    public byte[] getFastDigest(PathFragment path) throws IOException {
      return getDigest(path);
    }

    @Override
    public boolean isReadable(PathFragment path) {
      return true;
    }

    @Override
    public boolean isExecutable(PathFragment path) {
      return true;
    }

    @Override
    public boolean isWritable(PathFragment path) {
      return true;
    }
  }

  private FileSystem fsForPath(PathFragment path) {
    if (path.startsWith(externalDirectory)
        && !path.equals(externalDirectory)
        && remoteRepoMaterializationState
                .getOrDefault(path.getSegment(externalDirectorySegmentCount), LATCH_AT_0)
                .getCount()
            > 0) {
      return externalFs;
    } else {
      return nativeFs;
    }
  }

  // Always mirror deletions to the underlying native file system.

  @Override
  public boolean delete(PathFragment path) throws IOException {
    boolean deleted = nativeFs.delete(path);
    if (fsForPath(path) == externalFs) {
      deleted |= externalFs.delete(path);
    }
    return deleted;
  }

  @Override
  public void deleteTree(PathFragment path) throws IOException {
    nativeFs.deleteTree(path);
    if (fsForPath(path) == externalFs) {
      externalFs.deleteTree(path);
    }
  }

  @Override
  public void deleteTreesBelow(PathFragment dir) throws IOException {
    nativeFs.deleteTreesBelow(dir);
    if (fsForPath(dir) == externalFs) {
      externalFs.deleteTreesBelow(dir);
    }
  }

  // All methods below just delegate to fsForPath(path).

  @Override
  public byte[] getDigest(PathFragment path) throws IOException {
    return fsForPath(path).getDigest(path);
  }

  @Nullable
  @Override
  public byte[] getFastDigest(PathFragment path) throws IOException {
    return fsForPath(path).getFastDigest(path);
  }

  @Override
  public boolean supportsModifications(PathFragment path) {
    return fsForPath(path).supportsModifications(path);
  }

  @Override
  public boolean supportsSymbolicLinksNatively(PathFragment path) {
    return fsForPath(path).supportsSymbolicLinksNatively(path);
  }

  @Override
  public boolean supportsHardLinksNatively(PathFragment path) {
    return fsForPath(path).supportsHardLinksNatively(path);
  }

  @Override
  public boolean isFilePathCaseSensitive() {
    return nativeFs.isFilePathCaseSensitive();
  }

  @Override
  public boolean createDirectory(PathFragment path) throws IOException {
    return fsForPath(path).createDirectory(path);
  }

  @Override
  public void createDirectoryAndParents(PathFragment path) throws IOException {
    fsForPath(path).createDirectoryAndParents(path);
  }

  @Override
  public long getFileSize(PathFragment path, boolean followSymlinks) throws IOException {
    return fsForPath(path).getFileSize(path, followSymlinks);
  }

  @Override
  public long getLastModifiedTime(PathFragment path, boolean followSymlinks) throws IOException {
    return fsForPath(path).getLastModifiedTime(path, followSymlinks);
  }

  @Override
  public void setLastModifiedTime(PathFragment path, long newTime) throws IOException {
    fsForPath(path).setLastModifiedTime(path, newTime);
  }

  @Override
  public FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
    return fsForPath(path).stat(path, followSymlinks);
  }

  @Override
  public void createSymbolicLink(PathFragment linkPath, PathFragment targetFragment)
      throws IOException {
    fsForPath(linkPath).createSymbolicLink(linkPath, targetFragment);
  }

  @Override
  public PathFragment readSymbolicLink(PathFragment path) throws IOException {
    return fsForPath(path).readSymbolicLink(path);
  }

  @Override
  public boolean exists(PathFragment path, boolean followSymlinks) {
    return fsForPath(path).exists(path, followSymlinks);
  }

  @Override
  public Collection<String> getDirectoryEntries(PathFragment path) throws IOException {
    return fsForPath(path).getDirectoryEntries(path);
  }

  @Override
  public boolean isReadable(PathFragment path) throws IOException {
    return fsForPath(path).isReadable(path);
  }

  @Override
  public void setReadable(PathFragment path, boolean readable) throws IOException {
    fsForPath(path).setReadable(path, readable);
  }

  @Override
  public boolean isWritable(PathFragment path) throws IOException {
    return fsForPath(path).isWritable(path);
  }

  @Override
  public void setWritable(PathFragment path, boolean writable) throws IOException {
    fsForPath(path).setWritable(path, writable);
  }

  @Override
  public boolean isExecutable(PathFragment path) throws IOException {
    return fsForPath(path).isExecutable(path);
  }

  @Override
  public void setExecutable(PathFragment path, boolean executable) throws IOException {
    fsForPath(path).setExecutable(path, executable);
  }

  @Override
  public InputStream getInputStream(PathFragment path) throws IOException {
    return fsForPath(path).getInputStream(path);
  }

  @Override
  public SeekableByteChannel createReadWriteByteChannel(PathFragment path) throws IOException {
    return fsForPath(path).createReadWriteByteChannel(path);
  }

  @Override
  public OutputStream getOutputStream(PathFragment path, boolean append, boolean internal)
      throws IOException {
    return fsForPath(path).getOutputStream(path, append, internal);
  }

  @Override
  public void renameTo(PathFragment sourcePath, PathFragment targetPath) throws IOException {
    fsForPath(sourcePath).renameTo(sourcePath, targetPath);
  }

  @Override
  public void createFSDependentHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    fsForPath(originalPath).createFSDependentHardLink(linkPath, originalPath);
  }

  @Override
  public File getIoFile(PathFragment path) {
    return fsForPath(path).getIoFile(path);
  }

  @Override
  public java.nio.file.Path getNioPath(PathFragment path) {
    return fsForPath(path).getNioPath(path);
  }

  @Override
  public String getFileSystemType(PathFragment path) {
    return fsForPath(path).getFileSystemType(path);
  }

  @Override
  public boolean createWritableDirectory(PathFragment path) throws IOException {
    return fsForPath(path).createWritableDirectory(path);
  }

  @Override
  public byte[] getxattr(PathFragment path, String name, boolean followSymlinks)
      throws IOException {
    return fsForPath(path).getxattr(path, name, followSymlinks);
  }

  @Nullable
  @Override
  public PathFragment resolveOneLink(PathFragment path) throws IOException {
    return fsForPath(path).resolveOneLink(path);
  }

  @Override
  public Path resolveSymbolicLinks(PathFragment path) throws IOException {
    return fsForPath(path).resolveSymbolicLinks(path);
  }

  @Nullable
  @Override
  public FileStatus statNullable(PathFragment path, boolean followSymlinks) {
    return fsForPath(path).statNullable(path, followSymlinks);
  }

  @Nullable
  @Override
  public FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
    return fsForPath(path).statIfFound(path, followSymlinks);
  }

  @Override
  public boolean isFile(PathFragment path, boolean followSymlinks) {
    return fsForPath(path).isFile(path, followSymlinks);
  }

  @Override
  public boolean isSpecialFile(PathFragment path, boolean followSymlinks) {
    return fsForPath(path).isSpecialFile(path, followSymlinks);
  }

  @Override
  public boolean isSymbolicLink(PathFragment path) {
    return fsForPath(path).isSymbolicLink(path);
  }

  @Override
  public boolean isDirectory(PathFragment path, boolean followSymlinks) {
    return fsForPath(path).isDirectory(path, followSymlinks);
  }

  @Override
  public PathFragment readSymbolicLinkUnchecked(PathFragment path) throws IOException {
    return fsForPath(path).readSymbolicLinkUnchecked(path);
  }

  @Override
  public boolean exists(PathFragment path) {
    return fsForPath(path).exists(path);
  }

  @Override
  public Collection<Dirent> readdir(PathFragment path, boolean followSymlinks) throws IOException {
    return fsForPath(path).readdir(path, followSymlinks);
  }

  @Override
  public void chmod(PathFragment path, int mode) throws IOException {
    fsForPath(path).chmod(path, mode);
  }

  @Override
  public void createHardLink(PathFragment linkPath, PathFragment originalPath) throws IOException {
    fsForPath(linkPath).createHardLink(linkPath, originalPath);
  }

  @Override
  public void prefetchPackageAsync(PathFragment path, int maxDirs) {
    fsForPath(path).prefetchPackageAsync(path, maxDirs);
  }

  @Override
  public PathFragment createTempDirectory(PathFragment parent, String prefix) throws IOException {
    return fsForPath(parent).createTempDirectory(parent, prefix);
  }
}
