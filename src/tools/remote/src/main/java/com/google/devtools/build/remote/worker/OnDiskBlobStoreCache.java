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
package com.google.devtools.build.remote.worker;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;

import build.bazel.remote.execution.v2.ActionCacheUpdateCapabilities;
import build.bazel.remote.execution.v2.CacheCapabilities;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.SymlinkAbsolutePathStrategy;
import build.bazel.remote.execution.v2.SymlinkNode;
import com.google.common.flogger.GoogleLogger;
import com.google.common.hash.Hashing;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.CombinedCache;
import com.google.devtools.build.lib.remote.Store;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.Blob;
import com.google.devtools.build.lib.remote.disk.DiskCacheClient;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;

/** A {@link CombinedCache} backed by an {@link DiskCacheClient}. */
class OnDiskBlobStoreCache extends CombinedCache {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private record DigestAndInvocation(Digest digest, String invocationId) {}

  private final RemoteWorkerOptions remoteWorkerOptions;
  private final ConcurrentHashMap<DigestAndInvocation, Integer>
      numberOfDownloadsPerDigestAndInvocation = new ConcurrentHashMap<>();
  private final ConcurrentHashMap<Digest, Integer> numberOfUploadsPerLossCandidate =
      new ConcurrentHashMap<>();

  public OnDiskBlobStoreCache(
      Path cacheDir, DigestUtil digestUtil, RemoteWorkerOptions remoteWorkerOptions)
      throws IOException {
    super(
        /* remoteCacheClient= */ null,
        new DiskCacheClient(cacheDir, digestUtil),
        /* symlinkTemplate= */ null,
        digestUtil,
        /* chunkingEnabled= */ false);
    this.remoteWorkerOptions = remoteWorkerOptions;
  }

  @Override
  public CacheCapabilities getRemoteCacheCapabilities() {
    return CacheCapabilities.newBuilder()
        .setActionCacheUpdateCapabilities(
            ActionCacheUpdateCapabilities.newBuilder().setUpdateEnabled(true).build())
        .setSymlinkAbsolutePathStrategy(SymlinkAbsolutePathStrategy.Value.ALLOWED)
        .build();
  }

  /** If the given blob exists, updates its mtime and returns true. Otherwise, returns false. */
  boolean refresh(Digest digest) throws IOException {
    return diskCacheClient.refresh(diskCacheClient.toPath(digest, Store.CAS));
  }

  @SuppressWarnings("ProtoParseWithRegistry")
  public void downloadTree(
      RemoteActionExecutionContext context, Digest rootDigest, Path rootLocation)
      throws IOException, InterruptedException {
    rootLocation.createDirectoryAndParents();
    Directory directory = Directory.parseFrom(getFromFuture(downloadBlob(context, rootDigest)));
    HashSet<Path> childrenSeen = new HashSet<>();
    for (FileNode file : directory.getFilesList()) {
      Path dst = rootLocation.getRelative(unicodeToInternal(file.getName()));
      if (!childrenSeen.add(dst)) {
        throw new IOException("Duplicate child '%s' in directory %s".formatted(dst, directory));
      }
      getFromFuture(downloadFile(context, dst, file.getDigest()));
      dst.setExecutable(file.getIsExecutable());
    }
    for (SymlinkNode symlink : directory.getSymlinksList()) {
      Path dst = rootLocation.getRelative(unicodeToInternal(symlink.getName()));
      if (!childrenSeen.add(dst)) {
        throw new IOException("Duplicate child '%s' in directory %s".formatted(dst, directory));
      }
      // TODO(fmeum): The following line is not generally correct: The remote execution API allows
      //  for non-normalized symlink targets, but the normalization applied by PathFragment.create
      //  does not take directory symlinks into account. However, Bazel's file system API does not
      //  currently offer a way to specify a raw String as a symlink target.
      // https://github.com/bazelbuild/bazel/issues/14224
      dst.createSymbolicLink(PathFragment.create(unicodeToInternal(symlink.getTarget())));
    }
    for (DirectoryNode child : directory.getDirectoriesList()) {
      Path dst = rootLocation.getRelative(unicodeToInternal(child.getName()));
      if (!childrenSeen.add(dst)) {
        throw new IOException("Duplicate child '%s' in directory %s".formatted(dst, directory));
      }
      downloadTree(context, child.getDigest(), dst);
    }
  }

  @Override
  public ListenableFuture<byte[]> downloadBlob(
      RemoteActionExecutionContext context, Digest digest) {
    if (remoteWorkerOptions.getErrorOnDuplicateDownloads()) {
      // Only populate numberOfDownloadsPerDigestAndInvocation when fakeErrorForDuplicatedDownloads
      // is enabled to avoid unnecessary unbounded memory growth.
      int numberOfDownloads =
          numberOfDownloadsPerDigestAndInvocation.merge(
              new DigestAndInvocation(digest, context.getRequestMetadata().getToolInvocationId()),
              1,
              Integer::sum);
      if (numberOfDownloads > 1) {
        return Futures.immediateFailedFuture(
            new IOException(
                String.format(
                    "Duplicate download of blob digest %s for invocation id %s",
                    DigestUtil.toString(digest),
                    context.getRequestMetadata().getToolInvocationId())));
      }
    }

    return super.downloadBlob(context, digest);
  }

  @Override
  public ListenableFuture<Void> uploadFile(
      RemoteActionExecutionContext context, Digest digest, Path file) {
    if (shouldLoseUpload(digest)) {
      return Futures.immediateVoidFuture();
    }
    return super.uploadFile(context, digest, file);
  }

  @Override
  public ListenableFuture<Void> uploadBlob(
      RemoteActionExecutionContext context, Digest digest, Blob blob) {
    if (shouldLoseUpload(digest)) {
      return Futures.immediateVoidFuture();
    }
    return super.uploadBlob(context, digest, blob);
  }

  /**
   * Simulates a remote cache that loses CAS entries by silently dropping the upload of the blob
   * with the given digest for each of its first {@code --lost_blob_max_losses} uploads (see {@code
   * --lost_blob_percentage}): the upload is reported as successful to the client, but the blob is
   * not actually stored, so it only becomes available once it has been uploaded one more time.
   *
   * <p>Since the blob is never stored, all kinds of requests referencing it consistently observe it
   * as absent: findMissingBlobs reports it as missing, reads fail with NOT_FOUND, executions report
   * a FAILED_PRECONDITION with a MISSING violation when staging it as an input, and action cache
   * hits referencing it are treated as stale. Since only a bounded number of uploads of a blob is
   * affected, clients can always recover by re-uploading or regenerating it sufficiently often.
   */
  private boolean shouldLoseUpload(Digest digest) {
    if (!isLossCandidate(digest)) {
      return false;
    }
    int uploads = numberOfUploadsPerLossCandidate.merge(digest, 1, Integer::sum);
    int maxLosses = remoteWorkerOptions.getLostBlobMaxLosses();
    if (uploads > maxLosses) {
      return false;
    }
    logger.atInfo().log(
        "Simulated loss of CAS entry %s (loss %d of %d)",
        DigestUtil.toString(digest), uploads, maxLosses);
    return true;
  }

  private boolean isLossCandidate(Digest digest) {
    int lostBlobPercentage = remoteWorkerOptions.getLostBlobPercentage();
    // The empty blob is special-cased by clients and servers (including DiskCacheClient) as always
    // present, so it can't be lost.
    if (lostBlobPercentage == 0 || digest.getSizeBytes() == 0) {
      return false;
    }
    return isInBucket(/* category= */ "lose", digest.getHash(), lostBlobPercentage);
  }

  /**
   * Simulates a remote cache that has evicted CAS entries, e.g. due to an outage or a trimming
   * policy, by deleting a seeded sample of the entries that currently exist on disk (see {@code
   * --evict_existing_percentage}).
   *
   * <p>Unlike {@link #shouldLoseUpload}, which loses blobs as they are produced during a build,
   * this models the loss of cache contents that were populated by a previous build. It is meant to
   * be run on startup, before any requests are served, so that a subsequent build observes a warm
   * but partially evicted cache. Whether a blob is evicted is an independent sample from the set of
   * blobs lost via {@code --lost_blob_percentage}.
   *
   * @return the number of evicted entries.
   */
  int evictExistingEntries(int percentage) throws IOException {
    if (percentage <= 0) {
      return 0;
    }
    // toPath places an entry at <cas root>/<first 2 hash chars>/<full hash>, so going two levels up
    // from an arbitrary entry path yields the CAS store root.
    Path casRoot =
        diskCacheClient
            .toPath("0".repeat(64), Store.CAS)
            .getParentDirectory()
            .getParentDirectory();
    if (!casRoot.exists()) {
      return 0;
    }
    int evicted = 0;
    for (Path shard : casRoot.getDirectoryEntries()) {
      if (!shard.isDirectory()) {
        continue;
      }
      for (Path entry : shard.getDirectoryEntries()) {
        if (isInBucket(/* category= */ "evict", entry.getBaseName(), percentage)) {
          entry.delete();
          evicted++;
        }
      }
    }
    return evicted;
  }

  /**
   * Deterministically assigns the blob with the given hash to a bucket based on the seed and the
   * given category, returning whether it falls into the first {@code percentage} percent. The
   * category keeps independent uses (losing vs. evicting) from selecting correlated samples.
   */
  private boolean isInBucket(String category, String hash, int percentage) {
    long bucket =
        Hashing.murmur3_128()
            .newHasher()
            .putUnencodedChars(category)
            .putUnencodedChars(remoteWorkerOptions.getLostBlobSeed())
            .putUnencodedChars(hash)
            .hash()
            .asLong();
    return Math.floorMod(bucket, 100) < percentage;
  }

  public DigestUtil getDigestUtil() {
    return digestUtil;
  }

  public DiskCacheClient getDiskCacheClient() {
    return checkNotNull(diskCacheClient);
  }
}
