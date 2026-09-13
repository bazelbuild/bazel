// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.disk;

import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.DigestUtil.isOldStyleDigestFunction;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.remote.Store;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.MaybePathBacked;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.Blob;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistryLite;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.UUID;
import java.util.concurrent.Executors;

/**
 * An on-disk store for the remote action cache.
 *
 * <p>Concurrent Bazel processes can safely retrieve and store entries in a shared disk cache, even
 * when they collide.
 *
 * <p>The mtime of an entry reflects the most recent time the entry was stored *or* retrieved. This
 * property may be used to garbage collect the disk cache by deleting the least recently accessed
 * entries. This may be done by Bazel itself (see {@link DiskCacheGarbageCollectorIdleTask}), by
 * another Bazel process sharing the disk cache, or by an external process. Although we could have
 * arranged for an ongoing garbage collection to block a concurrent build, we judge it to not be
 * worth the extra complexity; assuming that the collection policy is not overly aggressive, the
 * likelihood of a race condition is fairly small, and an affected build is able to automatically
 * recover by retrying.
 */
public class DiskCacheClient {

  private static final String AC_DIR = "ac";
  private static final String CAS_DIR = "cas";
  private static final String TMP_DIR = "tmp";

  private final ImmutableMap<Store, Path> storeRootMap;
  private final Path tmpRoot;
  private final boolean checkActionResultIntegrity;

  // Disk cache operations are almost entirely I/O-bound as digests are only computed as part of
  // I/O operations, so using virtual threads is appropriate.
  @SuppressWarnings("AllowVirtualThreads")
  private final ListeningExecutorService executorService =
      MoreExecutors.listeningDecorator(
          Executors.newThreadPerTaskExecutor(Thread.ofVirtual().name("disk-cache-", 0).factory()));

  /**
   * Creates a new disk cache client.
   *
   * @param checkActionResultIntegrity whether {@link #downloadActionResult} should only return an
   *     action result whose referenced blobs are all present in the disk cache
   */
  public DiskCacheClient(Path root, DigestUtil digestUtil, boolean checkActionResultIntegrity)
      throws IOException {
    this.checkActionResultIntegrity = checkActionResultIntegrity;
    Path fnRoot =
        isOldStyleDigestFunction(digestUtil.getDigestFunction())
            ? root
            : root.getChild(
                Ascii.toLowerCase(digestUtil.getDigestFunction().getValueDescriptor().getName()));
    this.storeRootMap =
        ImmutableMap.of(Store.AC, fnRoot.getChild(AC_DIR), Store.CAS, fnRoot.getChild(CAS_DIR));

    this.tmpRoot = root.getChild(TMP_DIR);

    fnRoot.createDirectoryAndParents();
    tmpRoot.createDirectoryAndParents();
  }

  /**
   * If the given path exists, updates its mtime and returns true. Otherwise, returns false.
   *
   * <p>This provides a cheap way to identify candidates for deletion when trimming the cache. We
   * deliberately use the mtime because the atime is more likely to be externally modified and may
   * be unavailable on some filesystems.
   *
   * <p>Prefer calling {@link #downloadBlob} instead, which will automatically update the mtime.
   * This method should only be called by the remote worker implementation.
   *
   * @throws IOException if an I/O error other than a missing file occurs.
   */
  public boolean refresh(Path path) throws IOException {
    try {
      // Use NOW_SENTINEL_TIME instead of obtaining the current time so that the operation succeeds
      // even when the file has a different owner, as might be the case for a shared cache.
      path.setLastModifiedTime(Path.NOW_SENTINEL_TIME);
    } catch (FileNotFoundException e) {
      return false;
    }
    return true;
  }

  /**
   * Moves an existing file into the cache.
   *
   * <p>The caller must ensure that the digest is correct and the file has been recently modified.
   * This method should only be called by the combined cache implementation.
   */
  public void captureFile(Path src, Digest digest, Store store) throws IOException {
    Path target = toPath(digest, store);

    if (refresh(target)) {
      src.delete();
      return;
    }

    target.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.renameToleratingConcurrentCreation(src, target);
  }

  private ListenableFuture<Void> download(Digest digest, OutputStream out, Store store) {
    return executorService.submit(
        () -> {
          Path path = toPath(digest, store);
          if (!refresh(path)) {
            throw new CacheNotFoundException(digest);
          }
          Path outPath = null;
          if (out instanceof MaybePathBacked maybePathBacked) {
            outPath = maybePathBacked.maybeGetPath();
          }

          try {
            if (outPath != null) {
              // If the output stream is path-backed, the filesystem may be able to avoid copying
              // the file.
              FileSystemUtils.copyFile(path, outPath);
            } else {
              try (InputStream in = path.getInputStream()) {
                in.transferTo(out);
              }
            }
          } catch (FileNotFoundException e) {
            // The entry may have been deleted between the refresh above and the copy, for example
            // due to a concurrent garbage collection. Report this case as a cache miss rather than
            // a real I/O error.
            //
            // Note that a FileNotFoundException could also be thrown if a parent directory of the
            // destination doesn't exist, so we try to preserve the error reporting in that case.
            // TODO: When migrating to NIO exceptions, use NoSuchFileException#getFile to avoid this
            // check.
            if (outPath != null
                && outPath.getParentDirectory() != null
                && !outPath.getParentDirectory().exists()) {
              throw e;
            }
            var cacheNotFoundException = new CacheNotFoundException(digest);
            cacheNotFoundException.addSuppressed(e);
            throw cacheNotFoundException;
          }
          return null;
        });
  }

  public ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
    return Futures.transformAsync(
        download(digest, out, Store.CAS),
        (v) -> {
          try {
            out.flush();
            return immediateFuture(null);
          } catch (IOException e) {
            return Futures.immediateFailedFuture(e);
          }
        },
        directExecutor());
  }

  /**
   * If the blob with the given digest exists, marks it as recently used.
   *
   * @return whether the blob exists.
   * @throws IOException if an I/O error other than a missing file occurs.
   */
  private boolean refreshDigest(Digest digest) throws IOException {
    if (digest.getSizeBytes() == 0) {
      return true;
    }

    return refresh(toPath(digest, Store.CAS));
  }

  private boolean refreshOutputDirectory(Directory dir, boolean stopAtFirstMissing)
      throws IOException {
    boolean allPresent = true;
    for (var file : dir.getFilesList()) {
      allPresent &= refreshDigest(file.getDigest());
      if (!allPresent && stopAtFirstMissing) {
        return false;
      }
    }
    return allPresent;
  }

  /**
   * Marks all of the blobs referenced by the {@link ActionResult} that exist as recently used.
   *
   * @param stopAtFirstMissing whether to return as soon as a referenced blob is found to be
   *     missing, leaving the mtime of the remaining blobs untouched.
   * @return whether all of the referenced blobs exist.
   * @throws IOException if an I/O error other than a missing file occurs.
   */
  private boolean refreshActionResult(ActionResult actionResult, boolean stopAtFirstMissing)
      throws IOException {
    boolean allPresent = true;

    for (var outputFile : actionResult.getOutputFilesList()) {
      allPresent &= refreshDigest(outputFile.getDigest());
      if (!allPresent && stopAtFirstMissing) {
        return false;
      }
    }

    for (var outputDirectory : actionResult.getOutputDirectoriesList()) {
      var treeDigest = outputDirectory.getTreeDigest();
      if (!refreshDigest(treeDigest)) {
        // Without the Tree, the blobs it references can't be determined.
        if (stopAtFirstMissing) {
          return false;
        }
        allPresent = false;
        continue;
      }

      Tree tree;
      try (var in = toPath(treeDigest, Store.CAS).getInputStream()) {
        tree = Tree.parseFrom(in, ExtensionRegistryLite.getEmptyRegistry());
      } catch (FileNotFoundException e) {
        // The tree was deleted between the refresh above and the read, most likely by a concurrent
        // garbage collection. Treat it as missing rather than as a real I/O error.
        if (stopAtFirstMissing) {
          return false;
        }
        allPresent = false;
        continue;
      }
      allPresent &= refreshOutputDirectory(tree.getRoot(), stopAtFirstMissing);
      if (!allPresent && stopAtFirstMissing) {
        return false;
      }
      for (var dir : tree.getChildrenList()) {
        allPresent &= refreshOutputDirectory(dir, stopAtFirstMissing);
        if (!allPresent && stopAtFirstMissing) {
          return false;
        }
      }
    }

    if (actionResult.hasStdoutDigest()) {
      allPresent &= refreshDigest(actionResult.getStdoutDigest());
      if (!allPresent && stopAtFirstMissing) {
        return false;
      }
    }

    if (actionResult.hasStderrDigest()) {
      allPresent &= refreshDigest(actionResult.getStderrDigest());
    }

    return allPresent;
  }

  public ListenableFuture<ActionResult> downloadActionResult(ActionKey actionKey) {
    return Futures.transformAsync(
        // Update the mtime on the action result itself before any of the blobs it references.
        // This ensures that the blobs are always newer than the action result, so that trimming the
        // cache in LRU order cannot create dangling references.
        Utils.downloadAsActionResult(actionKey, (digest, out) -> download(digest, out, Store.AC)),
        actionResult -> {
          if (actionResult == null) {
            return immediateFuture(null);
          }

          boolean allBlobsPresent =
              refreshActionResult(
                  actionResult, /* stopAtFirstMissing= */ checkActionResultIntegrity);

          if (checkActionResultIntegrity && !allBlobsPresent) {
            // If at least one of the referenced blobs is missing, consider the action result to be
            // stale.
            return immediateFuture(null);
          }

          return immediateFuture(actionResult);
        },
        directExecutor());
  }

  public ListenableFuture<Void> uploadActionResult(ActionKey actionKey, ActionResult actionResult) {
    return executorService.submit(
        () -> {
          try (InputStream data = actionResult.toByteString().newInput()) {
            saveFile(actionKey.digest(), Store.AC, data);
          }
          return null;
        });
  }

  public void close() {
    executorService.close();
  }

  public ListenableFuture<Void> uploadFile(Digest digest, Path file) {
    return executorService.submit(
        () -> {
          saveFile(digest, Store.CAS, file);
          return null;
        });
  }

  public ListenableFuture<Void> uploadBlob(Digest digest, ByteString data) {
    return uploadBlob(digest, (Blob) data::newInput);
  }

  /** Uploads a blob from a stream supplier. */
  public ListenableFuture<Void> uploadBlob(Digest digest, Blob blob) {
    return executorService.submit(
        () -> {
          try (InputStream in = blob.get()) {
            saveFile(digest, Store.CAS, in);
          }
          return null;
        });
  }

  public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(Iterable<Digest> digests) {
    // Both upload and download check if the file exists before doing I/O. So we don't
    // have to do it here.
    return immediateFuture(ImmutableSet.copyOf(digests));
  }

  public Path getTempPath() {
    return tmpRoot.getChild(UUID.randomUUID().toString());
  }

  public Path toPath(Digest digest, Store store) {
    String hash = digest.getHash();
    return toPath(hash, store);
  }

  public Path toPath(String hash, Store store) {
    // Create the file in a subfolder to bypass possible folder file count limits.
    return storeRootMap.get(store).getChild(hash.substring(0, 2)).getChild(hash);
  }

  public void saveFile(Digest digest, Store store, InputStream in) throws IOException {
    save(
        digest,
        store,
        temp -> {
          try (OutputStream out = temp.getOutputStream()) {
            ByteStreams.copy(in, out);
            // Fsync temp before we rename it to avoid data loss in the case of machine
            // crashes (the OS may reorder the writes and the rename).
            if (out instanceof FileOutputStream fos) {
              fos.getFD().sync();
            }
          }
        });
  }

  /**
   * Saves an existing file into the cache.
   *
   * <p>The contents are copied through {@link FileSystemUtils#copyFile}, so a filesystem with
   * copy-on-write support (clonefile on macOS, copy_file_range on Linux) can serve the copy as a
   * clone, leaving the entry sharing its blocks with the file it was saved from.
   */
  private void saveFile(Digest digest, Store store, Path file) throws IOException {
    save(
        digest,
        store,
        temp -> {
          FileSystemUtils.copyFile(file, temp);
          // copyFile preserves the source's permissions and mtime, neither of which suits a cache
          // entry: an entry must remain readable by every user of a shared cache, and its mtime
          // records when it was last stored or retrieved.
          temp.chmod(0644);
          temp.setLastModifiedTime(Path.NOW_SENTINEL_TIME);
          // Fsync temp before we rename it to avoid data loss in the case of machine
          // crashes (the OS may reorder the writes and the rename).
          syncFile(temp);
        });
  }

  /** Writes the contents of a cache entry into a temporary file. */
  private interface TempFileWriter {
    void write(Path temp) throws IOException;
  }

  private void save(Digest digest, Store store, TempFileWriter writer) throws IOException {
    Path path = toPath(digest, store);

    // CAS entries are content-addressed and thus automatically have the correct content if they
    // exist.
    if (store == Store.CAS && refresh(path)) {
      return;
    }

    // Write a temporary file first, and then rename, to avoid data corruption in case of a crash.
    Path temp = getTempPath();

    try {
      writer.write(temp);
      path.getParentDirectory().createDirectoryAndParents();
      FileSystemUtils.renameToleratingConcurrentCreation(temp, path);
    } catch (IOException e) {
      try {
        temp.delete();
      } catch (IOException deleteErr) {
        e.addSuppressed(deleteErr);
      }
      throw e;
    }
  }

  /** Flushes a file's contents to stable storage, where the filesystem supports it. */
  private static void syncFile(Path path) throws IOException {
    try (InputStream in = path.getInputStream()) {
      if (in instanceof FileInputStream fileInputStream) {
        fileInputStream.getFD().sync();
      }
    }
  }
}
