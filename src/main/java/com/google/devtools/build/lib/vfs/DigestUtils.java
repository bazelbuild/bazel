// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.stats.CacheStats;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Longs;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Utility class for getting digests of files.
 *
 * <p>This class implements an optional cache of file digests when the computation of the digests is
 * costly (i.e. when {@link Path#getFastDigest()} is not available). The cache can be enabled via
 * the {@link #configureCache(long)} function, but note that enabling this cache might have an
 * impact on correctness because not all changes to files can be purely detected from their
 * metadata.
 *
 * <p>Note that this class is responsible for digesting file metadata in an order-independent
 * manner. Care must be taken to do this properly. The digest must be a function of the set of
 * (path, metadata) tuples. While the order of these pairs must not matter, it would <b>not</b> be
 * safe to make the digest be a function of the set of paths and the set of metadata.
 *
 * <p>Note that the (path, metadata) tuples must be unique, otherwise the XOR-based approach will
 * fail.
 */
public class DigestUtils {

  // Object to synchronize on when serializing large file reads.
  private static final Object DIGEST_LOCK = new Object();
  private static final AtomicBoolean MULTI_THREADED_DIGEST = new AtomicBoolean(false);

  // Typical size for a digest byte array.
  public static final int ESTIMATED_SIZE = 32;

  // Files of this size or less are assumed to be readable in one seek.
  // (This is the default readahead window on Linux.)
  @VisibleForTesting // the unittest is in a different package!
  public static final int MULTI_THREADED_DIGEST_MAX_FILE_SIZE = 128 * 1024;

  // The time that a digest computation has to take at least in order to be considered a slow-read.
  private static final long SLOW_READ_MILLIS = 5000L;

  // The average bytes-per-millisecond throughput that a digest computation has to go below in order
  // to be considered a slow-read.
  private static final long SLOW_READ_THROUGHPUT = (10 * 1024 * 1024) / 1000;

  /**
   * Keys used to cache the values of the digests for files where we don't have fast digests.
   *
   * <p>The cache keys are derived from many properties of the file metadata in an attempt to be
   * able to detect most file changes.
   */
  private static class CacheKey {
    /** Path to the file. */
    private final PathFragment path;

    /** File system identifier of the file (typically the inode number). */
    private final long nodeId;

    /** Last modification time of the file. */
    private final long modifiedTime;

    /** Size of the file. */
    private final long size;

    /**
     * Constructs a new cache key.
     *
     * @param path path to the file
     * @param status file status data from which to obtain the cache key properties
     * @throws IOException if reading the file status data fails
     */
    public CacheKey(Path path, FileStatus status) throws IOException {
      this.path = path.asFragment();
      this.nodeId = status.getNodeId();
      this.modifiedTime = status.getLastModifiedTime();
      this.size = status.getSize();
    }

    @Override
    public boolean equals(Object object) {
      if (object == this) {
        return true;
      } else if (!(object instanceof CacheKey)) {
        return false;
      } else {
        CacheKey key = (CacheKey) object;
        return path.equals(key.path)
            && nodeId == key.nodeId
            && modifiedTime == key.modifiedTime
            && size == key.size;
      }
    }

    @Override
    public int hashCode() {
      int result = 17;
      result = 31 * result + path.hashCode();
      result = 31 * result + Longs.hashCode(nodeId);
      result = 31 * result + Longs.hashCode(modifiedTime);
      result = 31 * result + Longs.hashCode(size);
      return result;
    }
  }

  /**
   * Global cache of files to their digests.
   *
   * <p>This is null when the cache is disabled.
   *
   * <p>Note that we do not use a {@link com.github.benmanes.caffeine.cache.LoadingCache} because
   * our keys represent the paths as strings, not as {@link Path} instances. As a result, the
   * loading function cannot actually compute the digests of the files so we have to handle this
   * externally.
   */
  private static Cache<CacheKey, byte[]> globalCache = null;

  /** Private constructor to prevent instantiation of utility class. */
  private DigestUtils() {}

  /**
   * Obtain file's digset using synchronized method, ensuring that system is not overloaded in case
   * when multiple threads are requesting digest calculations and underlying file system cannot
   * provide it via extended attribute.
   */
  private static byte[] getDigestInExclusiveMode(Path path) throws IOException {
    long startTime = Profiler.nanoTimeMaybe();
    synchronized (DIGEST_LOCK) {
      Profiler.instance().logSimpleTask(startTime, ProfilerTask.WAIT, path.getPathString());
      return getDigestInternal(path);
    }
  }

  private static byte[] getDigestInternal(Path path) throws IOException {
    long startTime = System.nanoTime();
    byte[] digest = path.getDigest();

    // When using multi-threaded digesting, it makes no sense to use the throughput of a single
    // digest operation to determine whether a read was abnormally slow (as the scheduler might just
    // have preferred other reads).
    if (!MULTI_THREADED_DIGEST.get()) {
      long millis = (System.nanoTime() - startTime) / 1000000;
      if (millis > SLOW_READ_MILLIS && (path.getFileSize() / millis) < SLOW_READ_THROUGHPUT) {
        System.err.printf(
            "Slow read: a %d-byte read from %s took %d ms.%n", path.getFileSize(), path, millis);
      }
    }

    return digest;
  }

  /**
   * Enables the caching of file digests based on file status data.
   *
   * <p>If the cache was already enabled, this causes the cache to be reinitialized thus losing all
   * contents. If the given size is zero, the cache is disabled altogether.
   *
   * @param maximumSize maximumSize of the cache in number of entries
   */
  public static void configureCache(long maximumSize) {
    if (maximumSize == 0) {
      globalCache = null;
    } else {
      globalCache = Caffeine.newBuilder().maximumSize(maximumSize).recordStats().build();
    }
  }

  /**
   * Obtains cache statistics.
   *
   * <p>The cache must have previously been enabled by a call to {@link #configureCache(long)}.
   *
   * @return an immutable snapshot of the cache statistics
   */
  public static CacheStats getCacheStats() {
    Cache<CacheKey, byte[]> cache = globalCache;
    Preconditions.checkNotNull(cache, "configureCache() must have been called with a size >= 0");
    return cache.stats();
  }

  /**
   * Enable or disable multi-threaded digesting even for large files.
   */
  public static void setMultiThreadedDigest(boolean multiThreadedDigest) {
    DigestUtils.MULTI_THREADED_DIGEST.set(multiThreadedDigest);
  }

  /**
   * Gets the digest of {@code path}, using a constant-time xattr call if the filesystem supports
   * it, and calculating the digest manually otherwise.
   *
   * <p>If {@link Path#getFastDigest} has already been attempted and was not available, call {@link
   * #manuallyComputeDigest} to skip an additional attempt to obtain the fast digest.
   *
   * @param path Path of the file.
   * @param fileSize Size of the file. Used to determine if digest calculation should be done
   *     serially or in parallel. Files larger than a certain threshold will be read serially, in
   *     order to avoid excessive disk seeks.
   */
  public static byte[] getDigestWithManualFallback(Path path, long fileSize) throws IOException {
    byte[] digest = path.getFastDigest();
    return digest != null ? digest : manuallyComputeDigest(path, fileSize);
  }

  /**
   * Gets the digest of {@code path}, using a constant-time xattr call if the filesystem supports
   * it, and calculating the digest manually otherwise.
   *
   * <p>Unlike {@link #getDigestWithManualFallback}, will not rate-limit manual digesting of files,
   * so only use this method if the file size is truly unknown and you don't expect many concurrent
   * manual digests of large files.
   *
   * @param path Path of the file.
   */
  public static byte[] getDigestWithManualFallbackWhenSizeUnknown(Path path) throws IOException {
    return getDigestWithManualFallback(path, -1);
  }

  /**
   * Calculates the digest manually.
   *
   * @param path Path of the file.
   * @param fileSize Size of the file. Used to determine if digest calculation should be done
   *     serially or in parallel. Files larger than a certain threshold will be read serially, in
   *     order to avoid excessive disk seeks.
   */
  public static byte[] manuallyComputeDigest(Path path, long fileSize) throws IOException {
    byte[] digest;

    // Attempt a cache lookup if the cache is enabled.
    Cache<CacheKey, byte[]> cache = globalCache;
    CacheKey key = null;
    if (cache != null) {
      key = new CacheKey(path, path.stat());
      digest = cache.getIfPresent(key);
      if (digest != null) {
        return digest;
      }
    }

    // Compute digest from the file contents.
    if (fileSize > MULTI_THREADED_DIGEST_MAX_FILE_SIZE && !MULTI_THREADED_DIGEST.get()) {
      // We'll have to read file content in order to calculate the digest.
      // We avoid overlapping this process for multiple large files, as
      // seeking back and forth between them will result in an overall loss of
      // throughput.
      digest = getDigestInExclusiveMode(path);
    } else {
      digest = getDigestInternal(path);
    }

    Preconditions.checkNotNull(digest, "Missing digest for %s (size %s)", path, fileSize);
    if (cache != null) {
      cache.put(key, digest);
    }
    return digest;
  }

  /** Compute lhs ^= rhs bitwise operation of the arrays. May clobber either argument. */
  public static byte[] xor(byte[] lhs, byte[] rhs) {
    int n = rhs.length;
    if (lhs.length >= n) {
      for (int i = 0; i < n; i++) {
        lhs[i] ^= rhs[i];
      }
      return lhs;
    }
    return xor(rhs, lhs);
  }

  /**
   * Run any pending maintenance operations on the global cache.
   *
   * <p>Used in tests to force caffeine to run any pending evictions.
   */
  @VisibleForTesting
  static void cleanupCacheForTest() {
    globalCache.cleanUp();
  }
}
