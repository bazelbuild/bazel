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
import com.google.common.base.Preconditions;
import com.google.common.primitives.Longs;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Utility class for getting digests of files.
 *
 * <p>This class implements an optional cache of file digests when the computation of the digests is
 * costly (i.e. when {@link Path#getFastDigest()} is not available). The cache can be enabled via
 * the {@link #configureCache(long)} function, but note that enabling this cache might have an
 * impact on correctness because not all changes to files can be purely detected from their
 * metadata.
 */
public class DigestUtils {
  // Typical size for a digest byte array.
  public static final int ESTIMATED_SIZE = 32;

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

    /** Last change time of the file. */
    private final long changeTime;

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
      this.changeTime = status.getLastChangeTime();
      this.modifiedTime = status.getLastModifiedTime();
      this.size = status.getSize();
    }

    @Override
    public boolean equals(Object object) {
      if (object == this) {
        return true;
      } else if (!(object instanceof CacheKey key)) {
        return false;
      } else {
        return path.equals(key.path)
            && nodeId == key.nodeId
            && changeTime == key.changeTime
            && modifiedTime == key.modifiedTime
            && size == key.size;
      }
    }

    @Override
    public int hashCode() {
      int result = 17;
      result = 31 * result + path.hashCode();
      result = 31 * result + Longs.hashCode(nodeId);
      result = 31 * result + Longs.hashCode(changeTime);
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
   * Clears the cache contents without changing its size. No-op if the cache hasn't yet been
   * initialized.
   */
  public static void clearCache() {
    if (globalCache != null) {
      globalCache.invalidateAll();
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
   * Gets the digest of {@code path}, using a constant-time xattr call if the filesystem supports
   * it, and calculating the digest manually otherwise.
   *
   * <p>If {@link Path#getFastDigest} has already been attempted and was not available, call {@link
   * #manuallyComputeDigest} to skip an additional attempt to obtain the fast digest.
   *
   * <p>Prefer calling {@link #manuallyComputeDigest(Path, FileStatus)} when a recently obtained
   * {@link FileStatus} is available.
   *
   * @param path the file path
   */
  public static byte[] getDigestWithManualFallback(Path path, XattrProvider xattrProvider)
      throws IOException {
    return getDigestWithManualFallback(path, xattrProvider, null);
  }

  /**
   * Same as {@link #getDigestWithManualFallback(Path, XattrProvider)}, but providing the ability to
   * reuse a recently obtained {@link FileStatus}.
   *
   * @param path the file path
   * @param status a recently obtained file status, if available
   */
  public static byte[] getDigestWithManualFallback(
      Path path, XattrProvider xattrProvider, @Nullable FileStatus status) throws IOException {
    byte[] digest = xattrProvider.getFastDigest(path);
    return digest != null ? digest : manuallyComputeDigest(path, status);
  }

  /**
   * Calculates a digest manually (i.e., assuming that a fast digest can't obtained).
   *
   * <p>Prefer calling {@link #manuallyComputeDigest(Path, FileStatus)} when a recently obtained
   * {@link FileStatus} is available.
   *
   * @param path the file path
   */
  public static byte[] manuallyComputeDigest(Path path) throws IOException {
    return manuallyComputeDigest(path, null);
  }

  /**
   * Same as {@link #manuallyComputeDigest(Path)}, but providing the ability to reuse a recently
   * obtained {@link FileStatus}.
   *
   * @param path the file path
   * @param status a recently obtained file status, if available
   */
  public static byte[] manuallyComputeDigest(Path path, @Nullable FileStatus status)
      throws IOException {
    byte[] digest;

    // Attempt a cache lookup if the cache is enabled.
    Cache<CacheKey, byte[]> cache = globalCache;
    CacheKey key = null;
    if (cache != null) {
      key = new CacheKey(path, status != null ? status : path.stat());
      digest = cache.getIfPresent(key);
      if (digest != null) {
        return digest;
      }
    }

    digest = path.getDigest();

    Preconditions.checkNotNull(digest, "Missing digest for %s", path);
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
}
