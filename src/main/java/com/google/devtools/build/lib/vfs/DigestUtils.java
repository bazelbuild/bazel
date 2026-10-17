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
import com.github.benmanes.caffeine.cache.stats.ConcurrentStatsCounter;
import com.github.benmanes.caffeine.cache.stats.StatsCounter;
import com.google.common.base.Preconditions;
import com.google.protobuf.ByteString;
import com.google.protobuf.UnsafeByteOperations;
import java.io.IOException;
import java.util.HexFormat;
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

  private static final HexFormat HEX_FORMAT = HexFormat.of();

  /**
   * The digest of a file together with the metadata of the file it was computed from, for files
   * where we don't have fast digests.
   *
   * <p>The cache holds one such value per file. A lookup only uses the digest if the file's current
   * metadata still matches, which is derived from many properties of the file metadata in an
   * attempt to detect most file changes.
   *
   * <p>The metadata lives in the value rather than the key so that a file that changes does not
   * leave a stale entry behind. The change time only moves forward, so a file's old metadata can
   * never be observed again.
   */
  private static final class CachedDigest {
    private final long nodeId;
    private final long changeTime;
    private final long size;
    private final byte[] digest;

    private CachedDigest(FileStatus status, byte[] digest) throws IOException {
      this.nodeId = status.getNodeId();
      this.changeTime = status.getLastChangeTime();
      this.size = status.getSize();
      this.digest = digest;
    }

    /** Returns whether the digest is still valid for a file with the given metadata. */
    private boolean matches(FileStatus status) throws IOException {
      return nodeId == status.getNodeId()
          && changeTime == status.getLastChangeTime()
          && size == status.getSize();
    }
  }

  /**
   * The cache together with its statistics counter.
   *
   * <p>Hits and misses are recorded by {@link #manuallyComputeDigest} itself, through {@code
   * stats}, so that a present but stale entry counts as a miss. The cache records evictions into
   * the same counter.
   */
  private record DigestCache(Cache<PathFragment, CachedDigest> cache, StatsCounter stats) {
    private static DigestCache create(long maximumSize) {
      var stats = new ConcurrentStatsCounter();
      return new DigestCache(
          Caffeine.newBuilder().maximumSize(maximumSize).recordStats(() -> stats).build(), stats);
    }
  }

  /**
   * Global cache of files to their digests.
   *
   * <p>This is null when the cache is disabled.
   *
   * <p>Note that we do not use a {@link com.github.benmanes.caffeine.cache.LoadingCache} because
   * our keys represent the paths as path fragments, not as {@link Path} instances. As a result, the
   * loading function cannot actually compute the digests of the files so we have to handle this
   * externally.
   */
  @Nullable private static DigestCache globalCache = null;

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
    globalCache = maximumSize == 0 ? null : DigestCache.create(maximumSize);
  }

  /**
   * Clears the cache contents without changing its size. No-op if the cache hasn't yet been
   * initialized.
   */
  public static void clearCache() {
    DigestCache cache = globalCache;
    if (cache != null) {
      cache.cache().invalidateAll();
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
    DigestCache cache = globalCache;
    Preconditions.checkNotNull(cache, "configureCache() must have been called with a size >= 0");
    return cache.cache().stats();
  }

  /**
   * Gets the digest of {@code path}, using a constant-time xattr call if the filesystem supports
   * it, and calculating the digest manually otherwise.
   *
   * <p>If {@link Path#getFastDigest} has already been attempted and was not available, call {@link
   * #manuallyComputeDigest} to skip an additional attempt to obtain the fast digest.
   *
   * @param path the file path
   * @param status a recently obtained file status, if available. Used to skip a stat.
   */
  public static byte[] getDigestWithManualFallback(
      Path path, XattrProvider xattrProvider, @Nullable FileStatus status) throws IOException {
    byte[] digest = xattrProvider.getFastDigest(path);
    return digest != null ? digest : manuallyComputeDigest(path, status);
  }

  /**
   * Calculates a digest manually (i.e., assuming that a fast digest can't obtained).
   *
   * @param path the file path
   * @param status a recently obtained file status, if available. Used to skip a stat.
   */
  public static byte[] manuallyComputeDigest(Path path, @Nullable FileStatus status)
      throws IOException {
    // Attempt a cache lookup if the cache is enabled.
    DigestCache cache = globalCache;
    PathFragment key = null;
    if (cache != null) {
      if (status == null) {
        status = path.stat();
      }
      key = path.asFragment();
      // Look up through the map view, which informs the eviction policy but does not record
      // statistics, so that only a usable digest counts as a hit.
      CachedDigest cached = cache.cache().asMap().get(key);
      if (cached != null && cached.matches(status)) {
        cache.stats().recordHits(1);
        return cached.digest;
      }
      cache.stats().recordMisses(1);
    }

    byte[] digest = path.getDigest();

    Preconditions.checkNotNull(digest, "Missing digest for %s", path);
    if (cache != null) {
      cache.cache().put(key, new CachedDigest(status, digest));
    }
    return digest;
  }

  /**
   * Combines two digests into one such that swapping the arguments results in the same result. May
   * clobber either argument.
   */
  public static byte[] combineUnordered(byte[] lhs, byte[] rhs) {
    int n = rhs.length;
    if (lhs.length >= n) {
      for (int i = 0; i < n; i++) {
        // Use + as in Guava's Hashing.combineUnordered.
        // This has a number of advantages over XOR, which was used in the past:
        // * Identical inputs will not cancel each other out.
        // * Due to the carry, addition isn't a linear operation on the level of bit vectors.
        //   This prevents adversaries from producing linear combinations (i.e., subsets of input
        //   sets) that collide with other inputs.
        lhs[i] += rhs[i];
      }
      return lhs;
    }
    return combineUnordered(rhs, lhs);
  }

  /** Returns the lowercase hex encoding of the given bytes. */
  public static ByteString toHexByteString(byte[] bytes) {
    byte[] hex = new byte[2 * bytes.length];
    for (int i = 0; i < bytes.length; i++) {
      hex[2 * i] = (byte) HEX_FORMAT.toHighHexDigit(bytes[i]);
      hex[2 * i + 1] = (byte) HEX_FORMAT.toLowHexDigit(bytes[i]);
    }
    return UnsafeByteOperations.unsafeWrap(hex);
  }
}
