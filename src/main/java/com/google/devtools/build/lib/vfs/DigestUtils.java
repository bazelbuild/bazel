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

import static com.google.common.base.Preconditions.checkArgument;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.stats.CacheStats;
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
 *
 * <p>The cache identifies a file by a {@link PathFragment}: the exec path of an action input (see
 * {@code ActionInput#getExecPath}), or the absolute path of a file that is only known by its {@link
 * Path}. The two kinds cannot collide since one is relative and the other absolute. Every consumer
 * that digests the file of an action input must key it by its exec path for their lookups to share
 * an entry, and should do so even when the file is read through another path, such as on an action
 * filesystem. The exec path is retained by the artifact for as long as the build refers to it, so
 * such a key costs the cache nothing, whereas an absolute path has to be built and retained for the
 * entry alone.
 *
 * <p>Callers that have an action input should use {@code ActionInputHelper}, {@code
 * FileArtifactValue} or {@code DigestUtil}, which derive the key from the input. The overloads
 * taking a key directly are for the few callers that only have the path, and they verify that the
 * path ends with the key, which holds for every exec path since a path is always its root followed
 * by its exec path.
 */
public class DigestUtils {
  // Typical size for a digest byte array.
  public static final int ESTIMATED_SIZE = 32;

  private static final HexFormat HEX_FORMAT = HexFormat.of();

  /**
   * Keys used to cache the values of the digests for files where we don't have fast digests.
   *
   * <p>The cache keys are derived from many properties of the file metadata in an attempt to be
   * able to detect most file changes.
   */
  private static record CacheKey(
      PathFragment path, long nodeId, long changeTime, long lastModifiedTime, long size) {
    /**
     * Constructs a new cache key.
     *
     * @param path the key identifying the file, see the class documentation
     * @param status file status data from which to obtain the cache key properties
     * @throws IOException if reading the file status data fails
     */
    private CacheKey(PathFragment path, FileStatus status) throws IOException {
      this(
          path,
          status.getNodeId(),
          status.getLastChangeTime(),
          status.getLastModifiedTime(),
          status.getSize());
    }
  }

  /**
   * Global cache of files to their digests.
   *
   * <p>This is null when the cache is disabled.
   *
   * <p>Note that we do not use a {@link com.github.benmanes.caffeine.cache.LoadingCache} because
   * our keys only identify the files and are not {@link Path} instances. As a result, the loading
   * function cannot actually compute the digests of the files so we have to handle this externally.
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
   * @param path the file path, which is also the key under which the file is cached
   * @param status a recently obtained file status, if available. Used to skip a stat.
   */
  public static byte[] getDigestWithManualFallback(
      Path path, XattrProvider xattrProvider, @Nullable FileStatus status) throws IOException {
    return getDigestWithManualFallback(path.asFragment(), path, xattrProvider, status);
  }

  /**
   * Same as {@link #getDigestWithManualFallback(Path, XattrProvider, FileStatus)}, with the file
   * cached under {@code cacheKey} rather than under its path.
   *
   * @param cacheKey the key under which the file is cached, see the class documentation
   * @param path the file path to stat and read
   */
  public static byte[] getDigestWithManualFallback(
      PathFragment cacheKey, Path path, XattrProvider xattrProvider, @Nullable FileStatus status)
      throws IOException {
    byte[] digest = xattrProvider.getFastDigest(path);
    return digest != null ? digest : manuallyComputeDigest(cacheKey, path, status);
  }

  /**
   * Calculates a digest manually (i.e., assuming that a fast digest can't obtained).
   *
   * @param path the file path, which is also the key under which the file is cached
   * @param status a recently obtained file status, if available. Used to skip a stat.
   */
  public static byte[] manuallyComputeDigest(Path path, @Nullable FileStatus status)
      throws IOException {
    return manuallyComputeDigest(path.asFragment(), path, status);
  }

  /**
   * Same as {@link #manuallyComputeDigest(Path, FileStatus)}, with the file cached under {@code
   * cacheKey} rather than under its path.
   *
   * @param cacheKey the key under which the file is cached, see the class documentation
   * @param path the file path to stat and read
   * @param status a recently obtained file status, if available. Used to skip a stat.
   */
  public static byte[] manuallyComputeDigest(
      PathFragment cacheKey, Path path, @Nullable FileStatus status) throws IOException {
    checkArgument(
        identifies(cacheKey, path), "digest cache key %s does not identify %s", cacheKey, path);
    byte[] digest;

    // Attempt a cache lookup if the cache is enabled.
    Cache<CacheKey, byte[]> cache = globalCache;
    CacheKey key = null;
    if (cache != null) {
      key = new CacheKey(cacheKey, status != null ? status : path.stat());
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

  /**
   * Returns whether {@code cacheKey} can identify the file at {@code path}: it is the path itself,
   * or an exec path the path ends with.
   */
  private static boolean identifies(PathFragment cacheKey, Path path) {
    return cacheKey.isAbsolute()
        ? cacheKey.equals(path.asFragment())
        : path.asFragment().endsWith(cacheKey);
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
