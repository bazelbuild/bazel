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
package com.google.devtools.build.lib.actions.cache;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheStats;
import com.google.common.io.BaseEncoding;
import com.google.common.primitives.Longs;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.VarInt;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;

/**
 * Utility class for getting md5 digests of files.
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
   * <p>Note that we do not use a {@link com.google.common.cache.LoadingCache} because our keys
   * represent the paths as strings, not as {@link Path} instances. As a result, the loading
   * function cannot actually compute the digests of the files so we have to handle this externally.
   */
  private static volatile Cache<CacheKey, byte[]> globalCache = null;

  /** Private constructor to prevent instantiation of utility class. */
  private DigestUtils() {}

  /**
   * Obtain file's MD5 metadata using synchronized method, ensuring that system
   * is not overloaded in case when multiple threads are requesting MD5
   * calculations and underlying file system cannot provide it via extended
   * attribute.
   */
  private static byte[] getDigestInExclusiveMode(Path path)
      throws IOException {
    long startTime = BlazeClock.nanoTime();
    synchronized (DIGEST_LOCK) {
      Profiler.instance().logSimpleTask(startTime, ProfilerTask.WAIT, path.getPathString());
      return getDigestInternal(path);
    }
  }

  private static byte[] getDigestInternal(Path path) throws IOException {
    long startTime = BlazeClock.nanoTime();
    byte[] digest = path.getDigest();

    long millis = (BlazeClock.nanoTime() - startTime) / 1000000;
    if (millis > 5000L) {
      System.err.println("Slow read: a " + path.getFileSize() + "-byte read from " + path
          + " took " +  millis + "ms.");
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
      globalCache = CacheBuilder.newBuilder().maximumSize(maximumSize).recordStats().build();
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
   * Get the digest of {@code path}, using a constant-time xattr call if the filesystem supports
   * it, and calculating the digest manually otherwise.
   *
   * @param path Path of the file.
   * @param fileSize size of the file. Used to determine if digest calculation should be done
   * serially or in parallel. Files larger than a certain threshold will be read serially, in order
   * to avoid excessive disk seeks.
   */
  public static byte[] getDigestOrFail(Path path, long fileSize)
      throws IOException {
    byte[] digest = path.getFastDigest();

    if (digest != null && !path.isValidDigest(digest)) {
      // Fail-soft in cases where md5bin is non-null, but not a valid digest.
      String msg = String.format("Malformed digest '%s' for file %s",
                                 BaseEncoding.base16().lowerCase().encode(digest),
                                 path);
      LoggingUtil.logToRemote(Level.SEVERE, msg, new IllegalStateException(msg));
      digest = null;
    }

    // At this point, either we could not get a fast digest or the fast digest we got is corrupt.
    // Attempt a cache lookup if the cache is enabled and return the cached digest if found.
    Cache<CacheKey, byte[]> cache = globalCache;
    CacheKey key = null;
    if (cache != null && digest == null) {
      key = new CacheKey(path, path.stat());
      digest = cache.getIfPresent(key);
    }
    if (digest != null) {
      return digest;
    }

    // All right, we have neither a fast nor a cached digest. Let's go through the costly process of
    // computing it from the file contents.
    if (fileSize > 4096 && !MULTI_THREADED_DIGEST.get()) {
      // We'll have to read file content in order to calculate the digest. In that case
      // it would be beneficial to serialize those calculations since there is a high
      // probability that MD5 will be requested for multiple output files simultaneously.
      // Exception is made for small (<=4K) files since they will not likely to introduce
      // significant delays (at worst they will result in two extra disk seeks by
      // interrupting other reads).
      digest = getDigestInExclusiveMode(path);
    } else {
      digest = getDigestInternal(path);
    }

    Preconditions.checkNotNull(
        digest,
        "We should have gotten a digest for %s at this point but we still don't have one",
        path);
    if (cache != null) {
      Preconditions.checkNotNull(
          key,
          "We should have computed a cache key earlier for %s because the cache is enabled and we"
              + " did not get a fast digest for this file, but we don't have a key here",
          path);
      cache.put(key, digest);
    }
    return digest;
  }

  /**
   * @param source the byte buffer source.
   * @return the digest from the given buffer.
   * @throws IOException if the byte buffer is incorrectly formatted.
   */
  public static Md5Digest read(ByteBuffer source) throws IOException {
    int size = VarInt.getVarInt(source);
    if (size != Md5Digest.MD5_SIZE) {
      throw new IOException("Unexpected digest length: " + size);
    }
    byte[] bytes = new byte[size];
    source.get(bytes);
    return new Md5Digest(bytes);
  }

  /** Write the digest to the output stream. */
  public static void write(Md5Digest digest, OutputStream sink) throws IOException {
    VarInt.putVarInt(digest.getDigestBytesUnsafe().length, sink);
    sink.write(digest.getDigestBytesUnsafe());
  }

  /**
   * @param mdMap A collection of (execPath, Metadata) pairs. Values may be null.
   * @return an <b>order-independent</b> digest from the given "set" of (path, metadata) pairs.
   */
  public static Md5Digest fromMetadata(Map<String, Metadata> mdMap) {
    byte[] result = new byte[Md5Digest.MD5_SIZE];
    // Profiling showed that MD5 engine instantiation was a hotspot, so create one instance for
    // this computation to amortize its cost.
    Fingerprint fp = new Fingerprint();
    for (Map.Entry<String, Metadata> entry : mdMap.entrySet()) {
      xorWith(result, getDigest(fp, entry.getKey(), entry.getValue()));
    }
    return new Md5Digest(result);
  }

  /**
   * @param env A collection of (String, String) pairs.
   * @return an order-independent digest of the given set of pairs.
   */
  public static Md5Digest fromEnv(Map<String, String> env) {
    byte[] result = new byte[Md5Digest.MD5_SIZE];
    Fingerprint fp = new Fingerprint();
    for (Map.Entry<String, String> entry : env.entrySet()) {
      fp.addString(entry.getKey());
      fp.addString(entry.getValue());
      xorWith(result, fp.digestAndReset());
    }
    return new Md5Digest(result);
  }

  private static byte[] getDigest(Fingerprint fp, String execPath, Metadata md) {
    fp.addString(execPath);

    if (md == null) {
      // Move along, nothing to see here.
    } else if (md.isFile()) {
      fp.addBytes(md.getDigest());
    } else {
      // Use the timestamp if the digest is not present, but not both. Modifying a timestamp while
      // keeping the contents of a file the same should not cause rebuilds.
      fp.addLong(md.getModifiedTime());
    }
    return fp.digestAndReset();
  }

  /** Compute lhs ^= rhs bitwise operation of the arrays. */
  private static void xorWith(byte[] lhs, byte[] rhs) {
    for (int i = 0; i < lhs.length; i++) {
      lhs[i] ^= rhs[i];
    }
  }
}
