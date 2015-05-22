// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.util.Objects;
import java.util.logging.Level;

import javax.annotation.Nullable;

/**
 * Utility class for getting md5 digests of files.
 */
public class DigestUtils {
  // Object to synchronize on when serializing large file reads.
  private static final Object MD5_LOCK = new Object();

  /** Private constructor to prevent instantiation of utility class. */
  private DigestUtils() {}

  /**
   * Returns true iff using MD5 digests is appropriate for an artifact.
   *
   * @param isFile whether or not Artifact is a file versus a directory, isFile() on its stat.
   * @param size size of Artifact on filesystem in bytes, getSize() on its stat.
   */
  public static boolean useFileDigest(boolean isFile, long size) {
    // Use timestamps for directories. Use digests for everything else.
    return isFile && size != 0;
  }

  /**
   * Obtain file's MD5 metadata using synchronized method, ensuring that system
   * is not overloaded in case when multiple threads are requesting MD5
   * calculations and underlying file system cannot provide it via extended
   * attribute.
   */
  private static byte[] getDigestInExclusiveMode(Path path) throws IOException {
    long startTime = BlazeClock.nanoTime();
    synchronized (MD5_LOCK) {
      Profiler.instance().logSimpleTask(startTime, ProfilerTask.WAIT, path.getPathString());
      return getDigestInternal(path);
    }
  }

  private static byte[] getDigestInternal(Path path) throws IOException {
    long startTime = BlazeClock.nanoTime();
    byte[] md5bin = path.getMD5Digest();

    long millis = (BlazeClock.nanoTime() - startTime) / 1000000;
    if (millis > 5000L) {
      System.err.println("Slow read: a " + path.getFileSize() + "-byte read from " + path
          + " took " +  millis + "ms.");
    }
    return md5bin;
  }

  private static boolean binaryDigestWellFormed(byte[] digest) {
    Preconditions.checkNotNull(digest);
    return digest.length == 16;
  }

  /**
   * Returns the the fast md5 digest of the file, or null if not available.
   */
  @Nullable
  public static byte[] getFastDigest(Path path) throws IOException {
    return path.getFastDigestFunctionType().equals("MD5") ? path.getFastDigest() : null;
  }

  /**
   * Get the md5 digest of {@code path}, using a constant-time xattr call if the filesystem supports
   * it, and calculating the digest manually otherwise.
   *
   * @param path Path of the file.
   * @param fileSize size of the file. Used to determine if digest calculation should be done
   * serially or in parallel. Files larger than a certain threshold will be read serially, in order
   * to avoid excessive disk seeks.
   */
  public static byte[] getDigestOrFail(Path path, long fileSize) throws IOException {
    // TODO(bazel-team): the action cache currently only works with md5 digests but it ought to
    // work with any opaque digest.
    byte[] md5bin = null;
    if (Objects.equals(path.getFastDigestFunctionType(), "MD5")) {
      md5bin = getFastDigest(path);
    }
    if (md5bin != null && !binaryDigestWellFormed(md5bin)) {
      // Fail-soft in cases where md5bin is non-null, but not a valid digest.
      String msg = String.format("Malformed digest '%s' for file %s",
                                 BaseEncoding.base16().lowerCase().encode(md5bin),
                                 path);
      LoggingUtil.logToRemote(Level.SEVERE, msg, new IllegalStateException(msg));
      md5bin = null;
    }
    if (md5bin != null) {
      return md5bin;
    } else if (fileSize > 4096) {
      // We'll have to read file content in order to calculate the digest. In that case
      // it would be beneficial to serialize those calculations since there is a high
      // probability that MD5 will be requested for multiple output files simultaneously.
      // Exception is made for small (<=4K) files since they will not likely to introduce
      // significant delays (at worst they will result in two extra disk seeks by
      // interrupting other reads).
      return getDigestInExclusiveMode(path);
    } else {
      return getDigestInternal(path);
    }
  }
}
