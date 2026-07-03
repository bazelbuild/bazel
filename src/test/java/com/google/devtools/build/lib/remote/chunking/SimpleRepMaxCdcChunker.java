// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote.chunking;

import static com.google.common.base.Preconditions.checkArgument;

import build.bazel.remote.execution.v2.Digest;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Simple reference implementation of RepMaxCDC, used for testing only.
 *
 * <p>This class provides the same behavior as {@link RepMaxCdcChunker}, but implements the
 * algorithm naively: for every chunk it rehashes the full lookahead window, and it repeats the
 * search with a shrinking horizon whenever the best cutting point would yield an oversized chunk.
 * This makes it easy to review for correctness, but slow due to repeated hashing of the input.
 *
 * <p>It is a port of {@code NewSimpleRepMaxContentDefinedChunker} from <a
 * href="https://github.com/buildbarn/go-cdc">buildbarn/go-cdc</a>, which plays the same role
 * there: a reference that the optimized implementation is tested against.
 */
final class SimpleRepMaxCdcChunker implements ContentDefinedChunker {

  private final int minSize;
  // The amount of data that must be available (unless the end of the input has been reached) to
  // determine the next chunk boundary: 2 * minSize + horizonSize. See cut() for why.
  private final int peekSize;
  private final DigestUtil digestUtil;

  SimpleRepMaxCdcChunker(RepMaxCdcChunkingConfig config, DigestUtil digestUtil) {
    this(config.minChunkSize(), config.horizonSize(), digestUtil);
  }

  SimpleRepMaxCdcChunker(int minSize, int horizonSize, DigestUtil digestUtil) {
    checkArgument(
        minSize >= GearTable.GEAR_HASH_WINDOW_SIZE,
        "minSize must be at least %s, got %s",
        GearTable.GEAR_HASH_WINDOW_SIZE,
        minSize);
    checkArgument(horizonSize >= 0, "horizonSize must not be negative, got %s", horizonSize);
    // The read buffer is sized at twice the peek size, so make sure that fits in an int.
    checkArgument(
        2L * minSize + horizonSize <= Integer.MAX_VALUE / 2,
        "2 * minSize + horizonSize must be at most %s, got %s",
        Integer.MAX_VALUE / 2,
        2L * minSize + horizonSize);

    this.minSize = minSize;
    this.peekSize = 2 * minSize + horizonSize;
    this.digestUtil = digestUtil;
  }

  /**
   * Finds the next chunk boundary.
   *
   * <p>The buffer must either contain at least {@link #peekSize} bytes, or all remaining bytes of
   * the input.
   */
  private int cut(byte[] buf, int off, int len) {
    // Look at no more data than a Peeker returning at most peekSize bytes would, so that chunk
    // boundaries are independent of how much input data happens to be buffered.
    int n = Math.min(len, peekSize);
    if (n < 2 * minSize) {
      // Too little data is left to form two chunks of at least minSize each. Return all of it as
      // a single final chunk, which is smaller than 2 * minSize (and smaller than minSize only if
      // the entire input is).
      return n;
    }

    long[] gear = GearTable.GEAR;

    // Compute the rolling hash leading up to the first position at which we may place a cut.
    long initialHash = 0;
    for (int i = minSize - GearTable.GEAR_HASH_WINDOW_SIZE; i < minSize; i++) {
      initialHash = (initialHash << 1) + gear[buf[off + i] & 0xFF];
    }

    // Leave at least minSize bytes behind, so that the final chunk of the input is at least
    // minSize in size as well.
    int limit = n - minSize;
    while (true) {
      // Scan the horizon for the position with the highest hash value, preferring the earliest
      // position in case of ties. Hash values are compared as unsigned 64-bit integers, matching
      // the Go reference implementation.
      long hash = initialHash;
      long bestHash = hash;
      int bestEnd = minSize;
      for (int i = minSize; i < limit; i++) {
        hash = (hash << 1) + gear[buf[off + i] & 0xFF];
        if (Long.compareUnsigned(bestHash, hash) < 0) {
          bestHash = hash;
          bestEnd = i + 1;
        }
      }
      if (bestEnd < 2 * minSize) {
        return bestEnd;
      }

      // If we were to cut at the most suitable position within the horizon, we would end up with
      // a chunk that is too large. Repeat the search, limiting the horizon to minSize bytes
      // before the position that was just obtained. This allows later chunks to still consider
      // this position again.
      limit = bestEnd - minSize;
    }
  }

  /**
   * Chunks a blob and returns chunk digests.
   *
   * <p>This method is used for building MerkleTree entries for large files. It returns the content
   * digests in order for each chunk.
   *
   * <p>Note: We don't need the raw data here. We can read from the original file (seekable) when
   * uploading, similar to how whole blobs work.
   */
  @Override
  public List<Digest> chunkToDigests(InputStream input) throws IOException {
    List<Digest> digests = new ArrayList<>();

    byte[] buf = new byte[peekSize * 2];
    int cursor = 0;
    int end = 0;
    boolean eof = false;

    while (true) {
      int available = end - cursor;
      if (available < peekSize && !eof) {
        if (cursor > 0 && available > 0) {
          System.arraycopy(buf, cursor, buf, 0, available);
        }
        cursor = 0;
        end = available;

        while (end < buf.length) {
          int n = input.read(buf, end, buf.length - end);
          if (n == -1) {
            eof = true;
            break;
          }
          end += n;
        }
        available = end - cursor;
      }

      if (available == 0) {
        break;
      }

      int chunkLen = cut(buf, cursor, available);
      digests.add(digestUtil.compute(buf, cursor, chunkLen));

      cursor += chunkLen;
    }

    return digests;
  }
}
