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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import java.io.IOException;
import java.io.InputStream;

/**
 * FastCDC 2020 implementation for splitting large blobs.
 *
 * <p>This module implements the canonical FastCDC algorithm as described in the
 * [paper](https://ieeexplore.ieee.org/document/9055082) by Wen Xia, et al., in 2020.
 */
public final class FastCdcChunker implements ContentDefinedChunker {

  // Masks for each of the desired number of bits, where 0 through 5 are unused.
  // The values for sizes 64 bytes through 128 kilo-bytes come from the C
  // reference implementation (found in the destor repository) while the extra
  // values come from the restic-FastCDC repository. The FastCDC paper claims that
  // the deduplication ratio is slightly improved when the mask bits are spread
  // relatively evenly, hence these seemingly "magic" values.
  // @formatter:off
  private static final long[] MASKS = {
    0, // 0: padding
    0, // 1: padding
    0, // 2: padding
    0, // 3: padding
    0, // 4: padding
    0x0000000001804110L, // 5: unused except for NC 3
    0x0000000001803110L, // 6: 64B
    0x0000000018035100L, // 7: 128B
    0x0000001800035300L, // 8: 256B
    0x0000019000353000L, // 9: 512B
    0x0000590003530000L, // 10: 1KB
    0x0000d90003530000L, // 11: 2KB
    0x0000d90103530000L, // 12: 4KB
    0x0000d90303530000L, // 13: 8KB
    0x0000d90313530000L, // 14: 16KB
    0x0000d90f03530000L, // 15: 32KB
    0x0000d90303537000L, // 16: 64KB
    0x0000d90703537000L, // 17: 128KB
    0x0000d90707537000L, // 18: 256KB
    0x0000d91707537000L, // 19: 512KB
    0x0000d91747537000L, // 20: 1MB
    0x0000d91767537000L, // 21: 2MB
    0x0000d93767537000L, // 22: 4MB
    0x0000d93777537000L, // 23: 8MB
    0x0000d93777577000L, // 24: 16MB
    0x0000db3777577000L, // 25: unused except for NC 3
  };
  // @formatter:on

  // The Gear table used for the rolling hash, shared with the other chunking algorithms in this
  // package. See GearTable for how it is derived.
  private static final long[] GEAR = GearTable.GEAR;

  private static final long[] GEAR_LS = computeGearLs();

  private static long[] computeGearLs() {
    long[] gearLs = new long[GEAR.length];
    for (int i = 0; i < GEAR.length; i++) {
      gearLs[i] = GEAR[i] << 1;
    }
    return gearLs;
  }

  private final int minSize;
  private final int maxSize;
  private final int avgSize;
  private final long maskS;
  private final long maskL;
  private final long maskSLs;
  private final long maskLLs;
  private final long seed;
  private final long shiftedSeed;
  private final DigestUtil digestUtil;

  public FastCdcChunker(DigestUtil digestUtil) {
    this(FastCdcChunkingConfig.defaults(), digestUtil);
  }

  public FastCdcChunker(FastCdcChunkingConfig config, DigestUtil digestUtil) {
    this(
        config.minChunkSize(),
        config.avgChunkSize(),
        config.maxChunkSize(),
        config.normalizationLevel(),
        Integer.toUnsignedLong(config.seed()),
        digestUtil);
  }

  public FastCdcChunker(
      int minSize, int avgSize, int maxSize, int normalization, long seed, DigestUtil digestUtil) {
    checkArgument(minSize > 0, "minSize must be positive");
    checkArgument(avgSize >= minSize, "avgSize must be >= minSize");
    checkArgument(maxSize >= avgSize, "maxSize must be >= avgSize");
    checkArgument((avgSize & (avgSize - 1)) == 0, "avgSize must be a power of 2, got %s", avgSize);
    checkArgument(normalization >= 0 && normalization <= 3, "normalization must be 0-3");

    this.minSize = minSize;
    this.avgSize = avgSize;
    this.maxSize = maxSize;
    this.digestUtil = digestUtil;

    int bits = 31 - Integer.numberOfLeadingZeros(avgSize);
    int smallBits = bits + normalization;
    int largeBits = bits - normalization;
    checkArgument(smallBits <= 25 && largeBits >= 5, "normalization level too extreme for avgSize");

    this.maskS = MASKS[smallBits];
    this.maskL = MASKS[largeBits];
    this.maskSLs = this.maskS << 1;
    this.maskLLs = this.maskL << 1;

    this.seed = seed;
    this.shiftedSeed = seed << 1;
  }

  /** Finds the next chunk boundary. */
  private int cut(byte[] buf, int off, int len) {
    if (len <= minSize) {
      return len;
    }

    int n = Math.min(len, maxSize);
    int center = Math.min(n, avgSize);

    // Round down to even boundaries for 2-byte processing so we don't need to
    // divide by 2 in the loop.
    int minLimit = minSize & ~1;
    int centerLimit = center & ~1;
    int remainingLimit = n & ~1;

    long s = this.seed;
    long sLs = this.shiftedSeed;
    long hash = 0;

    // Below avgSize: use maskS to discourage early cuts (too small chunks)
    for (int a = minLimit; a < centerLimit; a += 2) {
      hash = (hash << 2) + (GEAR_LS[buf[off + a] & 0xFF] ^ sLs);
      if ((hash & maskSLs) == 0) {
        return a;
      }
      hash = hash + (GEAR[buf[off + a + 1] & 0xFF] ^ s);
      if ((hash & maskS) == 0) {
        return a + 1;
      }
    }

    // Above avgSize: use maskL to encourage cuts (too large chunks)
    for (int a = centerLimit; a < remainingLimit; a += 2) {
      hash = (hash << 2) + (GEAR_LS[buf[off + a] & 0xFF] ^ sLs);
      if ((hash & maskLLs) == 0) {
        return a;
      }
      hash = hash + (GEAR[buf[off + a + 1] & 0xFF] ^ s);
      if ((hash & maskL) == 0) {
        return a + 1;
      }
    }

    return n;
  }

  /**
   * Chunks a file and returns chunk digests.
   *
   * <p>This method is used for building MerkleTree entries for large files. It returns the content
   * digests in order for each chunk.
   *
   * <p>Note: We don't need the raw data here. We can read from the original file (seekable) when
   * uploading, similar to how whole blobs work.
   */
  @Override
  public ImmutableList<Digest> chunkToDigests(InputStream input) throws IOException {
    ImmutableList.Builder<Digest> digests = ImmutableList.builder();

    byte[] buf = new byte[maxSize * 2];
    int cursor = 0;
    int end = 0;
    boolean eof = false;

    while (true) {
      int available = end - cursor;
      if (available < maxSize && !eof) {
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

    return digests.build();
  }
}
