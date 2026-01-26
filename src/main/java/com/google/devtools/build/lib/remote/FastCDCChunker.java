// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

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
public final class FastCDCChunker {

  /**
   * Reference to a chunk within a file. Stores position info, not data.
   *
   * <p>When uploading, we seek to {@code offset} in the parent file and read {@code length} bytes.
   */
  public record ChunkRef(Digest digest, long offset, int length) {}

  // Default values - prefer using ChunkingConfig from server capabilities
  public static final int CHUNKING_AVERAGE_SIZE = ChunkingConfig.DEFAULT_AVG_CHUNK_SIZE;
  public static final int CHUNKING_MIN_SIZE = CHUNKING_AVERAGE_SIZE / 4;
  public static final int CHUNKING_MAX_SIZE = CHUNKING_AVERAGE_SIZE * 4;
  public static final long CHUNKING_THRESHOLD = ChunkingConfig.DEFAULT_CHUNKING_THRESHOLD;
  public static final int CHUNKING_NORMALIZATION_LEVEL = ChunkingConfig.DEFAULT_NORMALIZATION_LEVEL;
  public static final int CHUNKING_SEED = ChunkingConfig.DEFAULT_SEED;

  // Masks for each of the desired number of bits, where 0 through 5 are unused. The values for
  // sizes 64 bytes through 128 kilo-bytes come from the C reference implementation (found in the
  // destor repository) while the extra values come from the restic-FastCDC repository. The FastCDC
  // paper claims that the deduplication ratio is slightly improved when the mask bits are spread
  // relatively evenly, hence these seemingly "magic" values.
  // @formatter:off
  private static final long[] MASKS = {
    0,                  // 0: padding
    0,                  // 1: padding
    0,                  // 2: padding
    0,                  // 3: padding
    0,                  // 4: padding
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

  // GEAR contains seemingly random numbers which are created by computing the MD5 digest of values
  // from 0 to 255, using only the high 8 bytes of the 16-byte digest. This is the "gear hash"
  // referred to in the FastCDC paper.
  private static final long[] GEAR = {
    0x3b5d3c7d207e37dcL, 0x784d68ba91123086L, 0xcd52880f882e7298L, 0xeacf8e4e19fdcca7L,
    0xc31f385dfbd1632bL, 0x1d5f27001e25abe6L, 0x83130bde3c9ad991L, 0xc4b225676e9b7649L,
    0xaa329b29e08eb499L, 0xb67fcbd21e577d58L, 0x0027baaada2acf6bL, 0xe3ef2d5ac73c2226L,
    0x0890f24d6ed312b7L, 0xa809e036851d7c7eL, 0xf0a6fe5e0013d81bL, 0x1d026304452cec14L,
    0x03864632648e248fL, 0xcdaacf3dcd92b9b4L, 0xf5e012e63c187856L, 0x8862f9d3821c00b6L,
    0xa82f7338750f6f8aL, 0x1e583dc6c1cb0b6fL, 0x7a3145b69743a7f1L, 0xabb20fee404807ebL,
    0xb14b3cfe07b83a5dL, 0xb9dc27898adb9a0fL, 0x3703f5e91baa62beL, 0xcf0bb866815f7d98L,
    0x3d9867c41ea9dcd3L, 0x1be1fa65442bf22cL, 0x14300da4c55631d9L, 0xe698e9cbc6545c99L,
    0x4763107ec64e92a5L, 0xc65821fc65696a24L, 0x76196c064822f0b7L, 0x485be841f3525e01L,
    0xf652bc9c85974ff5L, 0xcad8352face9e3e9L, 0x2a6ed1dceb35e98eL, 0xc6f483badc11680fL,
    0x3cfd8c17e9cf12f1L, 0x89b83c5e2ea56471L, 0xae665cfd24e392a9L, 0xec33c4e504cb8915L,
    0x3fb9b15fc9fe7451L, 0xd7fd1fd1945f2195L, 0x31ade0853443efd8L, 0x255efc9863e1e2d2L,
    0x10eab6008d5642cfL, 0x46f04863257ac804L, 0xa52dc42a789a27d3L, 0xdaaadf9ce77af565L,
    0x6b479cd53d87febbL, 0x6309e2d3f93db72fL, 0xc5738ffbaa1ff9d6L, 0x6bd57f3f25af7968L,
    0x67605486d90d0a4aL, 0xe14d0b9663bfbdaeL, 0xb7bbd8d816eb0414L, 0xdef8a4f16b35a116L,
    0xe7932d85aaaffed6L, 0x08161cbae90cfd48L, 0x855507beb294f08bL, 0x91234ea6ffd399b2L,
    0xad70cf4b2435f302L, 0xd289a97565bc2d27L, 0x8e558437ffca99deL, 0x96d2704b7115c040L,
    0x0889bbcdfc660e41L, 0x5e0d4e67dc92128dL, 0x72a9f8917063ed97L, 0x438b69d409e016e3L,
    0xdf4fed8a5d8a4397L, 0x00f41dcf41d403f7L, 0x4814eb038e52603fL, 0x9dafbacc58e2d651L,
    0xfe2f458e4be170afL, 0x4457ec414df6a940L, 0x06e62f1451123314L, 0xbd1014d173ba92ccL,
    0xdef318e25ed57760L, 0x9fea0de9dfca8525L, 0x459de1e76c20624bL, 0xaeec189617e2d666L,
    0x126a2c06ab5a83cbL, 0xb1321532360f6132L, 0x65421503dbb40123L, 0x2d67c287ea089ab3L,
    0x6c93bff5a56bd6b6L, 0x4ffb2036cab6d98dL, 0xce7b785b1be7ad4fL, 0xedb42ef6189fd163L,
    0xdc905288703988f6L, 0x365f9c1d2c691884L, 0xc640583680d99bfeL, 0x3cd4624c07593ec6L,
    0x7f1ea8d85d7c5805L, 0x014842d480b57149L, 0x0b649bcb5a828688L, 0xbcd5708ed79b18f0L,
    0xe987c862fbd2f2f0L, 0x982731671f0cd82cL, 0xbaf13e8b16d8c063L, 0x8ea3109cbd951bbaL,
    0xd141045bfb385cadL, 0x2acbc1a0af1f7d30L, 0xe6444d89df03bfdfL, 0xa18cc771b8188ff9L,
    0x9834429db01c39bbL, 0x214add07fe086a1fL, 0x8f07c19b1f6b3ff9L, 0x56a297b1bf4ffe55L,
    0x94d558e493c54fc7L, 0x40bfc24c764552cbL, 0x931a706f8a8520cbL, 0x32229d322935bd52L,
    0x2560d0f5dc4fefafL, 0x9dbcc48355969bb6L, 0x0fd81c3985c0b56aL, 0xe03817e1560f2bdaL,
    0xc1bb4f81d892b2d5L, 0xb0c4864f4e28d2d7L, 0x3ecc49f9d9d6c263L, 0x51307e99b52ba65eL,
    0x8af2b688da84a752L, 0xf5d72523b91b20b6L, 0x6d95ff1ff4634806L, 0x562f21555458339aL,
    0xc0ce47f889336346L, 0x487823e5089b40d8L, 0xe4727c7ebc6d9592L, 0x5a8f7277e94970baL,
    0xfca2f406b1c8bb50L, 0x5b1f8a95f1791070L, 0xd304af9fc9028605L, 0x5440ab7fc930e748L,
    0x312d25fbca2ab5a1L, 0x10f4a4b234a4d575L, 0x90301d55047e7473L, 0x3b6372886c61591eL,
    0x293402b77c444e06L, 0x451f34a4d3e97dd7L, 0x3158d814d81bc57bL, 0x034942425b9bda69L,
    0xe2032ff9e532d9bbL, 0x62ae066b8b2179e5L, 0x9545e10c2f8d71d8L, 0x7ff7483eb2d23fc0L,
    0x00945fcebdc98d86L, 0x8764bbbe99b26ca2L, 0x1b1ec62284c0bfc3L, 0x58e0fcc4f0aa362bL,
    0x5f4abefa878d458dL, 0xfd74ac2f9607c519L, 0xa4e3fb37df8cbfa9L, 0xbf697e43cac574e5L,
    0x86f14a3f68f4cd53L, 0x24a23d076f1ce522L, 0xe725cd8048868cc8L, 0xbf3c729eb2464362L,
    0xd8f6cd57b3cc1ed8L, 0x6329e52425541577L, 0x62aa688ad5ae1ac0L, 0x0a242566269bf845L,
    0x168b1a4753aca74bL, 0xf789afefff2e7e3cL, 0x6c3362093b6fccdbL, 0x4ce8f50bd28c09b2L,
    0x006a2db95ae8aa93L, 0x975b0d623c3d1a8cL, 0x18605d3935338c5bL, 0x5bb6f6136cad3c71L,
    0x0f53a20701f8d8a6L, 0xab8c5ad2e7e93c67L, 0x40b5ac5127acaa29L, 0x8c7bf63c2075895fL,
    0x78bd9f7e014a805cL, 0xb2c9e9f4f9c8c032L, 0xefd6049827eb91f3L, 0x2be459f482c16fbdL,
    0xd92ce0c5745aaa8cL, 0x0aaa8fb298d965b9L, 0x2b37f92c6c803b15L, 0x8c54a5e94e0f0e78L,
    0x95f9b6e90c0a3032L, 0xe7939faa436c7874L, 0xd16bfe8f6a8a40c9L, 0x44982b86263fd2faL,
    0xe285fb39f984e583L, 0x779a8df72d7619d3L, 0xf2d79a8de8d5dd1eL, 0xd1037354d66684e2L,
    0x004c82a4e668a8e5L, 0x31d40a7668b044e6L, 0xd70578538bd02c11L, 0xdb45431078c5f482L,
    0x977121bb7f6a51adL, 0x73d5ccbd34eff8ddL, 0xe437a07d356e17cdL, 0x47b2782043c95627L,
    0x9fb251413e41d49aL, 0xccd70b60652513d3L, 0x1c95b31e8a1b49b2L, 0xcae73dfd1bcb4c1bL,
    0x34d98331b1f5b70fL, 0x784e39f22338d92fL, 0x18613d4a064df420L, 0xf1d8dae25f0bcebeL,
    0x33f77c15ae855efcL, 0x3c88b3b912eb109cL, 0x956a2ec96bafeea5L, 0x1aa005b5e0ad0e87L,
    0x5500d70527c4bb8eL, 0xe36c57196421cc44L, 0x13c4d286cc36ee39L, 0x5654a23d818b2a81L,
    0x77b1dc13d161abdcL, 0x734f44de5f8d5eb5L, 0x60717e174a6c89a2L, 0xd47d9649266a211eL,
    0x5b13a4322bb69e90L, 0xf7669609f8b5fc3cL, 0x21e6ac55bedcdac9L, 0x9b56b62b61166deaL,
    0xf48f66b939797e9cL, 0x35f332f9c0e6ae9aL, 0xcc733f6a9a878db0L, 0x3da161e41cc108c2L,
    0xb7d74ae535914d51L, 0x4d493b0b11d36469L, 0xce264d1dfba9741aL, 0xa9d1f2dc7436dc06L,
    0x70738016604c2a27L, 0x231d36e96e93f3d5L, 0x7666881197838d19L, 0x4a2a83090aaad40cL,
    0xf1e761591668b35dL, 0x7363236497f730a7L, 0x301080e37379dd4dL, 0x502dea2971827042L,
    0xc2c5eb858f32625fL, 0x786afb9edfafbdffL, 0xdaee0d868490b2a4L, 0x617366b3268609f6L,
    0xae0e35a0fe46173eL, 0xd1a07de93e824f11L, 0x079b8b115ea4cca8L, 0x93a99274558faebbL,
    0xfb1e6e22e08a03b3L, 0xea635fdba3698dd0L, 0xcf53659328503a5cL, 0xcde3b31e6fd5d780L,
    0x8e3e4221d3614413L, 0xef14d0d86bf1a22cL, 0xe1d830d3f16c5ddbL, 0xaabd2b2a451504e1L,
  };
  // @formatter:on

  private final long[] GEAR_LS;
  private final int minSize;
  private final int maxSize;
  private final int avgSize;
  private final long maskS;
  private final long maskL;
  private final long maskSLs;
  private final long maskLLs;
  private final DigestUtil digestUtil;

  public FastCDCChunker(DigestUtil digestUtil) {
    this(ChunkingConfig.defaults(), digestUtil);
  }

  public FastCDCChunker(ChunkingConfig config, DigestUtil digestUtil) {
    this(config.minChunkSize(), config.avgChunkSize(), config.maxChunkSize(),
        config.normalizationLevel(), digestUtil);
  }

  public FastCDCChunker(int minSize, int avgSize, int maxSize, DigestUtil digestUtil) {
    this(minSize, avgSize, maxSize, CHUNKING_NORMALIZATION_LEVEL, digestUtil);
  }

  public FastCDCChunker(
      int minSize, int avgSize, int maxSize, int normalization, DigestUtil digestUtil) {
    checkArgument(minSize > 0, "minSize must be positive");
    checkArgument(avgSize >= minSize, "avgSize must be >= minSize");
    checkArgument(maxSize >= avgSize, "maxSize must be >= avgSize");
    checkArgument(normalization >= 0 && normalization <= 3, "normalization must be 0-3");

    this.minSize = minSize;
    this.avgSize = avgSize;
    this.maxSize = maxSize;
    this.digestUtil = digestUtil;

    this.GEAR_LS = new long[GEAR.length];
    for (int i = 0; i < GEAR.length; i++) {
      this.GEAR_LS[i] = this.GEAR[i] << 1;
    }

    int bits = (int) Math.round(Math.log(avgSize) / Math.log(2));
    this.maskS = MASKS[bits + normalization];
    this.maskL = MASKS[bits - normalization];
    this.maskSLs = this.maskS << 1;
    this.maskLLs = this.maskL << 1;
  }

  /**
   * Finds the next chunk boundary.
   */
  private int cut(byte[] buf, int off, int len) {
    if (len <= minSize) {
      return len;
    }

    int n = Math.min(len, maxSize);
    int center = Math.min(n, avgSize);

    // Round down to even boundaries for 2-byte processing so we don't need to divide by 2 in
    // the loop.
    int minLimit = minSize & ~1;
    int centerLimit = center & ~1;
    int remainingLimit = n & ~1;

    long hash = 0;

    // Below avgSize: use maskS to discourage early cuts (too small chunks)
    for (int a = minLimit; a < centerLimit; a += 2) {
      hash = (hash << 2) + GEAR_LS[buf[off + a] & 0xFF];
      if ((hash & maskSLs) == 0) {
        return a;
      }
      hash = hash + GEAR[buf[off + a + 1] & 0xFF];
      if ((hash & maskS) == 0) {
        return a + 1;
      }
    }

    // Above avgSize: use maskL to encourage cuts (too large chunks)
    for (int a = centerLimit; a < remainingLimit; a += 2) {
      hash = (hash << 2) + GEAR_LS[buf[off + a] & 0xFF];
      if ((hash & maskLLs) == 0) {
        return a;
      }
      hash = hash + GEAR[buf[off + a + 1] & 0xFF];
      if ((hash & maskL) == 0) {
        return a + 1;
      }
    }

    return n;
  }

  /**
   * Chunks a file and returns metadata (digest, offset, length) for each chunk.
   *
   * <p>This method is used for building MerkleTree entries for large files. It returns ChunkRef
   * objects that store the file offset and length of each chunk, allowing the chunk data to be
   * read later by seeking to the offset.
   *
   * <p>Note: We don't need the raw data here. We can read from the original file (seekable) when
   * uploading, similar to how whole blobs work.
   *
   * @param input the input stream to chunk (should be from a seekable file)
   * @return list of ChunkRefs with digest, file offset, and length for each chunk
   */
  public ImmutableList<ChunkRef> chunkToRefs(InputStream input) throws IOException {
    ImmutableList.Builder<ChunkRef> refs = ImmutableList.builder();
    long fileOffset = 0;

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
      Digest digest = digestUtil.compute(buf, cursor, chunkLen);
      refs.add(new ChunkRef(digest, fileOffset, chunkLen));

      cursor += chunkLen;
      fileOffset += chunkLen;
    }

    return refs.build();
  }
}
