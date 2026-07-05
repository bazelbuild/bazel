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
import static com.google.common.base.Preconditions.checkState;

import build.bazel.remote.execution.v2.Digest;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * RepMaxCDC implementation for splitting large blobs.
 *
 * <p>This module implements the RepMaxCDC algorithm as specified by the reference implementation
 * in <a href="https://github.com/buildbarn/go-cdc">buildbarn/go-cdc</a>. RepMaxCDC expands upon
 * MaxCDC, in that it repeatedly applies the chunking process until all chunks are within {@code
 * [minChunkSize, 2 * minChunkSize)} in size.
 *
 * <p>Cutting points are selected where the Gear rolling hash (a 64-byte window over the input) is
 * strictly maximal within a lookahead window. The size of the lookahead window is controlled by
 * the horizon size. Unlike FastCDC's maximum chunk size, the horizon size only denotes the quality
 * of the chunking that is performed: setting it to zero leads to uniform chunks of {@code
 * minChunkSize}, while setting it to a positive value {@code n} means that an optimal cutting
 * point within offsets {@code [minChunkSize, minChunkSize + n]} will always be respected. The
 * horizon size can be increased freely without reducing quality, though with diminishing returns.
 *
 * <p>The advantage of RepMaxCDC over FastCDC is that the bounds on the chunk size are tight: for a
 * given input it is trivial to check whether it is already chunked, purely by looking at its size.
 * It has been observed that RepMaxCDC provides a rate of deduplication that is comparable to
 * FastCDC.
 *
 * <p>All chunks are within {@code [minChunkSize, 2 * minChunkSize)} in size, except for the last
 * chunk of the input, which may be smaller than {@code minChunkSize} only if the entire input is.
 *
 * <p>Chunk boundaries produced by this implementation are identical to those produced by the Go
 * reference implementation for the same parameters, which is required for clients and servers to
 * benefit from each other's chunk data.
 *
 * <p>Implementation notes: this is a port of the optimized implementation in buildbarn/go-cdc
 * ({@code rep_max_content_defined_chunker.go}), which hashes every input byte exactly once by
 * carrying cutting point candidates and rolling hash state across chunks. Because the Gear hash at
 * a given position only depends on the 64 bytes preceding it, hash values are independent of chunk
 * boundaries and can be reused. The naive version of the algorithm ({@code
 * NewSimpleRepMaxContentDefinedChunker} in go-cdc) rehashes the full lookahead window for every
 * chunk, which makes it several times slower than FastCDC.
 */
public final class RepMaxCdcChunker implements ContentDefinedChunker {

  private final int minSize;
  // The amount of data that must be available (unless the end of the input has been reached) to
  // determine the next chunk boundary: 2 * minSize + horizonSize. An optimal cutting point is
  // searched within the horizonSize bytes following the earliest possible cutting point at
  // minSize, and a further minSize bytes are left behind so that the final chunk of the input is
  // at least minSize in size as well.
  private final int peekSize;
  private final DigestUtil digestUtil;

  public RepMaxCdcChunker(DigestUtil digestUtil) {
    this(RepMaxCdcChunkingConfig.defaults(), digestUtil);
  }

  public RepMaxCdcChunker(RepMaxCdcChunkingConfig config, DigestUtil digestUtil) {
    this(config.minChunkSize(), config.horizonSize(), digestUtil);
  }

  /**
   * Creates a chunker producing chunks of {@code [minSize, 2 * minSize)} bytes.
   *
   * @param minSize the minimum chunk size in bytes; must be at least the Gear hash window size (64
   *     bytes)
   * @param horizonSize the lookahead window in bytes used to find optimal cutting points; must not
   *     be negative
   * @throws IllegalArgumentException if the parameters are out of range
   */
  public RepMaxCdcChunker(int minSize, int horizonSize, DigestUtil digestUtil) {
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

    ChunkingSession session = new ChunkingSession(input);
    while (true) {
      int chunkSize = session.nextChunkSize();
      if (chunkSize == 0) {
        break;
      }
      digests.add(digestUtil.compute(session.buffer(), session.chunkOffset(), chunkSize));
    }

    return digests;
  }

  /**
   * The chunking state for a single input stream.
   *
   * <p>This is a port of {@code repMaxContentDefinedChunker.ReadNextChunk()} from buildbarn/go-cdc
   * on top of a sliding read buffer, and mirrors its structure closely. Comments are largely taken
   * from the Go implementation.
   */
  private final class ChunkingSession {
    private final InputStream input;
    private final byte[] buf = new byte[peekSize * 2];
    private int bufStart = 0;
    private int bufEnd = 0;
    private boolean eof = false;

    // The size of the previous chunk returned by nextChunkSize(). This amount of data will be
    // discarded from the buffer at the start of the next call to nextChunkSize().
    private int previousChunkSize = 0;

    // List of chunks for which no future data can influence their length. For each chunk, its
    // size is stored. Chunks are stored in reverse order, so that they can be popped from the
    // end.
    private final IntList completeChunks = new IntList();

    // List of cutting points that will determine the length of future chunks. The hashes at the
    // positions of the cutting points in this list will be strictly monotonically increasing.
    //
    // Cutting points are addressed relative to the first eligible position at which they may be
    // placed (i.e., the end of the last complete chunk, plus the minimum chunk size). This means
    // that the first entry is always equal to zero.
    //
    // Even though this list can grow to become proportional to the size of the horizon, this is
    // highly unlikely. As we progress, it becomes increasingly harder to find even more
    // preferable cutting points within the minimum chunk size.
    private final IntList incompleteChunks = new IntList();

    // The rolling hash value corresponding to the position up to where input data has been
    // processed.
    private long currentHash;

    // The rolling hash value corresponding to the position of the last incomplete chunk. Any new
    // incomplete chunk must have a hash value that is higher than this one.
    private long bestHash;

    ChunkingSession(InputStream input) {
      this.input = input;
    }

    byte[] buffer() {
      return buf;
    }

    int chunkOffset() {
      return bufStart;
    }

    /**
     * Makes up to {@code n} bytes available at {@link #bufStart} and returns how many of them are
     * available, which is only less than {@code n} at the end of the input.
     */
    private int peek(int n) throws IOException {
      int available = bufEnd - bufStart;
      if (available < n && !eof) {
        if (bufStart > 0 && available > 0) {
          System.arraycopy(buf, bufStart, buf, 0, available);
        }
        bufStart = 0;
        bufEnd = available;

        while (bufEnd < buf.length) {
          int read = input.read(buf, bufEnd, buf.length - bufEnd);
          if (read == -1) {
            eof = true;
            break;
          }
          bufEnd += read;
        }
        available = bufEnd - bufStart;
      }
      return Math.min(n, available);
    }

    /**
     * Returns the size of the next chunk, whose data is available in {@link #buffer()} at {@link
     * #chunkOffset()}, or zero at the end of the input.
     */
    int nextChunkSize() throws IOException {
      // Discard data that was handed out by the previous call.
      bufStart += previousChunkSize;
      previousChunkSize = 0;

      // If the previous iteration yielded multiple chunks, we can return them without peeking the
      // full horizon.
      if (!completeChunks.isEmpty()) {
        int firstChunk = completeChunks.popLast();
        checkState(peek(firstChunk) == firstChunk, "complete chunk no longer buffered");
        previousChunkSize = firstChunk;
        return firstChunk;
      }

      // Gain access to the data corresponding to the next chunk(s). If we're reaching the end of
      // the input, either consume all data or leave at least minSize bytes behind. This ensures
      // that all chunks of the file are at least minSize in size, assuming the file is as well.
      int dLen = peek(peekSize);
      if (dLen < 2 * minSize) {
        if (dLen == 0) {
          return 0;
        }
        previousChunkSize = dLen;
        return dLen;
      }
      dLen -= minSize;

      long[] gear = GearTable.GEAR;
      int base = bufStart;

      // Extract the final incomplete chunk from the list, as it denotes where the previous call
      // stopped hashing the input. incompleteChunks takes over the role of the Go
      // implementation's oldChunks from here on.
      int currentChunk;
      long hash;
      long best;
      if (incompleteChunks.size() >= 2) {
        currentChunk = incompleteChunks.popLast();
        hash = currentHash;
        best = bestHash;
      } else {
        // This is the very first chunk. We know that the first minSize positions can't contain a
        // cut. Skip them.
        incompleteChunks.clear();
        incompleteChunks.add(0);
        hash = 0;
        for (int i = minSize - GearTable.GEAR_HASH_WINDOW_SIZE; i < minSize; i++) {
          hash = (hash << 1) + gear[buf[base + i] & 0xFF];
        }
        best = hash;
        currentChunk = 0;
      }

      // The position of the next byte to hash, relative to the start of the peeked data.
      int pos = minSize + currentChunk;
      while (true) {
        // Start hashing data where the previous call left off. Stop hashing before the distance
        // between two consecutive potential cutting points becomes minSize in size, as this
        // allows us to complete a chunk.
        int hashRegionLen = dLen - pos;
        int originalChunkCount = -1;
        int bytesBeforeMinChunkSize = incompleteChunks.last() + minSize - 1 - currentChunk;
        if (hashRegionLen > bytesBeforeMinChunkSize) {
          hashRegionLen = bytesBeforeMinChunkSize;
          originalChunkCount = incompleteChunks.size();
        } else if (hashRegionLen == 0) {
          break;
        }

        // Preserve all offsets at which the hash increases. Hash values are compared as unsigned
        // 64-bit integers, matching the Go reference implementation.
        //
        // The Gear hash recurrence is linear, so the hashes of the next four positions can all be
        // computed directly from the hash at the block start: h(i+k) = (hash << k) + s(k), where
        // s(k) is the Gear sum of the k block bytes. This shortens the serial dependency chain
        // from one shift+add per byte to one per four bytes; the rest is independent work. The
        // unroll factor is an empirical optimum: at two-way the dependency chain still dominates,
        // while eight-way measured slower than four-way due to register pressure.
        int idx = 0;
        for (int p = base + pos; idx + 4 <= hashRegionLen; idx += 4, p += 4) {
          long s1 = gear[buf[p] & 0xFF];
          long s2 = (s1 << 1) + gear[buf[p + 1] & 0xFF];
          long s3 = (s2 << 1) + gear[buf[p + 2] & 0xFF];
          long s4 = (s3 << 1) + gear[buf[p + 3] & 0xFF];
          long h1 = (hash << 1) + s1;
          long h2 = (hash << 2) + s2;
          long h3 = (hash << 3) + s3;
          long h4 = (hash << 4) + s4;
          hash = h4;
          if (Long.compareUnsigned(best, h1) < 0) {
            best = h1;
            incompleteChunks.add(currentChunk + idx + 1);
          }
          if (Long.compareUnsigned(best, h2) < 0) {
            best = h2;
            incompleteChunks.add(currentChunk + idx + 2);
          }
          if (Long.compareUnsigned(best, h3) < 0) {
            best = h3;
            incompleteChunks.add(currentChunk + idx + 3);
          }
          if (Long.compareUnsigned(best, h4) < 0) {
            best = h4;
            incompleteChunks.add(currentChunk + idx + 4);
          }
        }
        for (; idx < hashRegionLen; idx++) {
          hash = (hash << 1) + gear[buf[base + pos + idx] & 0xFF];
          if (Long.compareUnsigned(best, hash) < 0) {
            best = hash;
            incompleteChunks.add(currentChunk + idx + 1);
          }
        }

        if (incompleteChunks.size() == originalChunkCount) {
          // The loop above did not yield any new cutting points, and the next byte is minSize
          // away from the last cutting point. This means we can complete all chunks up to this
          // point.
          int previousCompleteChunkCount = completeChunks.size();
          int nextChunk = incompleteChunks.last();
          for (int i = incompleteChunks.size() - 3; nextChunk >= minSize; i--) {
            int chunk = incompleteChunks.get(i);
            if (nextChunk - chunk >= minSize) {
              completeChunks.add(nextChunk - chunk);
              nextChunk = chunk;
              i--;
            }
          }
          completeChunks.add(minSize + nextChunk);
          completeChunks.reverse(previousCompleteChunkCount, completeChunks.size());

          incompleteChunks.truncate(1);
          currentChunk = 0;
          hash = (hash << 1) + gear[buf[base + pos + hashRegionLen] & 0xFF];
          best = hash;
          pos += hashRegionLen + 1;
        } else {
          currentChunk += hashRegionLen;
          pos += hashRegionLen;
        }
      }

      // Processed the full horizon. Return the first chunk.
      incompleteChunks.add(currentChunk);
      int firstChunk;
      if (!completeChunks.isEmpty()) {
        completeChunks.reverse(0, completeChunks.size());
        firstChunk = completeChunks.popLast();
      } else {
        // The process above did not yield any complete chunks, either because we reached the end
        // of the file or the horizon size wasn't large enough.
        //
        // Ensure that we pick a cutting point respecting the maximum chunk size, that still
        // allows us to pick the most optimal cutting point in the horizon later on.
        int firstChunkIndex = incompleteChunks.size() - 2;
        for (int maxChunk = incompleteChunks.get(firstChunkIndex) - minSize,
                i = firstChunkIndex - 2;
            maxChunk >= 0;
            i--) {
          int chunk = incompleteChunks.get(i);
          if (chunk <= maxChunk) {
            firstChunkIndex = i;
            maxChunk = chunk - minSize;
            i--;
          }
        }
        firstChunk = minSize + incompleteChunks.get(firstChunkIndex);

        // There will be potential cutting points after the selected one that are no longer
        // eligible, as those would violate the minimum chunk size. These should be removed from
        // the list.
        int reusableChunkIndex = firstChunkIndex + 1;
        while (true) {
          int offsetInSecondChunk = incompleteChunks.get(reusableChunkIndex) - firstChunk;
          if (offsetInSecondChunk >= 0) {
            // This cutting point and the ones after it should be kept.
            for (int i = reusableChunkIndex; i < incompleteChunks.size(); i++) {
              incompleteChunks.set(i, incompleteChunks.get(i) - firstChunk);
            }

            if (offsetInSecondChunk == 0) {
              // There is no need to recompute any cutting points.
              incompleteChunks.removePrefix(reusableChunkIndex);
            } else {
              // Because the first cutting point to keep resides at an offset beyond the minimum
              // chunk size, we may have glossed over potential cutting points before it.
              // Recompute these.
              //
              // This should only happen rarely, especially if the horizon size is sufficiently
              // large.
              int regionStart = base + firstChunk;
              long recomputedHash = 0;
              for (int i = minSize - GearTable.GEAR_HASH_WINDOW_SIZE; i < minSize; i++) {
                recomputedHash = (recomputedHash << 1) + gear[buf[regionStart + i] & 0xFF];
              }
              incompleteChunks.set(0, 0);
              long bestRecomputedHash = recomputedHash;
              int recomputedChunkIndex = 1;
              int originalChunksCount = incompleteChunks.size();
              for (int i = 0; i < offsetInSecondChunk - 1; i++) {
                recomputedHash =
                    (recomputedHash << 1) + gear[buf[regionStart + minSize + i] & 0xFF];
                if (Long.compareUnsigned(bestRecomputedHash, recomputedHash) < 0) {
                  bestRecomputedHash = recomputedHash;
                  int recomputedChunk = i + 1;
                  if (recomputedChunkIndex < reusableChunkIndex) {
                    incompleteChunks.set(recomputedChunkIndex, recomputedChunk);
                    recomputedChunkIndex++;
                  } else {
                    incompleteChunks.add(recomputedChunk);
                  }
                }
              }
              if (recomputedChunkIndex < reusableChunkIndex) {
                // Recomputing yielded fewer cutting points than we had previously. Make the
                // cutting points contiguous again.
                incompleteChunks.removeRange(recomputedChunkIndex, reusableChunkIndex);
              } else if (incompleteChunks.size() > originalChunksCount) {
                // Recomputing yielded more cutting points than we had previously. The excess
                // cutting points were stored at the end. Rotate them into place, so that the
                // list remains sorted.
                incompleteChunks.reverse(reusableChunkIndex, originalChunksCount);
                incompleteChunks.reverse(originalChunksCount, incompleteChunks.size());
                incompleteChunks.reverse(reusableChunkIndex, incompleteChunks.size());
              }
            }
            break;
          }

          // The cutting point should be removed.
          reusableChunkIndex++;
          if (reusableChunkIndex == incompleteChunks.size()) {
            incompleteChunks.truncate(1);
            break;
          }
        }
      }
      previousChunkSize = firstChunk;
      currentHash = hash;
      bestHash = best;
      return firstChunk;
    }
  }

  /** A minimal growable list of ints, to avoid boxing on the chunking path. */
  private static final class IntList {
    private int[] elements = new int[32];
    private int size = 0;

    int size() {
      return size;
    }

    boolean isEmpty() {
      return size == 0;
    }

    int get(int index) {
      return elements[index];
    }

    void set(int index, int value) {
      elements[index] = value;
    }

    int last() {
      return elements[size - 1];
    }

    void add(int value) {
      if (size == elements.length) {
        int[] grown = new int[elements.length * 2];
        System.arraycopy(elements, 0, grown, 0, size);
        elements = grown;
      }
      elements[size++] = value;
    }

    int popLast() {
      return elements[--size];
    }

    void clear() {
      size = 0;
    }

    void truncate(int newSize) {
      size = newSize;
    }

    /** Removes the first {@code count} elements, shifting the rest to the front. */
    void removePrefix(int count) {
      System.arraycopy(elements, count, elements, 0, size - count);
      size -= count;
    }

    /** Removes the elements in {@code [from, to)}, shifting later elements down. */
    void removeRange(int from, int to) {
      System.arraycopy(elements, to, elements, from, size - to);
      size -= to - from;
    }

    /** Reverses the elements in {@code [from, to)}. */
    void reverse(int from, int to) {
      for (int i = from, j = to - 1; i < j; i++, j--) {
        int tmp = elements[i];
        elements[i] = elements[j];
        elements[j] = tmp;
      }
    }
  }
}
