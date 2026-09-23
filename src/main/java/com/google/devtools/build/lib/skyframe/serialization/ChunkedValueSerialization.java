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
package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.compress.CompressionService;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.SequenceInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * Utilities for splitting large serialized values into chunks and reassembling them.
 *
 * <p>When a serialized byte stream exceeds the chunk size (default 1MB), it is compressed with
 * Zstandard, partitioned into chunks of at most that size, and each chunk is uploaded independently
 * to the {@link FingerprintValueService} under its {@link PackedFingerprint}. A compact "reference
 * blob" is stored in place of the original data, containing the chunk count and the fingerprints of
 * each chunk in order.
 */
public final class ChunkedValueSerialization {

  /** Signal byte indicating uncompressed payload follows. */
  public static final byte UNCOMPRESSED = 0;

  /** Signal byte indicating standard zstd-compressed stream follows. */
  public static final byte ZSTD_COMPRESSED = 1;

  /** Signal byte indicating a chunked reference blob follows. */
  public static final byte ZSTD_COMPRESSED_AND_CHUNKED = 2;

  /** Default chunk size */
  public static final int DEFAULT_CHUNK_SIZE = 512 * 1024;

  private static final AtomicInteger chunkSize = new AtomicInteger(DEFAULT_CHUNK_SIZE);

  @VisibleForTesting
  public static void setChunkSizeForTesting(int size) {
    chunkSize.set(size);
  }

  @VisibleForTesting
  public static void resetChunkSizeForTesting() {
    chunkSize.set(DEFAULT_CHUNK_SIZE);
  }

  public static int getChunkSize() {
    return chunkSize.get();
  }

  /** Result of maybe compressing and chunking a byte payload. */
  @SuppressWarnings("ArrayRecordComponent")
  public record ChunkingResult(byte[] serializedBytes, int bytesToUpload) {
    public boolean isChunked() {
      return ChunkedValueSerialization.isChunked(serializedBytes);
    }
  }

  /**
   * Compresses and chunks bytes for shared values if threshold is exceeded.
   *
   * <p>Returns {@code [0, bytes...]} if uncompressed, {@code [1, compressed...]} if compressed and
   * fits in one chunk, or {@code [2, referenceBlob...]} if chunked.
   */
  public static ChunkingResult maybeCompressAndChunk(
      byte[] uncompressedBytes,
      CompressionService compressionService,
      FingerprintValueService fingerprintValueService,
      Consumer<WriteStatus> writeStatusSink) {
    int currentChunkSize = getChunkSize();
    int compressionThreshold =
        Math.min(SharedValueSerializationContext.COMPRESSION_THRESHOLD_IN_BYTES, currentChunkSize);

    if (uncompressedBytes.length > compressionThreshold) {
      ByteArrayOutputStream compressedOut = new ByteArrayOutputStream();
      byte[] compressed = null;
      try {
        try (OutputStream zstdOut = compressionService.newZstdOutputStream(compressedOut)) {
          zstdOut.write(uncompressedBytes);
        }
        compressed = compressedOut.toByteArray();
      } catch (IOException e) {
        BugReporter.defaultInstance().sendBugReport(e);
        // Fall back to uncompressed.
        compressed = null;
      }
      if (compressed != null) {
        if (compressed.length > currentChunkSize) {
          return chunkCompressedBytes(
              compressed, currentChunkSize, fingerprintValueService, writeStatusSink);
        }
        byte[] result = new byte[compressed.length + 1];
        result[0] = ZSTD_COMPRESSED;
        System.arraycopy(compressed, 0, result, 1, compressed.length);
        return new ChunkingResult(result, /* bytesToUpload= */ result.length);
      }
    }

    byte[] result = new byte[uncompressedBytes.length + 1];
    result[0] = UNCOMPRESSED;
    System.arraycopy(uncompressedBytes, 0, result, 1, uncompressedBytes.length);
    return new ChunkingResult(result, /* bytesToUpload= */ result.length);
  }

  private static ChunkingResult chunkCompressedBytes(
      byte[] compressed,
      int currentChunkSize,
      FingerprintValueService fingerprintValueService,
      Consumer<WriteStatus> writeStatusSink) {
    int numChunks = Math.max(1, (compressed.length + currentChunkSize - 1) / currentChunkSize);
    ImmutableList.Builder<PackedFingerprint> chunkFingerprints =
        ImmutableList.builderWithExpectedSize(numChunks);

    for (int i = 0; i < numChunks; i++) {
      int start = i * currentChunkSize;
      int end = Math.min(start + currentChunkSize, compressed.length);
      byte[] chunk = Arrays.copyOfRange(compressed, start, end);
      PackedFingerprint chunkFp = fingerprintValueService.fingerprint(chunk);
      chunkFingerprints.add(chunkFp);
      WriteStatus ws = fingerprintValueService.put(chunkFp, chunk);
      writeStatusSink.accept(ws);
    }

    ByteArrayOutputStream refBlobOut = new ByteArrayOutputStream();
    refBlobOut.write(ZSTD_COMPRESSED_AND_CHUNKED);
    CodedOutputStream codedOut = CodedOutputStream.newInstance(refBlobOut);
    try {
      codedOut.writeInt32NoTag(numChunks);
      for (PackedFingerprint fp : chunkFingerprints.build()) {
        fp.writeTo(codedOut);
      }
      codedOut.flush();
    } catch (IOException e) {
      throw new AssertionError("ByteArrayOutputStream should not throw IOException", e);
    }
    byte[] refBlob = refBlobOut.toByteArray();
    return new ChunkingResult(refBlob, /* bytesToUpload= */ compressed.length + refBlob.length);
  }

  /** Returns true if {@code bytes} represents a chunked reference blob. */
  public static boolean isChunked(byte[] bytes) {
    return bytes.length > 0 && bytes[0] == ZSTD_COMPRESSED_AND_CHUNKED;
  }

  /** Parses the chunk fingerprints from a reference blob. */
  public static ImmutableList<PackedFingerprint> parseChunkFingerprints(byte[] bytes)
      throws IOException {
    if (bytes == null || bytes.length < 1 + 1 + PackedFingerprint.BYTES) {
      throw new IOException(
          "Corrupted chunked reference blob: too short ("
              + (bytes == null ? 0 : bytes.length)
              + " bytes)");
    }
    if (bytes[0] != ZSTD_COMPRESSED_AND_CHUNKED) {
      throw new IOException("Corrupted chunked reference blob: invalid header byte " + bytes[0]);
    }
    int offset = 1;
    CodedInputStream codedIn = CodedInputStream.newInstance(bytes, offset, bytes.length - offset);
    int numChunks = codedIn.readInt32();
    // 200 is an arbitrary limit. Hopefully no one will have a 100MB object to be serialized.
    if (numChunks < 1 || numChunks > 200) {
      throw new IOException("Corrupted chunked reference blob: invalid chunk count " + numChunks);
    }
    int expectedLength =
        1
            + CodedOutputStream.computeInt32SizeNoTag(numChunks)
            + numChunks * PackedFingerprint.BYTES;
    if (bytes.length != expectedLength) {
      throw new IOException(
          String.format(
              "Corrupted chunked reference blob: length mismatch (expected %d bytes, got %d)",
              expectedLength, bytes.length));
    }
    ImmutableList.Builder<PackedFingerprint> fps = ImmutableList.builderWithExpectedSize(numChunks);
    for (int i = 0; i < numChunks; i++) {
      fps.add(PackedFingerprint.readFrom(codedIn));
    }
    return fps.build();
  }

  /** Assembles a list of byte chunks into a contiguous {@link InputStream}. */
  public static InputStream toSequenceInputStream(List<byte[]> chunks) {
    List<InputStream> chunkStreams = new ArrayList<>(chunks.size());
    for (byte[] chunk : chunks) {
      chunkStreams.add(new ByteArrayInputStream(chunk));
    }
    return new SequenceInputStream(Collections.enumeration(chunkStreams));
  }

  private ChunkedValueSerialization() {}
}
