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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.compress.CompressionService;
import com.google.devtools.build.lib.compress.CompressionServiceImpl;
import com.google.devtools.build.lib.skyframe.serialization.ChunkedValueSerialization.ChunkingResult;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ChunkedValueSerializationTest {

  private final CompressionService compressionService = new CompressionServiceImpl();
  private FingerprintValueService fingerprintValueService;

  @Before
  public void setUp() {
    fingerprintValueService = FingerprintValueService.createForTesting();
  }

  @After
  public void tearDown() {
    ChunkedValueSerialization.resetChunkSizeForTesting();
  }

  @Test
  public void maybeCompressAndChunk_smallData_returnsUncompressed() {
    byte[] smallData = new byte[] {1, 2, 3, 4, 5};
    List<WriteStatus> writeStatuses = new ArrayList<>();

    ChunkingResult result =
        ChunkedValueSerialization.maybeCompressAndChunk(
            smallData, compressionService, fingerprintValueService, writeStatuses::add);

    assertThat(result.isChunked()).isFalse();
    assertThat(result.serializedBytes()[0]).isEqualTo(ChunkedValueSerialization.UNCOMPRESSED);
    assertThat(Arrays.copyOfRange(result.serializedBytes(), 1, result.serializedBytes().length))
        .isEqualTo(smallData);
    assertThat(result.bytesToUpload()).isEqualTo(smallData.length + 1);
    assertThat(writeStatuses).isEmpty();
  }

  @Test
  public void maybeCompressAndChunk_largeData_chunksAndRoundTrips() throws Exception {
    int testChunkSize = 256;
    ChunkedValueSerialization.setChunkSizeForTesting(testChunkSize);

    byte[] originalData = new byte[2000];
    new Random(42).nextBytes(originalData);

    List<WriteStatus> writeStatuses = new ArrayList<>();
    ChunkingResult result =
        ChunkedValueSerialization.maybeCompressAndChunk(
            originalData, compressionService, fingerprintValueService, writeStatuses::add);

    assertThat(result.isChunked()).isTrue();
    assertThat(ChunkedValueSerialization.isChunked(result.serializedBytes())).isTrue();
    assertThat(writeStatuses).isNotEmpty();

    ImmutableList<PackedFingerprint> chunkFingerprints =
        ChunkedValueSerialization.parseChunkFingerprints(result.serializedBytes());
    // 2000 bytes / 256 bytes/chunk = 8 chunks rounded up.
    assertThat(chunkFingerprints).hasSize(8);
    assertThat(writeStatuses).hasSize(8);

    List<byte[]> chunks = new ArrayList<>();
    for (PackedFingerprint fp : chunkFingerprints) {
      chunks.add(fingerprintValueService.get(fp).get());
    }

    int chunksTotalBytes = 0;
    for (byte[] chunk : chunks) {
      assertThat(chunk.length).isAtMost(testChunkSize);
      chunksTotalBytes += chunk.length;
    }
    assertThat(result.bytesToUpload())
        .isEqualTo(chunksTotalBytes + result.serializedBytes().length);

    InputStream seqStream = ChunkedValueSerialization.toSequenceInputStream(chunks);
    ByteArrayOutputStream decompressedOut = new ByteArrayOutputStream();
    try (InputStream zstdIn = compressionService.newZstdInputStream(seqStream)) {
      zstdIn.transferTo(decompressedOut);
    }

    assertThat(decompressedOut.toByteArray()).isEqualTo(originalData);
  }

  @Test
  public void maybeCompressAndChunk_compressedNotChunked_returnsCompressedWithCorrectByteCount() {
    byte[] compressibleData = new byte[2000];
    // Repeated pattern is highly compressible (much smaller than 512KB chunk size)
    Arrays.fill(compressibleData, (byte) 42);
    List<WriteStatus> writeStatuses = new ArrayList<>();

    ChunkingResult result =
        ChunkedValueSerialization.maybeCompressAndChunk(
            compressibleData, compressionService, fingerprintValueService, writeStatuses::add);

    assertThat(result.isChunked()).isFalse();
    assertThat(result.serializedBytes()[0]).isEqualTo(ChunkedValueSerialization.ZSTD_COMPRESSED);
    assertThat(result.bytesToUpload()).isEqualTo(result.serializedBytes().length);
    assertThat(writeStatuses).isEmpty();
  }

  @Test
  public void isChunked_validatesHeader() {
    assertThat(ChunkedValueSerialization.isChunked(new byte[0])).isFalse();
    assertThat(ChunkedValueSerialization.isChunked(new byte[10])).isFalse();

    byte[] testChunkSizeData = new byte[1000];
    new Random(99).nextBytes(testChunkSizeData);
    List<WriteStatus> writeStatuses = new ArrayList<>();
    ChunkedValueSerialization.setChunkSizeForTesting(200);
    ChunkingResult result =
        ChunkedValueSerialization.maybeCompressAndChunk(
            testChunkSizeData, compressionService, fingerprintValueService, writeStatuses::add);

    byte[] validRefBlob = result.serializedBytes();
    assertThat(ChunkedValueSerialization.isChunked(validRefBlob)).isTrue();

    // Corrupt tag
    byte[] corruptedTag = validRefBlob.clone();
    corruptedTag[0] = 1;
    assertThat(ChunkedValueSerialization.isChunked(corruptedTag)).isFalse();
  }

  @Test
  public void parseChunkFingerprints_detectsCorruption() {
    byte[] testChunkSizeData = new byte[1000];
    new Random(99).nextBytes(testChunkSizeData);
    List<WriteStatus> writeStatuses = new ArrayList<>();
    ChunkedValueSerialization.setChunkSizeForTesting(200);
    ChunkingResult result =
        ChunkedValueSerialization.maybeCompressAndChunk(
            testChunkSizeData, compressionService, fingerprintValueService, writeStatuses::add);

    byte[] validRefBlob = result.serializedBytes();

    // Truncated ref blob: isChunked returns true because header matches, but
    // parseChunkFingerprints throws IOException.
    byte[] truncated = Arrays.copyOf(validRefBlob, validRefBlob.length - 1);
    assertThat(ChunkedValueSerialization.isChunked(truncated)).isTrue();
    assertThrows(
        IOException.class, () -> ChunkedValueSerialization.parseChunkFingerprints(truncated));

    // Appended extra byte: isChunked returns true, parseChunkFingerprints throws IOException.
    byte[] extra = Arrays.copyOf(validRefBlob, validRefBlob.length + 1);
    assertThat(ChunkedValueSerialization.isChunked(extra)).isTrue();
    assertThrows(IOException.class, () -> ChunkedValueSerialization.parseChunkFingerprints(extra));

    // Corrupted tag throws IOException in parseChunkFingerprints
    byte[] corruptedTag = validRefBlob.clone();
    corruptedTag[0] = 1;
    assertThrows(
        IOException.class, () -> ChunkedValueSerialization.parseChunkFingerprints(corruptedTag));
  }
}
