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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import build.bazel.remote.execution.v2.Digest;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.runfiles.Runfiles;
import com.google.common.hash.Hashing;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FastCDCChunker}. */
@RunWith(JUnit4.class)
public class FastCDCChunkerTest {
  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  @Test
  public void chunkToDigests_emptyInput_returnsEmptyList() throws IOException {
    FastCDCChunker chunker = new FastCDCChunker(DIGEST_UTIL);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(new byte[0]));

    assertThat(digests).isEmpty();
  }

  @Test
  public void chunkToDigests_smallInput_returnsSingleChunk() throws IOException {
    ChunkingConfig config = new ChunkingConfig(1024, 2, 0);
    FastCDCChunker chunker = new FastCDCChunker(config, DIGEST_UTIL);
    byte[] data = new byte[100];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests).hasSize(1);
    assertThat(digests.get(0).getSizeBytes()).isEqualTo(100);
  }

  @Test
  public void chunkToDigests_dataAtMinSize_returnsSingleChunk() throws IOException {
    ChunkingConfig config = new ChunkingConfig(1024, 2, 0);
    FastCDCChunker chunker = new FastCDCChunker(config, DIGEST_UTIL);
    byte[] data = new byte[config.minChunkSize()];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests).hasSize(1);
    assertThat(digests.get(0).getSizeBytes()).isEqualTo(config.minChunkSize());
  }

  @Test
  public void chunkToDigests_largeInput_producesMultipleChunks() throws IOException {
    ChunkingConfig config = new ChunkingConfig(1024, 2, 0);
    FastCDCChunker chunker = new FastCDCChunker(config, DIGEST_UTIL);
    byte[] data = new byte[config.maxChunkSize() * 3];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests.size()).isGreaterThan(1);
    long totalSize = digests.stream().mapToLong(Digest::getSizeBytes).sum();
    assertThat(totalSize).isEqualTo(data.length);
  }

  @Test
  public void chunkToDigests_sameInputProducesSameChunks() throws IOException {
    FastCDCChunker chunker = new FastCDCChunker(DIGEST_UTIL);
    byte[] data = new byte[2 * 1024 * 1024];
    new Random(123).nextBytes(data);

    List<Digest> digests1 = chunker.chunkToDigests(new ByteArrayInputStream(data));
    List<Digest> digests2 = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests1).isEqualTo(digests2);
  }

  @Test
  public void chunkToDigests_chunkSizesWithinBounds() throws IOException {
    ChunkingConfig config = new ChunkingConfig(1024, 2, 0);
    FastCDCChunker chunker = new FastCDCChunker(config, DIGEST_UTIL);
    byte[] data = new byte[config.maxChunkSize() * 10];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    for (int i = 0; i < digests.size() - 1; i++) {
      long size = digests.get(i).getSizeBytes();
      assertThat(size).isAtLeast(config.minChunkSize());
      assertThat(size).isAtMost(config.maxChunkSize());
    }
  }

  @Test
  public void chunkToDigests_lastChunkCanBeSmallerThanMin() throws IOException {
    ChunkingConfig config = new ChunkingConfig(1024, 2, 0);
    FastCDCChunker chunker = new FastCDCChunker(config, DIGEST_UTIL);
    int dataSize = config.maxChunkSize() + config.minChunkSize() / 2;
    byte[] data = new byte[dataSize];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests.size()).isAtLeast(1);
    long totalSize = digests.stream().mapToLong(Digest::getSizeBytes).sum();
    assertThat(totalSize).isEqualTo(dataSize);
  }

  @Test
  public void chunkToDigests_digestsAreCorrect() throws IOException {
    ChunkingConfig config = new ChunkingConfig(1024, 2, 0);
    FastCDCChunker chunker = new FastCDCChunker(config, DIGEST_UTIL);
    byte[] data = new byte[500];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests).hasSize(1);
    Digest expected = DIGEST_UTIL.compute(data);
    assertThat(digests.get(0)).isEqualTo(expected);
  }

  @Test
  public void constructor_invalidMinSize_throws() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new FastCDCChunker(0, 1024, 4096, 2, 0, DIGEST_UTIL));
  }

  @Test
  public void constructor_avgSizeLessThanMinSize_throws() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new FastCDCChunker(1024, 512, 4096, 2, 0, DIGEST_UTIL));
  }

  @Test
  public void constructor_maxSizeLessThanAvgSize_throws() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new FastCDCChunker(256, 1024, 512, 2, 0, DIGEST_UTIL));
  }

  @Test
  public void constructor_avgSizeNotPowerOfTwo_throws() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new FastCDCChunker(256, 1000, 4096, 2, 0, DIGEST_UTIL));
  }

  @Test
  public void constructor_invalidNormalization_throws() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new FastCDCChunker(256, 1024, 4096, 4, 0, DIGEST_UTIL));
  }

  @Test
  public void chunkToDigests_withDefaultConfig() throws IOException {
    FastCDCChunker chunker = new FastCDCChunker(DIGEST_UTIL);
    byte[] data = new byte[4 * 1024 * 1024];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests.size()).isGreaterThan(1);
    long totalSize = digests.stream().mapToLong(Digest::getSizeBytes).sum();
    assertThat(totalSize).isEqualTo(data.length);
  }

  @Test
  public void chunkToDigests_testVectorsSeed0() throws Exception {
    verifyTestVectors(0, new int[][] {
        {0,     19186},
        {19186, 19279},
        {38465, 17354},
        {55819, 16387},
        {72206, 19940},
        {92146, 17320},
    }, new String[] {
        "0f9efa589121d5d9e9e2c4ace91337d77cae866537143f6f15a0ffd525a77c2d",
        "c7c86a165573c16448cda35c9169742e85645af42be22889f8b96b8ee0ec7cb0",
        "bc88521e28a8b4479cdea5f75aa721a24f3a0a7d0be903aa6d505c574e51e89d",
        "4b8dac2652e4685c629d2bb1ae9d4448e676b86f2e67ca0b2fff3d9580184b79",
        "c0a7062da6f2386c28e086ee0cedd5732252741269838773cff1ddb05b2df6ed",
        "7fa5b12134dc75cd2ac8dc60d3a8f3c8d22f0ee9d4cf74a4aa937e2a0d2d79a5",
    });
  }

  @Test
  public void chunkToDigests_testVectorsSeed666() throws Exception {
    verifyTestVectors(666, new int[][] {
        {0,     17635},
        {17635, 17334},
        {34969, 19136},
        {54105, 17467},
        {71572, 23593},
        {95165, 14301},
    }, new String[] {
        "cb3a9d80a3569772d4ed331ca37ab0c862c759897b890fc1aac90a4f2ea3a407",
        "d758c6b7b0b7eef1e996f8ccd17de6c645360b03a26c35541e7581348ac08944",
        "24846aefd89e510594bae3e9d7d5ea5012067601512610fed126a3c57ba993f5",
        "efa785e1fefb49f190e665f72fd246c1442079874508c312196da1fb3040d00b",
        "a2f557bdd8d40d8faada963ad5f91ec54b10ccee7c5ae72754a65137592dc607",
        "e131100b4a7147ccad19dc63c4a2fac1f5d8b644e1373eeb6803825024234efc",
    });
  }

  // Test vectors from the Remote Execution API specification:
  // https://github.com/bazelbuild/remote-apis/blob/v2.12.0/build/bazel/remote/execution/v2/fastcdc2020_test_vectors.txt
  // Test image: "Akashita" by Toriyama Sekien (1712-1788), public domain.
  // Source: https://commons.wikimedia.org/wiki/File:SekienAkashita.jpg
  private void verifyTestVectors(long seed, int[][] expectedChunks, String[] expectedHashes)
      throws Exception {
    Runfiles runfiles = Runfiles.create();
    String rlocationPath = System.getenv("SEKIEN_AKASHITA_PATH");
    Path testVectorPath = Path.of(runfiles.rlocation(rlocationPath));
    byte[] fileData = Files.readAllBytes(testVectorPath);

    FastCDCChunker chunker = new FastCDCChunker(4096, 16384, 65535, 2, seed, DIGEST_UTIL);
    List<Digest> digests;
    try (InputStream input = new ByteArrayInputStream(fileData)) {
      digests = chunker.chunkToDigests(input);
    }

    assertThat(digests).hasSize(expectedChunks.length);

    List<int[]> actualChunks = new ArrayList<>();
    int offset = 0;
    for (Digest digest : digests) {
      actualChunks.add(new int[] {offset, (int) digest.getSizeBytes()});
      offset += digest.getSizeBytes();
    }

    for (int i = 0; i < expectedChunks.length; i++) {
      assertThat(actualChunks.get(i)[0]).isEqualTo(expectedChunks[i][0]);
      assertThat(actualChunks.get(i)[1]).isEqualTo(expectedChunks[i][1]);

      byte[] chunkData = new byte[expectedChunks[i][1]];
      System.arraycopy(fileData, expectedChunks[i][0], chunkData, 0, chunkData.length);
      String chunkHash = Hashing.sha256().hashBytes(chunkData).toString();
      assertThat(chunkHash).isEqualTo(expectedHashes[i]);
    }
  }
}
