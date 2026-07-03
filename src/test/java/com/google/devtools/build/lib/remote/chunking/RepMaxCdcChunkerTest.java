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
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.runfiles.Runfiles;
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

/** Tests for {@link RepMaxCdcChunker}. */
@RunWith(JUnit4.class)
public class RepMaxCdcChunkerTest {
  private static final String TEST_VECTOR_PATH =
      "io_bazel/src/test/java/com/google/devtools/build/lib/remote/chunking/testdata/SekienAkashita.jpg";

  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  @Test
  public void chunkToDigests_emptyInput_returnsEmptyList() throws IOException {
    RepMaxCdcChunker chunker = new RepMaxCdcChunker(DIGEST_UTIL);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(new byte[0]));

    assertThat(digests).isEmpty();
  }

  @Test
  public void chunkToDigests_smallInput_returnsSingleChunk() throws IOException {
    RepMaxCdcChunker chunker = new RepMaxCdcChunker(1024, 8 * 1024, DIGEST_UTIL);
    byte[] data = new byte[100];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests).hasSize(1);
    assertThat(digests.get(0).getSizeBytes()).isEqualTo(100);
  }

  @Test
  public void chunkToDigests_dataAtMinSize_returnsSingleChunk() throws IOException {
    RepMaxCdcChunker chunker = new RepMaxCdcChunker(1024, 8 * 1024, DIGEST_UTIL);
    byte[] data = new byte[1024];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests).hasSize(1);
    assertThat(digests.get(0).getSizeBytes()).isEqualTo(1024);
  }

  @Test
  public void chunkToDigests_largeInput_producesMultipleChunks() throws IOException {
    RepMaxCdcChunkingConfig config = new RepMaxCdcChunkingConfig(1024, 8 * 1024);
    RepMaxCdcChunker chunker = new RepMaxCdcChunker(config, DIGEST_UTIL);
    byte[] data = new byte[config.maxChunkSize() * 3];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests.size()).isGreaterThan(1);
    long totalSize = digests.stream().mapToLong(Digest::getSizeBytes).sum();
    assertThat(totalSize).isEqualTo(data.length);
  }

  @Test
  public void chunkToDigests_sameInputProducesSameChunks() throws IOException {
    RepMaxCdcChunker chunker = new RepMaxCdcChunker(DIGEST_UTIL);
    byte[] data = new byte[2 * 1024 * 1024];
    new Random(123).nextBytes(data);

    List<Digest> digests1 = chunker.chunkToDigests(new ByteArrayInputStream(data));
    List<Digest> digests2 = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests1).isEqualTo(digests2);
  }

  @Test
  public void chunkToDigests_chunkSizesWithinBounds() throws IOException {
    int minSize = 1024;
    RepMaxCdcChunker chunker = new RepMaxCdcChunker(minSize, 8 * 1024, DIGEST_UTIL);
    byte[] data = new byte[minSize * 50];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    // All chunks, including the last one, are within [minSize, 2 * minSize) because the input is
    // a multiple of minSize.
    for (Digest digest : digests) {
      long size = digest.getSizeBytes();
      assertThat(size).isAtLeast(minSize);
      assertThat(size).isLessThan(2L * minSize);
    }
    long totalSize = digests.stream().mapToLong(Digest::getSizeBytes).sum();
    assertThat(totalSize).isEqualTo(data.length);
  }

  @Test
  public void chunkToDigests_lastChunkIsAtLeastMinSize() throws IOException {
    int minSize = 1024;
    RepMaxCdcChunker chunker = new RepMaxCdcChunker(minSize, 8 * 1024, DIGEST_UTIL);
    // An input that is not a multiple of the minimum chunk size.
    int dataSize = minSize * 10 + minSize / 2;
    byte[] data = new byte[dataSize];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    // Unlike FastCDC, RepMaxCDC guarantees that all chunks of an input of at least minSize are at
    // least minSize in size, including the last one.
    for (Digest digest : digests) {
      long size = digest.getSizeBytes();
      assertThat(size).isAtLeast(minSize);
      assertThat(size).isLessThan(2L * minSize);
    }
    long totalSize = digests.stream().mapToLong(Digest::getSizeBytes).sum();
    assertThat(totalSize).isEqualTo(dataSize);
  }

  @Test
  public void chunkToDigests_zeroHorizon_producesUniformChunks() throws IOException {
    int minSize = 1024;
    RepMaxCdcChunker chunker = new RepMaxCdcChunker(minSize, 0, DIGEST_UTIL);
    byte[] data = new byte[minSize * 10];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    // A zero horizon degenerates into uniform chunking of minSize.
    assertThat(digests).hasSize(10);
    for (Digest digest : digests) {
      assertThat(digest.getSizeBytes()).isEqualTo(minSize);
    }
  }

  @Test
  public void chunkToDigests_digestsAreCorrect() throws IOException {
    RepMaxCdcChunker chunker = new RepMaxCdcChunker(1024, 8 * 1024, DIGEST_UTIL);
    byte[] data = new byte[500];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests).hasSize(1);
    Digest expected = DIGEST_UTIL.compute(data);
    assertThat(digests.get(0)).isEqualTo(expected);
  }

  @Test
  public void constructor_minSizeBelowGearHashWindow_throws() {
    assertThrows(
        IllegalArgumentException.class, () -> new RepMaxCdcChunker(63, 8 * 1024, DIGEST_UTIL));
  }

  @Test
  public void constructor_negativeHorizonSize_throws() {
    assertThrows(IllegalArgumentException.class, () -> new RepMaxCdcChunker(1024, -1, DIGEST_UTIL));
  }

  @Test
  public void constructor_peekSizeOverflows_throws() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new RepMaxCdcChunker(1024 * 1024, Integer.MAX_VALUE / 2, DIGEST_UTIL));
  }

  @Test
  public void chunkToDigests_withDefaultConfig() throws IOException {
    RepMaxCdcChunker chunker = new RepMaxCdcChunker(DIGEST_UTIL);
    byte[] data = new byte[4 * 1024 * 1024];
    new Random(42).nextBytes(data);

    List<Digest> digests = chunker.chunkToDigests(new ByteArrayInputStream(data));

    assertThat(digests.size()).isGreaterThan(1);
    long totalSize = digests.stream().mapToLong(Digest::getSizeBytes).sum();
    assertThat(totalSize).isEqualTo(data.length);
  }

  @Test
  public void chunkToDigests_testVectors() throws Exception {
    verifyTestVectors(
        /* minSize= */ 4096,
        /* horizonSize= */ 32768,
        new int[][] {
          {0, 7026},
          {7026, 5198},
          {12224, 5310},
          {17534, 4271},
          {21805, 5278},
          {27083, 5588},
          {32671, 5017},
          {37688, 6761},
          {44449, 5541},
          {49990, 7612},
          {57602, 5501},
          {63103, 4695},
          {67798, 4362},
          {72160, 6221},
          {78381, 7190},
          {85571, 5858},
          {91429, 4382},
          {95811, 8022},
          {103833, 5633},
        },
        new String[] {
          "3c386537b8c3200dd2ab6623f6096a576d07844a4a0d5d2994d13e24887dae46",
          "cc7fb08694b471e3ba660b700b347775d1e7a55db8f308c176e2eb6776b5222c",
          "096a3e0e42f0a8b7a5a3a337e9b5b4c7fc59abcd6f69486fb68fe4c5c156300f",
          "3f4a0b00f9a04959806c9ce9e530fa5d761930ef6c8477117b58eb93196342e8",
          "d0c2f82664d8887bd2d289a75baea1d5688e6ee586f8767149bfef9c6df8fe63",
          "558e6b58c659f7463c846a0d18c155a86520d1ca69102f102635c08e8b0897e2",
          "c3c66478eb8e1632768e6e13dc31e492ac78befca38e9db85525b4b274c5b5d7",
          "a5ce22ca3a02323451ddca6db5319c26b5fc1abc5621a033c98cd3de154cd29d",
          "ec4752962eda083bf78b24cb087e07451ced85d8c0d719f32c474f7dae8fd764",
          "3d9e8dc0a5f14dee7db5eabf99bb636352081423f122a84fdb160cecc2ad01bb",
          "f8c747a292f7fd6433945356ee314b978b423ac4c3d248c8fdf36c0693eb6006",
          "71249e1031dc237ab6393381a30a10ef90d9a93f532c135b11cc0df60a767b5f",
          "95c42ec307be0ce1efde6de814c668f05a31887fde9501b478faca225cf9b8c7",
          "8869f4871a0f3fb63775e0fb7f10e170a59b71a6fa0a91a18fa3cf5fff987eba",
          "d2f91644457863f716071e76e9b1046408820aa95b36471d5e70ee5c91b611a2",
          "3507024624db4ff48ee546e984d7c60461d6a526aba7e4aa33833b162785e79e",
          "9b19e98bc5d9b6f486c7c7281fd61044b8e5333363048b20ce3dcb2ee0172980",
          "e56dbfa2a82695fed3c565857dbc644af91c983b55c13fe20d9e03bb89e009e2",
          "7d76fde9af9a911c4e580db9227e5109c5de28101f3f26637c527e80215aad38",
        });
  }

  @Test
  public void chunkToDigests_testVectorsZeroHorizon() throws Exception {
    verifyTestVectors(
        /* minSize= */ 8192,
        /* horizonSize= */ 0,
        new int[][] {
          {0, 8192},
          {8192, 8192},
          {16384, 8192},
          {24576, 8192},
          {32768, 8192},
          {40960, 8192},
          {49152, 8192},
          {57344, 8192},
          {65536, 8192},
          {73728, 8192},
          {81920, 8192},
          {90112, 8192},
          {98304, 11162},
        },
        new String[] {
          "716c5e702f4bee5f18426da9e9eb77d22aa936486741768b86fff472bc18c363",
          "33df8556634b77338abee984190d1aa21efb10411edeae3820fd61a93e489f58",
          "a05018146657a5c999868a9e3f6e94da715c519f929005ca50404eb5a30caf1f",
          "e124b2b4e8847def80e1b8da2fbfa5b483c3bcab6cf2d88e50d598d6c2d2e65c",
          "7855d8367f097e33d0d3afbdba0e17749309a0dfb48f3d373bdf94b9b8f378e1",
          "782a8f7b5367f177b94da16dc14026a3af3dcfc25b6b53f2060eafee0bc16922",
          "3be28a5d9d67d5f8f9b30fbe742406cfefb46c139b48a47eeb98837fbb55128b",
          "e6d7615aeefa30776bd0de14ad0e9ac5a738a767c63aa8c3310aa85c49d16d44",
          "316e9ca063049a490d8f8431f109f2eaf0bf74739b4c3f4be9da46144733c327",
          "2b4a043bd77df955a9fa30358597a90bca046474917e91d3715205f9d674a3bf",
          "78abb22658ea5642440139f1bf86382d4bca2d33858fd8e15a170f1c1ac19651",
          "076758af278b0fb9fd0313db8f6b796639c23241e2c8d831503e272e069d9aea",
          "e776b8d90b880e10e4fdc4f99ba3b0bfbe471f26362007a1775d4cab46a539c7",
        });
  }

  // Test vectors generated with the Go reference implementation of RepMaxCDC:
  // https://github.com/buildbarn/go-cdc
  // Test image: "Akashita" by Toriyama Sekien (1712-1788), public domain.
  // Source: https://commons.wikimedia.org/wiki/File:SekienAkashita.jpg
  private void verifyTestVectors(
      int minSize, int horizonSize, int[][] expectedChunks, String[] expectedHashes)
      throws Exception {
    Path testVectorPath =
        Path.of(Runfiles.preload().withSourceRepository("").rlocation(TEST_VECTOR_PATH));
    byte[] fileData = Files.readAllBytes(testVectorPath);

    RepMaxCdcChunker chunker = new RepMaxCdcChunker(minSize, horizonSize, DIGEST_UTIL);
    List<Digest> digests;
    try (InputStream input = new ByteArrayInputStream(fileData)) {
      digests = chunker.chunkToDigests(input);
    }

    assertThat(digests).hasSize(expectedChunks.length);

    List<int[]> actualChunks = new ArrayList<>();
    int offset = 0;
    for (Digest digest : digests) {
      actualChunks.add(new int[] {offset, (int) digest.getSizeBytes()});
      offset += (int) digest.getSizeBytes();
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
