// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs.bazel;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.hash.Hasher;
import com.google.devtools.build.runfiles.Runfiles;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Blake3MessageDigest}. */
@RunWith(JUnit4.class)
public class Blake3HasherTest {
  /**
   * A case from the official BLAKE3 test vectors, whose input of length n consists of the bytes i %
   * 251 for i in [0, n).
   */
  private record TestVector(int inputLength, String hash) {
    byte[] input() {
      var input = new byte[inputLength];
      for (int i = 0; i < inputLength; i++) {
        input[i] = (byte) (i % 251);
      }
      return input;
    }
  }

  private static ImmutableList<TestVector> testVectors;

  @BeforeClass
  public static void loadTestVectors() throws IOException {
    String path =
        Runfiles.preload().unmapped().rlocation(System.getProperty("blake3.test_vectors"));
    JsonObject json =
        JsonParser.parseString(Files.readString(Path.of(path), UTF_8)).getAsJsonObject();
    var builder = ImmutableList.<TestVector>builder();
    for (JsonElement testCase : json.getAsJsonArray("cases")) {
      var testCaseObject = testCase.getAsJsonObject();
      // The test vectors contain 131 bytes of extended output, which start with the 32-byte hash.
      builder.add(
          new TestVector(
              testCaseObject.get("input_len").getAsInt(),
              testCaseObject.get("hash").getAsString().substring(0, 64)));
    }
    testVectors = builder.build();
    assertThat(testVectors).isNotEmpty();
  }

  @Test
  public void testVectors_hashBytes() {
    for (TestVector testVector : testVectors) {
      assertWithMessage("input length %s", testVector.inputLength())
          .that(Blake3HashFunction.INSTANCE.hashBytes(testVector.input()).toString())
          .isEqualTo(testVector.hash());
    }
  }

  @Test
  public void testVectors_hashBytesWithOffset() {
    for (TestVector testVector : testVectors) {
      int length = testVector.inputLength();
      var input = new byte[length + 10];
      System.arraycopy(testVector.input(), 0, input, 3, length);

      assertWithMessage("input length %s", length)
          .that(Blake3HashFunction.INSTANCE.hashBytes(input, 3, length).toString())
          .isEqualTo(testVector.hash());
    }
  }

  @Test
  public void testVectors_hasherWithUnevenUpdatesOnRandomThreads() throws Exception {
    // Hashers may be used on other threads than the one that created them, e.g. when verifying
    // remote cache downloads. A fixed seed keeps the thread switches the same across runs.
    var random = new Random(0);
    try (ExecutorService thread1 = Executors.newSingleThreadExecutor();
        ExecutorService thread2 = Executors.newSingleThreadExecutor()) {
      var threads = ImmutableList.of(thread1, thread2);
      for (TestVector testVector : testVectors) {
        int length = testVector.inputLength();
        byte[] input = testVector.input();
        Hasher hasher = onRandomThread(random, threads, Blake3HashFunction.INSTANCE::newHasher);
        for (int start = 0, step = 1; start < length; start += step, step = step * 3 + 1) {
          int sliceStart = start;
          int sliceLength = Math.min(step, length - start);
          onRandomThread(random, threads, () -> hasher.putBytes(input, sliceStart, sliceLength));
        }
        String hash = onRandomThread(random, threads, () -> hasher.hash().toString());

        assertWithMessage("input length %s", length).that(hash).isEqualTo(testVector.hash());
      }
    }
  }

  private static <T> T onRandomThread(
      Random random, ImmutableList<ExecutorService> threads, Callable<T> task) throws Exception {
    return threads.get(random.nextInt(threads.size())).submit(task).get();
  }

  @Test
  public void hashString() {
    assertThat(Blake3HashFunction.INSTANCE.hashString("abc\u00ff", ISO_8859_1))
        .isEqualTo(Blake3HashFunction.INSTANCE.hashBytes(new byte[] {'a', 'b', 'c', (byte) 0xff}));
    assertThat(Blake3HashFunction.INSTANCE.hashString("abc\u00ff", UTF_8))
        .isEqualTo(
            Blake3HashFunction.INSTANCE.hashBytes(
                new byte[] {'a', 'b', 'c', (byte) 0xc3, (byte) 0xbf}));
    assertThat(Blake3HashFunction.INSTANCE.hashString("abc\u0100", ISO_8859_1))
        .isEqualTo(Blake3HashFunction.INSTANCE.hashBytes(new byte[] {'a', 'b', 'c', '?'}));
  }

  @Test
  public void engineUpdate_outOfBounds_throws() {
    Blake3MessageDigest digest = new Blake3MessageDigest();
    byte[] data = "hello world".getBytes(UTF_8);

    assertThrows(IndexOutOfBoundsException.class, () -> digest.engineUpdate(data, -1, 1));
    assertThrows(IndexOutOfBoundsException.class, () -> digest.engineUpdate(data, 0, -1));
    assertThrows(
        IndexOutOfBoundsException.class, () -> digest.engineUpdate(data, 0, data.length + 1));
    assertThrows(IndexOutOfBoundsException.class, () -> digest.engineUpdate(data, data.length, 1));
    assertThrows(
        IndexOutOfBoundsException.class, () -> digest.engineUpdate(data, 1, Integer.MAX_VALUE));
  }

  @Test
  public void engineUpdate_outOfBounds_leavesStateIntact() {
    byte[] data = "hello world".getBytes(UTF_8);
    Blake3MessageDigest digest = new Blake3MessageDigest();

    digest.engineUpdate(data, 0, 6);
    assertThrows(IndexOutOfBoundsException.class, () -> digest.engineUpdate(data, 6, data.length));
    digest.engineUpdate(data, 6, data.length - 6);

    Blake3Hasher expected = new Blake3Hasher(new Blake3MessageDigest());
    expected.putBytes(data);
    assertThat(new Blake3Hasher(digest).hash()).isEqualTo(expected.hash());
  }
}
