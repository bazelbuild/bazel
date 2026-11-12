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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.common.hash.Hasher;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Blake3MessageDigest}. */
@RunWith(JUnit4.class)
public class Blake3HasherTest {
  // Selected cases from
  // https://github.com/BLAKE3-team/BLAKE3/blob/master/test_vectors/test_vectors.json, where the
  // input of length n consists of the bytes i % 251 for i in [0, n).
  private static final ImmutableMap<Integer, String> TEST_VECTORS =
      ImmutableMap.<Integer, String>builder()
          .put(0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262")
          .put(1, "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213")
          .put(64, "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98")
          .put(65, "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee")
          .put(1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7")
          .put(1025, "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444")
          .put(8193, "bab6c09cb8ce8cf459261398d2e7aef35700bf488116ceb94a36d0f5f1b7bc3b")
          .put(31744, "62b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47")
          .put(102400, "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085")
          .buildOrThrow();

  private static byte[] testVectorInput(int length) {
    byte[] input = new byte[length];
    for (int i = 0; i < length; i++) {
      input[i] = (byte) (i % 251);
    }
    return input;
  }

  @Test
  public void emptyHash() {
    Blake3Hasher h = new Blake3Hasher(new Blake3MessageDigest());

    byte[] data = new byte[0];
    h.putBytes(data);

    assertEquals(
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262", h.hash().toString());
  }

  @Test
  public void helloWorld() {
    Blake3Hasher h = new Blake3Hasher(new Blake3MessageDigest());

    byte[] data = "hello world".getBytes(StandardCharsets.US_ASCII);
    h.putBytes(data);

    assertEquals(
        "d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24", h.hash().toString());
  }

  @Test
  public void engineUpdate_outOfBounds_throws() {
    Blake3MessageDigest digest = new Blake3MessageDigest();
    byte[] data = "hello world".getBytes(StandardCharsets.US_ASCII);

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
    byte[] data = "hello world".getBytes(StandardCharsets.US_ASCII);
    Blake3MessageDigest digest = new Blake3MessageDigest();

    digest.engineUpdate(data, 0, 6);
    assertThrows(IndexOutOfBoundsException.class, () -> digest.engineUpdate(data, 6, data.length));
    digest.engineUpdate(data, 6, data.length - 6);

    Blake3Hasher expected = new Blake3Hasher(new Blake3MessageDigest());
    expected.putBytes(data);
    assertEquals(expected.hash().toString(), new Blake3Hasher(digest).hash().toString());
  }

  @Test
  public void testVectors_hashBytes() {
    for (Map.Entry<Integer, String> testVector : TEST_VECTORS.entrySet()) {
      byte[] input = testVectorInput(testVector.getKey());

      assertThat(Blake3HashFunction.INSTANCE.hashBytes(input).toString())
          .isEqualTo(testVector.getValue());
    }
  }

  @Test
  public void testVectors_hashBytesWithOffset() {
    for (Map.Entry<Integer, String> testVector : TEST_VECTORS.entrySet()) {
      int length = testVector.getKey();
      byte[] input = new byte[length + 10];
      System.arraycopy(testVectorInput(length), 0, input, 3, length);

      assertThat(Blake3HashFunction.INSTANCE.hashBytes(input, 3, length).toString())
          .isEqualTo(testVector.getValue());
    }
  }

  @Test
  public void testVectors_hasherWithUnevenUpdates() {
    for (Map.Entry<Integer, String> testVector : TEST_VECTORS.entrySet()) {
      int length = testVector.getKey();
      byte[] input = testVectorInput(length);
      Hasher hasher = Blake3HashFunction.INSTANCE.newHasher();
      for (int start = 0, step = 1; start < length; start += step, step = step * 3 + 1) {
        hasher.putBytes(input, start, Math.min(step, length - start));
      }

      assertThat(hasher.hash().toString()).isEqualTo(testVector.getValue());
    }
  }
}
