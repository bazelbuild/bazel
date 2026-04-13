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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.collect.nestedset.DigestDeduper.DigestReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class DigestDeduperTest {

  private static DigestReference createDigest(String content) {
    DigestReference digest = new DigestReference();
    byte[] bytes = content.getBytes(UTF_8);
    digest.acceptBytes(bytes, 0, bytes.length);
    return digest;
  }

  @Test
  public void testAdd_newDigests() {
    DigestDeduper deduper = new DigestDeduper(10, 32);
    assertThat(deduper.add(createDigest("digest1"))).isTrue();
    assertThat(deduper.add(createDigest("digest2"))).isTrue();
    assertThat(deduper.add(createDigest("digest3"))).isTrue();
  }

  @Test
  public void testAdd_duplicateDigests() {
    DigestDeduper deduper = new DigestDeduper(10, 32);
    assertThat(deduper.add(createDigest("digest1"))).isTrue();
    assertThat(deduper.add(createDigest("digest1"))).isFalse();
  }

  @Test
  public void testAdd_collision() {
    DigestDeduper deduper = new DigestDeduper(2, 4);

    byte[] bytes1 = "aaaa".getBytes(UTF_8);
    byte[] bytes2 = "bbba".getBytes(UTF_8);

    DigestReference digest1 = new DigestReference();
    digest1.acceptBytes(bytes1, 0, bytes1.length);

    DigestReference digest2 = new DigestReference();
    digest2.acceptBytes(bytes2, 0, bytes2.length);

    assertThat(deduper.add(digest1)).isTrue();
    assertThat(deduper.add(digest2)).isTrue();
  }

  @Test
  public void testAdd_nearCapacity() {
    int maxSize = 10;
    DigestDeduper deduper = new DigestDeduper(maxSize, 32);

    for (int i = 0; i < maxSize; i++) {
      assertThat(deduper.add(createDigest("digest" + i))).isTrue();
    }

    // Adding a duplicate should fail.
    assertThat(deduper.add(createDigest("digest5"))).isFalse();

    // Adding one more new item should still succeed.
    assertThat(deduper.add(createDigest("digest" + maxSize))).isTrue();
  }

  @Test
  public void testAdd_nearCapacity_withProbing() {
    int maxSize = 10;
    DigestDeduper deduper = new DigestDeduper(maxSize, 32);

    for (int i = 0; i < maxSize; i++) {
      // These strings are chosen to have the same last 4 bytes, forcing hash collisions
      // and stressing the linear probing mechanism.
      assertThat(deduper.add(createDigest(i + "----------colliding_string"))).isTrue();
    }

    // Adding a duplicate should fail.
    assertThat(deduper.add(createDigest("5----------colliding_string"))).isFalse();

    // Adding one more new item should still succeed.
    assertThat(deduper.add(createDigest(maxSize + "----------colliding_string"))).isTrue();
  }

  @Test
  public void testSizeBitsFor_throwsExceptionForInvalidMaxSize() {
    assertThrows(IllegalArgumentException.class, () -> DigestDeduper.sizeBitsFor(0));
    assertThrows(IllegalArgumentException.class, () -> DigestDeduper.sizeBitsFor(-1));
  }

  @Test
  public void testSizeBitsFor_calculatesCorrectSize() {
    // A deduper with maxSize will have a capacity that is the smallest power of 2
    // greater than or equal to ceil(maxSize / 0.75).

    // maxSize = 10, minCapacity = ceil(10 / 0.75) = 14, size = 16 (4 bits)
    assertThat(DigestDeduper.sizeBitsFor(10)).isEqualTo(4);

    // maxSize = 12, minCapacity = ceil(12 / 0.75) = 16, size = 16 (4 bits)
    assertThat(DigestDeduper.sizeBitsFor(12)).isEqualTo(4);

    // maxSize = 13, minCapacity = ceil(13 / 0.75) = 18, size = 32 (5 bits)
    assertThat(DigestDeduper.sizeBitsFor(13)).isEqualTo(5);
  }
}
