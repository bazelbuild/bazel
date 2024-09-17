// Copyright 2024 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint.offsetZeros;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import java.util.HexFormat;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class PackedFingerprintTest {
  private final Random rng = new Random();

  @Test
  public void plainConversion_preservesBytes() {
    for (int i = 0; i < 10_000; i++) {
      byte[] bytes = randomFingerprintBytes();

      PackedFingerprint fingerprint = PackedFingerprint.fromBytes(bytes);
      assertThat(fingerprint.toBytes()).isEqualTo(bytes);
    }
  }

  @Test
  public void zeroOffsetConversion_correctBytes() {
    for (int i = 0; i < 10_000; i++) {
      byte[] bytes = randomFingerprintBytes();

      PackedFingerprint fingerprint = PackedFingerprint.fromBytesOffsetZeros(bytes);
      byte[] outBytes = fingerprint.toBytes();
      assertThat(bytes).hasLength(outBytes.length);
      for (int j = 0; j < bytes.length; j++) {
        byte nextByte = bytes[j];
        assertThat(outBytes[j]).isEqualTo(nextByte == 0 ? 1 : nextByte);
      }
    }
  }

  @Test
  public void offsetZeros_identity() {
    long input = HexFormat.fromHexDigitsToLong("0102030405060708");
    assertThat(offsetZeros(input)).isEqualTo(input);
  }

  @Test
  public void offsetZeros_convertsAllZeros() {
    assertThat(offsetZeros(0)).isEqualTo(0x0101_0101_0101_0101L);
  }

  @Test
  public void offsetZeros_preservesOnesWithCarry() {
    // This exercises the case of where the MSB becomes 1 in case (1b).
    assertThat(offsetZeros(0x0101_0101_0101_0100L)).isEqualTo(0x0101_0101_0101_0101L);
  }

  @Test
  public void offsetZeros_randomLongs() {
    for (int i = 0; i < 10_000; i++) {
      long next = rng.nextLong();
      long out = offsetZeros(next);
      for (int j = 0; j < 8; j++) {
        byte nextByte = kthByte(next, j);
        assertThat(kthByte(out, j)).isEqualTo(nextByte == 0 ? 1 : nextByte);
      }
    }
  }

  @Test
  public void concat_appendsBytes() {
    var fingerprint =
        PackedFingerprint.fromBytes(parseHex("deadbeef" + "facefeed" + "8badf00d" + "f005ba11"));
    assertThat(fingerprint.concat(parseHex("0ff1ce")))
        .isEqualTo(parseHex("deadbeeffacefeed8badf00df005ba110ff1ce"));
  }

  @Test
  public void copyTo_honorsOffset() {
    var fingerprint = PackedFingerprint.fromBytes(randomFingerprintBytes());

    byte[] target = new byte[4 + PackedFingerprint.BYTES];
    fingerprint.copyTo(target, 4);

    for (int i = 0; i < 4; i++) {
      assertThat(target[i]).isEqualTo(0);
    }

    byte[] fingerprintBytes = fingerprint.toBytes();
    for (int i = 0; i < PackedFingerprint.BYTES; i++) {
      assertThat(target[i + 4]).isEqualTo(fingerprintBytes[i]);
    }
  }

  @Test
  public void codec_roundTrips() throws Exception {
    var subjects = ImmutableList.<PackedFingerprint>builder();
    for (int i = 0; i < 10; i++) {
      byte[] bytes = randomFingerprintBytes();

      subjects.add(PackedFingerprint.fromBytes(bytes));
    }
    new SerializationTester(subjects.build()).runTests();
  }

  private byte[] randomFingerprintBytes() {
    byte[] bytes = new byte[PackedFingerprint.BYTES];
    rng.nextBytes(bytes);
    return bytes;
  }

  private static byte kthByte(long n, int k) {
    return (byte) ((n >>> (8 * k)) & (byte) 0xFF);
  }

  private static byte[] parseHex(String hex) {
    return HexFormat.of().parseHex(hex);
  }
}
