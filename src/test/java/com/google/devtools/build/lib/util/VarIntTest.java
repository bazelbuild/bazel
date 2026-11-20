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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.io.BaseEncoding;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link VarInt}. */
@RunWith(TestParameterInjector.class)
public final class VarIntTest {

  private static byte[] bytes(String hex) {
    return BaseEncoding.base16().lowerCase().decode(hex);
  }

  private enum IntTestCase {
    ZERO(0, bytes("00")),
    ONE(1, bytes("01")),
    FORTY_TWO(42, bytes("2a")),
    MAX_ONE_BYTE(127, bytes("7f")),
    MIN_TWO_BYTES(128, bytes("8001")),
    MAX_TWO_BYTES((1 << 14) - 1, bytes("ff7f")),
    MIN_THREE_BYTES(1 << 14, bytes("808001")),
    MAX_THREE_BYTES((1 << 21) - 1, bytes("ffff7f")),
    MIN_FOUR_BYTES(1 << 21, bytes("80808001")),
    MAX_FOUR_BYTES((1 << 28) - 1, bytes("ffffff7f")),
    MIN_FIVE_BYTES(1 << 28, bytes("8080808001")),
    MAX_VALUE(Integer.MAX_VALUE, bytes("ffffffff07")),
    MIN_VALUE(Integer.MIN_VALUE, bytes("8080808008")),
    MINUS_ONE(-1, bytes("ffffffff0f"));

    final int value;
    final byte[] encoded;

    IntTestCase(int value, byte[] encoded) {
      this.value = value;
      this.encoded = encoded;
    }
  }

  @Test
  public void varInt_roundTrip(@TestParameter IntTestCase testCase) throws IOException {
    var out = new ByteArrayOutputStream();
    VarInt.putVarInt(testCase.value, out);
    byte[] bytes = out.toByteArray();

    assertThat(bytes).isEqualTo(testCase.encoded);
    assertThat(bytes.length).isAtMost(VarInt.MAX_VARINT_SIZE);

    ByteBuffer src = ByteBuffer.wrap(bytes);
    assertThat(VarInt.getVarInt(src)).isEqualTo(testCase.value);
    assertThat(src.hasRemaining()).isFalse();
  }

  private enum LongTestCase {
    ZERO(0L, bytes("00")),
    ONE(1L, bytes("01")),
    FORTY_TWO(42L, bytes("2a")),
    MAX_ONE_BYTE(127L, bytes("7f")),
    MIN_TWO_BYTES(128L, bytes("8001")),
    MAX_TWO_BYTES((1L << 14) - 1, bytes("ff7f")),
    MIN_THREE_BYTES(1L << 14, bytes("808001")),
    MAX_THREE_BYTES((1L << 21) - 1, bytes("ffff7f")),
    MIN_FOUR_BYTES(1L << 21, bytes("80808001")),
    MAX_FOUR_BYTES((1L << 28) - 1, bytes("ffffff7f")),
    MIN_FIVE_BYTES(1L << 28, bytes("8080808001")),
    TWO_POW_32(1L << 32, bytes("8080808010")),
    MAX_FIVE_BYTES((1L << 35) - 1, bytes("ffffffff7f")),
    MIN_SIX_BYTES(1L << 35, bytes("808080808001")),
    MAX_SIX_BYTES((1L << 42) - 1, bytes("ffffffffff7f")),
    MIN_SEVEN_BYTES(1L << 42, bytes("80808080808001")),
    MAX_SEVEN_BYTES((1L << 49) - 1, bytes("ffffffffffff7f")),
    MIN_EIGHT_BYTES(1L << 49, bytes("8080808080808001")),
    MAX_EIGHT_BYTES((1L << 56) - 1, bytes("ffffffffffffff7f")),
    MIN_NINE_BYTES(1L << 56, bytes("808080808080808001")),
    MAX_VALUE(Long.MAX_VALUE, bytes("ffffffffffffffff7f")),
    MIN_VALUE(Long.MIN_VALUE, bytes("80808080808080808001")),
    MINUS_ONE(-1L, bytes("ffffffffffffffffff01"));

    final long value;
    final byte[] encoded;

    LongTestCase(long value, byte[] encoded) {
      this.value = value;
      this.encoded = encoded;
    }
  }

  @Test
  public void varLong_roundTrip(@TestParameter LongTestCase testCase) throws IOException {
    var out = new ByteArrayOutputStream();
    VarInt.putVarLong(testCase.value, out);
    byte[] bytes = out.toByteArray();

    assertThat(bytes).isEqualTo(testCase.encoded);
    assertThat(bytes.length).isAtMost(VarInt.MAX_VARLONG_SIZE);

    ByteBuffer src = ByteBuffer.wrap(bytes);
    assertThat(VarInt.getVarLong(src)).isEqualTo(testCase.value);
    assertThat(src.hasRemaining()).isFalse();
  }
}
