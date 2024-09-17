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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.annotations.VisibleForTesting;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * A compact in-memory representation of a 128-bit fingerprint.
 *
 * <p>A wrapper around the bytes in unavoidable because a {@code byte[]} doesn't implement
 * values-equality. Storing the bytes in longs is more direct and consumes less memory.
 *
 * <p>This class contains additional functionality for Memcached keys.
 *
 * @param lo the lower 64-bits of the fingerprint
 * @param hi the upper 64-bits of the fingerprint
 */
public record PackedFingerprint(long lo, long hi) {
  @VisibleForTesting static final int BYTES = 16;

  /**
   * Constructs a fingerprint directly from {@code bytes}.
   *
   * @throws IllegalArgumentException if {@code bytes} is not length {@link #BYTES}.
   */
  public static PackedFingerprint fromBytes(byte[] bytes) {
    checkArgument(bytes.length == BYTES, bytes.length);
    return new PackedFingerprint(
        (long) LONG_ARRAY_HANDLE.get(bytes, 0), (long) LONG_ARRAY_HANDLE.get(bytes, 8));
  }

  /**
   * Constructs a fingerprint from {@code bytes}, converting any 0 bytes to 1.
   *
   * <p>This is useful for creating Memcached keys, which must not contain the `\0` byte.
   *
   * @throws IllegalArgumentException if {@code bytes} is not length {@link #BYTES}.
   */
  public static PackedFingerprint fromBytesOffsetZeros(byte[] bytes) {
    checkArgument(bytes.length == BYTES, bytes.length);
    return new PackedFingerprint(
        offsetZeros((long) LONG_ARRAY_HANDLE.get(bytes, 0)),
        offsetZeros((long) LONG_ARRAY_HANDLE.get(bytes, 8)));
  }

  /** Produces the {@code byte[]} representation of this fingerprint. */
  public byte[] toBytes() {
    byte[] result = new byte[BYTES];
    copyTo(result, 0);
    return result;
  }

  /** Concatenates {@code bytes} to the {@code byte[]} representation of this fingerprint. */
  public byte[] concat(byte[] bytes) {
    byte[] result = new byte[BYTES + bytes.length];
    copyTo(result, 0);
    System.arraycopy(bytes, 0, result, 16, bytes.length);
    return result;
  }

  /** Copies the fingerprint bytes to {@code bytes} starting at the given {@code offset}. */
  public void copyTo(byte[] bytes, int offset) {
    LONG_ARRAY_HANDLE.set(bytes, offset, lo);
    LONG_ARRAY_HANDLE.set(bytes, offset + 8, hi);
  }

  @Override
  public int hashCode() {
    return (int) lo;
  }

  /** Changes all 0 bytes of {@code input} into 1. */
  @VisibleForTesting
  static long offsetZeros(long input) {
    // 1. (input - 0x0101_0101_0101_0101) produces a long with every byte having MSB as follows
    //    based on the input byte, `b`.
    //
    //    a.      b = 0    : MSB becomes 1
    //    b.      b = 1    : MSB may stay at 0 or become 1
    //    c.  1 < b < 0x81 : MSB may stay at or become 0
    //    d.      b = 0x81 : MSB may stay at 1 or become 0
    //    e.      b > 0x81 : MSB stays at 1
    //
    //    Note that while -1 is directly subtracted from each byte, it's possible for it to be
    //    effectively -2 if there are carries.
    //
    // 2. (~input & 0x8080_8080_8080_8080L) produces a long where every byte has its MSB set iff
    //    it was < 0x80 (and all other bits 0).
    //
    // 3. Combining (1) and (2) using the AND operation produces a long where every byte has its MSB
    //    set iff it was 0 and sometimes 1, case (a) and (b). Case (c) always has MSB 0 while cases
    //    (d) and (e) are masked out by (2).
    //
    // 4. Shifting (3) by 7 bits to the right produces a long where every byte has its LSB set iff
    //    that byte was originally 0 and sometimes 1.
    //
    // 5. Finally, combining (4) with the input using the OR operation produces a the long with all
    //    0 bytes turned into 1.
    return input | (((input - 0x0101_0101_0101_0101L) & (~input & 0x8080_8080_8080_8080L)) >>> 7);
  }

  @Keep
  private static class Codec extends LeafObjectCodec<PackedFingerprint> {
    @Override
    public Class<PackedFingerprint> getEncodedClass() {
      return PackedFingerprint.class;
    }

    @Override
    public void serialize(
        LeafSerializationContext context, PackedFingerprint obj, CodedOutputStream codedOut)
        throws IOException {
      codedOut.writeInt64NoTag(obj.lo());
      codedOut.writeInt64NoTag(obj.hi());
    }

    @Override
    public PackedFingerprint deserialize(
        LeafDeserializationContext context, CodedInputStream codedIn) throws IOException {
      return new PackedFingerprint(codedIn.readInt64(), codedIn.readInt64());
    }
  }

  private static final VarHandle LONG_ARRAY_HANDLE =
      MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.nativeOrder());
}
