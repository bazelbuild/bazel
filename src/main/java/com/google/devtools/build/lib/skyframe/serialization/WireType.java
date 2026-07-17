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

/**
 * Represents the wire type of an encoded object, specifying the "namespace" to use for the tag.
 *
 * <h3>Wire Types</h3>
 *
 * <p>The Bazel serialization format uses "tag numbers" in the data stream to identify codecs,
 * constants, and backreferences. The `WireType` provides a namespace for these tag numbers with up
 * to 8 possible namespaces. (In the past these tag numbers were from a single namespace and changed
 * their meanings from version to version.)
 *
 * <p>A "typed tag" is a combination of a wire type and a tag number. Tag numbers are non-negative
 * integers. To encode a typed tag, the `WireType` is stored in the 3 least significant bits, and
 * the tag number prepended to those 3 bits. The resulting integer is written as <a
 * href="https://protobuf.dev/programming-guides/encoding/#varints">a base-128 varint</a>. (This is
 * the same format as <a href="https://protobuf.dev/programming-guides/encoding/#structure">Google
 * Protocol Buffers' field number wire encoding</a>, but with our own semantics for the 3-bit "wire
 * type".)
 *
 * <p>As an example, the typed tag {@code 13} is represented as the binary value {@code 0b1101},
 * which is decoded as a wire type of {@code 5} and a tag number of {@code 1}. The wire type {@code
 * 5} is {@link WireType.CodecWireType#STABLE_PRIVATE "stable private codec"}.
 *
 * <p>{@code null} remains a special case, represented by an `UNSTABLE_CODEC` wire type with tag 0.
 *
 * <h3>Stable vs. Unstable Codecs/Constants</h3>
 *
 * <p>The "unstable" and "stable" definitions refer to whether the tag numbers are allowed to change
 * between format versions. "Unstable" tag numbers continue to change in meaning between Bazel
 * versions, while "stable" tag numbers are fixed and will refer to the same codec or constant until
 * the serialization format itself is versioned. Broadly speaking, a codec being "stable" means that
 * an object with equivalent semantics across Bazel versions will be encoded with the exact same
 * byte sequence.
 *
 * <p>If a class has a stable codec, it is still permitted to add new fields to that class as long
 * as the new fields are optional and are serialized after the old fields in the byte sequence.
 */
sealed interface WireType
    permits WireType.Backreference, WireType.CodecWireType, WireType.ConstantWireType {

  /** Returns the wire type index from a combined tag number and wire type index. */
  public static int getTagNumber(int typedTag) {
    return typedTag >> 3;
  }

  /** Returns the wire type index from a combined tag number and wire type index. */
  public static int getWireTypeIndex(int typedTag) {
    return typedTag & 0b111;
  }

  public static final byte UNSTABLE_CODEC_VALUE = 0b000;
  public static final byte UNSTABLE_CONSTANT_VALUE = 0b001;
  public static final byte BACKREFERENCE_VALUE = 0b010;
  public static final byte STABLE_CODEC_VALUE = 0b011;
  public static final byte STABLE_CONSTANT_VALUE = 0b100;
  public static final byte STABLE_PRIVATE_CODEC_VALUE = 0b101;
  public static final byte STABLE_PRIVATE_CONSTANT_VALUE = 0b110;

  public enum Backreference implements WireType {
    INSTANCE;

    @Override
    public byte getValue() {
      return WireType.BACKREFERENCE_VALUE;
    }

    @Override
    public int getTypedTagNumber(int tag) {
      return (tag << 3) | WireType.BACKREFERENCE_VALUE;
    }
  }

  public enum CodecWireType implements WireType {
    UNSTABLE(WireType.UNSTABLE_CODEC_VALUE),
    STABLE_PUBLIC(WireType.STABLE_CODEC_VALUE),
    STABLE_PRIVATE(WireType.STABLE_PRIVATE_CODEC_VALUE);

    private final byte value;

    private CodecWireType(byte value) {
      this.value = value;
    }

    @Override
    public byte getValue() {
      return value;
    }

    @Override
    public int getTypedTagNumber(int tag) {
      return (tag << 3) | value;
    }
  }

  public enum ConstantWireType implements WireType {
    UNSTABLE(WireType.UNSTABLE_CONSTANT_VALUE),
    STABLE_PUBLIC(WireType.STABLE_CONSTANT_VALUE),
    STABLE_PRIVATE(WireType.STABLE_PRIVATE_CONSTANT_VALUE);

    private final byte value;

    private ConstantWireType(byte value) {
      this.value = value;
    }

    @Override
    public byte getValue() {
      return value;
    }

    @Override
    public int getTypedTagNumber(int tag) {
      return (tag << 3) | value;
    }
  }

  byte getValue();

  int getTypedTagNumber(int tag);

  /** Returns the {@link WireType} associated with the given value. */
  public static WireType fromValue(int value) {
    return switch (value) {
      case WireType.UNSTABLE_CODEC_VALUE -> CodecWireType.UNSTABLE;
      case WireType.BACKREFERENCE_VALUE -> Backreference.INSTANCE;
      case WireType.STABLE_CODEC_VALUE -> CodecWireType.STABLE_PUBLIC;
      case WireType.STABLE_CONSTANT_VALUE -> ConstantWireType.STABLE_PUBLIC;
      case WireType.STABLE_PRIVATE_CODEC_VALUE -> CodecWireType.STABLE_PRIVATE;
      case WireType.STABLE_PRIVATE_CONSTANT_VALUE -> ConstantWireType.STABLE_PRIVATE;
      case WireType.UNSTABLE_CONSTANT_VALUE -> ConstantWireType.UNSTABLE;
      default -> throw new IllegalArgumentException("Unknown wire type: " + value);
    };
  }
}
