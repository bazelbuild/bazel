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
public sealed interface WireType
    permits WireType.Backreference, WireType.CodecWireType, WireType.ConstantWireType {

  /** Number of low-order bits used to store the wire type index in a typed tag. */
  public static final int WIRE_TYPE_BITS = 3;

  /** Bitmask extracting the wire type index from the low-order bits of a typed tag. */
  public static final int WIRE_TYPE_MASK = (1 << WIRE_TYPE_BITS) - 1;

  public static final byte UNSTABLE_CODEC_VALUE = 0b000;
  public static final byte UNSTABLE_CONSTANT_VALUE = 0b001;
  public static final byte BACKREFERENCE_VALUE = 0b010;
  public static final byte STABLE_CODEC_VALUE = 0b011;
  public static final byte STABLE_CONSTANT_VALUE = 0b100;
  public static final byte STABLE_PRIVATE_CODEC_VALUE = 0b101;
  public static final byte STABLE_PRIVATE_CONSTANT_VALUE = 0b110;

  /**
   * Typed tag {@code 0}, representing {@code null} on the wire ({@link CodecWireType#UNSTABLE} with
   * tag 0).
   */
  public static final int NULL_TYPED_TAG = (0 << WIRE_TYPE_BITS) | UNSTABLE_CODEC_VALUE;

  /**
   * Typed tag emitted by {@code LeafSerializationContext.serializeLeaf} to signal an immediate
   * (non-null, non-constant, non-backreferenced) value ({@link CodecWireType#UNSTABLE} with tag 1).
   */
  public static final int LEAF_IMMEDIATE_TYPED_TAG = (1 << WIRE_TYPE_BITS) | UNSTABLE_CODEC_VALUE;

  /** Returns the tag number from a combined tag number and wire type index. */
  static int getTagNumber(int typedTag) {
    return typedTag >> WIRE_TYPE_BITS;
  }

  /** Returns the wire type index from a combined tag number and wire type index. */
  static int getWireTypeIndex(int typedTag) {
    return typedTag & WIRE_TYPE_MASK;
  }

  /** Wire type for memoized backreferences. */
  public enum Backreference implements WireType {
    INSTANCE;

    @Override
    public byte getValue() {
      return WireType.BACKREFERENCE_VALUE;
    }

    @Override
    public int getTypedTagNumber(int tag) {
      return (tag << WIRE_TYPE_BITS) | WireType.BACKREFERENCE_VALUE;
    }

    @Override
    public boolean matches(int typedTag) {
      return getWireTypeIndex(typedTag) == WireType.BACKREFERENCE_VALUE;
    }
  }

  /** Wire types for unstable, public stable, and private stable codecs. */
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
      return (tag << WIRE_TYPE_BITS) | value;
    }

    @Override
    public boolean matches(int typedTag) {
      return getWireTypeIndex(typedTag) == value;
    }
  }

  /** Wire types for unstable, public stable, and private stable reference constants. */
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
      return (tag << WIRE_TYPE_BITS) | value;
    }

    @Override
    public boolean matches(int typedTag) {
      return getWireTypeIndex(typedTag) == value;
    }
  }

  byte getValue();

  int getTypedTagNumber(int tag);

  /** Returns true if {@code typedTag} encodes this {@link WireType} in its wire-type bits. */
  boolean matches(int typedTag);

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
