// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.coverage;

import java.util.Arrays;

/**
 * Abstracts bit field operations.
 */
public class BitField {

  private byte[] bytes;

  private static final int[] BIT_COUNT_LOOKUP = new int[256];

  static {
    BIT_COUNT_LOOKUP[0] = 0;
    BIT_COUNT_LOOKUP[1] = 1;
    for (int i = 2; i < 256; i += 2) {
      int count = BIT_COUNT_LOOKUP[i / 2];
      BIT_COUNT_LOOKUP[i] = count;
      BIT_COUNT_LOOKUP[i + 1] = count + 1;
    }
  }

  public BitField() {
    this(new byte[0]);
  }

  public BitField(byte[] bytes) {
    this.bytes = bytes.clone();
  }

  /**
   * Returns a copy of the underlying byte array.
   *
   * @return byte array copy
   */
  public byte[] getBytes() {
    return bytes.clone();
  }

  /**
   * Sets or clears a bit at the given index.
   *
   * @param index bit index
   */
  public void setBit(int index) {
    setBit(index, true);
  }

  /**
   * Sets or clears a bit at the given index.
   *
   * @param index bit index
   */
  private void setBit(int index, boolean isSet) {
    int byteIndex = index / 8;
    int newByteSize = byteIndex + 1;
    if (bytes.length < newByteSize) {
      bytes = Arrays.copyOf(bytes, newByteSize);
    }

    int bitIndex = index % 8;
    int mask = 1 << bitIndex;

    if (isSet) {
      bytes[byteIndex] = (byte) (bytes[byteIndex] | mask);
    } else {
      bytes[byteIndex] = (byte) (bytes[byteIndex] & ~mask);
    }
  }

  /**
   * Clears a bit at the given index
   *
   * @param index bit index
   */
  public void clearBit(int index) {
    setBit(index, false);
  }

  /**
   * Checks whether a bit at the given index is set.
   *
   * @param index bit index
   * @return true if set, false otherwise
   */
  public boolean isBitSet(int index) {
    int byteIndex = index / 8;

    if (byteIndex >= bytes.length) {
      return false;
    }

    int bitIndex = index % 8;
    int mask = 1 << bitIndex;
    return (bytes[byteIndex] & mask) != 0;
  }

  /** Performs a non-destructive bit-wise "and" of this bit field with another one. */
  public BitField and(BitField other) {
    int size = Math.min(bytes.length, other.bytes.length);
    byte[] result = new byte[size];

    for (int i = 0; i < size; i++) {
      result[i] = (byte) (bytes[i] & other.bytes[i]);
    }

    return new BitField(result);
  }

  /**
   * Performs a non-destructive bit-wise merge of this bit field and another one.
   *
   * @param other the other bit field
   * @return this bit field
   */
  public BitField or(BitField other) {
    byte[] largerArray;
    byte[] smallerArray;
    if (bytes.length < other.bytes.length) {
      largerArray = other.bytes;
      smallerArray = bytes;
    } else {
      largerArray = bytes;
      smallerArray = other.bytes;
    }

    // Start out with a copy of the larger of the two arrays.
    byte[] result = Arrays.copyOf(largerArray, largerArray.length);

    for (int i = 0; i < smallerArray.length; i++) {
      result[i] |= smallerArray[i];
    }

    return new BitField(result);
  }

  /**
   * Compares two bit fields for equality.
   *
   * @param obj another object
   * @return true if the other object is a bit field with the same bits set
   */
  @Override
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }
    if (!(obj instanceof BitField)) {
      return false;
    }
    return Arrays.equals(bytes, ((BitField) obj).bytes);
  }

  /**
   * Compare a BitField object with an array of bytes
   *
   * @param other a byte array to compare to
   * @return true if the underlying byte array is equal to the given byte array
   */
  public boolean equals(byte[] other) {
    return Arrays.equals(bytes, other);
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(bytes);
  }

  public int countBitsSet() {
    int count = 0;
    for (byte b : bytes) {
      // JAVA doesn't have the concept of unsigned byte; need to & with 255
      // to avoid exception of IndexOutOfBoundException when b < 0.
      count += BIT_COUNT_LOOKUP[0xFF & b];
    }
    return count;
  }

  public BitField not() {
    byte[] invertedBytes = new byte[bytes.length];
    for (int i = 0; i < bytes.length; i++) {
      invertedBytes[i] = Integer.valueOf(~bytes[i]).byteValue();
    }
    return new BitField(invertedBytes);
  }

  public int sizeInBits() {
    return bytes.length * 8;
  }

  public boolean any() {
    for (int i = 0; i < bytes.length; i++) {
      if (bytes[i] != (byte) 0) {
        return true;
      }
    }
    return false;
  }
}
