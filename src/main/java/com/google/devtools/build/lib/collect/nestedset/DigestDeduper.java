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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.util.BytesSink;
import com.google.devtools.build.lib.util.Fingerprint;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * A fixed-size deduplicator for digests.
 *
 * <p>This class is not thread safe.
 */
final class DigestDeduper {
  // Creates a VarHandle using generic type int[].class, telling it we want to read 'int' values.
  private static final VarHandle INT_HANDLE =
      MethodHandles.byteArrayViewVarHandle(int[].class, ByteOrder.nativeOrder());

  /** Mask for setting the top bit of the control byte to 1. */
  private static final byte CONTROL_BIT = (byte) 0x80;

  private final int digestLength;

  /**
   * Bytes indicating presence of data.
   *
   * <p>The first bit of the i-th byte is 1 iff the slot is occupied. The remaining bits in the byte
   * are the last 7 bits of the corresponding digest. This allows efficient probing, relying mostly
   * on just the control bytes.
   */
  private final byte[] control;

  private final byte[] data;

  /**
   * Bit mask that helps implement the modulo operation.
   *
   * <p>The size is always a power of 2, and this mask has a value of size - 1.
   */
  private final int sizeMask;

  static class DigestReference implements BytesSink {
    private byte[] buffer;
    private int offset;
    private int length;

    @Override
    public void acceptBytes(byte[] buffer, int offset, int length) {
      checkArgument(length >= 4, "length=%s < 4", length);
      this.buffer = buffer;
      this.offset = offset;
      this.length = length;
    }

    /**
     * Clears the reference.
     *
     * <p>The purpose of this method is to allow the client to avoid retaining {@link #buffer}
     * longer than necessary.
     */
    void clear() {
      this.buffer = null;
      this.offset = 0;
      this.length = 0;
    }

    int hash() {
      // Interprets the last 4 bytes of the digest as an integer. Since it is a digest, it should be
      // uniformly distributed.
      return (int) INT_HANDLE.get(buffer, offset + length - 4);
    }

    byte controlByte() {
      return (byte) (buffer[offset + length - 1] | CONTROL_BIT);
    }

    boolean equalsBytesAt(byte[] thatBuffer, int thatOffset) {
      // `length` might be less than `digestLength`. In such cases, the prefix contains a length
      // specifier that is implicitly matched by the following comparison.
      for (int i = 0; i < length; i++) {
        if (buffer[offset + i] != thatBuffer[thatOffset + i]) {
          return false;
        }
      }
      return true;
    }

    void copyTo(byte[] dest, int destOffset) {
      System.arraycopy(buffer, offset, dest, destOffset, length);
    }

    void addTo(Fingerprint fingerprint) {
      fingerprint.addBytes(buffer, offset, length);
    }
  }

  DigestDeduper(int maxSize, int digestLength) {
    this.digestLength = digestLength;
    int sizeBits = sizeBitsFor(maxSize);
    int size = 1 << sizeBits;
    this.sizeMask = size - 1;
    this.control = new byte[size];
    this.data = new byte[size * digestLength];
  }

  /**
   * Adds {@code digest} to this deduper.
   *
   * @return true if the digest was added and false if it was a duplicate
   */
  boolean add(DigestReference digest) {
    byte digestControlByte = digest.controlByte();
    int candidateSlot = digest.hash();
    while (true) {
      candidateSlot &= sizeMask; // fast modulo
      int controlByte = control[candidateSlot];
      if (controlByte == (byte) 0) {
        // The slot was empty. Adds `digest` to the slot.
        control[candidateSlot] = digestControlByte;
        digest.copyTo(data, candidateSlot * digestLength);
        return true;
      }
      if (controlByte == digestControlByte) { // likely match
        if (digest.equalsBytesAt(data, candidateSlot * digestLength)) {
          return false; // It was a duplicate.
        }
      }
      candidateSlot++;
    }
  }

  @VisibleForTesting
  static int sizeBitsFor(int maxSize) {
    checkArgument(maxSize > 0, "maxSize=%s not >0", maxSize);
    // 1. Calculate the minimum capacity required to satisfy the 0.75 load factor.
    // Formula: ceil(maxSize / 0.75)  =>  ceil((maxSize * 4) / 3)
    // Integer ceiling division trick: (A + B - 1) / B
    int minCapacity = (maxSize * 4 + 2) / 3;

    // 2. Find the smallest power of 2 >= minCapacity
    // If minCapacity is already a power of 2, this returns minCapacity.
    // If not, it returns the next power of 2.
    return 32 - Integer.numberOfLeadingZeros(minCapacity - 1);
  }
}
