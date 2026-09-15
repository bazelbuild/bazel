// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.zip;

import java.util.Arrays;

/**
 * A holder class for extra data in a ZIP entry.
 */
public final class ExtraData {
  static final int ID_OFFSET = 0;
  static final int LENGTH_OFFSET = 2;
  static final int FIXED_DATA_SIZE = 4;

  private final int index;
  private final byte[] buffer;

  /**
   * Creates a new {@link ExtraData} record with the specified id and data.
   *
   * @param id the ID tag for this extra data record
   * @param data the data payload for this extra data record
   */
  public ExtraData(short id, byte[] data) {
    if (data.length > 0xffff) {
      throw new IllegalArgumentException(String.format("Data is too long. Is %d; max %d",
          data.length, 0xffff));
    }
    index = 0;
    buffer = new byte[FIXED_DATA_SIZE + data.length];
    ZipUtil.shortToLittleEndian(buffer, ID_OFFSET, id);
    ZipUtil.shortToLittleEndian(buffer, LENGTH_OFFSET, (short) data.length);
    System.arraycopy(data, 0, buffer, FIXED_DATA_SIZE, data.length);
  }

  /**
   * Creates a new {@link ExtraData} record using the buffer as the backing data store.
   *
   * <p><em>NOTE:</em> does not perform any defensive copying. Any modification to the buffer will
   * alter the extra data record and can make it invalid.
   *
   * @param buffer the array containing the extra data record
   * @param index the index where the extra data record is located
   * @throws IllegalArgumentException if buffer does not contain a well formed extra data record
   *    at index
   */
  ExtraData(byte[] buffer, int index) {
    if (index >= buffer.length) {
      throw new IllegalArgumentException("index past end of buffer");
    }
    if (buffer.length - index < FIXED_DATA_SIZE) {
      throw new IllegalArgumentException("incomplete extra data entry in buffer");
    }
    int length = ZipUtil.getUnsignedShort(buffer, index + LENGTH_OFFSET);
    if (buffer.length - index - FIXED_DATA_SIZE < length) {
      throw new IllegalArgumentException("incomplete extra data entry in buffer");
    }
    this.buffer = buffer;
    this.index = index;
  }

  /** Returns the Id of the extra data record. */
  public short getId() {
    return ZipUtil.get16(buffer, index + ID_OFFSET);
  }

  /** Returns the total length of the extra data record in bytes. */
  public int getLength() {
    return getDataLength() + FIXED_DATA_SIZE;
  }

  /** Returns the length of the data payload of the extra data record in bytes. */
  public int getDataLength() {
    return ZipUtil.getUnsignedShort(buffer, index + LENGTH_OFFSET);
  }

  /** Returns a byte array copy of the data payload. */
  public byte[] getData() {
    return Arrays.copyOfRange(buffer, index + FIXED_DATA_SIZE, index + getLength());
  }

  /** Returns a byte array copy of the entire record. */
  public byte[] getBytes() {
    return Arrays.copyOfRange(buffer, index, index + getLength());
  }

  /** Returns the byte at index from the entire record. */
  byte getByte(int index) {
    return buffer[this.index + index];
  }
}
