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
package com.google.devtools.build.lib.util;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;

/**
 * Common methods to encode and decode varints and varlongs into ByteBuffers and
 * arrays.
 */
public class VarInt {

  /**
   * Maximum encoded size of 32-bit positive integers (in bytes)
   */
  public static final int MAX_VARINT_SIZE = 5;

  /**
   * maximum encoded size of 64-bit longs, and negative 32-bit ints (in bytes)
   */
  public static final int MAX_VARLONG_SIZE = 10;

  private VarInt() { }

  /** Returns the encoding size in bytes of its input value.
   * @param i the integer to be measured
   * @return the encoding size in bytes of its input value
   */
  public static int varIntSize(int i) {
    int result = 0;
    do {
      result++;
      i >>>= 7;
    } while (i != 0);
    return result;
  }

  /**
   * Reads a varint from the current position of the given ByteBuffer and returns the decoded value
   * as 32 bit integer.
   *
   * <p>The position of the buffer is advanced to the first byte after the decoded varint.
   *
   * @param src the ByteBuffer to get the var int from
   * @return The integer value of the decoded varint
   */
  public static int getVarInt(ByteBuffer src) {
    int tmp;
    if ((tmp = src.get()) >= 0) {
      return tmp;
    }
    int result = tmp & 0x7f;
    if ((tmp = src.get()) >= 0) {
      result |= tmp << 7;
    } else {
      result |= (tmp & 0x7f) << 7;
      if ((tmp = src.get()) >= 0) {
        result |= tmp << 14;
      } else {
        result |= (tmp & 0x7f) << 14;
        if ((tmp = src.get()) >= 0) {
          result |= tmp << 21;
        } else {
          result |= (tmp & 0x7f) << 21;
          result |= (tmp = src.get()) << 28;
          while (tmp < 0) {
            // We get into this loop only in the case of overflow.
            // By doing this, we can call getVarInt() instead of
            // getVarLong() when we only need an int.
            tmp = src.get();
          }
        }
      }
    }
    return result;
  }

  /**
   * Encodes an integer in a variable-length encoding, 7 bits per byte, into a destination byte[],
   * following the protocol buffer convention.
   *
   * @param v the int value to write to sink
   * @param sink the sink buffer to write to
   * @param offset the offset within sink to begin writing
   * @return the updated offset after writing the varint
   */
  public static int putVarInt(int v, byte[] sink, int offset) {
    do {
      // Encode next 7 bits + terminator bit
      int bits = v & 0x7F;
      v >>>= 7;
      byte b = (byte) (bits + ((v != 0) ? 0x80 : 0));
      sink[offset++] = b;
    } while (v != 0);
    return offset;
  }

  /**
   * Encodes an integer in a variable-length encoding, 7 bits per byte, and writes it to the given
   * OutputStream.
   *
   * @param v the value to encode
   * @param outputStream the OutputStream to write to
   */
  public static void putVarInt(int v, OutputStream outputStream) throws IOException {
    byte[] bytes = new byte[varIntSize(v)];
    putVarInt(v, bytes, 0);
    outputStream.write(bytes);
  }

  /**
   * Returns the encoding size in bytes of its input value.
   *
   * @param v the long to be measured
   * @return the encoding size in bytes of a given long value.
   */
  public static int varLongSize(long v) {
    int result = 0;
    do {
      result++;
      v >>>= 7;
    } while (v != 0);
    return result;
  }

  /**
   * Reads an up to 64 bit long varint from the current position of the
   * given ByteBuffer and returns the decoded value as long.
   *
   * <p>The position of the buffer is advanced to the first byte after the
   * decoded varint.
   *
   * @param src the ByteBuffer to get the var int from
   * @return The integer value of the decoded long varint
   */
  public static long getVarLong(ByteBuffer src) {
    long tmp;
    if ((tmp = src.get()) >= 0) {
      return tmp;
    }
    long result = tmp & 0x7f;
    if ((tmp = src.get()) >= 0) {
      result |= tmp << 7;
    } else {
      result |= (tmp & 0x7f) << 7;
      if ((tmp = src.get()) >= 0) {
        result |= tmp << 14;
      } else {
        result |= (tmp & 0x7f) << 14;
        if ((tmp = src.get()) >= 0) {
          result |= tmp << 21;
        } else {
          result |= (tmp & 0x7f) << 21;
          if ((tmp = src.get()) >= 0) {
            result |= tmp << 28;
          } else {
            result |= (tmp & 0x7f) << 28;
            if ((tmp = src.get()) >= 0) {
              result |= tmp << 35;
            } else {
              result |= (tmp & 0x7f) << 35;
              if ((tmp = src.get()) >= 0) {
                result |= tmp << 42;
              } else {
                result |= (tmp & 0x7f) << 42;
                if ((tmp = src.get()) >= 0) {
                  result |= tmp << 49;
                } else {
                  result |= (tmp & 0x7f) << 49;
                  if ((tmp = src.get()) >= 0) {
                    result |= tmp << 56;
                  } else {
                    result |= (tmp & 0x7f) << 56;
                    result |= ((long) src.get()) << 63;
                  }
                }
              }
            }
          }
        }
      }
    }
    return result;
  }

  /**
   * Encodes a long integer in a variable-length encoding, 7 bits per byte, to a
   * ByteBuffer sink.
   * @param v the value to encode
   * @param sink the ByteBuffer to add the encoded value
   */
  public static void putVarLong(long v, ByteBuffer sink) {
    while (true) {
      int bits = ((int) v) & 0x7f;
      v >>>= 7;
      if (v == 0) {
        sink.put((byte) bits);
        return;
      }
      sink.put((byte) (bits | 0x80));
    }
  }

  public static void putVarLong(long v, OutputStream outputStream) throws IOException {
    byte[] bytes = new byte[varLongSize(v)];
    ByteBuffer sink = ByteBuffer.wrap(bytes);
    putVarLong(v, sink);
    outputStream.write(bytes);
  }
}
