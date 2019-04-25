/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.dec;

/**
 * Bit reading helpers.
 */
final class BitReader {

  // Possible values: {5, 6}.  5 corresponds to 32-bit build, 6 to 64-bit. This value is used for
  // conditional compilation -> produced artifacts might be binary INCOMPATIBLE (JLS 13.2).
  private static final int LOG_BITNESS = 6;
  private static final int BITNESS = 1 << LOG_BITNESS;

  private static final int BYTENESS = BITNESS / 8;
  private static final int CAPACITY = 4096;
  // After encountering the end of the input stream, this amount of zero bytes will be appended.
  private static final int SLACK = 64;
  private static final int BUFFER_SIZE = CAPACITY + SLACK;
  // Don't bother to replenish the buffer while this number of bytes is available.
  private static final int SAFEGUARD = 36;
  private static final int WATERLINE = CAPACITY - SAFEGUARD;

  // "Half" refers to "half of native integer type", i.e. on 64-bit machines it is 32-bit type,
  // on 32-bit machines it is 16-bit.
  private static final int HALF_BITNESS = BITNESS / 2;
  private static final int HALF_SIZE = BYTENESS / 2;
  private static final int HALVES_CAPACITY = CAPACITY / HALF_SIZE;
  private static final int HALF_BUFFER_SIZE = BUFFER_SIZE / HALF_SIZE;
  private static final int HALF_WATERLINE = WATERLINE / HALF_SIZE;

  private static final int LOG_HALF_SIZE = LOG_BITNESS - 4;

  /**
   * Fills up the input buffer.
   *
   * <p> No-op if there are at least 36 bytes present after current position.
   *
   * <p> After encountering the end of the input stream, 64 additional zero bytes are copied to the
   * buffer.
   */
  static void readMoreInput(State s) {
    if (s.halfOffset > HALF_WATERLINE) {
      doReadMoreInput(s);
    }
  }

  static void doReadMoreInput(State s) {
    if (s.endOfStreamReached != 0) {
      if (halfAvailable(s) >= -2) {
        return;
      }
      throw new BrotliRuntimeException("No more input");
    }
    int readOffset = s.halfOffset << LOG_HALF_SIZE;
    int bytesInBuffer = CAPACITY - readOffset;
    // Move unused bytes to the head of the buffer.
    Utils.copyBytesWithin(s.byteBuffer, 0, readOffset, CAPACITY);
    s.halfOffset = 0;
    while (bytesInBuffer < CAPACITY) {
      int spaceLeft = CAPACITY - bytesInBuffer;
      int len = Utils.readInput(s.input, s.byteBuffer, bytesInBuffer, spaceLeft);
      // EOF is -1 in Java, but 0 in C#.
      if (len <= 0) {
        s.endOfStreamReached = 1;
        s.tailBytes = bytesInBuffer;
        bytesInBuffer += HALF_SIZE - 1;
        break;
      }
      bytesInBuffer += len;
    }
    bytesToNibbles(s, bytesInBuffer);
  }

  static void checkHealth(State s, int endOfStream) {
    if (s.endOfStreamReached == 0) {
      return;
    }
    int byteOffset = (s.halfOffset << LOG_HALF_SIZE) + ((s.bitOffset + 7) >> 3) - BYTENESS;
    if (byteOffset > s.tailBytes) {
      throw new BrotliRuntimeException("Read after end");
    }
    if ((endOfStream != 0) && (byteOffset != s.tailBytes)) {
      throw new BrotliRuntimeException("Unused bytes after end");
    }
  }

  static void fillBitWindow(State s) {
    if (s.bitOffset >= HALF_BITNESS) {
      // Same as doFillBitWindow. JVM fails to inline it.
      if (BITNESS == 64) {
        s.accumulator64 = ((long) s.intBuffer[s.halfOffset++] << HALF_BITNESS)
            | (s.accumulator64 >>> HALF_BITNESS);
      } else {
        s.accumulator32 = ((int) s.shortBuffer[s.halfOffset++] << HALF_BITNESS)
            | (s.accumulator32 >>> HALF_BITNESS);
      }
      s.bitOffset -= HALF_BITNESS;
    }
  }

  private static void doFillBitWindow(State s) {
    if (BITNESS == 64) {
      s.accumulator64 = ((long) s.intBuffer[s.halfOffset++] << HALF_BITNESS)
          | (s.accumulator64 >>> HALF_BITNESS);
    } else {
      s.accumulator32 = ((int) s.shortBuffer[s.halfOffset++] << HALF_BITNESS)
          | (s.accumulator32 >>> HALF_BITNESS);
    }
    s.bitOffset -= HALF_BITNESS;
  }

  static int peekBits(State s) {
    if (BITNESS == 64) {
      return (int) (s.accumulator64 >>> s.bitOffset);
    } else {
      return s.accumulator32 >>> s.bitOffset;
    }
  }

  static int readFewBits(State s, int n) {
    int val = peekBits(s) & ((1 << n) - 1);
    s.bitOffset += n;
    return val;
  }

  static int readBits(State s, int n) {
    if (HALF_BITNESS >= 24) {
      return readFewBits(s, n);
    } else {
      return (n <= 16) ? readFewBits(s, n) : readManyBits(s, n);
    }
  }

  private static int readManyBits(State s, int n) {
    int low = readFewBits(s, 16);
    doFillBitWindow(s);
    return low | (readFewBits(s, n - 16) << 16);
  }

  static void initBitReader(State s) {
    s.byteBuffer = new byte[BUFFER_SIZE];
    if (BITNESS == 64) {
      s.accumulator64 = 0;
      s.intBuffer = new int[HALF_BUFFER_SIZE];
    } else {
      s.accumulator32 = 0;
      s.shortBuffer = new short[HALF_BUFFER_SIZE];
    }
    s.bitOffset = BITNESS;
    s.halfOffset = HALVES_CAPACITY;
    s.endOfStreamReached = 0;
    prepare(s);
  }

  private static void prepare(State s) {
    readMoreInput(s);
    checkHealth(s, 0);
    doFillBitWindow(s);
    doFillBitWindow(s);
  }

  static void reload(State s) {
    if (s.bitOffset == BITNESS) {
      prepare(s);
    }
  }

  static void jumpToByteBoundary(State s) {
    int padding = (BITNESS - s.bitOffset) & 7;
    if (padding != 0) {
      int paddingBits = readFewBits(s, padding);
      if (paddingBits != 0) {
        throw new BrotliRuntimeException("Corrupted padding bits");
      }
    }
  }

  static int halfAvailable(State s) {
    int limit = HALVES_CAPACITY;
    if (s.endOfStreamReached != 0) {
      limit = (s.tailBytes + (HALF_SIZE - 1)) >> LOG_HALF_SIZE;
    }
    return limit - s.halfOffset;
  }

  static void copyBytes(State s, byte[] data, int offset, int length) {
    if ((s.bitOffset & 7) != 0) {
      throw new BrotliRuntimeException("Unaligned copyBytes");
    }

    // Drain accumulator.
    while ((s.bitOffset != BITNESS) && (length != 0)) {
      data[offset++] = (byte) peekBits(s);
      s.bitOffset += 8;
      length--;
    }
    if (length == 0) {
      return;
    }

    // Get data from shadow buffer with "sizeof(int)" granularity.
    int copyNibbles = Math.min(halfAvailable(s), length >> LOG_HALF_SIZE);
    if (copyNibbles > 0) {
      int readOffset = s.halfOffset << LOG_HALF_SIZE;
      int delta = copyNibbles << LOG_HALF_SIZE;
      System.arraycopy(s.byteBuffer, readOffset, data, offset, delta);
      offset += delta;
      length -= delta;
      s.halfOffset += copyNibbles;
    }
    if (length == 0) {
      return;
    }

    // Read tail bytes.
    if (halfAvailable(s) > 0) {
      // length = 1..3
      fillBitWindow(s);
      while (length != 0) {
        data[offset++] = (byte) peekBits(s);
        s.bitOffset += 8;
        length--;
      }
      checkHealth(s, 0);
      return;
    }

    // Now it is possible to copy bytes directly.
    while (length > 0) {
      int len = Utils.readInput(s.input, data, offset, length);
      if (len == -1) {
        throw new BrotliRuntimeException("Unexpected end of input");
      }
      offset += len;
      length -= len;
    }
  }

  /**
   * Translates bytes to halves (int/short).
   */
  static void bytesToNibbles(State s, int byteLen) {
    byte[] byteBuffer = s.byteBuffer;
    int halfLen = byteLen >> LOG_HALF_SIZE;
    if (BITNESS == 64) {
      int[] intBuffer = s.intBuffer;
      for (int i = 0; i < halfLen; ++i) {
        intBuffer[i] = ((byteBuffer[i * 4] & 0xFF))
            | ((byteBuffer[(i * 4) + 1] & 0xFF) << 8)
            | ((byteBuffer[(i * 4) + 2] & 0xFF) << 16)
            | ((byteBuffer[(i * 4) + 3] & 0xFF) << 24);
      }
    } else {
      short[] shortBuffer = s.shortBuffer;
      for (int i = 0; i < halfLen; ++i) {
        shortBuffer[i] = (short) ((byteBuffer[i * 2] & 0xFF)
            | ((byteBuffer[(i * 2) + 1] & 0xFF) << 8));
      }
    }
  }
}
