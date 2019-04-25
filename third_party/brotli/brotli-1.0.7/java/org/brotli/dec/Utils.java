/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.dec;

import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.nio.Buffer;

/**
 * A set of utility methods.
 */
final class Utils {

  private static final byte[] BYTE_ZEROES = new byte[1024];

  private static final int[] INT_ZEROES = new int[1024];

  /**
   * Fills byte array with zeroes.
   *
   * <p> Current implementation uses {@link System#arraycopy}, so it should be used for length not
   * less than 16.
   *
   * @param dest array to fill with zeroes
   * @param offset the first byte to fill
   * @param length number of bytes to change
   */
  static void fillBytesWithZeroes(byte[] dest, int start, int end) {
    int cursor = start;
    while (cursor < end) {
      int step = Math.min(cursor + 1024, end) - cursor;
      System.arraycopy(BYTE_ZEROES, 0, dest, cursor, step);
      cursor += step;
    }
  }

  /**
   * Fills int array with zeroes.
   *
   * <p> Current implementation uses {@link System#arraycopy}, so it should be used for length not
   * less than 16.
   *
   * @param dest array to fill with zeroes
   * @param offset the first item to fill
   * @param length number of item to change
   */
  static void fillIntsWithZeroes(int[] dest, int start, int end) {
    int cursor = start;
    while (cursor < end) {
      int step = Math.min(cursor + 1024, end) - cursor;
      System.arraycopy(INT_ZEROES, 0, dest, cursor, step);
      cursor += step;
    }
  }

  static void copyBytesWithin(byte[] bytes, int target, int start, int end) {
    System.arraycopy(bytes, start, bytes, target, end - start);
  }

  static int readInput(InputStream src, byte[] dst, int offset, int length) {
    try {
      return src.read(dst, offset, length);
    } catch (IOException e) {
      throw new BrotliRuntimeException("Failed to read input", e);
    }
  }

  static void closeInput(InputStream src) throws IOException {
    src.close();
  }

  static byte[] toUsAsciiBytes(String src) {
    try {
      // NB: String#getBytes(String) is present in JDK 1.1, while other variants require JDK 1.6 and
      // above.
      return src.getBytes("US-ASCII");
    } catch (UnsupportedEncodingException e) {
      throw new RuntimeException(e); // cannot happen
    }
  }

  // Crazy pills factory: code compiled for JDK8 does not work on JRE9.
  static void flipBuffer(Buffer buffer) {
    buffer.flip();
  }
}
