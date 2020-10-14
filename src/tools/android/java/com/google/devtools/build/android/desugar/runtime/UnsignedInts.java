// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.runtime;

/**
 * Static utility methods pertaining to {@code int} primitives that interpret values as
 * <i>unsigned</i> (that is, any negative value {@code x} is treated as the positive value {@code
 * 2^32 + x}). The methods for which signedness is not an issue are in {@link Ints}, as well as
 * signed versions of methods for which signedness is an issue.
 *
 * <p>See the Guava User Guide article on <a
 * href="https://github.com/google/guava/wiki/PrimitivesExplained#unsigned-support">unsigned
 * primitive utilities</a>.
 */
public final class UnsignedInts {
  static final long INT_MASK = 0xffffffffL;

  private UnsignedInts() {}

  private static int flip(int value) {
    return value ^ Integer.MIN_VALUE;
  }

  /**
   * Compares the two specified {@code int} values, treating them as unsigned values between {@code
   * 0} and {@code 2^32 - 1} inclusive.
   *
   * @param a the first unsigned {@code int} to compare
   * @param b the second unsigned {@code int} to compare
   * @return a negative value if {@code a} is less than {@code b}; a positive value if {@code a} is
   *     greater than {@code b}; or zero if they are equal
   */
  /*visible for testing*/ static int compare(int a, int b) {
    a = flip(a);
    b = flip(b);
    return (a < b) ? -1 : ((a > b) ? 1 : 0);
  }

  /**
   * Returns the value of the given {@code int} as a {@code long}, when treated as unsigned.
   */
  /*visible for testing*/ static long toLong(int value) {
    return value & INT_MASK;
  }

  /**
   * Returns dividend / divisor, where the dividend and divisor are treated as unsigned 32-bit
   * quantities.
   *
   * @param dividend the dividend (numerator)
   * @param divisor the divisor (denominator)
   * @throws ArithmeticException if divisor is 0
   */
  public static int divide(int dividend, int divisor) {
    return (int) (toLong(dividend) / toLong(divisor));
  }

  /**
   * Returns dividend % divisor, where the dividend and divisor are treated as unsigned 32-bit
   * quantities.
   *
   * @param dividend the dividend (numerator)
   * @param divisor the divisor (denominator)
   * @throws ArithmeticException if divisor is 0
   */
  public static int remainder(int dividend, int divisor) {
    return (int) (toLong(dividend) % toLong(divisor));
  }

  /**
   * Returns a string representation of x, where x is treated as unsigned.
   */
  public static String toString(int x) {
    return toString(x, 10);
  }

  /**
   * Returns a string representation of {@code x} for the given radix, where {@code x} is treated as
   * unsigned.
   *
   * @param x the value to convert to a string.
   * @param radix the radix to use while working with {@code x}
   * @throws IllegalArgumentException if {@code radix} is not between {@link Character#MIN_RADIX}
   *     and {@link Character#MAX_RADIX}.
   */
  public static String toString(int x, int radix) {
    long asLong = x & INT_MASK;
    return Long.toString(asLong, radix);
  }
}
