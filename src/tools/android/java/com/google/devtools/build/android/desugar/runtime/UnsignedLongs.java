// Copyright 2019 The Bazel Authors. All rights reserved.
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
 * Static utility methods pertaining to {@code long} primitives that interpret values as
 * <i>unsigned</i> (that is, any negative value {@code x} is treated as the positive value {@code
 * 2^64 + x}), based on Guava's implementation.
 *
 * <p>See the Guava User Guide article on <a
 * href="https://github.com/google/guava/wiki/PrimitivesExplained#unsigned-support">unsigned
 * primitive utilities</a>.
 */
public final class UnsignedLongs {
  private UnsignedLongs() {}

  /**
   * A (self-inverse) bijection which converts the ordering on unsigned longs to the ordering on
   * longs, that is, {@code a <= b} as unsigned longs if and only if {@code flip(a) <= flip(b)} as
   * signed longs.
   */
  private static long flip(long a) {
    return a ^ Long.MIN_VALUE;
  }

  /**
   * Compares the two specified {@code long} values, treating them as unsigned values between {@code
   * 0} and {@code 2^64 - 1} inclusive.
   *
   * <p><b>Java 8 users:</b> use {@link Long#compareUnsigned(long, long)} instead.
   *
   * @param a the first unsigned {@code long} to compare
   * @param b the second unsigned {@code long} to compare
   * @return a negative value if {@code a} is less than {@code b}; a positive value if {@code a} is
   *     greater than {@code b}; or zero if they are equal
   */
  /*visible for testing*/ static int compare(long a, long b) {
    a = flip(a);
    b = flip(b);
    // TODO(kmb): Could use Long.compare here if we desugared with LongCompareMethodRewriter
    return (a < b) ? -1 : ((a > b) ? 1 : 0);
  }

  /**
   * Returns dividend / divisor, where the dividend and divisor are treated as unsigned 64-bit
   * quantities.
   *
   * @param dividend the dividend (numerator)
   * @param divisor the divisor (denominator)
   * @throws ArithmeticException if divisor is 0
   */
  public static long divideUnsigned(long dividend, long divisor) {
    if (divisor < 0) { // i.e., divisor >= 2^63:
      if (compare(dividend, divisor) < 0) {
        return 0; // dividend < divisor
      } else {
        return 1; // dividend >= divisor
      }
    }

    // Optimization - use signed division if dividend < 2^63
    if (dividend >= 0) {
      return dividend / divisor;
    }

    /*
     * Otherwise, approximate the quotient, check, and correct if necessary. Our approximation is
     * guaranteed to be either exact or one less than the correct value. This follows from fact that
     * floor(floor(x)/i) == floor(x/i) for any real x and integer i != 0. The proof is not quite
     * trivial.
     */
    long quotient = ((dividend >>> 1) / divisor) << 1;
    long rem = dividend - quotient * divisor;
    return quotient + (compare(rem, divisor) >= 0 ? 1 : 0);
  }

  /**
   * Returns dividend % divisor, where the dividend and divisor are treated as unsigned 64-bit
   * quantities.
   *
   * @param dividend the dividend (numerator)
   * @param divisor the divisor (denominator)
   * @throws ArithmeticException if divisor is 0
   */
  public static long remainderUnsigned(long dividend, long divisor) {
    if (divisor < 0) { // i.e., divisor >= 2^63:
      if (compare(dividend, divisor) < 0) {
        return dividend; // dividend < divisor
      } else {
        return dividend - divisor; // dividend >= divisor
      }
    }

    // Optimization - use signed modulus if dividend < 2^63
    if (dividend >= 0) {
      return dividend % divisor;
    }

    /*
     * Otherwise, approximate the quotient, check, and correct if necessary. Our approximation is
     * guaranteed to be either exact or one less than the correct value. This follows from the fact
     * that floor(floor(x)/i) == floor(x/i) for any real x and integer i != 0. The proof is not
     * quite trivial.
     */
    long quotient = ((dividend >>> 1) / divisor) << 1;
    long rem = dividend - quotient * divisor;
    return rem - (compare(rem, divisor) >= 0 ? divisor : 0);
  }
}
