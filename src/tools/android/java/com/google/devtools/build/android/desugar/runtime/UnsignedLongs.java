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

  /** Returns a string representation of x, where x is treated as unsigned. */
  public static String toString(long x) {
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
  public static String toString(long x, int radix) {
    if (radix < Character.MIN_RADIX || radix > Character.MAX_RADIX) {
      throw new IllegalArgumentException(
          "radix (" + radix + ") must be between Character.MIN_RADIX and Character.MAX_RADIX");
    }
    if (x == 0) {
      // Simply return "0"
      return "0";
    } else if (x > 0) {
      return Long.toString(x, radix);
    } else {
      char[] buf = new char[64];
      int i = buf.length;
      if ((radix & (radix - 1)) == 0) {
        // Radix is a power of two so we can avoid division.
        int shift = Integer.numberOfTrailingZeros(radix);
        int mask = radix - 1;
        do {
          buf[--i] = Character.forDigit(((int) x) & mask, radix);
          x >>>= shift;
        } while (x != 0);
      } else {
        // Separate off the last digit using unsigned division. That will leave
        // a number that is nonnegative as a signed integer.
        long quotient;
        if ((radix & 1) == 0) {
          // Fast path for the usual case where the radix is even.
          quotient = (x >>> 1) / (radix >>> 1);
        } else {
          quotient = divideUnsigned(x, radix);
        }
        long rem = x - quotient * radix;
        buf[--i] = Character.forDigit((int) rem, radix);
        x = quotient;
        // Simple modulo/division approach
        while (x > 0) {
          buf[--i] = Character.forDigit((int) (x % radix), radix);
          x /= radix;
        }
      }
      // Generate string
      return new String(buf, i, buf.length - i);
    }
  }
}
