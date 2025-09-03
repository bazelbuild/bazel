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

package net.starlark.java.eval;

import java.math.BigInteger;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.types.StarlarkType;
import net.starlark.java.types.Types;

/** The Starlark int data type. */
@StarlarkBuiltin(
    name = "int",
    category = "core",
    doc =
        "The type of integers in Starlark. Starlark integers may be of any magnitude; arithmetic"
            + " is exact. Examples of integer expressions:<br>"
            + "<pre class=\"language-python\">153\n"
            + "0x2A  # hexadecimal literal\n"
            + "0o54  # octal literal\n"
            + "23 * 2 + 5\n"
            + "100 / -7\n"
            + "100 % -7  # -5 (unlike in some other languages)\n"
            + "int(\"18\")\n"
            + "</pre>")
public abstract class StarlarkInt implements StarlarkValue, Comparable<StarlarkInt> {
  @Override
  public StarlarkType getStarlarkType() {
    return Types.INT;
  }

  // A cache of small integers >= LEAST_SMALLINT.
  private static final int LEAST_SMALLINT = -128;
  private static final Int32[] smallints = new Int32[100_000];

  static final StarlarkInt ZERO = StarlarkInt.of(0);
  private static final StarlarkInt ONE = StarlarkInt.of(1);
  private static final StarlarkInt MINUS_ONE = StarlarkInt.of(-1);

  /** Only nested classes of {@code StarlarkInt} are allowed to inherit it. */
  private StarlarkInt() {}

  /** Returns the Starlark int value that represents x. */
  public static StarlarkInt of(int x) {
    int index = x - LEAST_SMALLINT; // (may overflow)
    if (0 <= index && index < smallints.length) {
      Int32 xi = smallints[index];
      if (xi == null) {
        xi = new Int32(x);
        smallints[index] = xi;
      }
      return xi;
    }
    return new Int32(x);
  }

  /** Returns the Starlark int value that represents x. */
  public static StarlarkInt of(long x) {
    if ((long) (int) x == x) {
      return StarlarkInt.of((int) x);
    }
    return new Int64(x);
  }

  /** Returns the Starlark int value that represents x. */
  public static StarlarkInt of(BigInteger x) {
    if (x.bitLength() < 64) {
      return StarlarkInt.of(x.longValue());
    }
    return new Big(x);
  }

  /**
   * Returns the StarlarkInt value that most closely approximates x.
   *
   * @throws IllegalArgumentException is x is not finite.
   */
  static StarlarkInt ofFiniteDouble(double x) {
    return StarlarkFloat.finiteDoubleToIntExact(x);
  }

  /**
   * Returns the int denoted by a literal string in the specified base, as if by the Starlark
   * expression {@code int(s, base)}.
   *
   * @throws NumberFormatException if the input is invalid.
   */
  public static StarlarkInt parse(String s, int base) {
    String stringForErrors = s;

    if (s.isEmpty()) {
      throw new NumberFormatException("empty string");
    }

    // +/- prefix?
    boolean isNegative = false;
    char c = s.charAt(0);
    if (c == '+') {
      s = s.substring(1);
    } else if (c == '-') {
      s = s.substring(1);
      isNegative = true;
    }

    String digits = s;

    // 0b 0o 0x prefix?
    if (s.length() > 1 && s.charAt(0) == '0') {
      int prefixBase = 0;
      c = s.charAt(1);
      if (c == 'b' || c == 'B') {
        prefixBase = 2;
      } else if (c == 'o' || c == 'O') {
        prefixBase = 8;
      } else if (c == 'x' || c == 'X') {
        prefixBase = 16;
      }
      if (prefixBase != 0) {
        if (base == 0 || base == prefixBase) {
          base = prefixBase;
          digits = s.substring(2); // strip prefix
        }
      }
    }

    // No prefix, no base? Use decimal.
    if (digits == s && base == 0) {
      // Don't infer base when input starts with '0' due to octal/decimal ambiguity.
      if (s.length() > 1 && s.charAt(0) == '0') {
        throw new NumberFormatException(
            "cannot infer base when string begins with a 0: " + Starlark.repr(stringForErrors));
      }
      base = 10;
    }
    if (base < 2 || base > 36) {
      throw new NumberFormatException(
          String.format("invalid base %d (want 2 <= base <= 36)", base));
    }

    // Do not allow Long.parseLong and new BigInteger to accept another +/- sign.
    if (digits.startsWith("+") || digits.startsWith("-")) {
      throw new NumberFormatException(
          String.format("invalid base-%d literal: %s", base, Starlark.repr(stringForErrors)));
    }

    StarlarkInt result;
    try {
      result = StarlarkInt.of(Long.parseLong(digits, base));
    } catch (NumberFormatException unused1) {
      try {
        result = StarlarkInt.of(new BigInteger(digits, base));
      } catch (NumberFormatException unused2) {
        throw new NumberFormatException(
            String.format("invalid base-%d literal: %s", base, Starlark.repr(stringForErrors)));
      }
    }
    return isNegative ? StarlarkInt.uminus(result) : result;
  }

  // Subclass for values exactly representable in a Java int.
  private static final class Int32 extends StarlarkInt {
    final int v;

    Int32(int v) {
      this.v = v;
    }

    @Override
    public int toInt(String what) {
      return v;
    }

    @Override
    public long toLong(String what) {
      return (long) v;
    }

    @Override
    protected long toLongFast() {
      return (long) v;
    }

    @Override
    public BigInteger toBigInteger() {
      return BigInteger.valueOf(v);
    }

    @Override
    public Number toNumber() {
      return v;
    }

    @Override
    public int signum() {
      return Integer.signum(v);
    }

    @Override
    public void repr(Printer printer) {
      printer.append(v);
    }

    @Override
    public int hashCode() {
      return 0x316c5239 * Integer.hashCode(v) ^ 0x67c4a7d5;
    }

    @Override
    public boolean equals(Object that) {
      return (that instanceof Int32 && this.v == ((Int32) that).v)
          || (that instanceof StarlarkFloat && intEqualsFloat(this, (StarlarkFloat) that));
    }
  }

  // Subclass for values exactly representable in a Java long.
  private static final class Int64 extends StarlarkInt {
    final long v;

    Int64(long v) {
      this.v = v;
    }

    @Override
    public long toLong(String what) {
      return v;
    }

    @Override
    protected long toLongFast() {
      return v;
    }

    @Override
    public BigInteger toBigInteger() {
      return BigInteger.valueOf(v);
    }

    @Override
    public Number toNumber() {
      return v;
    }

    @Override
    public int signum() {
      return Long.signum(v);
    }

    @Override
    public void repr(Printer printer) {
      printer.append(v);
    }

    @Override
    public int hashCode() {
      return 0x67c4a7d5 * Long.hashCode(v) ^ 0xee914a1b;
    }

    @Override
    public boolean equals(Object that) {
      return (that instanceof Int64 && this.v == ((Int64) that).v)
          || (that instanceof StarlarkFloat && intEqualsFloat(this, (StarlarkFloat) that));
    }
  }

  // Subclass for values not exactly representable in a long.
  private static final class Big extends StarlarkInt {
    final BigInteger v;

    Big(BigInteger v) {
      this.v = v;
    }

    @Override
    public BigInteger toBigInteger() {
      return v;
    }

    @Override
    public Number toNumber() {
      return v;
    }

    @Override
    public int signum() {
      return v.signum();
    }

    @Override
    public void repr(Printer printer) {
      printer.append(v.toString());
    }

    @Override
    public int hashCode() {
      return 0xee914a1b * v.hashCode() ^ 0x6406918f;
    }

    @Override
    public boolean equals(Object that) {
      return (that instanceof Big && this.v.equals(((Big) that).v))
          || (that instanceof StarlarkFloat && intEqualsFloat(this, (StarlarkFloat) that));
    }
  }

  /** Returns the value of this StarlarkInt as a Number (Integer, Long, or BigInteger). */
  public abstract Number toNumber();

  /** Returns the signum of this StarlarkInt (-1, 0, or +1). */
  public abstract int signum();

  /** Returns this StarlarkInt as a string of decimal digits. */
  @Override
  public String toString() {
    if (this instanceof Int32) {
      return Integer.toString(((Int32) this).v);
    } else if (this instanceof Int64) {
      return Long.toString(((Int64) this).v);
    } else {
      return toBigInteger().toString();
    }
  }

  @Override
  public abstract void repr(Printer printer);

  /** Returns the signed int32 value of this StarlarkInt, or fails if not exactly representable. */
  public int toInt(String what) throws EvalException {
    throw Starlark.errorf("got %s for %s, want value in signed 32-bit range", this, what);
  }

  /** Returns the signed int64 value of this StarlarkInt, or fails if not exactly representable. */
  public long toLong(String what) throws EvalException {
    throw Starlark.errorf("got %s for %s, want value in the signed 64-bit range", this, what);
  }

  // A preallocated exception used to indicate overflow errors without the cost of allocation.
  private static final Overflow OVERFLOW = new Overflow();

  private static final class Overflow extends Exception {}

  /**
   * Similar to {@link #toLong(String)}, but faster: exception is not allocated and stack trace is
   * not collected.
   */
  protected long toLongFast() throws Overflow {
    throw OVERFLOW;
  }

  /** Returns the nearest IEEE-754 double-precision value closest to this int, which may be ±Inf. */
  public double toDouble() {
    if (this instanceof Int32) {
      return ((Int32) this).v;
    } else if (this instanceof Int64) {
      return ((Int64) this).v;
    } else {
      return toBigInteger().doubleValue(); // may be ±Inf
    }
  }

  /**
   * Returns the nearest IEEE-754 double-precision value closest to this int.
   *
   * @throws EvalException is the int is to large to represent as a finite float value.
   */
  public double toFiniteDouble() throws EvalException {
    double d = toDouble();
    if (!Double.isFinite(d)) {
      throw Starlark.errorf("int too large to convert to float");
    }
    return d;
  }

  /** Returns the BigInteger value of this StarlarkInt. */
  public abstract BigInteger toBigInteger();

  /**
   * Returns the value of this StarlarkInt as a Java signed 32-bit int.
   *
   * @throws IllegalArgumentException if this int is not in that value range.
   */
  public final int toIntUnchecked() throws IllegalArgumentException {
    if (this instanceof Int32) {
      return ((Int32) this).v;
    }
    // This operator is provided for fast access and case discrimination.
    // Use toInt(String) for user-visible errors.
    throw new IllegalArgumentException("not a signed 32-bit value");
  }

  /** Returns the result of truncating this value into the signed 32-bit range. */
  public int truncateToInt() {
    if (this instanceof Int32) {
      return ((Int32) this).v;
    } else if (this instanceof Int64) {
      return (int) ((Int64) this).v;
    } else {
      return toBigInteger().intValue();
    }
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public boolean truth() {
    return this != ZERO;
  }

  @Override
  public int compareTo(StarlarkInt x) {
    return compare(this, x);
  }

  // binary operators

  /** Returns signum(x - y). */
  public static int compare(StarlarkInt x, StarlarkInt y) {
    // If both arguments are big, we compare BigIntegers.
    // If neither argument is big, we compare longs.
    // If only one argument is big, its magnitude is greater
    // than the other operand, so only its sign matters.
    //
    // We avoid unnecessary branches.
    try {
      long xl = x.toLongFast();
      try {
        long yl = y.toLongFast();
        return Long.compare(xl, yl); // (long, long)
      } catch (Overflow unused) {
        return -((Big) y).v.signum(); // (long, big)
      }
    } catch (Overflow unused) {
      return y instanceof Big
          ? ((Big) x).v.compareTo(((Big) y).v) // (big, big)
          : ((Big) x).v.signum(); // (big, long)
    }
  }

  /** Returns x + y. */
  public static StarlarkInt add(StarlarkInt x, StarlarkInt y) {
    if (x instanceof Int32 && y instanceof Int32) {
      long xl = ((Int32) x).v;
      long yl = ((Int32) y).v;
      return StarlarkInt.of(xl + yl);
    }

    // We avoid Math.addExact and its overheads of exception allocation.
    try {
      long xl = x.toLongFast();
      long yl = y.toLongFast();
      long zl = xl + yl;
      boolean overflow = ((xl ^ zl) & (yl ^ zl)) < 0; // see Hacker's Delight, chapter 2
      if (!overflow) {
        return StarlarkInt.of(zl);
      }
    } catch (Overflow unused) {
      /* fall through */
    }

    BigInteger xbig = x.toBigInteger();
    BigInteger ybig = y.toBigInteger();
    BigInteger zbig = xbig.add(ybig);
    return StarlarkInt.of(zbig);
  }

  /** Returns x - y. */
  public static StarlarkInt subtract(StarlarkInt x, StarlarkInt y) {
    if (x instanceof Int32 && y instanceof Int32) {
      long xl = ((Int32) x).v;
      long yl = ((Int32) y).v;
      return StarlarkInt.of(xl - yl);
    }

    // We avoid Math.subtractExact and its overhead of exception allocation.
    try {
      long xl = x.toLongFast();
      long yl = y.toLongFast();
      long zl = xl - yl;
      boolean overflow = ((xl ^ yl) & (xl ^ zl)) < 0; // see Hacker's Delight, chapter 2
      if (!overflow) {
        return StarlarkInt.of(zl);
      }
    } catch (Overflow unused) {
      /* fall through */
    }

    BigInteger xbig = x.toBigInteger();
    BigInteger ybig = y.toBigInteger();
    BigInteger zbig = xbig.subtract(ybig);
    return StarlarkInt.of(zbig);
  }

  /** Returns x * y. */
  public static StarlarkInt multiply(StarlarkInt x, StarlarkInt y) {
    // Fast path for common case: int32 * int32.
    if (x instanceof Int32 && y instanceof Int32) {
      long xl = ((Int32) x).v;
      long yl = ((Int32) y).v;
      return StarlarkInt.of(xl * yl);
    }

    try {
      long xl = x.toLongFast();
      long yl = y.toLongFast();

      // Signed int128 multiplication, using Hacker's Delight 8-2
      // (High-Order Half of 64-Bit Product) extended to 128 bits.
      // TODO(adonovan): use Math.multiplyHigh when Java 9 becomes available.
      long xlo = xl & 0xFFFFFFFFL;
      long xhi = xl >> 32;
      long ylo = yl & 0xFFFFFFFFL;
      long yhi = yl >> 32;
      long zlo = xlo * ylo;
      long t = xhi * ylo + (zlo >>> 32);
      long z1 = t & 0xFFFFFFFFL;
      long z2 = t >> 32;
      z1 += xlo * yhi;

      // high and low arms of result
      long z128hi = xhi * yhi + z2 + (z1 >> 32);
      long z128lo = xl * yl;

      // Check int128 result is within int64 range.
      if (z128hi == (z128lo & Long.MIN_VALUE) >> 63) {
        return StarlarkInt.of(z128lo);
      }

      /* overflow */

    } catch (Overflow unused) {
      /* fall through */
    }

    // Avoid unnecessary conversion to BigInteger if the other operand is -1, 0, 1.
    // (Also makes self-test below faster.)
    if (x == ZERO || y == ONE) {
      return x;
    } else if (y == ZERO || x == ONE) {
      return y;
    } else if (x == MINUS_ONE) {
      return StarlarkInt.uminus(y);
    } else if (y == MINUS_ONE) {
      return StarlarkInt.uminus(x);
    }

    BigInteger xbig = x.toBigInteger();
    BigInteger ybig = y.toBigInteger();
    BigInteger zbig = xbig.multiply(ybig);
    StarlarkInt z = StarlarkInt.of(zbig);
    // cheap self-test
    if (!(z instanceof Big)) {
      throw new AssertionError(
          String.format(
              "bug in multiplication: %s * %s = %s, must be long multiplication", x, y, z));
    }
    return z;
  }

  /** Returns x // y (floor of integer division). */
  public static StarlarkInt floordiv(StarlarkInt x, StarlarkInt y) throws EvalException {
    if (y == ZERO) {
      throw Starlark.errorf("integer division by zero");
    }
    try {
      long xl = x.toLongFast();
      long yl = y.toLongFast();
      // http://python-history.blogspot.com/2010/08/why-pythons-integer-division-floors.html
      if (xl == Long.MIN_VALUE && yl == -1) {
        /* sole case in which quotient doesn't fit in long */
      } else {
        long quo = Math.floorDiv(xl, yl);
        return StarlarkInt.of(quo);
      }
      /* overflow */
    } catch (Overflow unused) {
      /* fall through */
    }

    BigInteger xbig = x.toBigInteger();
    BigInteger ybig = y.toBigInteger();
    BigInteger[] quorem = xbig.divideAndRemainder(ybig);
    if ((xbig.signum() < 0) != (ybig.signum() < 0) && quorem[1].signum() != 0) {
      quorem[0] = quorem[0].subtract(BigInteger.ONE);
    }
    return StarlarkInt.of(quorem[0]);
  }

  /** Returns x % y. */
  public static StarlarkInt mod(StarlarkInt x, StarlarkInt y) throws EvalException {
    if (y == ZERO) {
      throw Starlark.errorf("integer modulo by zero");
    }
    try {
      long xl = x.toLongFast();
      long yl = y.toLongFast();
      // In Starlark, the sign of the result is the sign of the divisor.
      return StarlarkInt.of(Math.floorMod(xl, yl));
    } catch (Overflow unused) {
      /* fall through */
    }

    BigInteger xbig = x.toBigInteger();
    BigInteger ybig = y.toBigInteger();
    BigInteger zbig = xbig.remainder(ybig);
    if ((x.signum() < 0) != (y.signum() < 0) && zbig.signum() != 0) {
      zbig = zbig.add(ybig);
    }
    return StarlarkInt.of(zbig);
  }

  /** Returns x >> y. */
  public static StarlarkInt shiftRight(StarlarkInt x, StarlarkInt y) throws EvalException {
    int yi = y.toInt("shift count");
    if (yi < 0) {
      throw Starlark.errorf("negative shift count: %d", yi);
    }
    try {
      long xl = x.toLongFast();
      if (yi >= Long.SIZE) {
        return xl < 0 ? StarlarkInt.of(-1) : ZERO;
      }
      return StarlarkInt.of(xl >> yi);
    } catch (Overflow unused) {
      /* fall through */
    }

    BigInteger xbig = x.toBigInteger();
    BigInteger zbig = xbig.shiftRight(yi);
    return StarlarkInt.of(zbig);
  }

  /** Returns x << y. */
  public static StarlarkInt shiftLeft(StarlarkInt x, StarlarkInt y) throws EvalException {
    int yi = y.toInt("shift count");
    if (yi < 0) {
      throw Starlark.errorf("negative shift count: %d", yi);
    } else if (yi >= 512) {
      throw Starlark.errorf("shift count too large: %d", yi);
    }
    try {
      long xl = x.toLongFast();
      long z = xl << yi; // only uses low 6 bits of yi
      if ((z >> yi) == xl && yi < 64) {
        return StarlarkInt.of(z);
      }
      /* overflow */
    } catch (Overflow unused) {
      /* fall through */
    }

    BigInteger xbig = x.toBigInteger();
    BigInteger zbig = xbig.shiftLeft(yi);
    return StarlarkInt.of(zbig);
  }

  /** Returns x ^ y. */
  public static StarlarkInt xor(StarlarkInt x, StarlarkInt y) {
    try {
      long xl = x.toLongFast();
      long yl = y.toLongFast();
      return StarlarkInt.of(xl ^ yl);
    } catch (Overflow unused) {
      /* fall through */
    }

    BigInteger xbig = x.toBigInteger();
    BigInteger ybig = y.toBigInteger();
    BigInteger zbig = xbig.xor(ybig);
    return StarlarkInt.of(zbig);
  }

  /** Returns x | y. */
  public static StarlarkInt or(StarlarkInt x, StarlarkInt y) {
    try {
      long xl = x.toLongFast();
      long yl = y.toLongFast();
      return StarlarkInt.of(xl | yl);
    } catch (Overflow unused) {
      /* fall through */
    }

    BigInteger xbig = x.toBigInteger();
    BigInteger ybig = y.toBigInteger();
    BigInteger zbig = xbig.or(ybig);
    return StarlarkInt.of(zbig);
  }

  /** Returns x & y. */
  public static StarlarkInt and(StarlarkInt x, StarlarkInt y) {
    try {
      long xl = x.toLongFast();
      long yl = y.toLongFast();
      return StarlarkInt.of(xl & yl);
    } catch (Overflow unused) {
      /* fall through */
    }

    BigInteger xbig = x.toBigInteger();
    BigInteger ybig = y.toBigInteger();
    BigInteger zbig = xbig.and(ybig);
    return StarlarkInt.of(zbig);
  }

  /** Returns ~x. */
  public static StarlarkInt bitnot(StarlarkInt x) {
    try {
      long xl = x.toLongFast();
      return StarlarkInt.of(~xl);
    } catch (Overflow unused) {
      /* fall through */
    }

    BigInteger xbig = ((Big) x).v;
    return StarlarkInt.of(xbig.not());
  }

  /** Returns -x. */
  public static StarlarkInt uminus(StarlarkInt x) {
    if (x instanceof Int32) {
      long xl = ((Int32) x).v;
      return StarlarkInt.of(-xl);
    }

    if (x instanceof Int64) {
      long xl = ((Int64) x).v;
      if (xl != Long.MIN_VALUE) {
        return StarlarkInt.of(-xl);
      }
    }

    BigInteger xbig = x.toBigInteger();
    BigInteger ybig = xbig.negate();
    return StarlarkInt.of(ybig);
  }

  /** Reports whether int x exactly equals float y. */
  static boolean intEqualsFloat(StarlarkInt x, StarlarkFloat y) {
    double yf = y.toDouble();
    return !Double.isNaN(yf) && compareIntAndDouble(x, yf) == 0;
  }

  /** Returns an exact three-valued comparison of int x with (non-NaN) double y. */
  static int compareIntAndDouble(StarlarkInt x, double y) {
    if (Double.isInfinite(y)) {
      return y > 0 ? -1 : +1;
    }

    // For Int32 and some Int64s, the toDouble conversion is exact.
    if (x instanceof StarlarkInt.Int32
        || (x instanceof StarlarkInt.Int64 && longHasExactDouble(((Int64) x).v))) {
      // Avoid Double.compare: it believes -0.0 < 0.0.
      double xf = x.toDouble();
      if (xf > y) {
        return +1;
      } else if (xf < y) {
        return -1;
      }
      return 0;
    }

    // If signs differ, we needn't look at magnitude.
    int xsign = x.signum();
    int ysign = (int) Math.signum(y);
    if (xsign > ysign) {
      return +1;
    } else if (xsign < ysign) {
      return -1;
    }

    // Left-shift either the int or the float mantissa,
    // then compare the resulting integers.
    int shift = StarlarkFloat.getShift(y);
    BigInteger xbig = x.toBigInteger();
    if (shift < 0) {
      xbig = xbig.shiftLeft(-shift);
    }
    BigInteger ybig = BigInteger.valueOf(StarlarkFloat.getMantissa(y));
    if (shift > 0) {
      ybig = ybig.shiftLeft(shift);
    }
    return xbig.compareTo(ybig);
  }

  private static boolean longHasExactDouble(long x) {
    return (long) (double) x == x;
  }
}
