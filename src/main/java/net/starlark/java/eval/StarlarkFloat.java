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
import java.util.Locale;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.types.StarlarkType;
import net.starlark.java.types.Types;

/** The Starlark float data type. */
@StarlarkBuiltin(
    name = "float",
    category = "core",
    doc = "The type of floating-point numbers in Starlark.")
public final class StarlarkFloat implements StarlarkValue, Comparable<StarlarkFloat> {
  @Override
  public StarlarkType getStarlarkType() {
    return Types.FLOAT;
  }

  private final double v;

  private StarlarkFloat(double v) {
    this.v = v;
  }

  /** Returns the Starlark float value that represents x. */
  public static StarlarkFloat of(double v) {
    return new StarlarkFloat(v);
  }

  /** Returns the value of this float. */
  public double toDouble() {
    return v;
  }

  @Override
  public String toString() {
    return format(v, 'g');
  }

  @Override
  public void repr(Printer printer) {
    printer.append(toString());
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public boolean truth() {
    return this.v != 0.0;
  }

  /**
   * Defines a total order over float values. Positive and negative zero values compare equal. NaN
   * compares equal to itself and greater than +Inf.
   */
  @Override
  public int compareTo(StarlarkFloat that) {
    double x = this.v;
    double y = that.v;
    if (x > y) {
      return +1;
    } else if (x < y) {
      return -1;
    } else if (x == y) {
      return 0; // 0.0 == -0.0
    }

    // At least one operand is NaN.
    // Canonicalize NaNs using doubleToLongBits and compare bits.
    long xbits = Double.doubleToLongBits(x);
    long ybits = Double.doubleToLongBits(y);
    return Long.compare(xbits, ybits); // NaN > non-NaN
  }

  @Override
  public int hashCode() {
    // Equal float and int values must yield the same hash.
    if (Double.isFinite(v) && v == Math.rint(v)) {
      return StarlarkInt.ofFiniteDouble(v).hashCode();
    }

    // For non-integral values we can use a cheaper hash.
    // Hashing the bits is consistent with equals
    // because v is neither 0.0 nor -0.0.
    long bits = Double.doubleToLongBits(v); // canonicalizes NaNs
    return (int) (bits ^ (bits >>> 32));
  }

  @Override
  public boolean equals(Object that) {
    return (that instanceof StarlarkFloat && equal(this.v, ((StarlarkFloat) that).v))
        || (that instanceof StarlarkInt && StarlarkInt.intEqualsFloat((StarlarkInt) that, this));
  }

  // equal is an equivalence relation consistent with hashCode and compareTo.
  private static boolean equal(double x, double y) {
    return x == y || (Double.isNaN(x) && Double.isNaN(y));
  }

  // Performs printf-style string conversion of a double value v.
  // conv is one of [efgEFG].
  static String format(double v, char conv) {
    if (!Double.isFinite(v)) {
      if (v == Double.POSITIVE_INFINITY) {
        return "+inf";
      } else if (v == Double.NEGATIVE_INFINITY) {
        return "-inf";
      } else {
        return "nan";
      }
    }

    String s;
    switch (conv) {
      case 'e':
        s = String.format(Locale.US, "%e", v);
        break;
      case 'E':
        s = String.format(Locale.US, "%E", v);
        break;
      case 'f':
      case 'F': // an alias
        s = String.format(Locale.US, "%f", v);
        break;
      case 'g':
        s = String.format(Locale.US, "%.17g", v); // use DBL_DECIMAL_DIG places
        break;
      case 'G':
        s = String.format(Locale.US, "%.17G", v);
        break;
      default:
        throw new IllegalArgumentException("unsupported conversion: " + conv);
    }

    // %g is the default format used by str.
    // It always includes a '.' or an 'e', to make clear that
    // the value is a float, not an int.
    //
    // TODO(adonovan): round the value to the minimal precision required
    // to avoid ambiguity. This requires computing the decimal digit
    // strings of the adjacent floating-point values and then taking the
    // shortest prefix sufficient to distinguish v from them, or using a
    // more sophisticated algorithm such as Florian Loitsch's Grisu3 or
    // Ulf Adams' Ryu.  (Is there an easy way to compute the required
    // precision without materializing the digits? If so we could delegate
    // to format("%*g", prec, v).)
    //
    // For now, we just clean up the output of Java's %.17g implementation,
    // which is unambiguous, but may yield unnecessarily long digit strings
    // such as 1000000000000.0.
    if (conv == 'g' || conv == 'G') {
      int e = s.indexOf(conv == 'g' ? 'e' : 'E');
      if (e < 0) {
        int dot = s.indexOf('.');
        if (dot < 0) {
          // Ensure result always has a decimal point if no exponent.
          // "123" -> "123.0"
          s += ".0";
        } else {
          // Remove trailing zeros after decimal point.
          // "1.110" => "1.11"
          // "1.000" => "1.0"
          int i;
          for (i = s.length() - 1; i > dot + 1 && s.charAt(i) == '0'; i--) {}
          s = s.substring(0, i + 1);
        }
      } else {
        // Remove trailing zeros from mantissa.
        // "1.23000e+45" => "1.23e+45"
        // "1.00000e+45" => "1e+45"
        int i;
        for (i = e - 1; s.charAt(i) == '0'; i--) {}
        if (s.charAt(i) == '.') {
          i--;
        }
        if (i < e - 1) {
          s =
              new StringBuilder(i + 1 + s.length() - e)
                  .append(s, 0, i + 1) // "1.23"
                  .append(s, e, s.length()) // "e+45"
                  .toString();
        }
      }
    }

    return s;
  }

  /** Returns x // y (floor of division). */
  static StarlarkFloat floordiv(double x, double y) throws EvalException {
    if (y == 0.0) {
      throw Starlark.errorf("integer division by zero");
    }
    return StarlarkFloat.of(Math.floor(x / y));
  }

  /** Returns x / y (floating-point division). */
  static StarlarkFloat div(double x, double y) throws EvalException {
    if (y == 0.0) {
      throw Starlark.errorf("floating-point division by zero");
    }
    return StarlarkFloat.of(x / y);
  }

  /** Returns x % y (floating-point remainder). */
  static StarlarkFloat mod(double x, double y) throws EvalException {
    if (y == 0.0) {
      throw Starlark.errorf("floating-point modulo by zero");
    }
    // In Starlark, the sign of the result is the sign of the divisor.
    double z = x % y;
    if ((x < 0) != (y < 0) && z != 0) {
      z += y;
    }
    return StarlarkFloat.of(z);
  }

  /**
   * Returns the Starlark int value closest to x, truncating towards zero.
   *
   * @throws IllegalArgumentException if x is not finite (NaN or ±Inf).
   */
  static StarlarkInt finiteDoubleToIntExact(double x) {
    // small value?
    if (Long.MIN_VALUE <= x && x <= Long.MAX_VALUE) {
      return StarlarkInt.of((long) x);
    }

    // Shift must be positive, because we just handled all small values.
    int shift = getShift(x);
    if (shift <= 0) {
      throw new IllegalStateException("non-positive shift");
    }

    // Shift mantissa by exponent.
    long mantissa = getMantissa(x);
    return StarlarkInt.of(BigInteger.valueOf(mantissa).shiftLeft(shift));
  }

  private static final int EXPONENT_MASK = (1 << 11) - 1;

  // Returns the effective signed mantissa of x.
  // Precondition: x is finite.
  static long getMantissa(double x) {
    long bits = Double.doubleToRawLongBits(x);
    long mantissa = bits & ((1L << 52) - 1);
    int exp = ((int) (bits >>> 52)) & EXPONENT_MASK;
    switch (exp) {
      case 0: // denormal
        break;
      case EXPONENT_MASK:
        throw new IllegalArgumentException("not finite: " + x);
      default: // normal
        mantissa |= 1L << 52;
        break;
    }
    return x < 0 ? -mantissa : mantissa;
  }

  // Returns the effective left (+) or right (-) shift required of the value returned by
  // getMantissa(x).
  // Precondition: x is finite.
  static int getShift(double x) {
    long bits = Double.doubleToRawLongBits(x);
    int exp = ((int) (bits >>> 52)) & EXPONENT_MASK;
    switch (exp) {
      case 0: // denormal
        exp -= 1022;
        break;
      case EXPONENT_MASK:
        throw new IllegalArgumentException("not finite: " + x);
      default: // normal
        exp -= 1023;
        break;
    }
    return exp - 52;
  }
}
