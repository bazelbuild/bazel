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
package net.starlark.java.syntax;

import java.math.BigInteger;

/** Syntax node for an int literal. */
public final class IntLiteral extends Expression {
  private final String raw;
  private final int tokenOffset;
  private final Number value; // = Integer | Long | BigInteger

  IntLiteral(FileLocations locs, String raw, int tokenOffset, Number value) {
    super(locs, Kind.INT_LITERAL);
    this.raw = raw;
    this.tokenOffset = tokenOffset;
    this.value = value;
  }

  /**
   * Returns the value denoted by this literal as an Integer, Long, or BigInteger, using the
   * narrowest type capable of exactly representing the value.
   */
  public Number getValue() {
    return value;
  }

  /** Returns the raw source text of the literal. */
  public String getRaw() {
    return raw;
  }

  @Override
  public int getStartOffset() {
    return tokenOffset;
  }

  @Override
  public int getEndOffset() {
    return tokenOffset + raw.length();
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  /**
   * Returns the value denoted by a non-negative integer literal with an optional base prefix (but
   * no +/- sign), using the narrowest type of Integer, Long, or BigInteger capable of exactly
   * representing the value.
   *
   * @throws NumberFormatException if the string is not a valid literal.
   */
  public static Number scan(String str) {
    String orig = str;
    int radix = 10;
    if (str.length() > 1 && str.charAt(0) == '0') {
      switch (str.charAt(1)) {
        case 'x':
        case 'X':
          radix = 16;
          str = str.substring(2);
          break;
        case 'o':
        case 'O':
          radix = 8;
          str = str.substring(2);
          break;
        default:
          throw new NumberFormatException(
              "invalid octal literal: " + str + " (use '0o" + str.substring(1) + "')");
      }
    }

    try {
      long v = Long.parseLong(str, radix);
      if (v == (int) v) {
        return (int) v;
      }
      return v;
    } catch (NumberFormatException unused) {
      /* fall through */
    }
    try {
      return new BigInteger(str, radix);
    } catch (NumberFormatException unused) {
      throw new NumberFormatException("invalid base-" + radix + " integer literal: " + orig);
    }
  }
}
