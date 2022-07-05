// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.ziputils;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Static utility methods pertaining to {@code short} primitives that interpret values as
 * <i>unsigned</i> (that is, any negative value {@code x} is treated as the positive value {@code
 * 2^16 + x}).
 *
 * <p>Users of these utilities must be <i>extremely careful</i> not to mix up signed and unsigned
 * {@code short} values.
 */
public final class UnsignedShorts {
  public static final int SHORT_MASK = 0xffff;

  private UnsignedShorts() {}

  /**
   * Returns the value of the given {@code short} as a {@code int}, when treated as unsigned.
   *
   * <p><b>Java 8 users:</b> use {@link Short#toUnsignedInt(short)} instead.
   */
  public static int toInt(short value) {
    return value & SHORT_MASK;
  }

  /**
   * Returns the {@code short} value that, when treated as unsigned, is equal to {@code value}, if
   * possible.
   *
   * @param value a value between 0 and 2<sup>16</sup>-1 inclusive
   * @return the {@code short} value that, when treated as unsigned, equals {@code value}
   * @throws IllegalArgumentException if {@code value} is negative or greater than or equal to
   *     2<sup>16</sup>
   */
  public static short checkedCast(int value) {
    checkArgument((value >> Short.SIZE) == 0, "out of range: %s", value);
    return (short) value;
  }
}
