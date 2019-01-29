// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skyframe;

import java.math.BigInteger;
import javax.annotation.Nullable;

/** Utility class for fingerprint composition. */
public final class BigIntegerFingerprintUtils {
  private static final int BITS = 128;
  public static final int BYTES = BITS / 8;

  private static final BigInteger UINT128_LIMIT = BigInteger.ONE.shiftLeft(BITS);
  private static final BigInteger RELATIVE_PRIME = BigInteger.valueOf(31);

  private BigIntegerFingerprintUtils() {}

  public static BigInteger compose(BigInteger v1, BigInteger v2) {
    BigInteger temp = v1.add(v2);
    if (temp.compareTo(UINT128_LIMIT) >= 0) {
      return temp.subtract(UINT128_LIMIT);
    }
    return temp;
  }

  /**
   * Unordered, nullable composition.
   *
   * <p>null is absorbing and is used to convey errors and unavailability
   */
  @Nullable
  public static BigInteger composeNullable(@Nullable BigInteger v1, @Nullable BigInteger v2) {
    if (v1 == null || v2 == null) {
      return null;
    }
    return compose(v1, v2);
  }

  public static BigInteger composeOrdered(BigInteger accumulator, BigInteger v) {
    return compose(accumulator.multiply(RELATIVE_PRIME).mod(UINT128_LIMIT), v);
  }

  @Nullable
  public static BigInteger composeOrderedNullable(
      @Nullable BigInteger accumulator, @Nullable BigInteger v) {
    if (accumulator == null || v == null) {
      return null;
    }
    return compose(accumulator.multiply(RELATIVE_PRIME).mod(UINT128_LIMIT), v);
  }

}
