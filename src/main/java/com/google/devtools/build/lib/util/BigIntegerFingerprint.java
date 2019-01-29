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

package com.google.devtools.build.lib.util;

import com.google.devtools.build.lib.vfs.PathFragment;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import javax.annotation.Nullable;

/**
 * Wrapper for calculating a BigInteger fingerprint for an object. This BigInteger has a maximum of
 * 128 bits (16 bytes).
 */
public class BigIntegerFingerprint {

  // Limit for how big we want our BigInteger to be.
  private static final int BIT_LIMIT = 128;
  private static final int BYTE_LIMIT = BIT_LIMIT / 8;
  private static final BigInteger MAX_BIGINT = BigInteger.ONE.shiftLeft(BIT_LIMIT);
  private static final BigInteger BIGINT_TWO = BigInteger.valueOf(2);
  private static final BigInteger RELATIVE_PRIME = BigInteger.valueOf(31);

  // Non-final in order to manipulate and give an ordering.
  private BigInteger accumulator;

  public BigIntegerFingerprint() {
    this.accumulator = BigInteger.ONE;
  }

  public BigIntegerFingerprint addLong(long addition) {
    return addBigIntegerOrdered(BigInteger.valueOf(addition));
  }

  public BigIntegerFingerprint addString(String string) {
    return addBytes(string.getBytes(StandardCharsets.UTF_8));
  }

  public BigIntegerFingerprint addBytes(@Nullable byte[] bytes) {
    if (bytes == null) {
      return addBigIntegerOrdered(BigInteger.ZERO);
    }
    int numSegments = bytes.length / BYTE_LIMIT;
    if (numSegments == 0 || bytes.length == BYTE_LIMIT) {
      return addBigIntegerOrdered(new BigInteger(/*signum=*/ 1, bytes));
    }
    for (int segment = 0; segment < numSegments; segment++) {
      addBigIntegerOrdered(
          new BigInteger(
              /*signum=*/ 1,
              Arrays.copyOfRange(bytes, segment * BYTE_LIMIT, (segment + 1) * BYTE_LIMIT)));
    }
    if (numSegments * BYTE_LIMIT < bytes.length) {
      addBigIntegerOrdered(
          new BigInteger(
              /*signum=*/ 1, Arrays.copyOfRange(bytes, numSegments * BYTE_LIMIT, bytes.length)));
    }
    return this;
  }

  public BigIntegerFingerprint addBoolean(boolean bool) {
    if (bool) {
      addBigIntegerOrdered(BIGINT_TWO);
    } else {
      addBigIntegerOrdered(BigInteger.ONE);
    }
    return this;
  }

  public BigIntegerFingerprint addPath(PathFragment pathFragment) {
    return addString(pathFragment.getPathString());
  }

  private BigIntegerFingerprint addBigInteger(BigInteger bigInteger) {
    accumulator = accumulator.add(bigInteger).mod(MAX_BIGINT);
    return this;
  }

  public BigIntegerFingerprint addBigIntegerOrdered(BigInteger bigInteger) {
    accumulator = accumulator.multiply(RELATIVE_PRIME).mod(MAX_BIGINT);
    return addBigInteger(bigInteger);
  }

  public BigIntegerFingerprint addNullableBigIntegerOrdered(@Nullable BigInteger bigInteger) {
    if (bigInteger == null) {
      return this;
    }
    return addBigIntegerOrdered(bigInteger);
  }

  public BigInteger getFingerprint() {
    return accumulator;
  }

  public void reset() {
    accumulator = BigInteger.ONE;
  }
}
