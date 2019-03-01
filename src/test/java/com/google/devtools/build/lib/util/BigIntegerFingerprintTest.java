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

package com.google.devtools.build.lib.util;

import com.google.common.testing.EqualsTester;
import java.math.BigInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BigIntegerFingerprint}, mostly just checking basic collision avoidance. */
@RunWith(JUnit4.class)
public class BigIntegerFingerprintTest {
  @Test
  public void noClashes() {
    BigInteger long2Then1 = new BigIntegerFingerprint().addLong(2L).addLong(1L).getFingerprint();
    BigInteger long1Then32 =
        new BigIntegerFingerprint()
            .addLong(1L)
            .addLong(BigIntegerFingerprintUtils.RELATIVE_PRIME_INT + 1L)
            .getFingerprint();
    BigInteger trueThenFalse =
        new BigIntegerFingerprint().addBoolean(true).addBoolean(false).getFingerprint();
    BigInteger falseThen32 =
        new BigIntegerFingerprint()
            .addBoolean(false)
            .addLong(BigIntegerFingerprintUtils.RELATIVE_PRIME_INT + 1L)
            .getFingerprint();
    BigInteger bigIntOneThen0 =
        new BigIntegerFingerprint()
            .addBigIntegerOrdered(BigInteger.ONE)
            .addLong(0L)
            .getFingerprint();
    BigInteger long0ThenBigIntOne =
        new BigIntegerFingerprint()
            .addLong(0L)
            .addBigIntegerOrdered(BigInteger.ONE)
            .getFingerprint();
    BigInteger bigIntOneThenTrue =
        new BigIntegerFingerprint()
            .addBigIntegerOrdered(BigInteger.ONE)
            .addBoolean(true)
            .getFingerprint();
    BigInteger trueThenBigIntOne =
        new BigIntegerFingerprint()
            .addBoolean(true)
            .addBigIntegerOrdered(BigInteger.ONE)
            .getFingerprint();
    BigInteger long0ThenBigIntZero =
        new BigIntegerFingerprint()
            .addLong(0L)
            .addBigIntegerOrdered(BigInteger.ZERO)
            .getFingerprint();
    BigInteger bigIntZeroThenLong0 =
        new BigIntegerFingerprint()
            .addBigIntegerOrdered(BigInteger.ZERO)
            .addLong(1L)
            .getFingerprint();
    new EqualsTester()
        .addEqualityGroup(long2Then1)
        .addEqualityGroup(long1Then32)
        .addEqualityGroup(trueThenFalse)
        .addEqualityGroup(falseThen32)
        .addEqualityGroup(bigIntOneThen0)
        .addEqualityGroup(long0ThenBigIntOne)
        .addEqualityGroup(bigIntOneThenTrue)
        .addEqualityGroup(trueThenBigIntOne)
        .addEqualityGroup(long0ThenBigIntZero)
        .addEqualityGroup(bigIntZeroThenLong0)
        .testEquals();
  }
}
