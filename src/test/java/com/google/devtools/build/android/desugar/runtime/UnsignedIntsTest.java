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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link UnsignedInts}. */
@RunWith(JUnit4.class)
public class UnsignedIntsTest {
  private static final long[] UNSIGNED_INTS = {
    0L,
    1L,
    2L,
    3L,
    0x12345678L,
    0x5a4316b8L,
    0x6cf78a4bL,
    0xff1a618bL,
    0xfffffffdL,
    0xfffffffeL,
    0xffffffffL
  };

  @Test
  public void testToLong() {
    for (long a : UNSIGNED_INTS) {
      assertEquals(a, UnsignedInts.toLong((int) a));
    }
  }

  @Test
  public void testCompare() {
    for (long a : UNSIGNED_INTS) {
      for (long b : UNSIGNED_INTS) {
        int cmpAsLongs = Long.compare(a, b);
        int cmpAsUInt = UnsignedInts.compare((int) a, (int) b);
        assertEquals(Integer.signum(cmpAsLongs), Integer.signum(cmpAsUInt));
      }
    }
  }

  @Test
  public void testDivide() {
    for (long a : UNSIGNED_INTS) {
      for (long b : UNSIGNED_INTS) {
        try {
          assertEquals((int) (a / b), UnsignedInts.divide((int) a, (int) b));
          assertFalse(b == 0);
        } catch (ArithmeticException e) {
          assertEquals(0, b);
        }
      }
    }
  }

  @Test
  public void testRemainder() {
    for (long a : UNSIGNED_INTS) {
      for (long b : UNSIGNED_INTS) {
        try {
          assertEquals((int) (a % b), UnsignedInts.remainder((int) a, (int) b));
          assertFalse(b == 0);
        } catch (ArithmeticException e) {
          assertEquals(0, b);
        }
      }
    }
  }

  @Test
  public void testDivideRemainderEuclideanProperty() {
    // Use a seed so that the test is deterministic:
    Random r = new Random(0L);
    for (int i = 0; i < 1000000; i++) {
      int dividend = r.nextInt();
      int divisor = r.nextInt();
      // Test that the Euclidean property is preserved:
      assertEquals(
          0,
          dividend
              - (divisor * UnsignedInts.divide(dividend, divisor)
                  + UnsignedInts.remainder(dividend, divisor)));
    }
  }

  @Test
  public void testToString() {
    int[] bases = {2, 5, 7, 8, 10, 16};
    for (long a : UNSIGNED_INTS) {
      for (int base : bases) {
        assertEquals(UnsignedInts.toString((int) a, base), Long.toString(a, base));
      }
    }
  }
}
