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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link UnsignedLongs}. */
@RunWith(JUnit4.class)
public class UnsignedLongsTest {

  @Test
  public void testCompare() {
    // max value
    assertTrue(UnsignedLongs.compare(0, 0xffffffffffffffffL) < 0);
    assertTrue(UnsignedLongs.compare(0xffffffffffffffffL, 0) > 0);

    // both with high bit set
    assertTrue(UnsignedLongs.compare(0xff1a618b7f65ea12L, 0xffffffffffffffffL) < 0);
    assertTrue(UnsignedLongs.compare(0xffffffffffffffffL, 0xff1a618b7f65ea12L) > 0);

    // one with high bit set
    assertTrue(UnsignedLongs.compare(0x5a4316b8c153ac4dL, 0xff1a618b7f65ea12L) < 0);
    assertTrue(UnsignedLongs.compare(0xff1a618b7f65ea12L, 0x5a4316b8c153ac4dL) > 0);

    // neither with high bit set
    assertTrue(UnsignedLongs.compare(0x5a4316b8c153ac4dL, 0x6cf78a4b139a4e2aL) < 0);
    assertTrue(UnsignedLongs.compare(0x6cf78a4b139a4e2aL, 0x5a4316b8c153ac4dL) > 0);

    // same value
    assertTrue(UnsignedLongs.compare(0xff1a618b7f65ea12L, 0xff1a618b7f65ea12L) == 0);
  }

  @Test
  public void testDivide() {
    assertEquals(2, UnsignedLongs.divideUnsigned(14, 5));
    assertEquals(0, UnsignedLongs.divideUnsigned(0, 50));
    assertEquals(1, UnsignedLongs.divideUnsigned(0xfffffffffffffffeL, 0xfffffffffffffffdL));
    assertEquals(0, UnsignedLongs.divideUnsigned(0xfffffffffffffffdL, 0xfffffffffffffffeL));
    assertEquals(281479271743488L, UnsignedLongs.divideUnsigned(0xfffffffffffffffeL, 65535));
    assertEquals(0x7fffffffffffffffL, UnsignedLongs.divideUnsigned(0xfffffffffffffffeL, 2));
    assertEquals(3689348814741910322L, UnsignedLongs.divideUnsigned(0xfffffffffffffffeL, 5));
  }

  @Test
  public void testRemainder() {
    assertEquals(4, UnsignedLongs.remainderUnsigned(14, 5));
    assertEquals(0, UnsignedLongs.remainderUnsigned(0, 50));
    assertEquals(1, UnsignedLongs.remainderUnsigned(0xfffffffffffffffeL, 0xfffffffffffffffdL));
    assertEquals(
        0xfffffffffffffffdL,
        UnsignedLongs.remainderUnsigned(0xfffffffffffffffdL, 0xfffffffffffffffeL));
    assertEquals(65534L, UnsignedLongs.remainderUnsigned(0xfffffffffffffffeL, 65535));
    assertEquals(0, UnsignedLongs.remainderUnsigned(0xfffffffffffffffeL, 2));
    assertEquals(4, UnsignedLongs.remainderUnsigned(0xfffffffffffffffeL, 5));
  }

  @Test
  public void testDivideRemainderEuclideanProperty() {
    // Use a seed so that the test is deterministic:
    Random r = new Random(0L);
    for (int i = 0; i < 1000000; i++) {
      long dividend = r.nextLong();
      long divisor = r.nextLong();
      // Test that the Euclidean property is preserved:
      assertEquals(
          0,
          dividend
              - (divisor * UnsignedLongs.divideUnsigned(dividend, divisor)
                  + UnsignedLongs.remainderUnsigned(dividend, divisor)));
    }
  }
}
