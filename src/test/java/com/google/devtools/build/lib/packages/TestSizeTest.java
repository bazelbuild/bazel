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
package com.google.devtools.build.lib.packages;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests the various methods of {@link TestSize}
 */
@RunWith(JUnit4.class)
public class TestSizeTest {

  @Test
  public void testBasicConversion() {
    assertEquals(TestSize.SMALL, TestSize.valueOf("SMALL"));
    assertEquals(TestSize.MEDIUM, TestSize.valueOf("MEDIUM"));
    assertEquals(TestSize.LARGE, TestSize.valueOf("LARGE"));
    assertEquals(TestSize.ENORMOUS, TestSize.valueOf("ENORMOUS"));
  }

  @Test
  public void testGetDefaultTimeout() {
    assertEquals(TestTimeout.SHORT, TestSize.SMALL.getDefaultTimeout());
    assertEquals(TestTimeout.MODERATE, TestSize.MEDIUM.getDefaultTimeout());
    assertEquals(TestTimeout.LONG, TestSize.LARGE.getDefaultTimeout());
    assertEquals(TestTimeout.ETERNAL, TestSize.ENORMOUS.getDefaultTimeout());
  }

  @Test
  public void testGetDefaultShards() {
    assertEquals(2, TestSize.SMALL.getDefaultShards());
    assertEquals(10, TestSize.MEDIUM.getDefaultShards());
    assertEquals(20, TestSize.LARGE.getDefaultShards());
    assertEquals(30, TestSize.ENORMOUS.getDefaultShards());
  }

  @Test
  public void testGetTestSizeFromString() {
    assertNull(TestSize.getTestSize("Small"));
    assertNull(TestSize.getTestSize("Koala"));
    assertEquals(TestSize.SMALL, TestSize.getTestSize("small"));
    assertEquals(TestSize.MEDIUM, TestSize.getTestSize("medium"));
    assertEquals(TestSize.LARGE, TestSize.getTestSize("large"));
    assertEquals(TestSize.ENORMOUS, TestSize.getTestSize("enormous"));
  }

  @Test
  public void testGetTestSizeFromDefaultTimeout() {
    assertEquals(TestSize.SMALL, TestSize.getTestSize(TestTimeout.SHORT));
    assertEquals(TestSize.MEDIUM, TestSize.getTestSize(TestTimeout.MODERATE));
    assertEquals(TestSize.LARGE, TestSize.getTestSize(TestTimeout.LONG));
    assertEquals(TestSize.ENORMOUS, TestSize.getTestSize(TestTimeout.ETERNAL));
  }
}
