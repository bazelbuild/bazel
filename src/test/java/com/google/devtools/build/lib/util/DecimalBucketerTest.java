// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.util.DecimalBucketer.Bucket;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class DecimalBucketerTest {

  @Test
  public void testEmpty() {
    DecimalBucketer bucketer = new DecimalBucketer();
    assertThat(bucketer.getBuckets()).isEmpty();
  }

  @Test
  public void testSingleValue() {
    DecimalBucketer bucketer = new DecimalBucketer();
    bucketer.add(5);
    assertThat(bucketer.getBuckets()).containsExactly(new Bucket(5, 6, 1));
  }

  @Test
  public void testZero() {
    DecimalBucketer bucketer = new DecimalBucketer();
    bucketer.add(0);
    assertThat(bucketer.getBuckets()).containsExactly(new Bucket(0, 1, 1));
  }

  @Test
  public void testMultipleValuesSameBucket() {
    DecimalBucketer bucketer = new DecimalBucketer();
    bucketer.add(10);
    bucketer.add(15);
    bucketer.add(19);
    assertThat(bucketer.getBuckets()).containsExactly(new Bucket(10, 20, 3));
  }

  @Test
  public void testMultipleBuckets() {
    DecimalBucketer bucketer = new DecimalBucketer();
    bucketer.add(5);
    bucketer.add(12);
    bucketer.add(15);
    bucketer.add(25);
    bucketer.add(99);
    bucketer.add(100);

    assertThat(bucketer.getBuckets())
        .containsExactly(
            new Bucket(5, 6, 1),
            new Bucket(10, 20, 2),
            new Bucket(20, 30, 1),
            new Bucket(90, 100, 1),
            new Bucket(100, 200, 1))
        .inOrder();
  }

  @Test
  public void bucketsWithGap() {
    DecimalBucketer bucketer = new DecimalBucketer();
    bucketer.add(5);
    bucketer.add(61234);
    bucketer.add(69999);

    assertThat(bucketer.getBuckets())
        .containsExactly(new Bucket(5, 6, 1), new Bucket(60000, 70000, 2))
        .inOrder();
  }

  @Test
  public void testNegativeValue() {
    DecimalBucketer bucketer = new DecimalBucketer();
    assertThrows(IllegalArgumentException.class, () -> bucketer.add(-1));
  }

  @Test
  public void testLargeValues() {
    DecimalBucketer bucketer = new DecimalBucketer();
    long val = 9000000000000000000L; // 9 * 10^18
    bucketer.add(val);

    assertThat(bucketer.getBuckets()).containsExactly(new Bucket(val, Long.MAX_VALUE, 1));
  }
}
