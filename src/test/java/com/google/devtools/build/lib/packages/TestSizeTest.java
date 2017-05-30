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

import static com.google.common.truth.Truth.assertThat;

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
    assertThat(TestSize.valueOf("SMALL")).isEqualTo(TestSize.SMALL);
    assertThat(TestSize.valueOf("MEDIUM")).isEqualTo(TestSize.MEDIUM);
    assertThat(TestSize.valueOf("LARGE")).isEqualTo(TestSize.LARGE);
    assertThat(TestSize.valueOf("ENORMOUS")).isEqualTo(TestSize.ENORMOUS);
  }

  @Test
  public void testGetDefaultTimeout() {
    assertThat(TestSize.SMALL.getDefaultTimeout()).isEqualTo(TestTimeout.SHORT);
    assertThat(TestSize.MEDIUM.getDefaultTimeout()).isEqualTo(TestTimeout.MODERATE);
    assertThat(TestSize.LARGE.getDefaultTimeout()).isEqualTo(TestTimeout.LONG);
    assertThat(TestSize.ENORMOUS.getDefaultTimeout()).isEqualTo(TestTimeout.ETERNAL);
  }

  @Test
  public void testGetDefaultShards() {
    assertThat(TestSize.SMALL.getDefaultShards()).isEqualTo(2);
    assertThat(TestSize.MEDIUM.getDefaultShards()).isEqualTo(10);
    assertThat(TestSize.LARGE.getDefaultShards()).isEqualTo(20);
    assertThat(TestSize.ENORMOUS.getDefaultShards()).isEqualTo(30);
  }

  @Test
  public void testGetTestSizeFromString() {
    assertThat(TestSize.getTestSize("Small")).isNull();
    assertThat(TestSize.getTestSize("Koala")).isNull();
    assertThat(TestSize.getTestSize("small")).isEqualTo(TestSize.SMALL);
    assertThat(TestSize.getTestSize("medium")).isEqualTo(TestSize.MEDIUM);
    assertThat(TestSize.getTestSize("large")).isEqualTo(TestSize.LARGE);
    assertThat(TestSize.getTestSize("enormous")).isEqualTo(TestSize.ENORMOUS);
  }

  @Test
  public void testGetTestSizeFromDefaultTimeout() {
    assertThat(TestSize.getTestSize(TestTimeout.SHORT)).isEqualTo(TestSize.SMALL);
    assertThat(TestSize.getTestSize(TestTimeout.MODERATE)).isEqualTo(TestSize.MEDIUM);
    assertThat(TestSize.getTestSize(TestTimeout.LONG)).isEqualTo(TestSize.LARGE);
    assertThat(TestSize.getTestSize(TestTimeout.ETERNAL)).isEqualTo(TestSize.ENORMOUS);
  }
}
