// Copyright 2024 The Bazel Authors. All rights reserved.
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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link ShallowObjectSizeComputer}.
 *
 * <p>Most tests accept two values to account for the presence or absence of compressed OOPs.
 */
@RunWith(JUnit4.class)
public class ShallowObjectSizeComputerTest {
  private static class Empty {}

  @Test
  public void testEmptyClass() {
    assertThat(ShallowObjectSizeComputer.getClassShallowSize(Empty.class)).isEqualTo(16L);
  }

  @SuppressWarnings("unused")
  private static class OneReference {
    private Object ref;
  }

  @Test
  public void testOneReference() {
    assertThat(ShallowObjectSizeComputer.getClassShallowSize(OneReference.class)).isAnyOf(16L, 24L);
  }

  @SuppressWarnings("unused")
  private static class TwoReferences {
    private Object ref1;
    private Object ref2;
  }

  @Test
  public void testTwoReferences() {
    assertThat(ShallowObjectSizeComputer.getClassShallowSize(TwoReferences.class))
        .isAnyOf(24L, 32L);
  }

  @SuppressWarnings("unused")
  private static class ThreeBooleans {
    private boolean bool1;
    private boolean bool2;
    private boolean bool3;
  }

  @Test
  public void testThreeBooleans() {
    assertThat(ShallowObjectSizeComputer.getClassShallowSize(ThreeBooleans.class))
        .isAnyOf(16L, 24L);
  }

  @Test
  public void testObjectArray() {
    assertThat(ShallowObjectSizeComputer.getShallowSize(new Object[4])).isAnyOf(32L, 56L);
  }

  @Test
  public void testBooleanArray() {
    assertThat(ShallowObjectSizeComputer.getShallowSize(new boolean[4])).isAnyOf(24L, 32L);
  }
}
