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

import com.google.common.collect.ImmutableList;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.util.function.Supplier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link ShallowObjectSizeComputer}.
 *
 * <p>Tests by comparing the computed object size against measurements on the JVM.
 */
@RunWith(JUnit4.class)
public class ShallowObjectSizeComputerTest {
  private static class Empty {}

  @Test
  public void testEmptyClass() {
    assertComputedSizeIsCorrect(Empty::new);
  }

  @SuppressWarnings("unused")
  private static class OneReference {
    private Object ref;
  }

  @Test
  public void testOneReference() {
    assertComputedSizeIsCorrect(OneReference::new);
  }

  @SuppressWarnings("unused")
  private static class TwoReferences {
    private Object ref1;
    private Object ref2;
  }

  @Test
  public void testTwoReferences() {
    assertComputedSizeIsCorrect(TwoReferences::new);
  }

  @SuppressWarnings("unused")
  private static class ThreeBooleans {
    private boolean bool1;
    private boolean bool2;
    private boolean bool3;
  }

  @Test
  public void testThreeBooleans() {
    assertComputedSizeIsCorrect(ThreeBooleans::new);
  }

  @Test
  public void testObjectArray() {
    assertComputedSizeIsCorrect(() -> new Object[4]);
  }

  @Test
  public void testBooleanArray() {
    assertComputedSizeIsCorrect(() -> new boolean[4]);
  }

  // TODO(lberki): Lambdas without any values closed over must (eventually) be special-cased since
  // they don't require heap.

  @Test
  public void testClosureWithOneValue() {
    Object o = new Object();
    assertComputedSizeIsCorrect(() -> (Supplier<Object>) () -> o);
  }

  @Test
  public void testClosureWithThreeValues() {
    Object o1 = new Object();
    Object o2 = new Object();
    Object o3 = new Object();
    assertComputedSizeIsCorrect(() -> (Supplier<Object>) () -> ImmutableList.of(o1, o2, o3));
  }

  private void assertComputedSizeIsCorrect(Supplier<Object> createInstance) {
    Object sampleToCompute = createInstance.get();
    long computedSize = ShallowObjectSizeComputer.getShallowSize(sampleToCompute);
    long measuredSize = measureSize(createInstance);
    assertThat(computedSize).isEqualTo(measuredSize);
  }

  private static long measureSize(Supplier<Object> createInstance) {
    Object[] storage = new Object[1];

    // NB: this is com.sun.management.ThreadMXBean, NOT java.lang.management.ThreadMXBean
    ThreadMXBean bean = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    bean.setThreadAllocatedMemoryEnabled(true);

    // One would think that this is at least somewhat inaccurate, but according to measurements it's
    // accurate to the last byte. The mind boggles.
    long before = bean.getCurrentThreadAllocatedBytes();
    storage[0] = createInstance.get();
    long after = bean.getCurrentThreadAllocatedBytes();

    return after - before;
  }
}
