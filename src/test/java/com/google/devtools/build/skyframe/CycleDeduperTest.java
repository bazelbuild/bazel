// Copyright 2014 Google Inc. All rights reserved.
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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Simple tests for {@link CycleDeduper}. */
@RunWith(JUnit4.class)
public class CycleDeduperTest {

  private CycleDeduper<String> cycleDeduper = new CycleDeduper<>();

  @Test
  public void simple() throws Exception {
    assertTrue(cycleDeduper.seen(ImmutableList.of("a", "b")));
    assertFalse(cycleDeduper.seen(ImmutableList.of("a", "b")));
    assertFalse(cycleDeduper.seen(ImmutableList.of("b", "a")));

    assertTrue(cycleDeduper.seen(ImmutableList.of("a", "b", "c")));
    assertFalse(cycleDeduper.seen(ImmutableList.of("b", "c", "a")));
    assertFalse(cycleDeduper.seen(ImmutableList.of("c", "a", "b")));
    assertTrue(cycleDeduper.seen(ImmutableList.of("b", "a", "c")));
    assertFalse(cycleDeduper.seen(ImmutableList.of("c", "b", "a")));
  }

  @Test
  public void badCycle_Empty() throws Exception {
    try {
      cycleDeduper.seen(ImmutableList.<String>of());
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void badCycle_NonUniqueMembers() throws Exception {
    try {
      cycleDeduper.seen(ImmutableList.<String>of("a", "b", "a"));
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
  }
}
