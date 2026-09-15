// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Simple tests for {@link CycleDeduper}. */
@RunWith(JUnit4.class)
public class CycleDeduperTest {

  private CycleDeduper<String> cycleDeduper = new CycleDeduper<>();

  @Test
  public void simple() {
    assertThat(cycleDeduper.alreadySeen(ImmutableList.of("a", "b"))).isFalse();
    assertThat(cycleDeduper.alreadySeen(ImmutableList.of("a", "b"))).isTrue();
    assertThat(cycleDeduper.alreadySeen(ImmutableList.of("b", "a"))).isTrue();

    assertThat(cycleDeduper.alreadySeen(ImmutableList.of("a", "b", "c"))).isFalse();
    assertThat(cycleDeduper.alreadySeen(ImmutableList.of("b", "c", "a"))).isTrue();
    assertThat(cycleDeduper.alreadySeen(ImmutableList.of("c", "a", "b"))).isTrue();
    assertThat(cycleDeduper.alreadySeen(ImmutableList.of("b", "a", "c"))).isFalse();
    assertThat(cycleDeduper.alreadySeen(ImmutableList.of("c", "b", "a"))).isTrue();
  }

  @Test
  public void badCycle_empty() {
    assertThrows(IllegalStateException.class, () -> cycleDeduper.alreadySeen(ImmutableList.of()));
  }

  @Test
  public void badCycle_nonUniqueMembers() {
    assertThrows(
        IllegalStateException.class,
        () -> cycleDeduper.alreadySeen(ImmutableList.of("a", "b", "a")));
  }
}
