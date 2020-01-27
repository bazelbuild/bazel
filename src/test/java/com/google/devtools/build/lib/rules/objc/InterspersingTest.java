// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for the Interspersing utility. */
@RunWith(JUnit4.class)
public class InterspersingTest {
  @Test
  public void beforeEach_empty() {
    assertThat(Interspersing.beforeEach("foo", ImmutableList.<String>of())).isEmpty();
  }

  @Test
  public void beforeEach_isLazy() {
    Iterable<Integer> result = Interspersing.beforeEach(1011, Iterables.cycle(1, 2, 3));
    assertThat(Iterables.get(result, 0)).isEqualTo((Object) 1011);
    assertThat(Iterables.get(result, 1)).isEqualTo((Object) 1);
    assertThat(Iterables.get(result, 2)).isEqualTo((Object) 1011);
    assertThat(Iterables.get(result, 3)).isEqualTo((Object) 2);
  }

  @Test
  public void beforeEach_throwsForNullWhat() {
    assertThrows(
        NullPointerException.class, () -> Interspersing.beforeEach(null, ImmutableList.of("a")));
  }

  @Test
  public void prependEach_empty() {
    assertThat(Interspersing.prependEach("foo", ImmutableList.<String>of())).isEmpty();
  }

  @Test
  public void prependEach_isLazy() {
    Iterable<String> result = Interspersing.prependEach(
        "#", Iterables.cycle(1, 2, 3), Functions.toStringFunction());
    assertThat(Iterables.get(result, 0)).isEqualTo("#1");
  }

  @Test
  public void prependEach_typicalList() {
    Iterable<String> result = Interspersing.prependEach(
        "*", ImmutableList.of("a", "b", "c"));
    assertThat(result).containsExactly("*a", "*b", "*c").inOrder();
  }

  @Test
  public void prependEach_throwsForNullWhat() {
    assertThrows(
        NullPointerException.class, () -> Interspersing.prependEach(null, ImmutableList.of("a")));
  }
}
