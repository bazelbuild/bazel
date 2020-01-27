// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Extrema}. */
@RunWith(JUnit4.class)
public class ExtremaTest {
  @Test
  public void handlesDupes() {
    Extrema<Integer> extrema = Extrema.min(3);
    extrema.aggregate(4);
    extrema.aggregate(3);
    extrema.aggregate(1);
    extrema.aggregate(2);
    extrema.aggregate(1);
    extrema.aggregate(3);
    extrema.aggregate(1);
    assertThat(extrema.getExtremeElements()).containsExactly(1, 1, 1);
  }

  @Test
  public void customComparator() {
    class BoxedInt {
      private final int i;

      private BoxedInt(int i) {
        this.i = i;
      }
    }
    Extrema<BoxedInt> extrema =
        Extrema.max(
            2,
            new Comparator<BoxedInt>() {
              @Override
              public int compare(BoxedInt bi1, BoxedInt bi2) {
                return Integer.compare(bi1.i, bi2.i);
              }
            });
    extrema.aggregate(new BoxedInt(4));
    extrema.aggregate(new BoxedInt(1));
    extrema.aggregate(new BoxedInt(2));
    extrema.aggregate(new BoxedInt(3));
    extrema.aggregate(new BoxedInt(5));
    ImmutableList<Integer> extremeElements =
        extrema.getExtremeElements().stream()
            .map(bi -> bi.i)
            .collect(ImmutableList.toImmutableList());
    assertThat(extremeElements).containsExactly(5, 4).inOrder();
  }

  @Test
  public void minExtremaSmallK() {
    runRangeTest(Extrema.min(5), 1, 100, ImmutableList.of(1, 2, 3, 4, 5));
  }

  @Test
  public void minExtremaLargeK() {
    runRangeTest(Extrema.min(10), 1, 5, ImmutableList.of(1, 2, 3, 4, 5));
  }

  @Test
  public void maxExtremaSmallK() {
    runRangeTest(Extrema.max(5), 1, 100, ImmutableList.of(100, 99, 98, 97, 96));
  }

  @Test
  public void maxExtremaLargeK() {
    runRangeTest(Extrema.max(10), 1, 5, ImmutableList.of(5, 4, 3, 2, 1));
  }

  private void runRangeTest(
      Extrema<Integer> extrema,
      int leftEndpointInclusive,
      int rightEndpointInclusive,
      ImmutableList<Integer> expected) {
    assertThat(extrema.getExtremeElements()).isEmpty();
    closedRangeShuffled(leftEndpointInclusive, rightEndpointInclusive).forEach(extrema::aggregate);
    assertThat(extrema.getExtremeElements()).containsExactlyElementsIn(expected).inOrder();
    extrema.clear();
    assertThat(extrema.getExtremeElements()).isEmpty();
  }

  private static Stream<Integer> closedRangeShuffled(
      int leftEndpointInclusive, int rightEndpointInclusive) {
    List<Integer> list =
        IntStream.rangeClosed(leftEndpointInclusive, rightEndpointInclusive).boxed().collect(
            Collectors.toList());
    Collections.shuffle(list);
    return list.stream();
  }
}
