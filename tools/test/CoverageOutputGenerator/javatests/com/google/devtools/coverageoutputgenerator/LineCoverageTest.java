// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.coverageoutputgenerator;

import static com.google.common.truth.Truth.assertThat;
import static java.util.Map.entry;
import static org.junit.Assert.assertThrows;

import java.util.Iterator;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class LineCoverageTest {

  @Test
  public void testLineRetrieval() {
    LineCoverage lineCoverage = LineCoverage.create();
    lineCoverage.addLine(0, 0);
    lineCoverage.addLine(1, 1);
    lineCoverage.addLine(3, 5);
    lineCoverage.addLine(15, 2);

    assertThat(lineCoverage)
        .containsExactly(entry(0, 0L), entry(1, 1L), entry(3, 5L), entry(15, 2L));
  }

  @Test
  public void testLargeLineNumbers() {
    LineCoverage lineCoverage = LineCoverage.create();
    lineCoverage.addLine(1000, 1);
    lineCoverage.addLine(1000000, 2);
    lineCoverage.addLine(1499999, 3);
    lineCoverage.addLine(1500000, 4);
    lineCoverage.addLine(1500001, 5);

    assertThat(lineCoverage)
        .containsExactly(
            entry(1000, 1L),
            entry(1000000, 2L),
            entry(1499999, 3L),
            entry(1500000, 4L),
            entry(1500001, 5L));
  }

  @Test
  public void testLineIterator() {
    LineCoverage lineCoverage = LineCoverage.create();
    lineCoverage.addLine(1, 1);
    lineCoverage.addLine(3, 5);
    lineCoverage.addLine(15, 2);

    Iterator<Entry<Integer, Long>> iterator = lineCoverage.iterator();

    assertThat(iterator.hasNext()).isTrue();
    assertThat(iterator.next()).isEqualTo(entry(1, 1L));
    assertThat(iterator.hasNext()).isTrue();
    assertThat(iterator.next()).isEqualTo(entry(3, 5L));
    assertThat(iterator.hasNext()).isTrue();
    assertThat(iterator.next()).isEqualTo(entry(15, 2L));
    assertThat(iterator.hasNext()).isFalse();
    assertThrows(NoSuchElementException.class, iterator::next);
  }

  @Test
  public void testNegativeLineNumberThrows() {
    LineCoverage lineCoverage = LineCoverage.create();
    assertThrows(IllegalArgumentException.class, () -> lineCoverage.addLine(-1, 1));
  }
}
