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
import static org.junit.Assert.assertThrows;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class BranchCoverageTest {

  @Test
  public void testSimpleRetrieval() throws Exception {
    BranchCoverage branchCoverage = BranchCoverage.create();
    branchCoverage.addBranch(1, "", "0", false, 0);
    branchCoverage.addBranch(1, "", "1", false, 0);
    branchCoverage.addBranch(4, "", "0", true, 0);
    branchCoverage.addBranch(4, "", "1", true, 2);
    branchCoverage.addBranch(4, "", "2", true, 1);

    assertThat(branchCoverage.get(1, "", "0"))
        .isEqualTo(BranchCoverageItem.create(1, "", "0", false, 0));
    assertThat(branchCoverage.get(1, "", "1"))
        .isEqualTo(BranchCoverageItem.create(1, "", "1", false, 0));
    assertThat(branchCoverage.get(4, "", "0"))
        .isEqualTo(BranchCoverageItem.create(4, "", "0", true, 0));
    assertThat(branchCoverage.get(4, "", "1"))
        .isEqualTo(BranchCoverageItem.create(4, "", "1", true, 2));
    assertThat(branchCoverage.get(4, "", "2"))
        .isEqualTo(BranchCoverageItem.create(4, "", "2", true, 1));
  }

  @Test
  public void testNonExistentBranchReturnsNull() throws Exception {
    BranchCoverage branchCoverage = BranchCoverage.create();
    branchCoverage.addBranch(1, "", "0", false, 0);
    branchCoverage.addBranch(1, "", "1", false, 0);

    assertThat(branchCoverage.get(1, "", "2")).isNull();
    assertThat(branchCoverage.get(1, "1", "0")).isNull();
    assertThat(branchCoverage.get(2, "", "0")).isNull();
  }

  @Test
  public void testIterator() throws Exception {
    BranchCoverage branchCoverage = BranchCoverage.create();
    branchCoverage.addBranch(1, "", "0", false, 0);
    branchCoverage.addBranch(1, "", "1", false, 0);
    branchCoverage.addBranch(4, "", "0", true, 0);
    branchCoverage.addBranch(4, "", "1", true, 2);
    branchCoverage.addBranch(4, "", "2", true, 1);
    branchCoverage.addBranch(7, "id", "0", false, 0);
    branchCoverage.addBranch(7, "id", "1", false, 0);
    branchCoverage.addBranch(7, "id", "2", false, 0);

    Iterator<Entry<BranchCoverageKey, BranchCoverageItem>> it = branchCoverage.iterator();
    HashMap<BranchCoverageKey, BranchCoverageItem> result = new HashMap<>();
    while (it.hasNext()) {
      Entry<BranchCoverageKey, BranchCoverageItem> entry = it.next();
      result.put(entry.getKey(), entry.getValue());
    }

    assertThat(result)
        .containsExactly(
            BranchCoverageKey.create(1, "", "0"),
            BranchCoverageItem.create(1, "", "0", false, 0),
            BranchCoverageKey.create(1, "", "1"),
            BranchCoverageItem.create(1, "", "1", false, 0),
            BranchCoverageKey.create(4, "", "0"),
            BranchCoverageItem.create(4, "", "0", true, 0),
            BranchCoverageKey.create(4, "", "1"),
            BranchCoverageItem.create(4, "", "1", true, 2),
            BranchCoverageKey.create(4, "", "2"),
            BranchCoverageItem.create(4, "", "2", true, 1),
            BranchCoverageKey.create(7, "id", "0"),
            BranchCoverageItem.create(7, "id", "0", false, 0),
            BranchCoverageKey.create(7, "id", "1"),
            BranchCoverageItem.create(7, "id", "1", false, 0),
            BranchCoverageKey.create(7, "id", "2"),
            BranchCoverageItem.create(7, "id", "2", false, 0));
  }

  @Test
  public void testExhaustedIteratorThrows() throws Exception {
    BranchCoverage branchCoverage = BranchCoverage.create();
    branchCoverage.addBranch(1, "", "0", false, 0);

    Iterator<Entry<BranchCoverageKey, BranchCoverageItem>> it = branchCoverage.iterator();

    assertThat(it.hasNext()).isTrue();
    it.next();
    assertThat(it.hasNext()).isFalse();
    assertThrows(NoSuchElementException.class, () -> it.next());
  }

  @Test
  public void testCopy() throws Exception {
    BranchCoverage branchCoverage = BranchCoverage.create();
    branchCoverage.addBranch(1, "", "0", false, 0);
    branchCoverage.addBranch(1, "", "1", false, 0);
    branchCoverage.addBranch(4, "", "0", true, 0);
    branchCoverage.addBranch(4, "", "1", true, 2);

    BranchCoverage copy = BranchCoverage.copy(branchCoverage);
    HashMap<BranchCoverageKey, BranchCoverageItem> result = new HashMap<>();
    for (Entry<BranchCoverageKey, BranchCoverageItem> entry : copy) {
      result.put(entry.getKey(), entry.getValue());
    }

    assertThat(result)
        .containsExactly(
            BranchCoverageKey.create(1, "", "0"),
            BranchCoverageItem.create(1, "", "0", false, 0),
            BranchCoverageKey.create(1, "", "1"),
            BranchCoverageItem.create(1, "", "1", false, 0),
            BranchCoverageKey.create(4, "", "0"),
            BranchCoverageItem.create(4, "", "0", true, 0),
            BranchCoverageKey.create(4, "", "1"),
            BranchCoverageItem.create(4, "", "1", true, 2));
  }

  @Test
  public void testRepeatedBranchesAreMerged() throws Exception {
    BranchCoverage branchCoverage = BranchCoverage.create();
    branchCoverage.addBranch(1, "", "0", false, 0);
    branchCoverage.addBranch(1, "", "0", false, 0);
    branchCoverage.addBranch(1, "", "0", true, 1);
    branchCoverage.addBranch(1, "", "0", true, 2);
    branchCoverage.addBranch(2, "", "0", false, 0);
    branchCoverage.addBranch(2, "", "0", false, 0);

    assertThat(branchCoverage.get(1, "", "0"))
        .isEqualTo(BranchCoverageItem.create(1, "", "0", true, 3));
    assertThat(branchCoverage.get(2, "", "0"))
        .isEqualTo(BranchCoverageItem.create(2, "", "0", false, 0));
  }

  @Test
  public void testMerge() throws Exception {
    BranchCoverage branchCoverage1 = BranchCoverage.create();
    branchCoverage1.addBranch(1, "", "0", false, 0);
    branchCoverage1.addBranch(1, "", "1", false, 0);
    branchCoverage1.addBranch(4, "", "0", true, 0);
    branchCoverage1.addBranch(4, "", "1", true, 2);
    branchCoverage1.addBranch(6, "", "0", true, 1);
    branchCoverage1.addBranch(6, "id", "1", true, 0);
    BranchCoverage branchCoverage2 = BranchCoverage.create();
    branchCoverage2.addBranch(1, "", "0", true, 1);
    branchCoverage2.addBranch(1, "", "1", true, 2);
    branchCoverage2.addBranch(4, "", "0", true, 3);
    branchCoverage2.addBranch(4, "", "1", true, 4);
    branchCoverage2.addBranch(7, "id", "0", true, 5);
    branchCoverage2.addBranch(7, "id", "1", true, 6);

    BranchCoverage merged = BranchCoverage.merge(branchCoverage1, branchCoverage2);
    HashMap<BranchCoverageKey, BranchCoverageItem> result = new HashMap<>();
    for (Entry<BranchCoverageKey, BranchCoverageItem> entry : merged) {
      result.put(entry.getKey(), entry.getValue());
    }

    assertThat(result)
        .containsExactly(
            BranchCoverageKey.create(1, "", "0"),
            BranchCoverageItem.create(1, "", "0", true, 1),
            BranchCoverageKey.create(1, "", "1"),
            BranchCoverageItem.create(1, "", "1", true, 2),
            BranchCoverageKey.create(4, "", "0"),
            BranchCoverageItem.create(4, "", "0", true, 3),
            BranchCoverageKey.create(4, "", "1"),
            BranchCoverageItem.create(4, "", "1", true, 6),
            BranchCoverageKey.create(6, "", "0"),
            BranchCoverageItem.create(6, "", "0", true, 1),
            BranchCoverageKey.create(6, "id", "1"),
            BranchCoverageItem.create(6, "id", "1", true, 0),
            BranchCoverageKey.create(7, "id", "0"),
            BranchCoverageItem.create(7, "id", "0", true, 5),
            BranchCoverageKey.create(7, "id", "1"),
            BranchCoverageItem.create(7, "id", "1", true, 6));
  }

  @Test
  public void testContainsKey() throws Exception {
    BranchCoverage branchCoverage = BranchCoverage.create();
    branchCoverage.addBranch(1, "", "0", false, 0);
    branchCoverage.addBranch(1, "", "1", false, 0);

    assertThat(branchCoverage.containsKey(1, "", "0")).isTrue();
    assertThat(branchCoverage.containsKey(1, "", "1")).isTrue();
    assertThat(branchCoverage.containsKey(1, "", "2")).isFalse();
    assertThat(branchCoverage.containsKey(1, "1", "0")).isFalse();
    assertThat(branchCoverage.containsKey(2, "", "0")).isFalse();
  }

  @Test
  public void testGetKeys() throws Exception {
    BranchCoverage branchCoverage = BranchCoverage.create();
    branchCoverage.addBranch(1, "1", "0", false, 0);
    branchCoverage.addBranch(1, "1", "1", false, 0);
    branchCoverage.addBranch(4, "", "0", true, 0);
    branchCoverage.addBranch(4, "", "1", true, 2);
    branchCoverage.addBranch(4, "", "2", true, 1);
    branchCoverage.addBranch(7, "id", "0", false, 0);
    branchCoverage.addBranch(7, "id", "1", false, 0);
    branchCoverage.addBranch(7, "id", "2", false, 0);

    assertThat(branchCoverage.getKeys())
        .containsExactly(
            BranchCoverageKey.create(1, "1", "0"),
            BranchCoverageKey.create(1, "1", "1"),
            BranchCoverageKey.create(4, "", "0"),
            BranchCoverageKey.create(4, "", "1"),
            BranchCoverageKey.create(4, "", "2"),
            BranchCoverageKey.create(7, "id", "0"),
            BranchCoverageKey.create(7, "id", "1"),
            BranchCoverageKey.create(7, "id", "2"));
  }

  @Test
  public void testExtremeLineNumbers() throws Exception {
    BranchCoverage branchCoverage = BranchCoverage.create();
    String blockId = "abcdefghijklmnopqrstuvwxyz";
    branchCoverage.addBranch(Integer.MAX_VALUE, blockId, "1234567890", false, 0);
    branchCoverage.addBranch(Integer.MAX_VALUE, blockId, "12345678901", false, 0);

    assertThat(branchCoverage.get(Integer.MAX_VALUE, blockId, "1234567890"))
        .isEqualTo(BranchCoverageItem.create(Integer.MAX_VALUE, blockId, "1234567890", false, 0));
    assertThat(branchCoverage.get(Integer.MAX_VALUE, blockId, "12345678901"))
        .isEqualTo(BranchCoverageItem.create(Integer.MAX_VALUE, blockId, "12345678901", false, 0));
  }

  @Test
  public void testLargeNumberOfBranches() throws Exception {
    BranchCoverage branchCoverage = BranchCoverage.create();
    for (int i = 0; i < 100000; i++) {
      int lineNumber = (i / 100) + 1;
      branchCoverage.addBranch(lineNumber, "", String.valueOf(i), false, 0);
    }

    assertThat(branchCoverage.size()).isEqualTo(100000);
    for (int i = 0; i < 100000; i++) {
      int lineNumber = (i / 100) + 1;
      assertThat(branchCoverage.get(lineNumber, "", String.valueOf(i)))
          .isEqualTo(BranchCoverageItem.create(lineNumber, "", String.valueOf(i), false, 0));
    }
  }
}
