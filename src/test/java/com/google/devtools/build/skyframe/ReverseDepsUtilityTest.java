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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/** Test for {@code ReverseDepsUtility}. */
@RunWith(Parameterized.class)
public class ReverseDepsUtilityTest {

  private static final SkyFunctionName NODE_TYPE = SkyFunctionName.create("Type");
  private final int numElements;

  @Parameters(name = "numElements-{0}")
  public static List<Object[]> paramenters() {
    List<Object[]> params = new ArrayList<>();
    for (int i = 0; i < 20; i++) {
      params.add(new Object[] {i});
    }
    return params;
  }

  public ReverseDepsUtilityTest(int numElements) {
    this.numElements = numElements;
  }

  @Test
  public void testAddAndRemove() {
    for (int numRemovals = 0; numRemovals <= numElements; numRemovals++) {
      InMemoryNodeEntry example = new InMemoryNodeEntry();
      for (int j = 0; j < numElements; j++) {
        ReverseDepsUtility.addReverseDeps(
            example, Collections.singleton(LegacySkyKey.create(NODE_TYPE, j)));
      }
      // Not a big test but at least check that it does not blow up.
      assertThat(ReverseDepsUtility.toString(example)).isNotEmpty();
      assertThat(ReverseDepsUtility.getReverseDeps(example)).hasSize(numElements);
      for (int i = 0; i < numRemovals; i++) {
        ReverseDepsUtility.removeReverseDep(example, LegacySkyKey.create(NODE_TYPE, i));
      }
      assertThat(ReverseDepsUtility.getReverseDeps(example)).hasSize(numElements - numRemovals);
      assertThat(example.getReverseDepsDataToConsolidateForReverseDepsUtil()).isNull();
    }
  }

  // Same as testAdditionAndRemoval but we add all the reverse deps in one call.
  @Test
  public void testAddAllAndRemove() {
    for (int numRemovals = 0; numRemovals <= numElements; numRemovals++) {
      InMemoryNodeEntry example = new InMemoryNodeEntry();
      List<SkyKey> toAdd = new ArrayList<>();
      for (int j = 0; j < numElements; j++) {
        toAdd.add(LegacySkyKey.create(NODE_TYPE, j));
      }
      ReverseDepsUtility.addReverseDeps(example, toAdd);
      assertThat(ReverseDepsUtility.getReverseDeps(example)).hasSize(numElements);
      for (int i = 0; i < numRemovals; i++) {
        ReverseDepsUtility.removeReverseDep(example, LegacySkyKey.create(NODE_TYPE, i));
      }
      assertThat(ReverseDepsUtility.getReverseDeps(example)).hasSize(numElements - numRemovals);
      assertThat(example.getReverseDepsDataToConsolidateForReverseDepsUtil()).isNull();
    }
  }

  @Test
  public void testDuplicateCheckOnGetReverseDeps() {
    InMemoryNodeEntry example = new InMemoryNodeEntry();
    for (int i = 0; i < numElements; i++) {
      ReverseDepsUtility.addReverseDeps(
          example, Collections.singleton(LegacySkyKey.create(NODE_TYPE, i)));
    }
    // Should only fail when we call getReverseDeps().
    ReverseDepsUtility.addReverseDeps(
        example, Collections.singleton(LegacySkyKey.create(NODE_TYPE, 0)));
    try {
      ReverseDepsUtility.getReverseDeps(example);
      assertThat(numElements).isEqualTo(0);
    } catch (Exception expected) {
    }
  }

  @Test
  public void doubleAddThenRemove() {
    InMemoryNodeEntry example = new InMemoryNodeEntry();
    SkyKey key = LegacySkyKey.create(NODE_TYPE, 0);
    ReverseDepsUtility.addReverseDeps(example, Collections.singleton(key));
    // Should only fail when we call getReverseDeps().
    ReverseDepsUtility.addReverseDeps(example, Collections.singleton(key));
    ReverseDepsUtility.removeReverseDep(example, key);
    try {
      ReverseDepsUtility.getReverseDeps(example);
      fail();
    } catch (IllegalStateException expected) {
    }
  }

  @Test
  public void doubleAddThenRemoveCheckedOnSize() {
    InMemoryNodeEntry example = new InMemoryNodeEntry();
    SkyKey fixedKey = LegacySkyKey.create(NODE_TYPE, 0);
    SkyKey key = LegacySkyKey.create(NODE_TYPE, 1);
    ReverseDepsUtility.addReverseDeps(example, ImmutableList.of(fixedKey, key));
    // Should only fail when we reach the limit.
    ReverseDepsUtility.addReverseDeps(example, Collections.singleton(key));
    ReverseDepsUtility.removeReverseDep(example, key);
    ReverseDepsUtility.checkReverseDep(example, fixedKey);
    try {
      ReverseDepsUtility.checkReverseDep(example, fixedKey);
      fail();
    } catch (IllegalStateException expected) {
    }
  }

  @Test
  public void addRemoveAdd() {
    InMemoryNodeEntry example = new InMemoryNodeEntry();
    SkyKey fixedKey = LegacySkyKey.create(NODE_TYPE, 0);
    SkyKey key = LegacySkyKey.create(NODE_TYPE, 1);
    ReverseDepsUtility.addReverseDeps(example, ImmutableList.of(fixedKey, key));
    ReverseDepsUtility.removeReverseDep(example, key);
    ReverseDepsUtility.addReverseDeps(example, Collections.singleton(key));
    assertThat(ReverseDepsUtility.getReverseDeps(example)).containsExactly(fixedKey, key);
  }

  @Test
  public void testMaybeCheck() {
    InMemoryNodeEntry example = new InMemoryNodeEntry();
    for (int i = 0; i < numElements; i++) {
      ReverseDepsUtility.addReverseDeps(
          example, Collections.singleton(LegacySkyKey.create(NODE_TYPE, i)));
      // This should always succeed, since the next element is still not present.
      ReverseDepsUtility.maybeCheckReverseDepNotPresent(
          example, LegacySkyKey.create(NODE_TYPE, i + 1));
    }
    try {
      ReverseDepsUtility.maybeCheckReverseDepNotPresent(example, LegacySkyKey.create(NODE_TYPE, 0));
      // Should only fail if empty or above the checking threshold.
      assertThat(numElements == 0 || numElements >= ReverseDepsUtility.MAYBE_CHECK_THRESHOLD)
          .isTrue();
    } catch (Exception expected) {
    }
  }
}
