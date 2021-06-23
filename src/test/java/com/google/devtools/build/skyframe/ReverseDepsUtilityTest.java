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

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/** Test for {@code ReverseDepsUtility}. */
@RunWith(Parameterized.class)
public class ReverseDepsUtilityTest {
  private final int numElements;

  @Parameters(name = "numElements-{0}")
  public static List<Object[]> parameters() {
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
        ReverseDepsUtility.addReverseDep(example, Key.create(j));
      }
      // Not a big test but at least check that it does not blow up.
      assertThat(ReverseDepsUtility.toString(example)).isNotEmpty();
      assertThat(ReverseDepsUtility.getReverseDeps(example)).hasSize(numElements);
      for (int i = 0; i < numRemovals; i++) {
        ReverseDepsUtility.removeReverseDep(example, Key.create(i));
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
      for (int j = 0; j < numElements; j++) {
        ReverseDepsUtility.addReverseDep(example, Key.create(j));
      }
      assertThat(ReverseDepsUtility.getReverseDeps(example)).hasSize(numElements);
      for (int i = 0; i < numRemovals; i++) {
        ReverseDepsUtility.removeReverseDep(example, Key.create(i));
      }
      assertThat(ReverseDepsUtility.getReverseDeps(example)).hasSize(numElements - numRemovals);
      assertThat(example.getReverseDepsDataToConsolidateForReverseDepsUtil()).isNull();
    }
  }

  @Test
  public void testDuplicateCheckOnGetReverseDeps() {
    InMemoryNodeEntry example = new InMemoryNodeEntry();
    for (int i = 0; i < numElements; i++) {
      ReverseDepsUtility.addReverseDep(example, Key.create(i));
    }
    // Should only fail when we call getReverseDeps().
    ReverseDepsUtility.addReverseDep(example, Key.create(0));
    if (numElements == 0) {
      // Will not throw.
      ReverseDepsUtility.getReverseDeps(example);
    } else {
      assertThrows(Exception.class, () -> ReverseDepsUtility.getReverseDeps(example));
    }
  }

  @Test
  public void doubleAddThenRemove() {
    InMemoryNodeEntry example = new InMemoryNodeEntry();
    SkyKey key = Key.create(0);
    ReverseDepsUtility.addReverseDep(example, key);
    // Should only fail when we call getReverseDeps().
    ReverseDepsUtility.addReverseDep(example, key);
    ReverseDepsUtility.removeReverseDep(example, key);
    assertThrows(IllegalStateException.class, () -> ReverseDepsUtility.getReverseDeps(example));
  }

  @Test
  public void doubleAddThenRemoveCheckedOnSize() {
    InMemoryNodeEntry example = new InMemoryNodeEntry();
    SkyKey fixedKey = Key.create(0);
    ReverseDepsUtility.addReverseDep(example, fixedKey);
    SkyKey key = Key.create(1);
    ReverseDepsUtility.addReverseDep(example, key);
    // Should only fail when we reach the limit.
    ReverseDepsUtility.addReverseDep(example, key);
    ReverseDepsUtility.removeReverseDep(example, key);
    ReverseDepsUtility.checkReverseDep(example, fixedKey);
    assertThrows(
        IllegalStateException.class, () -> ReverseDepsUtility.checkReverseDep(example, fixedKey));
  }

  @Test
  public void addRemoveAdd() {
    InMemoryNodeEntry example = new InMemoryNodeEntry();
    SkyKey fixedKey = Key.create(0);
    ReverseDepsUtility.addReverseDep(example, fixedKey);
    SkyKey key = Key.create(1);
    ReverseDepsUtility.addReverseDep(example, key);
    ReverseDepsUtility.removeReverseDep(example, key);
    ReverseDepsUtility.addReverseDep(example, key);
    assertThat(ReverseDepsUtility.getReverseDeps(example)).containsExactly(fixedKey, key);
  }

  private static class Key extends AbstractSkyKey<Integer> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(Integer arg) {
      super(arg);
    }

    private static Key create(Integer arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctionName.FOR_TESTING;
    }
  }
}
