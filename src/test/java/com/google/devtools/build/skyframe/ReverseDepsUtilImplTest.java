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

import com.google.common.collect.ImmutableList;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Test for {@code ReverseDepsUtilImpl}.
 */
@RunWith(Parameterized.class)
public class ReverseDepsUtilImplTest {

  private static final SkyFunctionName NODE_TYPE = SkyFunctionName.create("Type");
  private final int numElements;

  @Parameters
  public static List<Object[]> paramenters() {
    List<Object[]> params = new ArrayList<>();
    for (int i = 0; i < 20; i++) {
      params.add(new Object[] {i});
    }
    return params;
  }

  public ReverseDepsUtilImplTest(int numElements) {
    this.numElements = numElements;
  }

  private static final ReverseDepsUtil<Example> REVERSE_DEPS_UTIL =
      new ReverseDepsUtilImpl<Example>() {
        @Override
        void setReverseDepsObject(Example container, Object object) {
          container.reverseDeps = object;
        }

        @Override
        void setSingleReverseDep(Example container, boolean singleObject) {
          container.single = singleObject;
        }

        @Override
        void setDataToConsolidate(Example container, List<Object> dataToConsolidate) {
          container.dataToConsolidate = dataToConsolidate;
        }

        @Override
        Object getReverseDepsObject(Example container) {
          return container.reverseDeps;
        }

        @Override
        boolean isSingleReverseDep(Example container) {
          return container.single;
        }

        @Override
        List<Object> getDataToConsolidate(Example container) {
          return container.dataToConsolidate;
        }
      };

  private class Example {

    Object reverseDeps = ImmutableList.of();
    boolean single;
    List<Object> dataToConsolidate;

    @Override
    public String toString() {
      return "Example: " + reverseDeps + ", " + single + ", " + dataToConsolidate;
    }
  }

  @Test
  public void testAddAndRemove() {
    for (int numRemovals = 0; numRemovals <= numElements; numRemovals++) {
      Example example = new Example();
      for (int j = 0; j < numElements; j++) {
        REVERSE_DEPS_UTIL.addReverseDeps(
            example, Collections.singleton(SkyKey.create(NODE_TYPE, j)));
      }
      // Not a big test but at least check that it does not blow up.
      assertThat(REVERSE_DEPS_UTIL.toString(example)).isNotEmpty();
      assertThat(REVERSE_DEPS_UTIL.getReverseDeps(example)).hasSize(numElements);
      for (int i = 0; i < numRemovals; i++) {
        REVERSE_DEPS_UTIL.removeReverseDep(example, SkyKey.create(NODE_TYPE, i));
      }
      assertThat(REVERSE_DEPS_UTIL.getReverseDeps(example)).hasSize(numElements - numRemovals);
      assertThat(example.dataToConsolidate).isNull();
    }
  }

  // Same as testAdditionAndRemoval but we add all the reverse deps in one call.
  @Test
  public void testAddAllAndRemove() {
    for (int numRemovals = 0; numRemovals <= numElements; numRemovals++) {
      Example example = new Example();
      List<SkyKey> toAdd = new ArrayList<>();
      for (int j = 0; j < numElements; j++) {
        toAdd.add(SkyKey.create(NODE_TYPE, j));
      }
      REVERSE_DEPS_UTIL.addReverseDeps(example, toAdd);
      assertThat(REVERSE_DEPS_UTIL.getReverseDeps(example)).hasSize(numElements);
      for (int i = 0; i < numRemovals; i++) {
        REVERSE_DEPS_UTIL.removeReverseDep(example, SkyKey.create(NODE_TYPE, i));
      }
      assertThat(REVERSE_DEPS_UTIL.getReverseDeps(example)).hasSize(numElements - numRemovals);
      assertThat(example.dataToConsolidate).isNull();
    }
  }

  @Test
  public void testDuplicateCheckOnGetReverseDeps() {
    Example example = new Example();
    for (int i = 0; i < numElements; i++) {
      REVERSE_DEPS_UTIL.addReverseDeps(example, Collections.singleton(SkyKey.create(NODE_TYPE, i)));
    }
    // Should only fail when we call getReverseDeps().
    REVERSE_DEPS_UTIL.addReverseDeps(example, Collections.singleton(SkyKey.create(NODE_TYPE, 0)));
    try {
      REVERSE_DEPS_UTIL.getReverseDeps(example);
      assertThat(numElements).isEqualTo(0);
    } catch (Exception expected) {
    }
  }

  @Test
  public void doubleAddThenRemove() {
    Example example = new Example();
    SkyKey key = SkyKey.create(NODE_TYPE, 0);
    REVERSE_DEPS_UTIL.addReverseDeps(example, Collections.singleton(key));
    // Should only fail when we call getReverseDeps().
    REVERSE_DEPS_UTIL.addReverseDeps(example, Collections.singleton(key));
    REVERSE_DEPS_UTIL.removeReverseDep(example, key);
    try {
      REVERSE_DEPS_UTIL.getReverseDeps(example);
      Assert.fail();
    } catch (IllegalStateException expected) {
    }
  }

  @Test
  public void doubleAddThenRemoveCheckedOnSize() {
    Example example = new Example();
    SkyKey fixedKey = SkyKey.create(NODE_TYPE, 0);
    SkyKey key = SkyKey.create(NODE_TYPE, 1);
    REVERSE_DEPS_UTIL.addReverseDeps(example, ImmutableList.of(fixedKey, key));
    // Should only fail when we reach the limit.
    REVERSE_DEPS_UTIL.addReverseDeps(example, Collections.singleton(key));
    REVERSE_DEPS_UTIL.removeReverseDep(example, key);
    REVERSE_DEPS_UTIL.checkReverseDep(example, fixedKey);
    try {
      REVERSE_DEPS_UTIL.checkReverseDep(example, fixedKey);
      Assert.fail();
    } catch (IllegalStateException expected) {
    }
  }

  @Test
  public void addRemoveAdd() {
    Example example = new Example();
    SkyKey fixedKey = SkyKey.create(NODE_TYPE, 0);
    SkyKey key = SkyKey.create(NODE_TYPE, 1);
    REVERSE_DEPS_UTIL.addReverseDeps(example, ImmutableList.of(fixedKey, key));
    REVERSE_DEPS_UTIL.removeReverseDep(example, key);
    REVERSE_DEPS_UTIL.addReverseDeps(example, Collections.singleton(key));
    assertThat(REVERSE_DEPS_UTIL.getReverseDeps(example)).containsExactly(fixedKey, key);
  }

  @Test
  public void testMaybeCheck() {
    Example example = new Example();
    for (int i = 0; i < numElements; i++) {
      REVERSE_DEPS_UTIL.addReverseDeps(example, Collections.singleton(SkyKey.create(NODE_TYPE, i)));
      // This should always succeed, since the next element is still not present.
      REVERSE_DEPS_UTIL.maybeCheckReverseDepNotPresent(example, SkyKey.create(NODE_TYPE, i + 1));
    }
    try {
      REVERSE_DEPS_UTIL.maybeCheckReverseDepNotPresent(example, SkyKey.create(NODE_TYPE, 0));
      // Should only fail if empty or above the checking threshold.
      assertThat(numElements == 0 || numElements >= ReverseDepsUtilImpl.MAYBE_CHECK_THRESHOLD)
          .isTrue();
    } catch (Exception expected) {
    }
  }
}
