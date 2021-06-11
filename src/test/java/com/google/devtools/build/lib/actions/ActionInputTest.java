// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.ActionInput.EXEC_PATH_COMPARATOR;
import static java.util.Comparator.naturalOrder;

import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.Comparator;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link ActionInput}. */
@RunWith(TestParameterInjector.class)
public final class ActionInputTest {

  enum TestedComparator {
    NATURAL_ORDER(naturalOrder()),
    EXEC_PATH_COMPARATOR(ActionInput.EXEC_PATH_COMPARATOR);

    TestedComparator(Comparator<ActionInput> comparator) {
      this.comparator = comparator;
    }

    @SuppressWarnings("ImmutableEnumChecker")
    final Comparator<ActionInput> comparator;
  }

  @Test
  public void compare_differentPaths_returnsCorrectValue(
      @TestParameter TestedComparator testedComparator) {
    ActionInput lower = ActionInputHelper.fromPath("a");
    ActionInput higher = ActionInputHelper.fromPath("b");

    assertThat(testedComparator.comparator.compare(lower, higher)).isLessThan(0);
    assertThat(testedComparator.comparator.compare(higher, lower)).isGreaterThan(0);
  }

  @Test
  public void compare_nestedPaths_returnsCorrectValue(
      @TestParameter TestedComparator testedComparator) {
    ActionInput lower = ActionInputHelper.fromPath("a/b/c");
    ActionInput higher = ActionInputHelper.fromPath("a/b/c/d");

    assertThat(testedComparator.comparator.compare(lower, higher)).isLessThan(0);
    assertThat(testedComparator.comparator.compare(higher, lower)).isGreaterThan(0);
  }

  @Test
  public void compare_sameExecPath_returnsZero(@TestParameter TestedComparator testedComparator) {
    ActionInput input1 = ActionInputHelper.fromPath("a");
    ActionInput input2 =
        new ActionInput() {
          @Override
          public String getExecPathString() {
            return "a";
          }

          @Override
          public PathFragment getExecPath() {
            return PathFragment.create("a");
          }

          @Override
          public boolean isSymlink() {
            throw new UnsupportedOperationException();
          }
        };

    assertThat(testedComparator.comparator.compare(input1, input2)).isEqualTo(0);
    assertThat(testedComparator.comparator.compare(input2, input1)).isEqualTo(0);
  }

  @Test
  public void compare_nullPathIsLowest(@TestParameter TestedComparator testedComparator) {
    ActionInput input = ActionInputHelper.fromPath("a");

    assertThat(testedComparator.comparator.compare(input, null)).isGreaterThan(0);
  }

  @Test
  public void execPathComparatorCompare_nullFirstParameter_returnsLowerThanZero() {
    ActionInput input = ActionInputHelper.fromPath("a");

    assertThat(EXEC_PATH_COMPARATOR.compare(null, input)).isLessThan(0);
  }

  @SuppressWarnings("EqualsWithItself")
  @Test
  public void execPathComparatorCompare_nulls_returnsZero() {
    assertThat(EXEC_PATH_COMPARATOR.compare(null, null)).isEqualTo(0);
  }
}
