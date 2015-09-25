// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.util;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Actions;

/**
 * Test helper for testing {@link Action} implementations.
 */
public class ActionTester {

  /**
   * A generator for action instances.
   */
  public interface ActionCombinationFactory {

    /**
     * Returns a new action instance. The parameter {@code i} is used to vary the parameters used to
     * create the action. Implementations should do something like this:
     * <code><pre>
     * return new MyAction(owner, inputs, outputs, configuration,
     *     (i & 1) == 0 ? a1 : a2,
     *     (i & 2) == 0 ? b1 : b2,
     *     (i & 4) == 0 ? c1 : c2);
     *     (i & 16) == 0 ? d1 : d2);
     * </pre></code>
     *
     * <p>The wrap-around (in this case at 32) is intentional and is checked for by the testing
     * method.
     *
     * <p>To reduce the combinatorial complexity of testing an action class, all elements that are
     * only used to change the executed command line should go into a single parameter, and the key
     * computation should take the generated command line into account.
     *
     * <p>Furthermore, when called with identical parameters, this method should return different
     * instances (i.e. according to {@code ==}), but they should have the same key.
     */
    Action generate(int i);
  }

  /**
   * Tests that different actions have different keys. The count should specify how many different
   * permutations the {@link ActionCombinationFactory} can generate.
   */
  public static void runTest(int count, ActionCombinationFactory factory) throws Exception {
    Action[] actions = new Action[count];
    for (int i = 0; i < actions.length; i++) {
      actions[i] = factory.generate(i);
    }
    // Sanity check that the count is correct.
    assertThat(Actions.canBeShared(actions[0], factory.generate(count))).isTrue();

    for (int i = 0; i < actions.length; i++) {
      assertThat(Actions.canBeShared(actions[i], factory.generate(i))).isTrue();
      for (int j = i + 1; j < actions.length; j++) {
        assertWithMessage(i + " and " + j).that(Actions.canBeShared(actions[i], actions[j]))
            .isFalse();
      }
    }
  }
}
