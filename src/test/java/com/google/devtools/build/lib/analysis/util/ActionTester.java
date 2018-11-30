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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Actions;
import java.util.BitSet;
import java.util.EnumSet;

/**
 * Test helper for testing {@link Action} implementations.
 */
public class ActionTester {

  /** A generator for action instances. */
  public interface ActionCombinationFactory<E extends Enum<E>> {

    /**
     * Returns a new action instance. The parameter {@code attributesToFlip} is used to vary the
     * parameters used to create the action. Implementations should do something like this: <code>
     * <pre>
     * private enum KeyAttributes { ATTR_1, ATTR_2, ATTR_3, ATTR_4 }
     * return new MyAction(owner, inputs, outputs, configuration,
     *     attributesToFlip.contains(ATTR_0) ? a1 : a2,
     *     attributesToFlip.contains(ATTR_1) ? b1 : b2,
     *     attributesToFlip.contains(ATTR_2) ? c1 : c2,
     *     attributesToFlip.contains(ATTR_3) ? d1 : d2);
     * </pre>
     * </code>
     *
     * <p>To reduce the combinatorial complexity of testing an action class, all elements that are
     * only used to change the executed command line should go into a single parameter, and the key
     * computation should take the generated command line into account.
     *
     * <p>Furthermore, when called with identical parameters, this method should return different
     * instances (i.e. according to {@code ==}), but they should have the same key.
     *
     * @param attributesToFlip
     */
    Action generate(ImmutableSet<E> attributesToFlip)
        throws InterruptedException;
  }

  /**
   * Tests that different actions have different keys. The attributeCount should specify how many
   * different permutations the {@link ActionCombinationFactory} should generate.
   */
  public static <E extends Enum<E>> void runTest(
      Class<E> attributeClass,
      ActionCombinationFactory<E> factory,
      ActionKeyContext actionKeyContext)
      throws Exception {
    int attributesCount = attributeClass.getEnumConstants().length;
    Preconditions.checkArgument(
        attributesCount <= 30,
        "Maximum attribute count is 30, more will overflow the max array size.");
    int count = (int) Math.pow(2, attributesCount);
    Action[] actions = new Action[count];
    for (int i = 0; i < actions.length; i++) {
      actions[i] = factory.generate(makeEnumSetInitializedTo(attributeClass, i));
    }
    // Sanity check that the count is correct.
    assertThat(
            Actions.areTheSame(
                actionKeyContext,
                actions[0],
                factory.generate(makeEnumSetInitializedTo(attributeClass, count))))
        .isTrue();

    for (int i = 0; i < actions.length; i++) {
      assertThat(
              Actions.areTheSame(
                  actionKeyContext,
                  actions[i],
                  factory.generate(makeEnumSetInitializedTo(attributeClass, i))))
          .isTrue();
      for (int j = i + 1; j < actions.length; j++) {
        assertWithMessage(i + " and " + j)
            .that(Actions.areTheSame(actionKeyContext, actions[i], actions[j]))
            .isFalse();
      }
    }
  }

  private static <E extends Enum<E>> ImmutableSet<E> makeEnumSetInitializedTo(
      Class<E> attributeClass, int seed) {
    EnumSet<E> result = EnumSet.<E>noneOf(attributeClass);
    BitSet b = BitSet.valueOf(new long[] {seed});
    E[] attributes = attributeClass.getEnumConstants();
    for (int i = 0; i < attributes.length; i++) {
      if (b.get(i)) {
        result.add(attributes[i]);
      }
    }
    return Sets.immutableEnumSet(result);
  }
}
