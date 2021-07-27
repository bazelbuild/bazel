// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;

/**
 * A transition factory that composes two other transition factories in an ordered sequence.
 *
 * <p>Example:
 *
 * <pre>
 *   transitionFactory1: { someSetting = $oldVal + " foo" }
 *   transitionFactory2: { someSetting = $oldVal + " bar" }
 *   ComposingTransitionFactory(transitionFactory1, transitionFactory2):
 *     { someSetting = $oldVal + " foo bar" }
 * </pre>
 */
@AutoValue
public abstract class ComposingTransitionFactory<T extends TransitionFactory.Data>
    implements TransitionFactory<T> {

  /**
   * Creates a {@link ComposingTransitionFactory} that applies the given factories in sequence:
   * {@code fromOptions -> transition1 -> transition2 -> toOptions }.
   *
   * <p>Note that this method checks for transition factories that cannot be composed, such as if
   * one of the transitions is {@link NoTransition} or the host transition, and returns an
   * efficiently composed transition.
   */
  public static <T extends TransitionFactory.Data> TransitionFactory<T> of(
      TransitionFactory<T> transitionFactory1, TransitionFactory<T> transitionFactory2) {

    Preconditions.checkNotNull(transitionFactory1);
    Preconditions.checkNotNull(transitionFactory2);
    Preconditions.checkArgument(
        !transitionFactory1.isSplit() || !transitionFactory2.isSplit(),
        "can't compose two split transition factories");

    if (isFinal(transitionFactory1)) {
      // Since no other transition can be composed with transitionFactory1, use it directly.
      return transitionFactory1;
    } else if (NoTransition.isInstance(transitionFactory1)) {
      // Since transitionFactory1 causes no changes, use transitionFactory2 directly.
      return transitionFactory2;
    }

    if (NoTransition.isInstance(transitionFactory2)) {
      // Since transitionFactory2 causes no changes, use transitionFactory1 directly.
      return transitionFactory1;
    } else if (isFinal(transitionFactory2)) {
      // When the second transition is null or a HOST transition, there's no need to compose. But
      // this also
      // improves performance: host transitions are common, and ConfiguredTargetFunction has special
      // optimized logic to handle them. If they were buried in the last segment of a
      // ComposingTransition, those optimizations wouldn't trigger.
      return transitionFactory2;
    }

    return create(transitionFactory1, transitionFactory2);
  }

  private static <T extends TransitionFactory.Data> boolean isFinal(
      TransitionFactory<T> transitionFactory) {
    return NullTransition.isInstance(transitionFactory) || transitionFactory.isHost();
  }

  private static <T extends TransitionFactory.Data> TransitionFactory<T> create(
      TransitionFactory<T> transitionFactory1, TransitionFactory<T> transitionFactory2) {
    return new AutoValue_ComposingTransitionFactory<T>(transitionFactory1, transitionFactory2);
  }

  abstract TransitionFactory<T> transitionFactory1();

  abstract TransitionFactory<T> transitionFactory2();

  @Override
  public ConfigurationTransition create(T data) {
    ConfigurationTransition transition1 = transitionFactory1().create(data);
    ConfigurationTransition transition2 = transitionFactory2().create(data);
    return new ComposingTransition(transition1, transition2);
  }

  @Override
  public boolean isHost() {
    return transitionFactory1().isHost() || transitionFactory2().isHost();
  }

  @Override
  public boolean isTool() {
    return transitionFactory1().isTool() || transitionFactory2().isTool();
  }

  @Override
  public boolean isSplit() {
    return transitionFactory1().isSplit() || transitionFactory2().isSplit();
  }
}
