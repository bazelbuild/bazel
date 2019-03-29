package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory.TransitionFactoryData;

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
public abstract class ComposingTransitionFactory<T extends TransitionFactoryData>
    implements TransitionFactory<T> {

  /**
   * Creates a {@link ComposingTransitionFactory} that applies the given factories in sequence:
   * {@code fromOptions -> transition1 -> transition2 -> toOptions }.
   *
   * <p>Note that this method checks for transition factories that cannot be composed, such as if
   * one of the transitions is {@link NoTransition} or the host transition, and returns an
   * efficiently composed transition.
   */
  public static <T extends TransitionFactoryData> TransitionFactory<T> of(
      TransitionFactory<T> transitionFactory1, TransitionFactory<T> transitionFactory2) {

    Preconditions.checkNotNull(transitionFactory1);
    Preconditions.checkNotNull(transitionFactory2);

    if (transitionFactory1.isFinal()) {
      // Since no other transition can be composed with transitionFactory1, use it directly.
      return transitionFactory1;
    } else if (NoTransition.isInstance(transitionFactory1)) {
      // Since transitionFactory1 causes no changes, use transitionFactory2 directly.
      return transitionFactory2;
    }

    if (NoTransition.isInstance(transitionFactory2)) {
      // Since transitionFactory2 causes no changes, use transitionFactory1 directly.
      return transitionFactory1;
    } else if (NullTransition.isInstance(transitionFactory2) || transitionFactory2.isHost()) {
      // When the second transition is null or a HOST transition, there's no need to compose. But
      // this also
      // improves performance: host transitions are common, and ConfiguredTargetFunction has special
      // optimized logic to handle them. If they were buried in the last segment of a
      // ComposingTransition, those optimizations wouldn't trigger.
      return transitionFactory2;
    }

    return create(transitionFactory1, transitionFactory2);
  }

  private static <T extends TransitionFactoryData> TransitionFactory<T> create(
      TransitionFactory<T> transitionFactory1, TransitionFactory<T> transitionFactory2) {
    return new AutoValue_ComposingTransitionFactory(transitionFactory1, transitionFactory2);
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
  public boolean isSplit() {
    return transitionFactory1().isSplit() || transitionFactory2().isSplit();
  }

  @Override
  public boolean isFinal() {
    return transitionFactory1().isFinal() || transitionFactory2().isFinal();
  }

  // TODO(https://github.com/bazelbuild/bazel/issues/7814): Move ComposingTransition here and make
  // it private.
}
