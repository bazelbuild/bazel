package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory.TransitionFactoryData;

/**
 * Helper for the types of transitions that are statically declared but must be instantiated for
 * each use.
 */
public interface TransitionFactory<T extends TransitionFactoryData> {

  /** Interface for types of data that a {@link TransitionFactory} can use. */
  interface TransitionFactoryData {}

  /** Returns a new {@link ConfigurationTransition}, based on the given data. */
  ConfigurationTransition create(T data);

  /** Returns {@code true} if the result of this {@link TransitionFactory} is a host transition. */
  default boolean isHost() {
    return false;
  }

  /** Returns {@code true} if the result of this {@link TransitionFactory} is a split transition. */
  default boolean isSplit() {
    return false;
  }

  /** Returns {@code true} if the result of this {@link TransitionFactory} is a final transition. */
  default boolean isFinal() {
    return false;
  }
}
