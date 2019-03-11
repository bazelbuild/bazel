package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory.TransitionFactoryData;

/**
 * Helper for the types of transitions that are statically declared but must be instantiated for
 * each use.
 */
public interface TransitionFactory<T extends TransitionFactoryData> {
  abstract class TransitionFactoryData {}

  /** Returns a new {@link ConfigurationTransition}. */
  ConfigurationTransition create(T data);

  default boolean isHost() {
    return false;
  }

  default boolean isSplit() {
    return false;
  }

  default boolean isFinal() {
    return false;
  }
}
