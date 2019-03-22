package com.google.devtools.build.lib.analysis.config;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory.TransitionFactoryData;

/** Useful implementations of {@link TransitionFactory}. */
// This class is in lib.analysis.config in order to access HostTransition, which is not visible to
// lib.analysis.config.transitions.
public final class TransitionFactories {
  // Don't instantiate this class.
  private TransitionFactories() {}

  /** Returns a {@link TransitionFactory} that wraps a static transition. */
  public static <T extends TransitionFactoryData> TransitionFactory<T> of(
      ConfigurationTransition transition) {
    if (transition instanceof HostTransition) {
      return HostTransition.createFactory();
    } else if (transition instanceof NoTransition) {
      return NoTransition.createFactory();
    } else if (transition instanceof NullTransition) {
      return NullTransition.createFactory();
    } else if (transition instanceof SplitTransition) {
      return split((SplitTransition) transition);
    }
    return new AutoValue_TransitionFactories_IdentityFactory(transition);
  }

  /** Returns a {@link TransitionFactory} that wraps a static split transition. */
  public static <T extends TransitionFactoryData> TransitionFactory<T> split(
      SplitTransition splitTransition) {
    return new AutoValue_TransitionFactories_SplitTransitionFactory<T>(splitTransition);
  }

  /** A {@link TransitionFactory} implementation that wraps a static transition. */
  @AutoValue
  abstract static class IdentityFactory<T extends TransitionFactoryData>
      implements TransitionFactory<T> {

    abstract ConfigurationTransition transition();

    @Override
    public ConfigurationTransition create(T data) {
      return transition();
    }
  }

  /** A {@link TransitionFactory} implementation that wraps a split transition. */
  @AutoValue
  abstract static class SplitTransitionFactory<T extends TransitionFactoryData>
      implements TransitionFactory<T> {
    abstract SplitTransition splitTransition();

    @Override
    public SplitTransition create(T data) {
      return splitTransition();
    }

    @Override
    public boolean isSplit() {
      return true;
    }
  }
}
