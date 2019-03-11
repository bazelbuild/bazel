package com.google.devtools.build.lib.analysis.config;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory.TransitionFactoryData;

public final class TransitionFactories {
  private TransitionFactories() {}

  // A static factory that returns a singleton transition.
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

  // A factory specifically for split transitions.
  public static <T extends TransitionFactoryData> TransitionFactory<T> split(
      SplitTransition splitTransition) {
    return new AutoValue_TransitionFactories_SplitTransitionFactory<T>(splitTransition);
  }

  @AutoValue
  abstract static class IdentityFactory<T extends TransitionFactoryData>
      implements TransitionFactory<T> {

    abstract ConfigurationTransition transition();

    @Override
    public ConfigurationTransition create(T data) {
      return transition();
    }
  }

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
