package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory.TransitionFactoryData;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import java.util.List;
import java.util.Objects;

public class ComposingTransitionFactory<T extends TransitionFactoryData>
    implements TransitionFactory<T> {

  /*
  public static <T extends TransitionFactoryData> TransitionFactory<T> create(
      ConfigurationTransition transition1, ConfigurationTransition transition2) {
    return create(TransitionFactories.of(transition1), TransitionFactories.of(transition2));
  }

  public static <T extends TransitionFactoryData> TransitionFactory<T> create(
      ConfigurationTransition transition1, TransitionFactory<T> factory2) {
    return create(TransitionFactories.of(transition1), factory2);
  }

  public static <T extends TransitionFactoryData> TransitionFactory<T> create(
      TransitionFactory<T> factory1, ConfigurationTransition transition2) {
    return create(factory1, TransitionFactories.of(transition2));
  }
  */

  public static <T extends TransitionFactoryData> TransitionFactory<T> create(
      TransitionFactory<T> factory1, TransitionFactory<T> factory2) {

    Preconditions.checkNotNull(factory1);
    Preconditions.checkNotNull(factory2);
    if (factory1.isFinal() || factory2 instanceof NoTransition.NoTransitionFactory) {
      return factory1;
    } else if (factory2.isFinal() || factory1 instanceof NoTransition.NoTransitionFactory) {
      // When the second transition is a HOST transition, there's no need to compose. But this also
      // improves performance: host transitions are common, and ConfiguredTargetFunction has special
      // optimized logic to handle them. If they were buried in the last segment of a
      // ComposingTransition, those optimizations wouldn't trigger.
      return factory2;
    }

    return new ComposingTransitionFactory<>(factory1, factory2);
  }

  private final TransitionFactory<T> factory1;
  private final TransitionFactory<T> factory2;

  @VisibleForSerialization
  ComposingTransitionFactory(TransitionFactory<T> factory1, TransitionFactory<T> factory2) {
    this.factory1 = factory1;
    this.factory2 = factory2;
  }

  @Override
  public ConfigurationTransition create(T data) {
    ConfigurationTransition transition1 = factory1.create(data);
    ConfigurationTransition transition2 = factory2.create(data);
    return new ComposingTransition(transition1, transition2);
  }

  @Override
  public boolean isHost() {
    return factory1.isHost() || factory2.isHost();
  }

  @Override
  public boolean isSplit() {
    return factory1.isSplit() || factory2.isSplit();
  }

  @Override
  public boolean isFinal() {
    return factory1.isFinal() || factory2.isFinal();
  }

  @Override
  public int hashCode() {
    return Objects.hash(factory1, factory2);
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof ComposingTransitionFactory
        && ((ComposingTransitionFactory) other).factory1.equals(this.factory1)
        && ((ComposingTransitionFactory) other).factory2.equals(this.factory2);
  }

  @VisibleForSerialization
  static class ComposingTransition implements ConfigurationTransition {
    private final ConfigurationTransition transition1;
    private final ConfigurationTransition transition2;

    @VisibleForSerialization
    ComposingTransition(
        ConfigurationTransition transition1, ConfigurationTransition transition2) {
      this.transition1 = transition1;
      this.transition2 = transition2;
    }

    @Override
    public List<BuildOptions> apply(BuildOptions buildOptions) {
      ImmutableList.Builder<BuildOptions> toOptions = ImmutableList.builder();
      for (BuildOptions transition1Options : transition1.apply(buildOptions)) {
        toOptions.addAll(transition2.apply(transition1Options));
      }
      return toOptions.build();
    }

    @Override
    public String reasonForOverride() {
      return "Basic abstraction for combining other transitions";
    }

    @Override
    public String getName() {
      return "(" + transition1.getName() + " + " + transition2.getName() + ")";
    }

    @Override
    public int hashCode() {
      return Objects.hash(transition1, transition2);
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof ComposingTransition
          && ((ComposingTransition) other).transition1.equals(this.transition1)
          && ((ComposingTransition) other).transition2.equals(this.transition2);
    }
  }
}
