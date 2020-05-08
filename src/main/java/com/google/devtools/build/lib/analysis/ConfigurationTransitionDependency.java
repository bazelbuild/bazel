package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import java.util.Objects;

/**
 * Implementation of a dependency with a given configuration transition.
 */
public final class ConfigurationTransitionDependency {
  private final Label label;
  private final ConfigurationTransition transition;
  private final AspectCollection aspects;

  private ConfigurationTransitionDependency(
      Label label, ConfigurationTransition transition, AspectCollection aspects) {
    this.label = label;
    this.transition = Preconditions.checkNotNull(transition);
    this.aspects = Preconditions.checkNotNull(aspects);
  }

  /** Returns the label of the target this dependency points to. */
  public Label getLabel() {
    return label;
  }

  public ConfigurationTransition getTransition() {
    return transition;
  }

  public AspectCollection getAspects() {
    return aspects;
  }

  @Override
  public int hashCode() {
    return Objects.hash(label, transition, aspects);
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof ConfigurationTransitionDependency)) {
      return false;
    }
    ConfigurationTransitionDependency otherDep = (ConfigurationTransitionDependency) other;
    return label.equals(otherDep.label)
        && transition.equals(otherDep.transition)
        && aspects.equals(otherDep.aspects);
  }

  @Override
  public String toString() {
    return String.format(
        "%s{label=%s, transition=%s, aspects=%s}",
        getClass().getSimpleName(), label, transition, aspects);
  }

  public static ConfigurationTransitionBuilder builder() {
    return new ConfigurationTransitionBuilder();
  }

  /** Builder to assist in creating dependency instances with a configuration transition. */
  public static class ConfigurationTransitionBuilder {
    private Label label;
    private ConfigurationTransition transition;
    private AspectCollection aspects = AspectCollection.EMPTY;

    private ConfigurationTransitionBuilder() {
    }

    public ConfigurationTransitionBuilder setLabel(Label label) {
      this.label = label;
      return this;
    }

    public ConfigurationTransitionBuilder setTransition(ConfigurationTransition transition) {
      this.transition = transition;
      return this;
    }

    /**
     * Add aspects to this Dependency.
     */
    public ConfigurationTransitionBuilder addAspects(AspectCollection aspects) {
      this.aspects = aspects;
      return this;
    }

    /** Returns the full Dependency instance. */
    public ConfigurationTransitionDependency build() {
      return new ConfigurationTransitionDependency(label, transition, aspects);
    }
  }
}
