package com.google.devtools.build.lib.analysis;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Implementation of a dependency with a given configuration transition.
 */
@AutoValue
public abstract class ConfigurationTransitionDependency {
  /** Returns the label of the target this dependency points to. */
  public abstract Label getLabel();

  public abstract ConfigurationTransition getTransition();

  public abstract AspectCollection getAspects();

  public static ConfigurationTransitionBuilder builder() {
    return new AutoValue_ConfigurationTransitionDependency.Builder()
        .setAspects(AspectCollection.EMPTY);
  }

  /** Builder to assist in creating dependency instances with a configuration transition. */
  @AutoValue.Builder
  public static abstract class ConfigurationTransitionBuilder {
    public abstract ConfigurationTransitionBuilder setLabel(Label label);

    public abstract ConfigurationTransitionBuilder setTransition(ConfigurationTransition transition);

    /**
     * Add aspects to this Dependency.
     */
    public abstract ConfigurationTransitionBuilder setAspects(AspectCollection aspectCollection);

    /** Returns the full Dependency instance. */
    public abstract ConfigurationTransitionDependency build();
  }
}
