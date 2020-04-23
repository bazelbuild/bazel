// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A dependency of a configured target through a label.
 *
 * <p>The dep's configuration can be specified in one of two ways:
 *
 * <p>Explicit configurations: includes the target and the configuration of the dependency
 * configured target and any aspects that may be required, as well as the configurations for these
 * aspects and an optional transition key. {@link Dependency#getTransitionKey} provides some more
 * context on transition keys.
 *
 * <p>Configuration transitions: includes the target and the desired configuration transitions that
 * should be applied to produce the dependency's configuration. It's the caller's responsibility to
 * construct an actual configuration out of that. A set of aspects is also included; the caller must
 * also construct configurations for each of these.
 *
 * <p>Note that the presence of an aspect here does not necessarily mean that it will be created.
 * They will be filtered based on the {@link TransitiveInfoProvider} instances their associated
 * configured targets have (we cannot do that here because the configured targets are not available
 * yet). No error or warning is reported in this case, because it is expected that rules sometimes
 * over-approximate the providers they supply in their definitions.
 */
public abstract class Dependency {

  /**
   * Creates a new {@link Dependency} with a null configuration, suitable for edges with no
   * configuration.
   */
  public static Dependency withNullConfiguration(Label label) {
    return new NullConfigurationDependency(label);
  }

  /**
   * Creates a new {@link Dependency} with the given explicit configuration.
   *
   * <p>The configuration must not be {@code null}.
   *
   * <p>A {@link Dependency} created this way will have no associated aspects.
   */
  public static Dependency withConfiguration(Label label, BuildConfiguration configuration) {
    return new ExplicitConfigurationDependency(
        label,
        configuration,
        AspectCollection.EMPTY,
        ImmutableMap.<AspectDescriptor, BuildConfiguration>of(),
        null);
  }

  /**
   * Creates a new {@link Dependency} with the given configuration and aspects. The configuration
   * is also applied to all aspects.
   *
   * <p>The configuration and aspects must not be {@code null}.
   */
  public static Dependency withConfigurationAndAspects(
      Label label, BuildConfiguration configuration, AspectCollection aspects) {
    ImmutableMap.Builder<AspectDescriptor, BuildConfiguration> aspectBuilder =
        new ImmutableMap.Builder<>();
    for (AspectDescriptor aspect : aspects.getAllAspects()) {
      aspectBuilder.put(aspect, configuration);
    }
    return new ExplicitConfigurationDependency(
        label, configuration, aspects, aspectBuilder.build(), null);
  }

  /**
   * Creates a new {@link Dependency} with the given configuration, aspects, and transition key. The
   * configuration is also applied to all aspects. This should be preferred over {@link
   * Dependency#withConfigurationAndAspects} if the {@code configuration} was derived from a {@link
   * ConfigurationTransition}, and so there is a corresponding transition key.
   *
   * <p>The configuration and aspects must not be {@code null}.
   */
  public static Dependency withConfigurationAspectsAndTransitionKey(
      Label label,
      BuildConfiguration configuration,
      AspectCollection aspects,
      @Nullable String transitionKey) {
    ImmutableMap.Builder<AspectDescriptor, BuildConfiguration> aspectBuilder =
        new ImmutableMap.Builder<>();
    for (AspectDescriptor aspect : aspects.getAllAspects()) {
      aspectBuilder.put(aspect, configuration);
    }
    return new ExplicitConfigurationDependency(
        label, configuration, aspects, aspectBuilder.build(), transitionKey);
  }

  /**
   * Creates a new {@link Dependency} with the given configuration and aspects, suitable for
   * storing the output of a configuration trimming step. The aspects each have their own
   * configuration.
   *
   * <p>The aspects and configurations must not be {@code null}.
   */
  public static Dependency withConfiguredAspects(
      Label label, BuildConfiguration configuration,
      AspectCollection aspects,
      Map<AspectDescriptor, BuildConfiguration> aspectConfigurations) {
    return new ExplicitConfigurationDependency(
        label, configuration, aspects, ImmutableMap.copyOf(aspectConfigurations), null);
  }

  /**
   * Creates a new {@link Dependency} with the given transition and aspects.
   */
  public static Dependency withTransitionAndAspects(
      Label label, ConfigurationTransition transition, AspectCollection aspects) {
    return new ConfigurationTransitionDependency(label, transition, aspects);
  }

  protected final Label label;

  /**
   * Only the implementations below should extend this class. Use the factory methods above to
   * create new Dependencies.
   */
  private Dependency(Label label) {
    this.label = Preconditions.checkNotNull(label);
  }

  /** Returns the label of the target this dependency points to. */
  public Label getLabel() {
    return label;
  }

  /**
   * Returns true if this dependency specifies an explicit configuration, false if it specifies
   * a configuration transition.
   */
  public abstract boolean hasExplicitConfiguration();

  /**
   * Returns the explicit configuration intended for this dependency.
   *
   * @throws IllegalStateException if {@link #hasExplicitConfiguration} returns false.
   */
  @Nullable
  public abstract BuildConfiguration getConfiguration();

  /**
   * Returns the configuration transition to apply to reach the target this dependency points to.
   *
   * @throws IllegalStateException if {@link #hasExplicitConfiguration} returns true.
   */
  public abstract ConfigurationTransition getTransition();

  /**
   * Returns the set of aspects which should be evaluated and combined with the configured target
   * pointed to by this dependency.
   *
   * @see #getAspectConfiguration(AspectDescriptor)
   */
  public abstract AspectCollection getAspects();

  /**
   * Returns the configuration an aspect should be evaluated with
   **
   * @throws IllegalStateException if {@link #hasExplicitConfiguration()} returns false.
   */
  public abstract BuildConfiguration getAspectConfiguration(AspectDescriptor aspect);

  /**
   * Returns the key of a configuration transition, if exists, associated with this dependency. See
   * {@link ConfigurationTransition#apply} for more details.
   *
   * @throws IllegalStateException if {@link #hasExplicitConfiguration()} returns false.
   */
  public abstract String getTransitionKey();

  /**
   * Implementation of a dependency with no configuration, suitable for, e.g., source files or
   * visibility.
   */
  private static final class NullConfigurationDependency extends Dependency {
    public NullConfigurationDependency(Label label) {
      super(label);
    }

    @Override
    public boolean hasExplicitConfiguration() {
      return true;
    }

    @Nullable
    @Override
    public BuildConfiguration getConfiguration() {
      return null;
    }

    @Override
    public ConfigurationTransition getTransition() {
      throw new IllegalStateException(
          "This dependency has an explicit configuration, not a transition.");
    }

    @Override
    public AspectCollection getAspects() {
      return AspectCollection.EMPTY;
    }

    @Override
    public BuildConfiguration getAspectConfiguration(AspectDescriptor aspect) {
      return null;
    }

    @Nullable
    @Override
    public String getTransitionKey() {
      return null;
    }

    @Override
    public int hashCode() {
      return label.hashCode();
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof NullConfigurationDependency)) {
        return false;
      }
      NullConfigurationDependency otherDep = (NullConfigurationDependency) other;
      return label.equals(otherDep.label);
    }

    @Override
    public String toString() {
      return String.format("NullConfigurationDependency{label=%s}", label);
    }
  }

  /**
   * Implementation of a dependency with an explicitly set configuration.
   */
  private static final class ExplicitConfigurationDependency extends Dependency {
    private final BuildConfiguration configuration;
    private final AspectCollection aspects;
    private final ImmutableMap<AspectDescriptor, BuildConfiguration> aspectConfigurations;
    @Nullable private final String transitionKey;

    public ExplicitConfigurationDependency(
        Label label,
        BuildConfiguration configuration,
        AspectCollection aspects,
        ImmutableMap<AspectDescriptor, BuildConfiguration> aspectConfigurations,
        @Nullable String transitionKey) {
      super(label);
      this.configuration = Preconditions.checkNotNull(configuration);
      this.aspects = Preconditions.checkNotNull(aspects);
      this.aspectConfigurations = Preconditions.checkNotNull(aspectConfigurations);
      this.transitionKey = transitionKey;
    }

    @Override
    public boolean hasExplicitConfiguration() {
      return true;
    }

    @Override
    public BuildConfiguration getConfiguration() {
      return configuration;
    }

    @Override
    public ConfigurationTransition getTransition() {
      throw new IllegalStateException(
          "This dependency has an explicit configuration, not a transition.");
    }

    @Override
    public AspectCollection getAspects() {
      return aspects;
    }

    @Override
    public BuildConfiguration getAspectConfiguration(AspectDescriptor aspect) {
      return aspectConfigurations.get(aspect);
    }

    @Nullable
    @Override
    public String getTransitionKey() {
      return transitionKey;
    }

    @Override
    public int hashCode() {
      return Objects.hash(label, configuration, aspectConfigurations, transitionKey);
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof ExplicitConfigurationDependency)) {
        return false;
      }
      ExplicitConfigurationDependency otherDep = (ExplicitConfigurationDependency) other;
      return label.equals(otherDep.label)
          && configuration.equals(otherDep.configuration)
          && aspects.equals(otherDep.aspects)
          && aspectConfigurations.equals(otherDep.aspectConfigurations);
    }

    @Override
    public String toString() {
      return String.format(
          "%s{label=%s, configuration=%s, aspectConfigurations=%s}",
          getClass().getSimpleName(), label, configuration, aspectConfigurations);
    }
  }

  /**
   * Implementation of a dependency with a given configuration transition.
   */
  private static final class ConfigurationTransitionDependency extends Dependency {
    private final ConfigurationTransition transition;
    private final AspectCollection aspects;

    public ConfigurationTransitionDependency(
        Label label, ConfigurationTransition transition, AspectCollection aspects) {
      super(label);
      this.transition = Preconditions.checkNotNull(transition);
      this.aspects = Preconditions.checkNotNull(aspects);
    }

    @Override
    public boolean hasExplicitConfiguration() {
      return false;
    }

    @Override
    public BuildConfiguration getConfiguration() {
      throw new IllegalStateException(
          "This dependency has a transition, not an explicit configuration.");
    }

    @Override
    public ConfigurationTransition getTransition() {
      return transition;
    }

    @Override
    public AspectCollection getAspects() {
      return aspects;
    }

    @Override
    public BuildConfiguration getAspectConfiguration(AspectDescriptor aspect) {
      throw new IllegalStateException(
          "This dependency has a transition, not an explicit aspect configuration.");
    }

    @Override
    public String getTransitionKey() {
      throw new IllegalStateException(
          "This dependency has a transition, not an explicit configuration.");
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
  }
}
