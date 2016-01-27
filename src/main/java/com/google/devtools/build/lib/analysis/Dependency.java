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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.Map;
import java.util.Objects;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A dependency of a configured target through a label.
 *
 * <p>For static configurations: includes the target and the configuration of the dependency
 * configured target and any aspects that may be required, as well as the configurations for
 * these aspects.
 *
 * <p>For dynamic configurations: includes the target and the desired configuration transitions
 * that should be applied to produce the dependency's configuration. It's the caller's
 * responsibility to construct an actual configuration out of that. A set of aspects is also
 * included; the caller must also construct configurations for each of these.
 *
 * <p>Note that the presence of an aspect here does not necessarily mean that it will be created.
 * They will be filtered based on the {@link TransitiveInfoProvider} instances their associated
 * configured targets have (we cannot do that here because the configured targets are not
 * available yet). No error or warning is reported in this case, because it is expected that rules
 * sometimes over-approximate the providers they supply in their definitions.
 */
public abstract class Dependency {

  /**
   * Creates a new {@link Dependency} with a null configuration, suitable for edges with no
   * configuration in static configuration builds.
   */
  public static Dependency withNullConfiguration(Label label) {
    return new NullConfigurationDependency(label);
  }

  /**
   * Creates a new {@link Dependency} with the given configuration, suitable for static
   * configuration builds.
   *
   * <p>The configuration must not be {@code null}.
   *
   * <p>A {@link Dependency} created this way will have no associated aspects.
   */
  public static Dependency withConfiguration(Label label, BuildConfiguration configuration) {
    return new StaticConfigurationDependency(
        label, configuration, ImmutableMap.<Aspect, BuildConfiguration>of());
  }

  /**
   * Creates a new {@link Dependency} with the given configuration and aspects, suitable for
   * static configuration builds. The configuration is also applied to all aspects.
   *
   * <p>The configuration and aspects must not be {@code null}.
   */
  public static Dependency withConfigurationAndAspects(
      Label label, BuildConfiguration configuration, Set<Aspect> aspects) {
    ImmutableMap.Builder<Aspect, BuildConfiguration> aspectBuilder = new ImmutableMap.Builder<>();
    for (Aspect aspect : aspects) {
      aspectBuilder.put(aspect, configuration);
    }
    return new StaticConfigurationDependency(label, configuration, aspectBuilder.build());
  }

  /**
   * Creates a new {@link Dependency} with the given configuration and aspects, suitable for
   * storing the output of a dynamic configuration trimming step. The aspects each have their own
   * configuration.
   *
   * <p>The aspects and configurations must not be {@code null}.
   */
  public static Dependency withConfiguredAspects(
      Label label, BuildConfiguration configuration,
      Map<Aspect, BuildConfiguration> aspectConfigurations) {
    return new StaticConfigurationDependency(
        label, configuration, ImmutableMap.copyOf(aspectConfigurations));
  }

  /**
   * Creates a new {@link Dependency} with the given transition and aspects, suitable for dynamic
   * configuration builds.
   */
  public static Dependency withTransitionAndAspects(
      Label label, Attribute.Transition transition, Set<Aspect> aspects) {
    return new DynamicConfigurationDependency(label, transition, ImmutableSet.copyOf(aspects));
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
   * Returns true if this dependency specifies a static configuration, false if it specifies
   * a dynamic transition.
   */
  public abstract boolean hasStaticConfiguration();

  /**
   * Returns the static configuration for the target this dependency points to.
   *
   * @throws IllegalStateException if {@link #hasStaticConfiguration} returns false.
   */
  @Nullable
  public abstract BuildConfiguration getConfiguration();

  /**
   * Returns the dynamic transition to be applied to reach the target this dependency points to.
   *
   * @throws IllegalStateException if {@link #hasStaticConfiguration} returns true.
   */
  public abstract Attribute.Transition getTransition();

  /**
   * Returns the set of aspects which should be evaluated and combined with the configured target
   * pointed to by this dependency.
   *
   * @see #getAspectConfigurations()
   */
  public abstract ImmutableSet<Aspect> getAspects();

  /**
   * Returns the mapping from aspects to the static configurations they should be evaluated with.
   *
   * <p>The {@link Map#keySet()} of this map is equal to that returned by {@link #getAspects()}.
   *
   * @throws IllegalStateException if {@link #hasStaticConfiguration()} returns false.
   */
  public abstract ImmutableMap<Aspect, BuildConfiguration> getAspectConfigurations();

  /**
   * Implementation of a dependency with no configuration, suitable for static configuration
   * builds of edges to source files or e.g. for visibility.
   */
  private static final class NullConfigurationDependency extends Dependency {
    public NullConfigurationDependency(Label label) {
      super(label);
    }

    @Override
    public boolean hasStaticConfiguration() {
      return true;
    }

    @Nullable
    @Override
    public BuildConfiguration getConfiguration() {
      return null;
    }

    @Override
    public Attribute.Transition getTransition() {
      throw new IllegalStateException(
          "A dependency with a static configuration does not have a transition.");
    }

    @Override
    public ImmutableSet<Aspect> getAspects() {
      return ImmutableSet.of();
    }

    @Override
    public ImmutableMap<Aspect, BuildConfiguration> getAspectConfigurations() {
      return ImmutableMap.of();
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
   * Implementation of a dependency with static configurations, suitable for static configuration
   * builds.
   */
  private static final class StaticConfigurationDependency extends Dependency {
    private final BuildConfiguration configuration;
    private final ImmutableMap<Aspect, BuildConfiguration> aspectConfigurations;

    public StaticConfigurationDependency(
        Label label, BuildConfiguration configuration,
        ImmutableMap<Aspect, BuildConfiguration> aspects) {
      super(label);
      this.configuration = Preconditions.checkNotNull(configuration);
      this.aspectConfigurations = Preconditions.checkNotNull(aspects);
    }

    @Override
    public boolean hasStaticConfiguration() {
      return true;
    }

    @Override
    public BuildConfiguration getConfiguration() {
      return configuration;
    }

    @Override
    public Attribute.Transition getTransition() {
      throw new IllegalStateException(
          "A dependency with a static configuration does not have a transition.");
    }

    @Override
    public ImmutableSet<Aspect> getAspects() {
      return aspectConfigurations.keySet();
    }

    @Override
    public ImmutableMap<Aspect, BuildConfiguration> getAspectConfigurations() {
      return aspectConfigurations;
    }

    @Override
    public int hashCode() {
      return Objects.hash(label, configuration, aspectConfigurations);
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof StaticConfigurationDependency)) {
        return false;
      }
      StaticConfigurationDependency otherDep = (StaticConfigurationDependency) other;
      return label.equals(otherDep.label)
          && configuration.equals(otherDep.configuration)
          && aspectConfigurations.equals(otherDep.aspectConfigurations);
    }

    @Override
    public String toString() {
      return String.format(
          "StaticConfigurationDependency{label=%s, configuration=%s, aspectConfigurations=%s}",
          label, configuration, aspectConfigurations);
    }
  }

  /**
   * Implementation of a dependency with a given configuration transition, suitable for dynamic
   * configuration builds.
   */
  private static final class DynamicConfigurationDependency extends Dependency {
    private final Attribute.Transition transition;
    private final ImmutableSet<Aspect> aspects;

    public DynamicConfigurationDependency(
        Label label, Attribute.Transition transition, ImmutableSet<Aspect> aspects) {
      super(label);
      this.transition = Preconditions.checkNotNull(transition);
      this.aspects = Preconditions.checkNotNull(aspects);
    }

    @Override
    public boolean hasStaticConfiguration() {
      return false;
    }

    @Override
    public BuildConfiguration getConfiguration() {
      throw new IllegalStateException(
          "A dependency with a dynamic configuration transition does not have a configuration.");
    }

    @Override
    public Attribute.Transition getTransition() {
      return transition;
    }

    @Override
    public ImmutableSet<Aspect> getAspects() {
      return aspects;
    }

    @Override
    public ImmutableMap<Aspect, BuildConfiguration> getAspectConfigurations() {
      throw new IllegalStateException(
          "A dependency with a dynamic configuration transition does not have aspect "
          + "configurations.");
    }

    @Override
    public int hashCode() {
      return Objects.hash(label, transition, aspects);
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof DynamicConfigurationDependency)) {
        return false;
      }
      DynamicConfigurationDependency otherDep = (DynamicConfigurationDependency) other;
      return label.equals(otherDep.label)
          && transition.equals(otherDep.transition)
          && aspects.equals(otherDep.aspects);
    }

    @Override
    public String toString() {
      return String.format(
          "DynamicConfigurationDependency{label=%s, transition=%s, aspects=%s}",
          label, transition, aspects);
    }
  }
}
