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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
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
 * aspects and transition keys. {@link Dependency#getTransitionKeys} provides some more context on
 * transition keys.
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
  /** Builder to assist in creating dependency instances. */
  public static class Builder {
    private final Label label;

    private Builder(Label label) {
      this.label = Preconditions.checkNotNull(label);
    }

    /**
     * Returns a sub-builder for a new {@link Dependency} with a null configuration, suitable for edges with no configuration.
     */
    public NullConfigurationBuilder withNullConfiguration() {
      return new NullConfigurationBuilder(label);
    }

    /**
     * Returns a sub-builder for a new {@link Dependency} with the given explicit configuration. Should only be used for a dependency with no configuration changes.
     *
     * <p>The configuration must not be {@code null}.
     */
    public ExplicitConfigurationBuilder withConfiguration(BuildConfiguration configuration) {
      return new ExplicitConfigurationBuilder(label, configuration);
    }
  }

  /** Builder to assist in creating dependency instances with no configuration. */
  public static class NullConfigurationBuilder {
    private final Label label;
    private final List<String> transitionKeys = new ArrayList<>();

    private NullConfigurationBuilder(Label label) {
      this.label = Preconditions.checkNotNull(label);
    }

    /**
     * Add transition keys to this Dependency, used when edges with a split transition were resolved to null configuration targets.
     */
    public NullConfigurationBuilder addTransitionKeys(Collection<String> transitionKeys) {
      this.transitionKeys.addAll(transitionKeys);
      return this;
    }

    /** Returns the full Dependency instance. */
    public Dependency build() {
      return new NullConfigurationDependency(label, ImmutableList.copyOf(transitionKeys));
    }
  }

  /** Builder to assist in creating dependency instances with an explicit configuration. */
  public static class ExplicitConfigurationBuilder {
    private final Label label;
    private final BuildConfiguration configuration;
    private AspectCollection aspects = AspectCollection.EMPTY;
    private Map<AspectDescriptor, BuildConfiguration> aspectConfigurations = new HashMap<>();
    private final List<String> transitionKeys = new ArrayList<>();

    private ExplicitConfigurationBuilder(Label label, BuildConfiguration configuration) {
      this.label = Preconditions.checkNotNull(label);
      this.configuration = Preconditions.checkNotNull(configuration);
    }

    /**
     * Add aspects to this Dependency. Unless {@link #addAspectConfigurations} is also used, the same configuration is
     * applied to all aspects.
     */
    public ExplicitConfigurationBuilder addAspects(AspectCollection aspects) {
      this.aspects = aspects;
      return this;
    }

    /**
     * Add transition keys to this Dependency, used for edges with a split transition.
     */
    public ExplicitConfigurationBuilder addTransitionKey(String transitionKey) {
      if (transitionKey != null) {
        this.transitionKeys.add(transitionKey);
      }
      return this;
    }

    /**
     * Set explicit configurations for aspects on this Dependency (for example, after configuration trimming). Any aspects not specified will use the target's configuration.
     */
    public ExplicitConfigurationBuilder addAspectConfigurations(Map<AspectDescriptor, BuildConfiguration> aspectConfigurations) {
      this.aspectConfigurations.putAll(aspectConfigurations);
      return this;
    }

    /** Returns the full Dependency instance. */
    public Dependency build() {
      // Use the target configuration for all aspects with none of their own.
      for (AspectDescriptor aspect : aspects.getAllAspects()) {
        aspectConfigurations.putIfAbsent(aspect, configuration);
      }
      return new ExplicitConfigurationDependency(
          label,
          configuration,
          aspects,
          ImmutableMap.copyOf(aspectConfigurations),
          ImmutableList.copyOf(transitionKeys));
    }
  }

  public static Builder builder(Label label) {
    return new Builder(label);
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
   * Returns the keys of a configuration transition, if exist, associated with this dependency. See
   * {@link ConfigurationTransition#apply} for more details. Normally, this returns an empty list,
   * when there was no configuration transition in effect, or one with a single entry, when there
   * was a specific configuration transition result that led to this. It may also return a list with
   * multiple entries if the dependency has a null configuration, yet the outgoing edge has a split
   * transition. In such cases all transition keys returned by the transition are tagged to the
   * dependency.
   *
   * @throws IllegalStateException if {@link #hasExplicitConfiguration()} returns false.
   */
  public abstract ImmutableList<String> getTransitionKeys();

  /** Returns the ConfiguredTargetKey needed to fetch this dependency. */
  public abstract ConfiguredTargetKey getConfiguredTargetKey();

  /**
   * Implementation of a dependency with no configuration, suitable for, e.g., source files or
   * visibility.
   */
  private static final class NullConfigurationDependency extends Dependency {
    private final ImmutableList<String> transitionKeys;

    public NullConfigurationDependency(Label label, ImmutableList<String> transitionKeys) {
      super(label);
      this.transitionKeys = Preconditions.checkNotNull(transitionKeys);
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

    @Override
    public ImmutableList<String> getTransitionKeys() {
      return transitionKeys;
    }

    @Override
    public ConfiguredTargetKey getConfiguredTargetKey() {
      return ConfiguredTargetKey.of(getLabel(), null);
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
    private final ImmutableList<String> transitionKeys;

    public ExplicitConfigurationDependency(
        Label label,
        BuildConfiguration configuration,
        AspectCollection aspects,
        ImmutableMap<AspectDescriptor, BuildConfiguration> aspectConfigurations,
        ImmutableList<String> transitionKeys) {
      super(label);
      this.configuration = Preconditions.checkNotNull(configuration);
      this.aspects = Preconditions.checkNotNull(aspects);
      this.aspectConfigurations = Preconditions.checkNotNull(aspectConfigurations);
      this.transitionKeys = Preconditions.checkNotNull(transitionKeys);
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

    @Override
    public ImmutableList<String> getTransitionKeys() {
      return transitionKeys;
    }

    @Override
    public ConfiguredTargetKey getConfiguredTargetKey() {
      return ConfiguredTargetKey.of(getLabel(), getConfiguration());
    }

    @Override
    public int hashCode() {
      return Objects.hash(label, configuration, aspectConfigurations, transitionKeys);
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
}
