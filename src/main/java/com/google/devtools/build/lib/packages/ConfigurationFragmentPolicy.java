// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkInterfaceUtils;

/**
 * Policy used to express the set of configuration fragments which are legal for a rule or aspect to
 * access.
 */
@AutoCodec
public final class ConfigurationFragmentPolicy {

  /**
   * How to handle the case if the configuration is missing fragments that are required according
   * to the rule class.
   */
  public enum MissingFragmentPolicy {
    /**
     * Some rules are monolithic across languages, and we want them to continue to work even when
     * individual languages are disabled. Use this policy if the rule implementation is handling
     * missing fragments.
     */
    IGNORE,

    /**
     * Use this policy to generate fail actions for the target rather than failing the analysis
     * outright. Again, this is used when rules are monolithic across languages, but we still need
     * to analyze the dependent libraries. (Instead of this mechanism, consider annotating
     * attributes as unused if certain fragments are unavailable.)
     */
    CREATE_FAIL_ACTIONS,

    /**
     * Use this policy to fail the analysis of that target with an error message; this is the
     * default.
     */
    FAIL_ANALYSIS;
  }

  /**
   * Builder to construct a new ConfigurationFragmentPolicy.
   */
  public static final class Builder {
    /**
     * Sets of configuration fragment classes required by this rule, a set for each configuration.
     * Duplicate entries will automatically be ignored by the SetMultimap.
     */
    private final SetMultimap<ConfigurationTransition, Class<?>> requiredConfigurationFragments
        = LinkedHashMultimap.create();

    /**
     * Sets of configuration fragments required by this rule, as defined by their Starlark names
     * (see {@link StarlarkBuiltin}, a set for each configuration.
     *
     * <p>Duplicate entries will automatically be ignored by the SetMultimap.
     */
    private final SetMultimap<ConfigurationTransition, String>
        starlarkRequiredConfigurationFragments = LinkedHashMultimap.create();

    private final Map<Class<?>, MissingFragmentPolicy> missingFragmentPolicy =
        new LinkedHashMap<>();

    /**
     * Declares that the implementation of the associated rule class requires the given
     * fragments to be present in this rule's target configuration only.
     *
     * <p>The value is inherited by subclasses.
     */
    public Builder requiresConfigurationFragments(Collection<Class<?>> configurationFragments) {
      requiresConfigurationFragments(NoTransition.INSTANCE, configurationFragments);
      return this;
    }

    /**
     * Declares that the implementation of the associated rule class requires the given
     * fragments to be present in the specified configuration. Valid transition values are
     * HOST for the host configuration and NONE for the target configuration.
     *
     * <p>The value is inherited by subclasses.
     */
    public Builder requiresConfigurationFragments(ConfigurationTransition transition,
        Collection<Class<?>> configurationFragments) {
      // We can relax this assumption if needed. But it's already sketchy to let a rule see more
      // than its own configuration. So we don't want to casually proliferate this pattern.
      Preconditions.checkArgument(
          transition == NoTransition.INSTANCE || transition.isHostTransition());
      requiredConfigurationFragments.putAll(transition, configurationFragments);
      return this;
    }

    /**
     * Declares that the implementation of the associated rule class requires the given fragments to
     * be present in this rule's target configuration only.
     *
     * <p>In contrast to {@link #requiresConfigurationFragments(Collection)}, this method takes the
     * names of fragments (as determined by {@link StarlarkBuiltin}) instead of their classes.
     *
     * <p>The value is inherited by subclasses.
     */
    public Builder requiresConfigurationFragmentsByStarlarkBuiltinName(
        Collection<String> configurationFragmentNames) {

      requiresConfigurationFragmentsByStarlarkBuiltinName(
          NoTransition.INSTANCE, configurationFragmentNames);
      return this;
    }

    /**
     * Declares the configuration fragments that are required by this rule for the specified
     * configuration. Valid transition values are HOST for the host configuration and NONE for the
     * target configuration.
     *
     * <p>In contrast to {@link #requiresConfigurationFragments(ConfigurationTransition,
     * Collection)}, this method takes the names of fragments (as determined by {@link
     * StarlarkBuiltin}) instead of their classes.
     */
    public Builder requiresConfigurationFragmentsByStarlarkBuiltinName(
        ConfigurationTransition transition, Collection<String> configurationFragmentNames) {
      // We can relax this assumption if needed. But it's already sketchy to let a rule see more
      // than its own configuration. So we don't want to casually proliferate this pattern.
      Preconditions.checkArgument(
          transition == NoTransition.INSTANCE || transition.isHostTransition());
      starlarkRequiredConfigurationFragments.putAll(transition, configurationFragmentNames);
      return this;
    }

    /**
     * Adds the configuration fragments from the {@code other} policy to this Builder.
     *
     * <p>Missing fragment policy is also copied over, overriding previously set values.
     */
    public Builder includeConfigurationFragmentsFrom(ConfigurationFragmentPolicy other) {
      requiredConfigurationFragments.putAll(other.requiredConfigurationFragments);
      starlarkRequiredConfigurationFragments.putAll(other.starlarkRequiredConfigurationFragments);
      missingFragmentPolicy.putAll(other.missingFragmentPolicy);
      return this;
    }

    /**
     * Sets the policy for the case where the configuration is missing specified required fragment
     * class (see {@link #requiresConfigurationFragments}).
     */
    public Builder setMissingFragmentPolicy(
        Class<?> fragmentClass, MissingFragmentPolicy missingFragmentPolicy) {
      this.missingFragmentPolicy.put(fragmentClass, missingFragmentPolicy);
      return this;
    }

    public ConfigurationFragmentPolicy build() {
      return new ConfigurationFragmentPolicy(
          ImmutableSetMultimap.copyOf(requiredConfigurationFragments),
          ImmutableSetMultimap.copyOf(starlarkRequiredConfigurationFragments),
          ImmutableMap.copyOf(missingFragmentPolicy));
    }
  }

  /**
   * A dictionary that maps configurations (NONE for target configuration, HOST for host
   * configuration) to required configuration fragments.
   */
  private final ImmutableSetMultimap<ConfigurationTransition, Class<?>>
      requiredConfigurationFragments;

  /**
   * A dictionary that maps configurations (NONE for target configuration, HOST for host
   * configuration) to lists of Starlark module names of required configuration fragments.
   */
  private final ImmutableSetMultimap<ConfigurationTransition, String>
      starlarkRequiredConfigurationFragments;

  /** What to do during analysis if a configuration fragment is missing. */
  private final ImmutableMap<Class<?>, MissingFragmentPolicy> missingFragmentPolicy;

  @AutoCodec.VisibleForSerialization
  ConfigurationFragmentPolicy(
      ImmutableSetMultimap<ConfigurationTransition, Class<?>> requiredConfigurationFragments,
      ImmutableSetMultimap<ConfigurationTransition, String> starlarkRequiredConfigurationFragments,
      ImmutableMap<Class<?>, MissingFragmentPolicy> missingFragmentPolicy) {
    this.requiredConfigurationFragments = requiredConfigurationFragments;
    this.starlarkRequiredConfigurationFragments = starlarkRequiredConfigurationFragments;
    this.missingFragmentPolicy = missingFragmentPolicy;
  }

  /**
   * The set of required configuration fragments; this contains all fragments that can be
   * accessed by the rule implementation under any configuration.
   */
  public Set<Class<?>> getRequiredConfigurationFragments() {
    return ImmutableSet.copyOf(requiredConfigurationFragments.values());
  }

  /**
   * Returns the fragments required by Starlark definitions (e.g. <code>fragments = ["cpp"]</code>
   * with the naming form seen in the Starlark API.
   *
   * <p>{@link
   * com.google.devtools.build.lib.analysis.config.BuildConfiguration#getStarlarkFragmentByName} can
   * be used to convert this to Java fragment instances.
   */
  public Collection<String> getRequiredStarlarkFragments() {
    return starlarkRequiredConfigurationFragments.values();
  }
  /**
   * Checks if the configuration fragment may be accessed (i.e., if it's declared) in the specified
   * configuration (target or host).
   *
   * <p>Note that, currently, all native fragments are included regardless of whether they were
   * specified in the same configuration that was passed.
   */
  public boolean isLegalConfigurationFragment(
      Class<?> configurationFragment, ConfigurationTransition config) {
    return requiredConfigurationFragments.containsValue(configurationFragment)
        || hasLegalFragmentName(configurationFragment, config);
  }

  /**
   * Checks if the configuration fragment may be accessed (i.e., if it's declared) in any
   * configuration.
   */
  public boolean isLegalConfigurationFragment(Class<?> configurationFragment) {
    return requiredConfigurationFragments.containsValue(configurationFragment)
        || hasLegalFragmentName(configurationFragment);
  }

  /**
   * Checks whether the name of the given fragment class was declared as required in the
   * specified configuration (target or host).
   */
  private boolean hasLegalFragmentName(
      Class<?> configurationFragment, ConfigurationTransition transition) {
    StarlarkBuiltin fragmentModule =
        StarlarkInterfaceUtils.getStarlarkBuiltin(configurationFragment);

    return fragmentModule != null
        && starlarkRequiredConfigurationFragments.containsEntry(transition, fragmentModule.name());
  }

  /**
   * Checks whether the name of the given fragment class was declared as required in any
   * configuration.
   */
  private boolean hasLegalFragmentName(Class<?> configurationFragment) {
    StarlarkBuiltin fragmentModule =
        StarlarkInterfaceUtils.getStarlarkBuiltin(configurationFragment);

    return fragmentModule != null
        && starlarkRequiredConfigurationFragments.containsValue(fragmentModule.name());
  }

  /**
   * Whether to fail analysis if any of the specified configuration fragment class is missing.
   *
   * <p>If unset for the specific fragment class, defaults to FAIL_ANALYSIS
   */
  public MissingFragmentPolicy getMissingFragmentPolicy(Class<?> fragmentClass) {
    return missingFragmentPolicy.getOrDefault(fragmentClass, MissingFragmentPolicy.FAIL_ANALYSIS);
  }
}
