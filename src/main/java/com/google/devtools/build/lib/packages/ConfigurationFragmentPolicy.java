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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.Transition;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import java.util.Collection;
import java.util.Map;
import java.util.Set;

/**
 * Policy used to express the set of configuration fragments which are legal for a rule or aspect
 * to access.
 */
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
    private final SetMultimap<Transition, Class<?>> requiredConfigurationFragments
        = LinkedHashMultimap.create();
    /**
     * Sets of configuration fragment names required by this rule, a set for each configuration.
     * Duplicate entries will automatically be ignored by the SetMultimap.
     */
    private final SetMultimap<Transition, String> requiredConfigurationFragmentNames
        = LinkedHashMultimap.create();
    private MissingFragmentPolicy missingFragmentPolicy = MissingFragmentPolicy.FAIL_ANALYSIS;

    /**
     * Declares that the implementation of the associated rule class requires the given
     * fragments to be present in this rule's target configuration only.
     *
     * <p>The value is inherited by subclasses.
     */
    public Builder requiresConfigurationFragments(Collection<Class<?>> configurationFragments) {
      requiresConfigurationFragments(ConfigurationTransition.NONE, configurationFragments);
      return this;
    }

    /**
     * Declares that the implementation of the associated rule class requires the given
     * fragments to be present in the specified configuration. Valid transition values are
     * HOST for the host configuration and NONE for the target configuration.
     *
     * <p>The value is inherited by subclasses.
     */
    public Builder requiresConfigurationFragments(Attribute.Transition transition,
        Collection<Class<?>> configurationFragments) {
      // We can relax this assumption if needed. But it's already sketchy to let a rule see more
      // than its own configuration. So we don't want to casually proliferate this pattern.
      Preconditions.checkArgument(
          transition == Attribute.ConfigurationTransition.NONE || transition.isHostTransition());
      requiredConfigurationFragments.putAll(transition, configurationFragments);
      return this;
    }

    /**
     * Declares that the implementation of the associated rule class requires the given
     * fragments to be present in this rule's target configuration only.
     *
     * <p>In contrast to {@link #requiresConfigurationFragments(Collection)}, this method takes the
     * names of fragments (as determined by {@link SkylarkModule.Resolver}) instead of their
     * classes.
     *
     * <p>The value is inherited by subclasses.
     */
    public Builder requiresConfigurationFragmentsBySkylarkModuleName(
        Collection<String> configurationFragmentNames) {

      requiresConfigurationFragmentsBySkylarkModuleName(
          Attribute.ConfigurationTransition.NONE, configurationFragmentNames);
      return this;
    }

    /**
     * Declares the configuration fragments that are required by this rule for the specified
     * configuration. Valid transition values are HOST for the host configuration and NONE for
     * the target configuration.
     *
     * <p>In contrast to {@link #requiresConfigurationFragments(Transition,
     * Collection)}, this method takes the names of fragments (as determined by
     * {@link SkylarkModule.Resolver}) instead of their
     * classes.
     */
    public Builder requiresConfigurationFragmentsBySkylarkModuleName(Transition transition,
        Collection<String> configurationFragmentNames) {
      // We can relax this assumption if needed. But it's already sketchy to let a rule see more
      // than its own configuration. So we don't want to casually proliferate this pattern.
      Preconditions.checkArgument(
          transition == ConfigurationTransition.NONE || transition.isHostTransition());
      requiredConfigurationFragmentNames.putAll(transition, configurationFragmentNames);
      return this;
    }

    /**
     * Adds the configuration fragments from the {@code other} policy to this Builder.
     *
     * <p>Does not change the missing fragment policy.
     */
    public Builder includeConfigurationFragmentsFrom(ConfigurationFragmentPolicy other) {
      requiredConfigurationFragments.putAll(other.requiredConfigurationFragments);
      requiredConfigurationFragmentNames.putAll(other.requiredConfigurationFragmentNames);
      return this;
    }

    /**
     * Sets the policy for the case where the configuration is missing required fragments (see
     * {@link #requiresConfigurationFragments}).
     */
    public Builder setMissingFragmentPolicy(MissingFragmentPolicy missingFragmentPolicy) {
      this.missingFragmentPolicy = missingFragmentPolicy;
      return this;
    }

    public ConfigurationFragmentPolicy build() {
      return new ConfigurationFragmentPolicy(
          ImmutableSetMultimap.copyOf(requiredConfigurationFragments),
          ImmutableSetMultimap.copyOf(requiredConfigurationFragmentNames),
          missingFragmentPolicy);
    }
  }

  /**
   * A dictionary that maps configurations (NONE for target configuration, HOST for host
   * configuration) to required configuration fragments.
   */
  private final ImmutableSetMultimap<Transition, Class<?>>
      requiredConfigurationFragments;

  /**
   * A dictionary that maps configurations (NONE for target configuration, HOST for host
   * configuration) to lists of Skylark module names of required configuration fragments.
   */
  private final ImmutableSetMultimap<Transition, String>
      requiredConfigurationFragmentNames;

  /**
   * What to do during analysis if a configuration fragment is missing.
   */
  private final MissingFragmentPolicy missingFragmentPolicy;

  private ConfigurationFragmentPolicy(
      ImmutableSetMultimap<Transition, Class<?>> requiredConfigurationFragments,
      ImmutableSetMultimap<Transition, String> requiredConfigurationFragmentNames,
      MissingFragmentPolicy missingFragmentPolicy) {
    this.requiredConfigurationFragments = requiredConfigurationFragments;
    this.requiredConfigurationFragmentNames = requiredConfigurationFragmentNames;
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
   * Checks if the configuration fragment may be accessed (i.e., if it's declared) in the specified
   * configuration (target or host).
   *
   * <p>Note that, currently, all native fragments are included regardless of whether they were
   * specified in the same configuration that was passed.
   */
  public boolean isLegalConfigurationFragment(
      Class<?> configurationFragment, Transition config) {
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
      Class<?> configurationFragment, Transition transition) {
    String name = SkylarkModule.Resolver.resolveName(configurationFragment);
    if (requiredConfigurationFragmentNames.containsEntry(transition, name)) {
      return true;
    } else if (transition == ConfigurationTransition.HOST) {
      // We can get rid of this check when we remove ConfigurationTransition.HOST. This essentially
      // makes ConfigurationTransition.HOST and HostTransition.INSTANCE interchangeable.
      for (Map.Entry<Transition, String> entry : requiredConfigurationFragmentNames.entries()) {
        if (entry.getKey().isHostTransition() && entry.getValue().equals(name)) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Checks whether the name of the given fragment class was declared as required in any
   * configuration.
   */
  private boolean hasLegalFragmentName(Class<?> configurationFragment) {
    return requiredConfigurationFragmentNames.containsValue(
        SkylarkModule.Resolver.resolveName(configurationFragment));
  }

  /**
   * Whether to fail analysis if any of the required configuration fragments are missing.
   */
  public MissingFragmentPolicy getMissingFragmentPolicy() {
    return missingFragmentPolicy;
  }
}
