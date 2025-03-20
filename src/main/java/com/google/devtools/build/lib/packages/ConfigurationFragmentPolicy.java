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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentClassSet;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import net.starlark.java.annot.StarlarkAnnotations;
import net.starlark.java.annot.StarlarkBuiltin;

/**
 * Policy used to express the set of configuration fragments which are legal for a rule or aspect to
 * access.
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
    FAIL_ANALYSIS
  }

  /**
   * Builder to construct a new ConfigurationFragmentPolicy.
   */
  public static final class Builder {
    /** Configuration fragment classes required by this rule. */
    private final Set<Class<? extends Fragment>> requiredConfigurationFragments = new HashSet<>();

    /**
     * Sets of configuration fragments required by this rule, as defined by their Starlark names
     * (see {@link StarlarkBuiltin}.
     *
     * <p>Duplicate entries will automatically be ignored by the SetMultimap.
     */
    private final Set<String> starlarkRequiredConfigurationFragments = new LinkedHashSet<>();

    private final Map<Class<?>, MissingFragmentPolicy> missingFragmentPolicy =
        new LinkedHashMap<>();

    /**
     * Declares that the implementation of the associated rule class requires the given fragments to
     * be present.
     *
     * <p>The value is inherited by subclasses.
     */
    @CanIgnoreReturnValue
    public Builder requiresConfigurationFragments(
        Collection<Class<? extends Fragment>> configurationFragments) {
      requiredConfigurationFragments.addAll(configurationFragments);
      return this;
    }

    /**
     * Declares that the implementation of the associated rule class requires the given fragments to
     * be present for this rule.
     *
     * <p>In contrast to {@link #requiresConfigurationFragments(Collection)}, this method takes the
     * names of fragments (as determined by {@link StarlarkBuiltin}) instead of their classes.
     *
     * <p>The value is inherited by subclasses.
     */
    @CanIgnoreReturnValue
    public Builder requiresConfigurationFragmentsByStarlarkBuiltinName(
        Collection<String> configurationFragmentNames) {
      starlarkRequiredConfigurationFragments.addAll(configurationFragmentNames);
      return this;
    }

    /**
     * Adds the configuration fragments from the {@code other} policy to this Builder.
     *
     * <p>Missing fragment policy is also copied over, overriding previously set values.
     */
    @CanIgnoreReturnValue
    public Builder includeConfigurationFragmentsFrom(ConfigurationFragmentPolicy other) {
      requiredConfigurationFragments.addAll(other.requiredConfigurationFragments);
      starlarkRequiredConfigurationFragments.addAll(other.starlarkRequiredConfigurationFragments);
      missingFragmentPolicy.putAll(other.missingFragmentPolicy);
      return this;
    }

    /**
     * Sets the policy for the case where the configuration is missing specified required fragment
     * class (see {@link #requiresConfigurationFragments}).
     */
    @CanIgnoreReturnValue
    public Builder setMissingFragmentPolicy(
        Class<?> fragmentClass, MissingFragmentPolicy missingFragmentPolicy) {
      this.missingFragmentPolicy.put(fragmentClass, missingFragmentPolicy);
      return this;
    }

    public ConfigurationFragmentPolicy build() {
      return new ConfigurationFragmentPolicy(
          FragmentClassSet.of(requiredConfigurationFragments),
          ImmutableSet.copyOf(starlarkRequiredConfigurationFragments),
          ImmutableMap.copyOf(missingFragmentPolicy));
    }
  }

  private final FragmentClassSet requiredConfigurationFragments;

  /** A set of Starlark module names of required configuration fragments. */
  private final ImmutableSet<String> starlarkRequiredConfigurationFragments;

  /** What to do during analysis if a configuration fragment is missing. */
  private final ImmutableMap<Class<?>, MissingFragmentPolicy> missingFragmentPolicy;

  private ConfigurationFragmentPolicy(
      FragmentClassSet requiredConfigurationFragments,
      ImmutableSet<String> starlarkRequiredConfigurationFragments,
      ImmutableMap<Class<?>, MissingFragmentPolicy> missingFragmentPolicy) {
    this.requiredConfigurationFragments = requiredConfigurationFragments;
    this.starlarkRequiredConfigurationFragments = starlarkRequiredConfigurationFragments;
    this.missingFragmentPolicy = missingFragmentPolicy;
  }

  /**
   * The set of required configuration fragments; this contains all fragments that can be accessed
   * by the rule implementation under any configuration.
   */
  public FragmentClassSet getRequiredConfigurationFragments() {
    return requiredConfigurationFragments;
  }

  /**
   * Returns the fragments required by Starlark definitions (e.g. <code>fragments = ["cpp"]</code>
   * with the naming form seen in the Starlark API.
   *
   * <p>{@link
   * com.google.devtools.build.lib.analysis.config.BuildConfigurationValue#getStarlarkFragmentByName}
   * can be used to convert this to Java fragment instances.
   */
  public ImmutableCollection<String> getRequiredStarlarkFragments() {
    return starlarkRequiredConfigurationFragments;
  }

  /**
   * Checks if the configuration fragment may be accessed (i.e., if it's declared) in any
   * configuration.
   */
  public boolean isLegalConfigurationFragment(Class<?> configurationFragment) {
    return requiredConfigurationFragments.contains(configurationFragment)
        || hasLegalFragmentName(configurationFragment);
  }

  /**
   * Checks whether the name of the given fragment class was declared as required in any
   * configuration.
   */
  private boolean hasLegalFragmentName(Class<?> configurationFragment) {
    StarlarkBuiltin fragmentModule = StarlarkAnnotations.getStarlarkBuiltin(configurationFragment);

    return fragmentModule != null
        && starlarkRequiredConfigurationFragments.contains(fragmentModule.name());
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
