// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config;

import com.google.devtools.build.lib.packages.RuleClass;
import java.util.Collection;

/**
 * A wrapper for {@link RuleClass.Builder} that supports access to {@link
 * com.google.devtools.build.lib.analysis} classes.
 *
 * <p>{@link RuleClass.Builder} is in {@link com.google.devtools.build.lib.packages}, so it can't
 * reference analysis-time objects.
 */
public class ConfigAwareRuleClassBuilder {
  private final RuleClass.Builder ruleClassBuilder;

  /**
   * Instantiates a builder wrapped around the given {@link RuleClass.Builder}.
   */
  private ConfigAwareRuleClassBuilder(RuleClass.Builder ruleClassBuilder) {
    this.ruleClassBuilder = ruleClassBuilder;
  }

  /**
   * Instantiates a builder wrapped around the given {@link RuleClass.Builder}.
   */
  public static ConfigAwareRuleClassBuilder of(RuleClass.Builder ruleClassBuilder) {
    return new ConfigAwareRuleClassBuilder(ruleClassBuilder);
  }

  /**
   * Returns the {@link RuleClass.Builder} this object wraps.
   *
   * <p>Call this when done with config-specific building to re-expose the builder methods in
   * {@link RuleClass.Builder}
   */
  public RuleClass.Builder originalBuilder() {
    return ruleClassBuilder;
  }

  /**
   * Declares the fragments required by this rule for the configuration of the host machine this
   * rule's actions execute on.
   *
   * <p>This is not the same as the rule's configuration. The rule's configuration is its
   * <i>target</i> configuration, which determines the platform it builds outputs for. The actions
   * created by this rule run on a host machine, which is what the host configuration corresponds
   * to.
   *
   * <p>The value is inherited by subclasses.
   */
  public ConfigAwareRuleClassBuilder requiresHostConfigurationFragments(
      Class<? extends Fragment>... configurationFragments) {
    ruleClassBuilder.requiresConfigurationFragments(HostTransition.INSTANCE,
        configurationFragments);
    return this;
  }

  /**
   * Declares the fragments required by this rule for the configuration of the host machine this
   * rule's actions execute on.
   *
   * <p>This is not the same as the rule's configuration. The rule's configuration is its
   * <i>target</i> configuration, which determines the platform it builds outputs for. The actions
   * created by this rule run on a host machine, which is what the host configuration corresponds
   * to.
   *
   * <p>In contrast to {@link #requiresHostConfigurationFragments(Class...)}, this method takes
   * Starlark module names of fragments instead of their classes.
   */
  public ConfigAwareRuleClassBuilder requiresHostConfigurationFragmentsBySkylarkModuleName(
      Collection<String> configurationFragmentNames) {
    ruleClassBuilder.requiresConfigurationFragmentsBySkylarkModuleName(HostTransition.INSTANCE,
        configurationFragmentNames);
    return this;
  }
}
