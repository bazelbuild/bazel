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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.Dependency;
import com.google.devtools.build.lib.analysis.LabelAndConfiguration;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ConflictException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Build a post-processed ConfiguredTarget, vetting it for action conflict issues.
 */
public class PostConfiguredTargetFunction implements SkyFunction {
  private static final Function<Dependency, SkyKey> TO_KEYS =
      new Function<Dependency, SkyKey>() {
    @Override
    public SkyKey apply(Dependency input) {
      return PostConfiguredTargetValue.key(
          new ConfiguredTargetKey(input.getLabel(), input.getConfiguration()));
    }
  };

  private final SkyframeExecutor.BuildViewProvider buildViewProvider;
  private final RuleClassProvider ruleClassProvider;

  public PostConfiguredTargetFunction(
      SkyframeExecutor.BuildViewProvider buildViewProvider,
      RuleClassProvider ruleClassProvider) {
    this.buildViewProvider = Preconditions.checkNotNull(buildViewProvider);
    this.ruleClassProvider = ruleClassProvider;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    ImmutableMap<Action, ConflictException> badActions = PrecomputedValue.BAD_ACTIONS.get(env);
    ConfiguredTargetValue ctValue = (ConfiguredTargetValue)
        env.getValue(ConfiguredTargetValue.key((ConfiguredTargetKey) skyKey.argument()));
    if (env.valuesMissing()) {
      return null;
    }

    for (Action action : ctValue.getActions()) {
      if (badActions.containsKey(action)) {
        throw new ActionConflictFunctionException(badActions.get(action));
      }
    }

    ConfiguredTarget ct = ctValue.getConfiguredTarget();
    TargetAndConfiguration ctgValue =
        new TargetAndConfiguration(ct.getTarget(), ct.getConfiguration());

    Set<ConfigMatchingProvider> configConditions =
        getConfigurableAttributeConditions(ctgValue, env);
    if (configConditions == null) {
      return null;
    }

    ListMultimap<Attribute, Dependency> deps;
    try {
      BuildConfiguration hostConfiguration =
          buildViewProvider.getSkyframeBuildView().getHostConfiguration(ct.getConfiguration());
      SkyframeDependencyResolver resolver =
          buildViewProvider.getSkyframeBuildView().createDependencyResolver(env);
      deps =
          resolver.dependentNodeMap(
              ctgValue, hostConfiguration, /*aspect=*/ null, configConditions);
      if (ct.getConfiguration() != null && ct.getConfiguration().useDynamicConfigurations()) {
        deps = ConfiguredTargetFunction.trimConfigurations(env, ctgValue, deps, hostConfiguration,
            ruleClassProvider);
      }
    } catch (EvalException e) {
      throw new PostConfiguredTargetFunctionException(e);
    } catch (ConfiguredTargetFunction.DependencyEvaluationException e) {
      throw new PostConfiguredTargetFunctionException(e);
    }

    env.getValues(Iterables.transform(deps.values(), TO_KEYS));
    if (env.valuesMissing()) {
      return null;
    }

    return new PostConfiguredTargetValue(ct);
  }

  /**
   * Returns the configurable attribute conditions necessary to evaluate the given configured
   * target, or null if not all dependencies have yet been SkyFrame-evaluated.
   */
  @Nullable
  private Set<ConfigMatchingProvider> getConfigurableAttributeConditions(
      TargetAndConfiguration ctg, Environment env) {
    if (!(ctg.getTarget() instanceof Rule)) {
      return ImmutableSet.of();
    }
    Rule rule = (Rule) ctg.getTarget();
    RawAttributeMapper mapper = RawAttributeMapper.of(rule);
    Set<SkyKey> depKeys = new LinkedHashSet<>();
    for (Attribute attribute : rule.getAttributes()) {
      for (Label label : mapper.getConfigurabilityKeys(attribute.getName(), attribute.getType())) {
        if (!BuildType.Selector.isReservedLabel(label)) {
          depKeys.add(ConfiguredTargetValue.key(label, ctg.getConfiguration()));
        }
      }
    }
    Map<SkyKey, SkyValue> cts = env.getValues(depKeys);
    if (env.valuesMissing()) {
      return null;
    }
    ImmutableSet.Builder<ConfigMatchingProvider> conditions = ImmutableSet.builder();
    for (SkyValue ctValue : cts.values()) {
      ConfiguredTarget ct = ((ConfiguredTargetValue) ctValue).getConfiguredTarget();
      conditions.add(Preconditions.checkNotNull(ct.getProvider(ConfigMatchingProvider.class)));
    }
    return conditions.build();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((LabelAndConfiguration) skyKey.argument()).getLabel());
  }

  private static class ActionConflictFunctionException extends SkyFunctionException {
    public ActionConflictFunctionException(ConflictException e) {
      super(e, Transience.PERSISTENT);
    }
  }

  private static class PostConfiguredTargetFunctionException extends SkyFunctionException {
    public PostConfiguredTargetFunctionException(Exception e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
