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
package com.google.devtools.build.lib.query2.cquery;

import static com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.getExecutionPlatformConstraints;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multimaps;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.ExecGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetNotFoundException;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryVisibility;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ToolchainContextKey;
import com.google.devtools.build.lib.skyframe.UnloadedToolchainContext;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A {@link TargetAccessor} for {@link ConfiguredTarget} objects.
 *
 * <p>Incomplete; we'll implement getVisibility when needed.
 */
public class ConfiguredTargetAccessor implements TargetAccessor<KeyedConfiguredTarget> {

  private final WalkableGraph walkableGraph;
  private final ConfiguredTargetQueryEnvironment queryEnvironment;

  public ConfiguredTargetAccessor(
      WalkableGraph walkableGraph, ConfiguredTargetQueryEnvironment queryEnvironment) {
    this.walkableGraph = walkableGraph;
    this.queryEnvironment = queryEnvironment;
  }

  @Override
  public String getTargetKind(KeyedConfiguredTarget target) {
    Target actualTarget = getTarget(target);
    return actualTarget.getTargetKind();
  }

  @Override
  public String getLabel(KeyedConfiguredTarget target) {
    return target.getLabel().toString();
  }

  @Override
  public String getPackage(KeyedConfiguredTarget target) {
    return target.getLabel().getPackageIdentifier().getPackageFragment().toString();
  }

  @Override
  public boolean isRule(KeyedConfiguredTarget target) {
    Target actualTarget = getTarget(target);
    return actualTarget instanceof Rule;
  }

  @Override
  public boolean isTestRule(KeyedConfiguredTarget target) {
    Target actualTarget = getTarget(target);
    return TargetUtils.isTestRule(actualTarget);
  }

  @Override
  public boolean isTestSuite(KeyedConfiguredTarget target) {
    Target actualTarget = getTarget(target);
    return TargetUtils.isTestSuiteRule(actualTarget);
  }

  @Override
  public List<KeyedConfiguredTarget> getPrerequisites(
      QueryExpression caller,
      KeyedConfiguredTarget keyedConfiguredTarget,
      String attrName,
      String errorMsgPrefix)
      throws QueryException, InterruptedException {
    // Process aliases.
    KeyedConfiguredTarget actual = keyedConfiguredTarget.getActual();

    Preconditions.checkArgument(
        isRule(actual), "%s %s is not a rule configured target", errorMsgPrefix, getLabel(actual));

    Multimap<Label, KeyedConfiguredTarget> depsByLabel =
        Multimaps.index(
            queryEnvironment.getFwdDeps(ImmutableList.of(actual)), kct -> kct.getLabel());

    Rule rule = (Rule) getTarget(actual);
    ImmutableMap<Label, ConfigMatchingProvider> configConditions = actual.getConfigConditions();
    ConfiguredAttributeMapper attributeMapper =
        ConfiguredAttributeMapper.of(
            rule, configConditions, keyedConfiguredTarget.getConfigurationChecksum());
    if (!attributeMapper.has(attrName)) {
      throw new QueryException(
          caller,
          String.format(
              "%sconfigured target of type %s does not have attribute '%s'",
              errorMsgPrefix, rule.getRuleClass(), attrName),
          ConfigurableQuery.Code.ATTRIBUTE_MISSING);
    }
    ImmutableList.Builder<KeyedConfiguredTarget> toReturn = ImmutableList.builder();
    attributeMapper.visitLabels(attributeMapper.getAttributeDefinition(attrName)).stream()
        .forEach(depEdge -> toReturn.addAll(depsByLabel.get(depEdge.getLabel())));
    return toReturn.build();
  }

  @Override
  public List<String> getStringListAttr(KeyedConfiguredTarget target, String attrName) {
    Target actualTarget = getTarget(target);
    return TargetUtils.getStringListAttr(actualTarget, attrName);
  }

  @Override
  public String getStringAttr(KeyedConfiguredTarget target, String attrName) {
    Target actualTarget = getTarget(target);
    return TargetUtils.getStringAttr(actualTarget, attrName);
  }

  @Override
  public Iterable<String> getAttrAsString(KeyedConfiguredTarget target, String attrName) {
    Target actualTarget = getTarget(target);
    return TargetUtils.getAttrAsString(actualTarget, attrName);
  }

  @Override
  public ImmutableSet<QueryVisibility<KeyedConfiguredTarget>> getVisibility(
      QueryExpression caller, KeyedConfiguredTarget from) throws QueryException {
    // TODO(bazel-team): implement this if needed.
    throw new QueryException(
        "visible() is not supported on configured targets",
        ConfigurableQuery.Code.VISIBLE_FUNCTION_NOT_SUPPORTED);
  }

  public Target getTarget(KeyedConfiguredTarget keyedConfiguredTarget) {
    // Dereference any aliases that might be present.
    Label label = keyedConfiguredTarget.getConfiguredTarget().getOriginalLabel();
    try {
      return queryEnvironment.getTarget(label);
    } catch (InterruptedException e) {
      throw new IllegalStateException("Thread interrupted in the middle of getting a Target.", e);
    } catch (TargetNotFoundException e) {
      throw new IllegalStateException("Unable to get target from package in accessor.", e);
    }
  }

  /** Returns the rule that generates the given output file. */
  RuleConfiguredTarget getGeneratingConfiguredTarget(KeyedConfiguredTarget kct)
      throws InterruptedException {
    Preconditions.checkArgument(kct.getConfiguredTarget() instanceof OutputFileConfiguredTarget);
    return (RuleConfiguredTarget)
        ((ConfiguredTargetValue)
                walkableGraph.getValue(
                    ConfiguredTargetKey.builder()
                        .setLabel(
                            ((OutputFileConfiguredTarget) kct.getConfiguredTarget())
                                .getGeneratingRule()
                                .getLabel())
                        .setConfiguration(queryEnvironment.getConfiguration(kct))
                        .build()))
            .getConfiguredTarget();
  }

  @Nullable
  ToolchainCollection<ToolchainContext> getToolchainContexts(
      Target target, BuildConfiguration config) {
    return getToolchainContexts(target, config, walkableGraph);
  }

  @Nullable
  private static ToolchainCollection<ToolchainContext> getToolchainContexts(
      Target target, BuildConfiguration config, WalkableGraph walkableGraph) {
    if (!(target instanceof Rule)) {
      return null;
    }

    Rule rule = ((Rule) target);
    if (!rule.getRuleClassObject().useToolchainResolution()) {
      return null;
    }

    ImmutableSet<Label> requiredToolchains = rule.getRuleClassObject().getRequiredToolchains();
    // Collect local (target, rule) constraints for filtering out execution platforms.
    ImmutableSet<Label> execConstraintLabels =
        getExecutionPlatformConstraints(rule, config.getFragment(PlatformConfiguration.class));
    ImmutableMap<String, ExecGroup> execGroups = rule.getRuleClassObject().getExecGroups();

    ToolchainCollection.Builder<UnloadedToolchainContext> toolchainContexts =
        ToolchainCollection.builder();
    BuildConfigurationValue.Key configurationKey = BuildConfigurationValue.key(config);
    try {
      for (Map.Entry<String, ExecGroup> group : execGroups.entrySet()) {
        ExecGroup execGroup = group.getValue();
        UnloadedToolchainContext context =
            (UnloadedToolchainContext)
                walkableGraph.getValue(
                    ToolchainContextKey.key()
                        .configurationKey(configurationKey)
                        .requiredToolchainTypeLabels(execGroup.requiredToolchains())
                        .execConstraintLabels(execGroup.execCompatibleWith())
                        .build());
        if (context == null) {
          return null;
        }
        toolchainContexts.addContext(group.getKey(), context);
      }
      UnloadedToolchainContext defaultContext =
          (UnloadedToolchainContext)
              walkableGraph.getValue(
                  ToolchainContextKey.key()
                      .configurationKey(configurationKey)
                      .requiredToolchainTypeLabels(requiredToolchains)
                      .execConstraintLabels(execConstraintLabels)
                      .build());
      if (defaultContext == null) {
        return null;
      }
      toolchainContexts.addDefaultContext(defaultContext);
      return toolchainContexts.build().asToolchainContexts();
    } catch (InterruptedException e) {
      throw new IllegalStateException(
          "Thread interrupted in the middle of getting a ToolchainContext.", e);
    }
  }
}
