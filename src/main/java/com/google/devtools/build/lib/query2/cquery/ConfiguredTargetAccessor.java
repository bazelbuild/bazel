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
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryVisibility;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.UnloadedToolchainContext;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A {@link TargetAccessor} for {@link ConfiguredTarget} objects.
 *
 * <p>Incomplete; we'll implement getVisibility when needed.
 */
public class ConfiguredTargetAccessor implements TargetAccessor<ConfiguredTarget> {

  private final WalkableGraph walkableGraph;
  private final ConfiguredTargetQueryEnvironment queryEnvironment;

  public ConfiguredTargetAccessor(
      WalkableGraph walkableGraph, ConfiguredTargetQueryEnvironment queryEnvironment) {
    this.walkableGraph = walkableGraph;
    this.queryEnvironment = queryEnvironment;
  }

  @Override
  public String getTargetKind(ConfiguredTarget target) {
    Target actualTarget = getTargetFromConfiguredTarget(target);
    return actualTarget.getTargetKind();
  }

  @Override
  public String getLabel(ConfiguredTarget target) {
    return target.getLabel().toString();
  }

  @Override
  public String getPackage(ConfiguredTarget target) {
    return target.getLabel().getPackageIdentifier().getPackageFragment().toString();
  }

  @Override
  public boolean isRule(ConfiguredTarget target) {
    Target actualTarget = getTargetFromConfiguredTarget(target);
    return actualTarget instanceof Rule;
  }

  @Override
  public boolean isTestRule(ConfiguredTarget target) {
    Target actualTarget = getTargetFromConfiguredTarget(target);
    return TargetUtils.isTestRule(actualTarget);
  }

  @Override
  public boolean isTestSuite(ConfiguredTarget target) {
    Target actualTarget = getTargetFromConfiguredTarget(target);
    return TargetUtils.isTestSuiteRule(actualTarget);
  }

  @Override
  public List<ConfiguredTarget> getPrerequisites(
      QueryExpression caller,
      ConfiguredTarget configuredTarget,
      String attrName,
      String errorMsgPrefix)
      throws QueryException, InterruptedException {
    Preconditions.checkArgument(
        isRule(configuredTarget),
        "%s %s is not a rule configured target",
        errorMsgPrefix,
        getLabel(configuredTarget));

    Multimap<Label, ConfiguredTarget> depsByLabel =
        Multimaps.index(
            queryEnvironment.getFwdDeps(ImmutableList.of(configuredTarget)),
            ConfiguredTarget::getLabel);

    Rule rule = (Rule) getTargetFromConfiguredTarget(configuredTarget);
    ImmutableMap<Label, ConfigMatchingProvider> configConditions =
        ((RuleConfiguredTarget) configuredTarget).getConfigConditions();
    ConfiguredAttributeMapper attributeMapper =
        ConfiguredAttributeMapper.of(rule, configConditions);
    if (!attributeMapper.has(attrName)) {
      throw new QueryException(
          caller,
          String.format(
              "%s %s of type %s does not have attribute '%s'",
              errorMsgPrefix, configuredTarget, rule.getRuleClass(), attrName));
    }
    ImmutableList.Builder<ConfiguredTarget> toReturn = ImmutableList.builder();
    attributeMapper.visitLabels(attributeMapper.getAttributeDefinition(attrName)).stream()
        .forEach(depEdge -> toReturn.addAll(depsByLabel.get(depEdge.getLabel())));
    return toReturn.build();
  }

  @Override
  public List<String> getStringListAttr(ConfiguredTarget target, String attrName) {
    Target actualTarget = getTargetFromConfiguredTarget(target);
    return TargetUtils.getStringListAttr(actualTarget, attrName);
  }

  @Override
  public String getStringAttr(ConfiguredTarget target, String attrName) {
    Target actualTarget = getTargetFromConfiguredTarget(target);
    return TargetUtils.getStringAttr(actualTarget, attrName);
  }

  @Override
  public Iterable<String> getAttrAsString(ConfiguredTarget target, String attrName) {
    Target actualTarget = getTargetFromConfiguredTarget(target);
    return TargetUtils.getAttrAsString(actualTarget, attrName);
  }

  @Override
  public Set<QueryVisibility<ConfiguredTarget>> getVisibility(ConfiguredTarget from)
      throws QueryException, InterruptedException {
    // TODO(bazel-team): implement this if needed.
    throw new UnsupportedOperationException();
  }

  public Target getTargetFromConfiguredTarget(ConfiguredTarget configuredTarget) {
    return getTargetFromConfiguredTarget(configuredTarget, walkableGraph);
  }

  public static Target getTargetFromConfiguredTarget(
      ConfiguredTarget configuredTarget, WalkableGraph walkableGraph) {
    Target target = null;
    try {
      // Dereference any aliases that might be present.
      Label label = configuredTarget.getOriginalLabel();
      target =
          ((PackageValue) walkableGraph.getValue(PackageValue.key(label.getPackageIdentifier())))
              .getPackage()
              .getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      throw new IllegalStateException("Unable to get target from package in accessor.", e);
    } catch (InterruptedException e2) {
      throw new IllegalStateException("Thread interrupted in the middle of getting a Target.", e2);
    }
    return target;
  }

  /** Returns the rule that generates the given output file. */
  public RuleConfiguredTarget getGeneratingConfiguredTarget(OutputFileConfiguredTarget oct)
      throws InterruptedException {
    return (RuleConfiguredTarget)
        ((ConfiguredTargetValue)
                walkableGraph.getValue(
                    ConfiguredTargetKey.of(
                        oct.getGeneratingRule().getLabel(),
                        queryEnvironment.getConfiguration(oct))))
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
        new ToolchainCollection.Builder<>();
    try {
      for (Map.Entry<String, ExecGroup> group : execGroups.entrySet()) {
        ExecGroup execGroup = group.getValue();
        UnloadedToolchainContext context =
            (UnloadedToolchainContext)
                walkableGraph.getValue(
                    UnloadedToolchainContext.key()
                        .configurationKey(BuildConfigurationValue.key(config))
                        .requiredToolchainTypeLabels(execGroup.getRequiredToolchains())
                        .execConstraintLabels(execGroup.getExecutionPlatformConstraints())
                        .build());
        if (context == null) {
          return null;
        }
        toolchainContexts.addContext(group.getKey(), context);
      }
      UnloadedToolchainContext defaultContext =
          (UnloadedToolchainContext)
              walkableGraph.getValue(
                  UnloadedToolchainContext.key()
                      .configurationKey(BuildConfigurationValue.key(config))
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
