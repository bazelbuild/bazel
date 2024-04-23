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


import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimaps;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetNotFoundException;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryVisibility;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.build.skyframe.state.EnvironmentForUtilities;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * A {@link TargetAccessor} for {@link ConfiguredTarget} objects.
 *
 * <p>Incomplete; we'll implement getVisibility when needed.
 */
public class ConfiguredTargetAccessor implements TargetAccessor<CqueryNode> {

  private final WalkableGraph walkableGraph;
  private final ConfiguredTargetQueryEnvironment queryEnvironment;
  private final SkyFunction.LookupEnvironment lookupEnvironment;

  public ConfiguredTargetAccessor(
      WalkableGraph walkableGraph, ConfiguredTargetQueryEnvironment queryEnvironment) {
    this.walkableGraph = walkableGraph;
    this.queryEnvironment = queryEnvironment;
    this.lookupEnvironment =
        new EnvironmentForUtilities(
            key -> {
              try {
                SkyValue value = walkableGraph.getValue(key);
                if (value != null) {
                  return value;
                }
                return walkableGraph.getException(key);
              } catch (InterruptedException e) {
                throw new IllegalStateException(
                    "Thread interrupted in the middle of looking up: " + key, e);
              }
            });
  }

  @Override
  public String getTargetKind(CqueryNode target) {
    Target actualTarget = getTarget(target);
    return actualTarget.getTargetKind();
  }

  @Override
  public String getLabel(CqueryNode target) {
    return target.getOriginalLabel().toString();
  }

  @Override
  public String getPackage(CqueryNode target) {
    return target.getOriginalLabel().getPackageIdentifier().getPackageFragment().toString();
  }

  @Override
  public boolean isRule(CqueryNode target) {
    Target actualTarget = getTarget(target);
    return actualTarget instanceof Rule;
  }

  @Override
  public boolean isTestRule(CqueryNode target) {
    Target actualTarget = getTarget(target);
    return TargetUtils.isTestRule(actualTarget);
  }

  @Override
  public boolean isTestSuite(CqueryNode target) {
    Target actualTarget = getTarget(target);
    return TargetUtils.isTestSuiteRule(actualTarget);
  }

  /**
   * Returns all of {@code keyedConfiguredTarget}'s prerequisites.
   *
   * <p>Does not resolve aliases. So for aliases, this returns their {@code actual} attribute deps
   * (plus any implicit deps).
   *
   * <p>Use sparingly: this doesn't distinguish where those prerequisites come from. For example if
   * {@code keyedConfiguredTarget} depends on aspect A which depends on {@code //foo}, whether
   * {@code //foo} is returned here depends on the values of {@link
   * QueryEnvironment.Setting#INCLUDE_ASPECTS} or {@link QueryEnvironment.Setting#EXPLICIT_ASPECTS}
   *
   * <p>So this method returns the canonical direct dependencies as determined by cquery. But it
   * doesn't expose the logic cquery uses to determine that, nor the command-line flags that toggle
   * cquery's choices.
   */
  Set<CqueryNode> getPrerequisites(CqueryNode keyedConfiguredTarget) throws InterruptedException {
    return queryEnvironment.getFwdDeps(ImmutableList.of(keyedConfiguredTarget));
  }

  @Override
  public List<CqueryNode> getPrerequisites(
      QueryExpression caller,
      CqueryNode keyedConfiguredTarget,
      String attrName,
      String errorMsgPrefix)
      throws QueryException, InterruptedException {
    // Process aliases.
    CqueryNode actual = keyedConfiguredTarget.getActual();

    Preconditions.checkArgument(
        isRule(actual), "%s %s is not a rule configured target", errorMsgPrefix, getLabel(actual));

    ImmutableListMultimap<Label, CqueryNode> depsByLabel =
        Multimaps.index(
            queryEnvironment.getFwdDeps(ImmutableList.of(actual)), CqueryNode::getOriginalLabel);

    Rule rule = (Rule) getTarget(actual);
    ImmutableMap<Label, ConfigMatchingProvider> configConditions = actual.getConfigConditions();
    ConfiguredAttributeMapper attributeMapper =
        ConfiguredAttributeMapper.of(
            rule,
            configConditions,
            keyedConfiguredTarget.getConfigurationChecksum(),
            /*alwaysSucceed=*/ false);
    if (!attributeMapper.has(attrName)) {
      throw new QueryException(
          caller,
          String.format(
              "%sconfigured target of type %s does not have attribute '%s'",
              errorMsgPrefix, rule.getRuleClass(), attrName),
          ConfigurableQuery.Code.ATTRIBUTE_MISSING);
    }
    ImmutableList.Builder<CqueryNode> toReturn = ImmutableList.builder();
    attributeMapper.visitLabels(attrName, label -> toReturn.addAll(depsByLabel.get(label)));
    return toReturn.build();
  }

  @Override
  public List<String> getStringListAttr(CqueryNode target, String attrName) {
    ConfiguredAttributeMapper attributeMapper = getAttributes(target);
    return attributeMapper.get(attrName, Types.STRING_LIST);
  }

  @Override
  public String getStringAttr(CqueryNode target, String attrName) {
    ConfiguredAttributeMapper attributeMapper = getAttributes(target);
    return attributeMapper.get(attrName, Type.STRING);
  }

  @Override
  public Iterable<String> getAttrAsString(CqueryNode target, String attrName) {
    ConfiguredAttributeMapper attributeMapper = getAttributes(target);
    Attribute attribute = attributeMapper.getAttributeDefinition(attrName);
    if (attribute == null) {
      // Ignore unknown attributes.
      return ImmutableList.of();
    }
    Type<?> attributeType = attribute.getType();

    Object value = attributeMapper.get(attrName, attributeType);
    if (value == null) {
      return ImmutableList.of();
    }

    if (Objects.equals(attrName, "visibility")
        && attributeType.equals(BuildType.NODEP_LABEL_LIST)) {
      // This special case for the visibility attribute is needed because its value is replaced
      // with an empty list during package loading if it is public or private in order not to visit
      // the package called 'visibility'.
      Target actualTarget = getTarget(target);
      Preconditions.checkArgument(actualTarget instanceof Rule);
      Rule rule = (Rule) actualTarget;
      value = attributeType.cast(rule.getVisibilityDeclaredLabels());
    }

    // Return a single-valued list, because a configured target only has one value for the
    // attribute. Flatten to a string regardless of the actual type so that regex-based matches can
    // be performed.
    return ImmutableList.of(TargetUtils.convertAttributeValue(attributeType, value));
  }

  private ConfiguredAttributeMapper getAttributes(CqueryNode target) {
    Target actualTarget = getTarget(target);
    Preconditions.checkArgument(actualTarget instanceof Rule);
    Rule rule = (Rule) actualTarget;
    ImmutableMap<Label, ConfigMatchingProvider> configConditions = target.getConfigConditions();
    return ConfiguredAttributeMapper.of(
        rule, configConditions, target.getConfigurationChecksum(), /* alwaysSucceed= */ false);
  }

  @Override
  public ImmutableSet<QueryVisibility<CqueryNode>> getVisibility(
      QueryExpression caller, CqueryNode from) throws QueryException {
    // TODO(bazel-team): implement this if needed.
    throw new QueryException(
        "visible() is not supported on configured targets",
        ConfigurableQuery.Code.VISIBLE_FUNCTION_NOT_SUPPORTED);
  }

  public Target getTarget(CqueryNode configuredTarget) {
    // Dereference any aliases that might be present.
    Label label = configuredTarget.getOriginalLabel();
    try {
      return queryEnvironment.getTarget(label);
    } catch (InterruptedException e) {
      throw new IllegalStateException("Thread interrupted in the middle of getting a Target.", e);
    } catch (TargetNotFoundException e) {
      throw new IllegalStateException("Unable to get target from package in accessor.", e);
    }
  }

  SkyFunction.LookupEnvironment getLookupEnvironment() {
    return lookupEnvironment;
  }

  /** Returns the rule that generates the given output file. */
  RuleConfiguredTarget getGeneratingConfiguredTarget(CqueryNode kct) throws InterruptedException {
    Preconditions.checkArgument(kct instanceof OutputFileConfiguredTarget);
    return (RuleConfiguredTarget)
        ((ConfiguredTargetValue)
                walkableGraph.getValue(
                    ConfiguredTargetKey.builder()
                        .setLabel(((OutputFileConfiguredTarget) kct).getGeneratingRule().getLabel())
                        .setConfigurationKey(kct.getConfigurationKey())
                        .build()))
            .getConfiguredTarget();
  }
}
