// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.aquery;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.query2.engine.KeyExtractor;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetLookup;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetNotFoundException;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryVisibility;
import com.google.devtools.build.lib.server.FailureDetails.ActionQuery;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A {@link TargetAccessor} for {@link ConfiguredTargetValue} objects.
 *
 * <p>Incomplete; we'll implement getPrerequisites and getVisibility when needed.
 */
public class ConfiguredTargetValueAccessor implements TargetAccessor<ConfiguredTargetValue> {

  private final WalkableGraph walkableGraph;
  private final TargetLookup targetLookup;
  private final KeyExtractor<ConfiguredTargetValue, ConfiguredTargetKey>
      configuredTargetKeyExtractor;

  public ConfiguredTargetValueAccessor(
      WalkableGraph walkableGraph,
      TargetLookup targetLookup,
      KeyExtractor<ConfiguredTargetValue, ConfiguredTargetKey> configuredTargetKeyExtractor) {
    this.walkableGraph = walkableGraph;
    this.targetLookup = targetLookup;
    this.configuredTargetKeyExtractor = configuredTargetKeyExtractor;
  }

  @Override
  public String getTargetKind(ConfiguredTargetValue configuredTargetValue) {
    Target actualTarget = getTargetFromConfiguredTargetValue(configuredTargetValue);
    return actualTarget.getTargetKind();
  }

  @Override
  public String getLabel(ConfiguredTargetValue configuredTargetValue) {
    return configuredTargetValue.getConfiguredTarget().getLabel().toString();
  }

  @Override
  public String getPackage(ConfiguredTargetValue configuredTargetValue) {
    return configuredTargetValue
        .getConfiguredTarget()
        .getLabel()
        .getPackageIdentifier()
        .getPackageFragment()
        .toString();
  }

  @Override
  public boolean isRule(ConfiguredTargetValue configuredTargetValue) {
    Target actualTarget = getTargetFromConfiguredTargetValue(configuredTargetValue);
    return actualTarget instanceof Rule;
  }

  @Override
  public boolean isTestRule(ConfiguredTargetValue configuredTargetValue) {
    Target actualTarget = getTargetFromConfiguredTargetValue(configuredTargetValue);
    return TargetUtils.isTestRule(actualTarget);
  }

  @Override
  public boolean isTestSuite(ConfiguredTargetValue configuredTargetValue) {
    Target actualTarget = getTargetFromConfiguredTargetValue(configuredTargetValue);
    return TargetUtils.isTestSuiteRule(actualTarget);
  }

  @Override
  public List<ConfiguredTargetValue> getPrerequisites(
      QueryExpression caller,
      ConfiguredTargetValue configuredTargetValue,
      String attrName,
      String errorMsgPrefix)
      throws QueryException, InterruptedException {
    // TODO(bazel-team): implement this if needed.
    throw new QueryException(
        "labels() is not supported for aquery", ActionQuery.Code.LABELS_FUNCTION_NOT_SUPPORTED);
  }

  @Override
  public List<String> getStringListAttr(
      ConfiguredTargetValue configuredTargetValue, String attrName) {
    Target actualTarget = getTargetFromConfiguredTargetValue(configuredTargetValue);
    return TargetUtils.getStringListAttr(actualTarget, attrName);
  }

  @Override
  public String getStringAttr(ConfiguredTargetValue configuredTargetValue, String attrName) {
    Target actualTarget = getTargetFromConfiguredTargetValue(configuredTargetValue);
    return TargetUtils.getStringAttr(actualTarget, attrName);
  }

  @Override
  public Iterable<String> getAttrAsString(
      ConfiguredTargetValue configuredTargetValue, String attrName) {
    Target actualTarget = getTargetFromConfiguredTargetValue(configuredTargetValue);
    return TargetUtils.getAttrAsString(actualTarget, attrName);
  }

  @Override
  public ImmutableSet<QueryVisibility<ConfiguredTargetValue>> getVisibility(
      QueryExpression caller, ConfiguredTargetValue from) throws QueryException {
    // TODO(bazel-team): implement this if needed.
    throw new QueryException(
        "visible() is not supported on configured targets",
        ConfigurableQuery.Code.VISIBLE_FUNCTION_NOT_SUPPORTED);
  }

  private Target getTargetFromConfiguredTargetValue(ConfiguredTargetValue configuredTargetValue) {
    // Dereference any aliases that might be present.
    Label label = configuredTargetValue.getConfiguredTarget().getOriginalLabel();
    try {
      return targetLookup.getTarget(label);
    } catch (InterruptedException e) {
      throw new IllegalStateException("Thread interrupted in the middle of getting a Target.", e);
    } catch (TargetNotFoundException e) {
      throw new IllegalStateException("Unable to get target from package in accessor.", e);
    }
  }

  /** Returns the AspectValues that are attached to the given configuredTarget. */
  public Set<AspectValue> getAspectValues(ConfiguredTargetValue configuredTargetValue)
      throws InterruptedException {
    Set<AspectValue> result = new HashSet<>();
    SkyKey skyKey = configuredTargetKeyExtractor.extractKey(configuredTargetValue);
    Iterable<SkyKey> revDeps =
        Iterables.concat(walkableGraph.getReverseDeps(ImmutableList.of(skyKey)).values());
    Label label = configuredTargetValue.getConfiguredTarget().getLabel();
    for (SkyKey revDep : revDeps) {
      SkyFunctionName skyFunctionName = revDep.functionName();
      if (SkyFunctions.ASPECT.equals(skyFunctionName)) {
        AspectValue aspectValue = (AspectValue) walkableGraph.getValue(revDep);
        if (((AspectKey) revDep).getLabel().equals(label)) {
          result.add(aspectValue);
        }
      }
    }
    return result;
  }
}
