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
package com.google.devtools.build.lib.query2;

import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryVisibility;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.List;
import java.util.Set;

/** A {@link TargetAccessor} for {@link ConfiguredTargetValue} objects.
 *
 * Incomplete; we'll implement getLabelListAttr and getVisibility when needed.
 */
class ConfiguredTargetValueAccessor implements TargetAccessor<ConfiguredTargetValue> {

  private final WalkableGraph walkableGraph;

  ConfiguredTargetValueAccessor(WalkableGraph walkableGraph) {
    this.walkableGraph = walkableGraph;
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
  public List<ConfiguredTargetValue> getLabelListAttr(
      QueryExpression caller,
      ConfiguredTargetValue configuredTargetValue,
      String attrName,
      String errorMsgPrefix)
      throws QueryException, InterruptedException {
    // TODO(bazel-team): implement this if needed.
    throw new UnsupportedOperationException();
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
  public Set<QueryVisibility<ConfiguredTargetValue>> getVisibility(ConfiguredTargetValue from)
      throws QueryException, InterruptedException {
    // TODO(bazel-team): implement this if needed.
    throw new UnsupportedOperationException();
  }

  private Target getTargetFromConfiguredTargetValue(ConfiguredTargetValue configuredTargetValue) {
    return ConfiguredTargetAccessor.getTargetFromConfiguredTarget(
        configuredTargetValue.getConfiguredTarget(), walkableGraph);
  }
}
