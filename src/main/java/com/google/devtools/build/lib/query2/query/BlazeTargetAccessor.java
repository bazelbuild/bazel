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
package com.google.devtools.build.lib.query2.query;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.PackageGroupsRuleVisibility;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetNotFoundException;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryVisibility;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Implementation of {@link TargetAccessor &lt;Target&gt;} that uses an
 * {@link AbstractBlazeQueryEnvironment &lt;Target&gt;} internally to report issues and resolve
 * targets.
 */
public final class BlazeTargetAccessor implements TargetAccessor<Target> {
  private final AbstractBlazeQueryEnvironment<Target> queryEnvironment;

  public BlazeTargetAccessor(AbstractBlazeQueryEnvironment<Target> queryEnvironment) {
    this.queryEnvironment = queryEnvironment;
  }

  @Override
  public String getTargetKind(Target target) {
    return target.getTargetKind();
  }

  @Override
  public String getLabel(Target target) {
    return target.getLabel().toString();
  }

  @Override
  public String getPackage(Target target) {
    return target.getPackage().getNameFragment().toString();
  }

  @Override
  public Iterable<Target> getPrerequisites(
      QueryExpression caller, Target target, String attrName, String errorMsgPrefix)
      throws QueryException, InterruptedException {
    Preconditions.checkArgument(target instanceof Rule);

    Rule rule = (Rule) target;

    AggregatingAttributeMapper attrMap = AggregatingAttributeMapper.of(rule);
    Type<?> attrType = attrMap.getAttributeType(attrName);
    if (attrType == null) {
      // Return an empty list if the attribute isn't defined for this rule.
      return ImmutableList.of();
    }

    Set<Label> labels = attrMap.getReachableLabels(attrName, false);
    // TODO(nharmata): Figure out how to make use of the package semaphore in the transitive
    // callsites of this method.
    Map<Label, Target> labelTargetMap = queryEnvironment.getTargets(labels);
    // Optimize for the common-case of no missing targets.
    if (labelTargetMap.size() != labels.size()) {
      for (Label label : labels) {
        if (!labelTargetMap.containsKey(label)) {
          // If a target was missing, fetch it directly for the sole purpose of getting a useful
          // error message.
          try {
            queryEnvironment.getTarget(label);
          } catch (TargetNotFoundException e) {
            queryEnvironment.reportBuildFileError(caller, errorMsgPrefix + e.getMessage());
          }
        }
      }

    }
    return labelTargetMap.values();
  }

  @Override
  public List<String> getStringListAttr(Target target, String attrName) {
    return TargetUtils.getStringListAttr(target, attrName);
  }

  @Override
  public String getStringAttr(Target target, String attrName) {
    return TargetUtils.getStringAttr(target, attrName);
  }

  @Override
  public Iterable<String> getAttrAsString(Target target, String attrName) {
    return TargetUtils.getAttrAsString(target, attrName);
  }

  @Override
  public boolean isRule(Target target) {
    return target instanceof Rule;
  }

  @Override
  public boolean isTestRule(Target target) {
    return TargetUtils.isTestRule(target);
  }

  @Override
  public boolean isTestSuite(Target target) {
    return TargetUtils.isTestSuiteRule(target);
  }

  @Override
  public Set<QueryVisibility<Target>> getVisibility(Target target)
      throws QueryException, InterruptedException {
    ImmutableSet.Builder<QueryVisibility<Target>> result = ImmutableSet.builder();
    result.add(QueryVisibility.samePackage(target, this));
    convertVisibility(result, target);
    return result.build();
  }

  // CAUTION: keep in sync with ConfiguredTargetFactory#convertVisibility()
  private void convertVisibility(
      ImmutableSet.Builder<QueryVisibility<Target>> packageSpecifications, Target target)
      throws QueryException, InterruptedException {
   RuleVisibility ruleVisibility = target.getVisibility();
   if (ruleVisibility instanceof ConstantRuleVisibility) {
     if (((ConstantRuleVisibility) ruleVisibility).isPubliclyVisible()) {
       packageSpecifications.add(QueryVisibility.<Target>everything());
     }
     return;
   } else if (ruleVisibility instanceof PackageGroupsRuleVisibility) {
     PackageGroupsRuleVisibility packageGroupsVisibility =
         (PackageGroupsRuleVisibility) ruleVisibility;
     for (Label groupLabel : packageGroupsVisibility.getPackageGroups()) {
       try {
          maybeConvertGroupVisibility(groupLabel, packageSpecifications);
       } catch (TargetNotFoundException e) {
         throw new QueryException(e.getMessage());
       }
     }
      packageSpecifications.add(
          new BlazeQueryVisibility(packageGroupsVisibility.getDirectPackages()));
     return;
   } else {
     throw new IllegalStateException("unknown visibility: " + ruleVisibility.getClass());
   }
  }

  private void maybeConvertGroupVisibility(
      Label groupLabel, ImmutableSet.Builder<QueryVisibility<Target>> packageSpecifications)
      throws QueryException, TargetNotFoundException, InterruptedException {
    Target groupTarget = queryEnvironment.getTarget(groupLabel);
    if (groupTarget instanceof PackageGroup) {
      convertGroupVisibility(
          (PackageGroup) queryEnvironment.getTarget(groupLabel), packageSpecifications);
    }
  }

  private void convertGroupVisibility(
      PackageGroup group, ImmutableSet.Builder<QueryVisibility<Target>> packageSpecifications)
      throws QueryException, TargetNotFoundException, InterruptedException {
    for (Label include : group.getIncludes()) {
      maybeConvertGroupVisibility(include, packageSpecifications);
    }
    packageSpecifications.add(new BlazeQueryVisibility(group.getPackageSpecifications()));
  }
}
