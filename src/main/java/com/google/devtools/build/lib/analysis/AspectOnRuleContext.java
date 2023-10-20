// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.LinkedHashMap;
import java.util.Map;

/** Extends {@link RuleContext} to provide all data available during the analysis of an aspect. */
public final class AspectOnRuleContext extends RuleContext {

  /**
   * A list of all aspects applied to the target.
   *
   * <p>The last aspect in the list is the main aspect that this context is for.
   */
  private final ImmutableList<Aspect> aspects;

  private final ImmutableList<AspectDescriptor> aspectDescriptors;

  AspectOnRuleContext(
      RuleContext.Builder builder,
      AttributeMap ruleAttributes,
      ImmutableListMultimap<DependencyKind, ConfiguredTargetAndData> prerequisitesCollection,
      ExecGroupCollection execGroupCollection) {
    super(
        builder,
        new AspectAwareAttributeMapper(
            ruleAttributes, mergeAspectsAttributes(builder.getAspects())),
        prerequisitesCollection,
        execGroupCollection);

    this.aspects = builder.getAspects();
    this.aspectDescriptors = aspects.stream().map(Aspect::getDescriptor).collect(toImmutableList());
  }

  /**
   * Merge the attributes of the aspects in the aspects path.
   *
   * <p>For attributes with the same name, the one that is first encountered takes precedence.
   */
  private static ImmutableMap<String, Attribute> mergeAspectsAttributes(
      ImmutableList<Aspect> aspects) {
    if (aspects.isEmpty()) {
      return ImmutableMap.of();
    } else if (aspects.size() == 1) {
      return aspects.get(0).getDefinition().getAttributes();
    } else {
      LinkedHashMap<String, Attribute> aspectAttributes = new LinkedHashMap<>();
      for (Aspect aspect : aspects) {
        ImmutableMap<String, Attribute> currentAttributes = aspect.getDefinition().getAttributes();
        for (Map.Entry<String, Attribute> kv : currentAttributes.entrySet()) {
          aspectAttributes.putIfAbsent(kv.getKey(), kv.getValue());
        }
      }
      return ImmutableMap.copyOf(aspectAttributes);
    }
  }

  @Override
  public ImmutableList<Aspect> getAspects() {
    return aspects;
  }

  /**
   * Return the main aspect of this context.
   *
   * <p>It is the last aspect in the list of aspects applied to a target; all other aspects are the
   * ones main aspect sees as specified by its "required_aspect_providers").
   */
  @Override
  public Aspect getMainAspect() {
    return Iterables.getLast(aspects);
  }

  /** All aspects applied to the rule. */
  @Override
  public ImmutableList<AspectDescriptor> getAspectDescriptors() {
    return aspectDescriptors;
  }

  @Override
  public boolean useAutoExecGroups() {
    ImmutableMap<String, Attribute> aspectAttributes =
        getMainAspect().getDefinition().getAttributes();
    if (aspectAttributes.containsKey("$use_auto_exec_groups")) {
      return (boolean) aspectAttributes.get("$use_auto_exec_groups").getDefaultValueUnchecked();
    } else {
      return getConfiguration().useAutoExecGroups();
    }
  }
}
