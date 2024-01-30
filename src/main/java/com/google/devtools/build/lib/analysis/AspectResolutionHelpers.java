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
import static java.util.Collections.reverse;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.AspectCollection.AspectCycleOnPathException;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.ArrayList;
import javax.annotation.Nullable;

/** Helpers for aspect resolution. */
public final class AspectResolutionHelpers {
  private AspectResolutionHelpers() {}

  public static boolean aspectMatchesConfiguredTarget(
      ConfiguredTarget ct, boolean isRule, Aspect aspect) {
    if (!aspect.getDefinition().applyToFiles()
        && !aspect.getDefinition().applyToGeneratingRules()
        && !isRule) {
      return false;
    }
    if (ct.getConfigurationKey() == null) {
      // Aspects cannot apply to PackageGroups or InputFiles, the only cases where this is null.
      return false;
    }
    return ct.satisfies(aspect.getDefinition().getRequiredProviders());
  }

  public static boolean aspectMatchesConfiguredTarget(ConfiguredTargetAndData ctad, Aspect aspect) {
    return aspectMatchesConfiguredTarget(ctad.getConfiguredTarget(), ctad.isTargetRule(), aspect);
  }

  /**
   * Computes the set of aspects that could be applied to a dependency.
   *
   * <p>This is composed of two parts:
   *
   * <ol>
   *   <li>The aspects that are visible to this aspect being evaluated, if any. If another aspect is
   *       visible on the configured target, it should also be visible on the dependencies for
   *       consistency. This is the argument {@code aspectsPath}.
   *   <li>The aspects propagated by the attributes of this configured target / aspect.
   * </ol>
   *
   * <p>The presence of an aspect here does not necessarily mean that it will be available on a
   * dependency: it can still be filtered out because it requires a provider that the configured
   * target it should be attached to it doesn't advertise. This is taken into account in {@link
   * #computeAspectCollection} once the {@link ConfiguredTargetAndData} instances for the
   * dependencies are known.
   */
  public static ImmutableList<Aspect> computePropagatingAspects(
      DependencyKind kind, ImmutableList<Aspect> aspectsPath, Rule rule) {
    Attribute attribute = kind.getAttribute();
    if (attribute == null) {
      return ImmutableList.of();
    }
    var aspectsBuilder = new ImmutableList.Builder<Aspect>().addAll(attribute.getAspects(rule));
    collectPropagatingAspects(
        aspectsPath, attribute.getName(), kind.getOwningAspect(), aspectsBuilder);
    return aspectsBuilder.build();
  }

  /**
   * Computes the way aspects should be computed for the direct dependencies.
   *
   * <p>This is done by filtering the aspects that can be propagated on any attribute according to
   * the providers advertised by direct dependencies and by creating the {@link AspectCollection}
   * that tells how to compute the final set of providers based on the interdependencies between the
   * propagating aspects.
   */
  public static AspectCollection computeAspectCollection(
      ConfiguredTargetAndData prerequisite, ImmutableList<Aspect> propagatingAspects)
      throws InconsistentAspectOrderException {
    var aspects = propagatingAspects;
    if (prerequisite.isTargetOutputFile()) {
      // If `applyToGeneratingRules` holds, the aspect has no required providers so some of the
      // filtering logic below may be skipped, but it doesn't seem to be worth added complexity.
      aspects =
          aspects.stream()
              .filter(aspect -> aspect.getDefinition().applyToGeneratingRules())
              .collect(toImmutableList());
    }

    if (aspects.isEmpty() || (!prerequisite.isTargetOutputFile() && !prerequisite.isTargetRule())) {
      return AspectCollection.EMPTY;
    }

    var filteredAspectPath = new ArrayList<Aspect>();

    int aspectsCount = aspects.size();
    for (int i = aspectsCount - 1; i >= 0; i--) {
      Aspect aspect = aspects.get(i);
      if (prerequisite.satisfies(aspect.getDefinition().getRequiredProviders())
          || isAspectRequired(aspect, filteredAspectPath)) {
        // Adds the aspect if the configured target satisfies its required providers or it is
        // required by an aspect already in the {@code filteredAspectPath}.
        filteredAspectPath.add(aspect);
      }
    }
    reverse(filteredAspectPath);
    try {
      return AspectCollection.create(filteredAspectPath);
    } catch (AspectCycleOnPathException e) {
      throw new InconsistentAspectOrderException(
          prerequisite.getTargetLabel(), prerequisite.getLocation(), e);
    }
  }

  /**
   * Collects the aspects from {@code aspectsPath} that need to be propagated along the attribute
   * {@code attributeName}.
   *
   * <p>It can happen that some of the aspects cannot be propagated if the dependency doesn't have a
   * provider that's required by them. These will be filtered out after the rule class of the
   * dependency is known.
   */
  private static void collectPropagatingAspects(
      ImmutableList<Aspect> aspectsPath,
      String attributeName,
      @Nullable AspectClass aspectOwningAttribute,
      ImmutableList.Builder<Aspect> allFilteredAspects) {
    int aspectsNum = aspectsPath.size();
    var filteredAspectsPath = new ArrayList<Aspect>();

    // `aspectsPath` is ordered bottom up. Iterating backwards traverses top-down so the following
    // loop captures aspects that propagate along the given attribute and all their transitive
    // requirements.
    for (int i = aspectsNum - 1; i >= 0; i--) {
      Aspect aspect = aspectsPath.get(i);
      if (aspect.getAspectClass().equals(aspectOwningAttribute)) {
        // Do not propagate over the aspect's own attributes.
        continue;
      }

      if (aspect.getDefinition().propagateAlong(attributeName)
          || isAspectRequired(aspect, filteredAspectsPath)) {
        // Add the aspect if it can propagate over this {@code attributeName} based on its
        // attr_aspects or it is required by an aspect already in the {@code filteredAspectsPath}.
        filteredAspectsPath.add(aspect);
      }
    }
    // Reverse filteredAspectsPath to return it to the same order as the input aspectsPath.
    reverse(filteredAspectsPath);
    allFilteredAspects.addAll(filteredAspectsPath);
  }

  /** Checks if {@code aspect} is required by any {@link Aspect} in {@code aspectsPath}. */
  private static boolean isAspectRequired(Aspect aspect, Iterable<Aspect> aspectsPath) {
    for (Aspect existingAspect : aspectsPath) {
      if (existingAspect.getDefinition().requires(aspect)) {
        return true;
      }
    }
    return false;
  }
}
