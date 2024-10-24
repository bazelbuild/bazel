// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.List;

/**
 * Reports cycles between targets. These may be in the form of {@link
 * com.google.devtools.build.lib.analysis.ConfiguredTargetValue}s or {@link TransitiveTargetValue}s.
 */
class TargetCycleReporter extends AbstractLabelCycleReporter {

  private static final Predicate<SkyKey> CONFIGURED_TARGET_OR_TRANSITIVE_RDEP =
      Predicates.or(
          SkyFunctions.isSkyFunction(SkyFunctions.CONFIGURED_TARGET),
          SkyFunctions.isSkyFunction(SkyFunctions.ASPECT),
          SkyFunctions.isSkyFunction(SkyFunctions.TOP_LEVEL_ASPECTS),
          SkyFunctions.isSkyFunction(TransitiveTargetKey.NAME),
          SkyFunctions.isSkyFunction(SkyFunctions.PREPARE_ANALYSIS_PHASE),
          SkyFunctions.isSkyFunction(SkyFunctions.BUILD_DRIVER));

  TargetCycleReporter(PackageProvider packageProvider) {
    super(packageProvider);
  }

  @Override
  protected boolean shouldSkipOnPathToCycle(SkyKey key) {
    return SkyFunctions.PREPARE_ANALYSIS_PHASE.equals(key.functionName())
        // BuildDriverKeys don't provide any relevant info for the end user.
        || SkyFunctions.BUILD_DRIVER.equals(key.functionName());
  }

  @Override
  protected boolean canReportCycle(SkyKey topLevelKey, CycleInfo cycleInfo) {
    return CONFIGURED_TARGET_OR_TRANSITIVE_RDEP.apply(topLevelKey)
        && cycleInfo.getPathToCycle().stream().allMatch(CONFIGURED_TARGET_OR_TRANSITIVE_RDEP)
        && cycleInfo.getCycle().stream().allMatch(CONFIGURED_TARGET_OR_TRANSITIVE_RDEP);
  }

  @Override
  public String prettyPrint(Object key) {
    if (key instanceof ConfiguredTargetKey configuredTargetKey) {
      return configuredTargetKey.prettyPrint();
    } else if (key instanceof AspectKey aspectKey) {
      return aspectKey.prettyPrint();
    } else {
      return getLabel((SkyKey) key).toString();
    }
  }

  @Override
  public Label getLabel(SkyKey key) {
    if (key instanceof ActionLookupKey) {
      return Preconditions.checkNotNull(((ActionLookupKey) key.argument()).getLabel(), key);
    } else if (key instanceof TransitiveTargetKey transitiveTargetKey) {
      return transitiveTargetKey.getLabel();
    } else {
      throw new UnsupportedOperationException(key.toString());
    }
  }

  @Override
  protected String getAdditionalMessageAboutCycle(
      ExtendedEventHandler eventHandler, SkyKey topLevelKey, CycleInfo cycleInfo) {
    List<SkyKey> keys = Lists.newArrayList();
    if (!cycleInfo.getPathToCycle().isEmpty()) {
      if (!shouldSkipOnPathToCycle(topLevelKey)) {
        keys.add(topLevelKey);
      }
      cycleInfo.getPathToCycle().stream()
          .filter(key -> !shouldSkipOnPathToCycle(key))
          .forEach(keys::add);
    }
    keys.addAll(cycleInfo.getCycle());
    // Make sure we check the edge from the last element of the cycle to the first element of the
    // cycle.
    keys.add(cycleInfo.getCycle().get(0));

    Target currentTarget = getTargetForLabel(eventHandler, getLabel(keys.get(0)));
    for (SkyKey nextKey : keys) {
      Label nextLabel = getLabel(nextKey);
      Target nextTarget = getTargetForLabel(eventHandler, nextLabel);
      // TODO(aranguyen): remove this code as a result of b/128716030
      // This is inefficient but it's no big deal since we only do this when there's a cycle.
      if (!nextTarget.getTargetKind().equals(PackageGroup.targetKind())
          && Iterables.contains(currentTarget.getVisibilityDependencyLabels(), nextLabel)) {
        return "\nThe cycle is caused by a visibility edge from "
            + currentTarget.getLabel()
            + " to the non-package_group target "
            + nextTarget.getLabel()
            + ". Note that "
            + "visibility labels are supposed to be package_group targets, which prevents cycles "
            + "of this form.";
      }
      currentTarget = nextTarget;
    }
    return "";
  }
}
