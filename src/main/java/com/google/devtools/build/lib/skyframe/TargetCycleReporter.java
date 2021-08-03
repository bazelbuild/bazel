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
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.List;

/**
 * Reports cycles between targets. These may be in the form of {@link ConfiguredTargetValue}s or
 * {@link TransitiveTargetValue}s.
 */
class TargetCycleReporter extends AbstractLabelCycleReporter {

  private static final Predicate<SkyKey> SUPPORTED_SKY_KEY =
      Predicates.or(
          SkyFunctions.isSkyFunction(SkyFunctions.CONFIGURED_TARGET),
          SkyFunctions.isSkyFunction(SkyFunctions.ASPECT),
          SkyFunctions.isSkyFunction(SkyFunctions.LOAD_STARLARK_ASPECT),
          SkyFunctions.isSkyFunction(TransitiveTargetKey.NAME),
          SkyFunctions.isSkyFunction(SkyFunctions.PREPARE_ANALYSIS_PHASE));

  TargetCycleReporter(PackageProvider packageProvider) {
    super(packageProvider);
  }

  @Override
  protected boolean shouldSkipOnPathToCycle(SkyKey key) {
    return SkyFunctions.PREPARE_ANALYSIS_PHASE.equals(key.functionName());
  }

  @Override
  protected boolean canReportCycle(SkyKey topLevelKey, CycleInfo cycleInfo) {
    return SUPPORTED_SKY_KEY.apply(topLevelKey)
        && cycleInfo.getPathToCycle().stream().allMatch(SUPPORTED_SKY_KEY)
        && cycleInfo.getCycle().stream().allMatch(SUPPORTED_SKY_KEY);
  }

  @Override
  public String prettyPrint(SkyKey key) {
    if (key instanceof ConfiguredTargetKey) {
      return ((ConfiguredTargetKey) key.argument()).prettyPrint();
    } else if (key instanceof AspectKey) {
      return ((AspectKey) key.argument()).prettyPrint();
    } else if (key instanceof AspectValueKey) {
      return ((AspectValueKey) key).getDescription();
    } else {
      return getLabel(key).toString();
    }
  }

  @Override
  public Label getLabel(SkyKey key) {
    if (key instanceof ActionLookupKey) {
      return Preconditions.checkNotNull(((ActionLookupKey) key.argument()).getLabel(), key);
    } else if (key instanceof TransitiveTargetKey) {
      return ((TransitiveTargetKey) key).getLabel();
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
      // This is inefficient but it's no big deal since we only do this when there's a cycle.
      if (currentTarget.getVisibility().getDependencyLabels().contains(nextLabel)
          && !nextTarget.getTargetKind().equals(PackageGroup.targetKind())) {
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
