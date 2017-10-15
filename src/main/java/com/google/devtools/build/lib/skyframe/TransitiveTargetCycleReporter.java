// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.List;

/**
 * Reports cycles between {@link TransitiveTargetValue}s. These indicates cycles between targets
 * (e.g. '//a:foo' depends on '//b:bar' and '//b:bar' depends on '//a:foo').
 */
class TransitiveTargetCycleReporter extends AbstractLabelCycleReporter {

  private static final Predicate<SkyKey> IS_TRANSITIVE_TARGET_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.TRANSITIVE_TARGET);

  TransitiveTargetCycleReporter(PackageProvider packageProvider) {
    super(packageProvider);
  }

  @Override
  protected boolean canReportCycle(SkyKey topLevelKey, CycleInfo cycleInfo) {
    return Iterables.all(Iterables.concat(ImmutableList.of(topLevelKey),
        cycleInfo.getPathToCycle(), cycleInfo.getCycle()),
        IS_TRANSITIVE_TARGET_SKY_KEY);
  }

  @Override
  public String prettyPrint(SkyKey key) {
    return getLabel(key).toString();
  }

  @Override
  protected Label getLabel(SkyKey key) {
    return (Label) key.argument();
  }

  @Override
  protected String getAdditionalMessageAboutCycle(
      ExtendedEventHandler eventHandler, SkyKey topLevelKey, CycleInfo cycleInfo) {
    Target currentTarget = getTargetForLabel(eventHandler, getLabel(topLevelKey));
    List<SkyKey> keys = Lists.newArrayList();
    if (!cycleInfo.getPathToCycle().isEmpty()) {
      keys.add(topLevelKey);
      keys.addAll(cycleInfo.getPathToCycle());
    }
    keys.addAll(cycleInfo.getCycle());
    // Make sure we check the edge from the last element of the cycle to the first element of the
    // cycle.
    keys.add(cycleInfo.getCycle().get(0));
    for (SkyKey nextKey : keys) {
      Label nextLabel = getLabel(nextKey);
      Target nextTarget = getTargetForLabel(eventHandler, nextLabel);
      // This is inefficient but it's no big deal since we only do this when there's a cycle.
      if (currentTarget.getVisibility().getDependencyLabels().contains(nextLabel)
          && !nextTarget.getTargetKind().equals(PackageGroup.targetKind())) {
        return "\nThe cycle is caused by a visibility edge from " + currentTarget.getLabel()
            + " to the non-package-group target " + nextTarget.getLabel() + " . Note that "
            + "visibility labels are supposed to be package group targets (which prevents cycles "
            + "of this form)";
      }
      currentTarget = nextTarget;
    }
    return "";
  }
}
