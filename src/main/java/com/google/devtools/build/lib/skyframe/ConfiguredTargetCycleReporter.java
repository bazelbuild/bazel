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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * Reports cycles between {@link ConfiguredTargetValue}s. Similar to
 * {@link TransitiveTargetCycleReporter}, these indicate cycles between targets, but during the
 * analysis phase. In the current target-parsing, loading, analysis, and execution phase
 * distinction, such cycles can only occur due to the presence of a specific configuration (if
 * such a cycle occurs regardless of the configuration, then it would have been caught during the
 * target parsing or loading phase).
 */
class ConfiguredTargetCycleReporter extends AbstractLabelCycleReporter {

  private static final Predicate<SkyKey> IS_CONFIGURED_TARGET_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.CONFIGURED_TARGET);

  ConfiguredTargetCycleReporter(PackageProvider packageProvider) {
    super(packageProvider);
  }

  @Override
  protected boolean canReportCycle(SkyKey topLevelKey, CycleInfo cycleInfo) {
    return Iterables.all(Iterables.concat(ImmutableList.of(topLevelKey),
        cycleInfo.getPathToCycle(), cycleInfo.getCycle()), IS_CONFIGURED_TARGET_SKY_KEY);
  }

  @Override
  protected String getAdditionalMessageAboutCycle(
      EventHandler eventHandler, SkyKey topLevelKey, CycleInfo cycleInfo) {
    return "\nThis cycle occurred because of a configuration option";
  }

  @Override
  public String prettyPrint(SkyKey key) {
    return ((ConfiguredTargetKey) key.argument()).prettyPrint();
  }

  @Override
  public Label getLabel(SkyKey key) {
    return ((ConfiguredTargetKey) key.argument()).getLabel();
  }
}