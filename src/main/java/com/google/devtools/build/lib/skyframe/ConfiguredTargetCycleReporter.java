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

import static com.google.devtools.build.lib.skyframe.SkyFunctions.TRANSITIVE_TARGET;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectKey;
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
      Predicates.or(
          SkyFunctions.isSkyFunction(SkyFunctions.CONFIGURED_TARGET),
          SkyFunctions.isSkyFunction(SkyFunctions.ASPECT),
          SkyFunctions.isSkyFunction(SkyFunctions.LOAD_STARLARK_ASPECT));

  private static final Predicate<SkyKey> IS_TRANSITIVE_TARGET_SKY_KEY =
      SkyFunctions.isSkyFunction(TRANSITIVE_TARGET);

  private final TransitiveTargetCycleReporter targetReporter;

  ConfiguredTargetCycleReporter(PackageProvider packageProvider) {
    super(packageProvider);
    targetReporter = new TransitiveTargetCycleReporter(packageProvider);
  }

  @Override
  protected boolean canReportCycle(SkyKey topLevelKey, CycleInfo cycleInfo) {
    if (!IS_CONFIGURED_TARGET_SKY_KEY.apply(topLevelKey)) {
      return false;
    }
    Iterable<SkyKey> cycleKeys = Iterables.concat(cycleInfo.getPathToCycle(), cycleInfo.getCycle());
    // The top-level key should be a ConfiguredTargetValue key, but cycles and paths to it can
    // travel through TransitiveTargetValue keys because ConfiguredTargetFunction visits
    // visits TransitiveTargetFunction as a part of dynamic configuration computation.
    return Iterables.all(cycleKeys,
        Predicates.<SkyKey>or(IS_CONFIGURED_TARGET_SKY_KEY, IS_TRANSITIVE_TARGET_SKY_KEY));
  }

  @Override
  protected String getAdditionalMessageAboutCycle(
      ExtendedEventHandler eventHandler, SkyKey topLevelKey, CycleInfo cycleInfo) {
    if (Iterables.all(cycleInfo.getCycle(), IS_TRANSITIVE_TARGET_SKY_KEY)) {
      // The problem happened strictly in loading, so delegate the explanation to
      // TransitiveTargetCycleReporter.
      Iterable<SkyKey> pathAsTargetKeys = Iterables.transform(cycleInfo.getPathToCycle(),
          new Function<SkyKey, SkyKey>() {
            @Override
            public SkyKey apply(SkyKey key) {
              return asTransitiveTargetKey(key);
            }
          });
      return targetReporter.getAdditionalMessageAboutCycle(eventHandler,
          asTransitiveTargetKey(topLevelKey),
          new CycleInfo(pathAsTargetKeys, cycleInfo.getCycle()));
    } else {
      return "\nThis cycle occurred because of a configuration option";
    }
  }

  private SkyKey asTransitiveTargetKey(SkyKey key) {
    if (IS_TRANSITIVE_TARGET_SKY_KEY.apply(key)) {
      return key;
    } else if (key.argument() instanceof ActionLookupKey) {
      return TransitiveTargetKey.of(((ActionLookupKey) key.argument()).getLabel());
    } else {
      throw new IllegalArgumentException("Unknown type: " + key);
    }
  }

  @Override
  public String prettyPrint(SkyKey key) {
    if (SkyFunctions.isSkyFunction(SkyFunctions.CONFIGURED_TARGET).apply(key)) {
      return ((ConfiguredTargetKey) key.argument()).prettyPrint();
    } else if (SkyFunctions.isSkyFunction(SkyFunctions.ASPECT).apply(key)) {
      return ((AspectKey) key.argument()).prettyPrint();
    } else if (key instanceof AspectValue.AspectValueKey) {
      return ((AspectValue.AspectValueKey) key).getDescription();
    } else {
      return getLabel(key).toString();
    }
  }

  @Override
  public Label getLabel(SkyKey key) {
    if (key instanceof ActionLookupKey) {
      return Preconditions.checkNotNull(((ActionLookupKey) key.argument()).getLabel(), key);
    } else if (SkyFunctions.isSkyFunction(TRANSITIVE_TARGET).apply(key)) {
      return ((TransitiveTargetKey) key).getLabel();
    } else {
      throw new UnsupportedOperationException(key.toString());
    }
  }
}
