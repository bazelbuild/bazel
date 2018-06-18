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

package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.skyframe.NotComparableSkyValue;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** A {@link SkyValue} for a {@link ConfiguredTarget}. */
public interface ConfiguredTargetValue extends NotComparableSkyValue {
  static SkyKey key(Label label, BuildConfiguration configuration) {
    return ConfiguredTargetKey.of(label, configuration);
  }

  static ImmutableList<SkyKey> keys(Iterable<ConfiguredTargetKey> lacs) {
    ImmutableList.Builder<SkyKey> keys = ImmutableList.builder();
    for (ConfiguredTargetKey lac : lacs) {
      keys.add(lac);
    }
    return keys.build();
  }

  /**
   * Returns the configured target for this value.
   */
  ConfiguredTarget getConfiguredTarget();

  /**
   * Returns the set of packages transitively loaded by this value. Must only be used for
   * constructing the package -> source root map needed for some builds. If the caller has not
   * specified that this map needs to be constructed (via the constructor argument in {@link
   * ConfiguredTargetFunction#ConfiguredTargetFunction}), calling this will crash.
   */
  NestedSet<Package> getTransitivePackagesForPackageRootResolution();

  /** Returns the actions registered by the configured target for this value. */
  ImmutableList<ActionAnalysisMetadata> getActions();

  /**
   * Returns the number of {@link Action} objects present in this value.
   */
  int getNumActions();

  /**
   * Clears configured target data from this value, leaving only the artifact->generating action
   * map.
   *
   * <p>Should only be used when user specifies --discard_analysis_cache. Must be called at most
   * once per value, after which {@link #getConfiguredTarget} and {@link #getActions} cannot be
   * called.
   */
  void clear(boolean clearEverything);
}
