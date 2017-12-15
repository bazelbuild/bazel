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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Actions.GeneratingActions;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A configured target in the context of a Skyframe graph.
 */
@Immutable
@ThreadSafe
@VisibleForTesting
public final class ConfiguredTargetValue extends ActionLookupValue {

  // These variables are only non-final because they may be clear()ed to save memory.
  // configuredTarget is null only after it is cleared.
  @Nullable private ConfiguredTarget configuredTarget;

  // May be null either after clearing or because transitive packages are not tracked.
  @Nullable private NestedSet<Package> transitivePackagesForPackageRootResolution;

  ConfiguredTargetValue(
      ConfiguredTarget configuredTarget,
      GeneratingActions generatingActions,
      @Nullable NestedSet<Package> transitivePackagesForPackageRootResolution,
      boolean removeActionsAfterEvaluation) {
    super(generatingActions, removeActionsAfterEvaluation);
    this.configuredTarget = Preconditions.checkNotNull(configuredTarget, generatingActions);
    this.transitivePackagesForPackageRootResolution = transitivePackagesForPackageRootResolution;
  }

  @VisibleForTesting
  public ConfiguredTarget getConfiguredTarget() {
    Preconditions.checkNotNull(configuredTarget);
    return configuredTarget;
  }

  @VisibleForTesting
  public List<ActionAnalysisMetadata> getActions() {
    Preconditions.checkNotNull(configuredTarget, this);
    return actions;
  }

  /**
   * Returns the set of packages transitively loaded by this value. Must only be used for
   * constructing the package -> source root map needed for some builds. If the caller has not
   * specified that this map needs to be constructed (via the constructor argument in {@link
   * ConfiguredTargetFunction#ConfiguredTargetFunction}), calling this will crash.
   */
  public NestedSet<Package> getTransitivePackagesForPackageRootResolution() {
    return Preconditions.checkNotNull(transitivePackagesForPackageRootResolution);
  }

  /**
   * Clears configured target data from this value, leaving only the artifact->generating action
   * map.
   *
   * <p>Should only be used when user specifies --discard_analysis_cache. Must be called at most
   * once per value, after which {@link #getConfiguredTarget} and {@link #getActions} cannot be
   * called.
   *
   * @param clearEverything if true, clear the {@link #configuredTarget}. If not, only the {@link
   *     #transitivePackagesForPackageRootResolution} field is cleared. Top-level targets need their
   *     {@link #configuredTarget} preserved, so should pass false here.
   */
  public void clear(boolean clearEverything) {
    Preconditions.checkNotNull(configuredTarget);
    Preconditions.checkNotNull(transitivePackagesForPackageRootResolution);
    if (clearEverything) {
      configuredTarget = null;
    }
    transitivePackagesForPackageRootResolution = null;
  }

  @VisibleForTesting
  public static SkyKey key(Label label, BuildConfiguration configuration) {
    return key(new ConfiguredTargetKey(label, configuration));
  }

  static ImmutableList<SkyKey> keys(Iterable<ConfiguredTargetKey> lacs) {
    ImmutableList.Builder<SkyKey> keys = ImmutableList.builder();
    for (ConfiguredTargetKey lac : lacs) {
      keys.add(key(lac));
    }
    return keys.build();
  }

  /**
   * Returns a label of ConfiguredTargetValue.
   */
  @ThreadSafe
  static Label extractLabel(SkyKey value) {
    Object valueName = value.argument();
    Preconditions.checkState(valueName instanceof ConfiguredTargetKey, valueName);
    return ((ConfiguredTargetKey) valueName).getLabel();
  }

  @Override
  public String toString() {
    return getStringHelper().add("configuredTarget", configuredTarget).toString();
  }
}
