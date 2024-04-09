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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.analysis.config.CommonOptions.EMPTY_OPTIONS;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.RuleConfiguredObjectValue;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Package;
import javax.annotation.Nullable;

/** A configured target in the context of a Skyframe graph. */
@Immutable
@ThreadSafe
public final class RuleConfiguredTargetValue
    implements RuleConfiguredObjectValue, ConfiguredTargetValue {

  // This variable is non-final because it may be clear()ed to save memory. It is null only after
  // clear(true) is called.
  @Nullable private RuleConfiguredTarget configuredTarget;

  /**
   * Operations accessing actions, for example, executing them, should be performed in the same
   * Bazel instance that constructs the {@code RuleConfiguredTargetValue} instance and not on a
   * Bazel instance that retrieves it remotely using deserialization.
   */
  @Nullable // Null if deserialized.
  private final transient ImmutableList<ActionAnalysisMetadata> actions;

  // May be null after clearing; because transitive packages are not tracked; or after
  // deserialization.
  @Nullable private transient NestedSet<Package> transitivePackages;

  public RuleConfiguredTargetValue(
      RuleConfiguredTarget configuredTarget, @Nullable NestedSet<Package> transitivePackages) {
    this.configuredTarget = Preconditions.checkNotNull(configuredTarget);
    this.transitivePackages = transitivePackages;
    // These are specifically *not* copied to save memory.
    this.actions = configuredTarget.getActions();
  }

  @Nullable // May be null after clearing.
  @Override
  public RuleConfiguredTarget getConfiguredTarget() {
    return configuredTarget;
  }

  @Override
  public ImmutableList<ActionAnalysisMetadata> getActions() {
    return checkNotNull(actions, "actions are not available on deserialized instances");
  }

  @Nullable
  @Override
  public NestedSet<Package> getTransitivePackages() {
    return transitivePackages;
  }

  @Override
  public void clear(boolean clearEverything) {
    if (configuredTarget != null
        && configuredTarget.getConfigurationKey() != null
        && configuredTarget.getConfigurationChecksum().equals(EMPTY_OPTIONS.checksum())) {
      // Keep these to avoid the need to re-create them later, they are dependencies of the empty
      // configuration key and will never change.
      return;
    }
    if (clearEverything) {
      configuredTarget = null;
    }
    transitivePackages = null;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("actions", actions)
        .add("configuredTarget", configuredTarget)
        .toString();
  }
}
