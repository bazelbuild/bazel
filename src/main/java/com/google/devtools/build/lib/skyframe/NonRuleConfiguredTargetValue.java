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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import javax.annotation.Nullable;

/** A non-rule configured target in the context of a Skyframe graph. */
@Immutable
@ThreadSafe
// Reached via OutputFileConfiguredTarget.
@AutoCodec(explicitlyAllowClass = RuleConfiguredTarget.class)
public final class NonRuleConfiguredTargetValue implements ConfiguredTargetValue {
  // These variables are only non-final because they may be clear()ed to save memory.
  // configuredTarget is null only after it is cleared.
  @Nullable private ConfiguredTarget configuredTarget;

  // May be null either after clearing or because transitive packages are not tracked.
  @Nullable private NestedSet<Package> transitivePackages;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  NonRuleConfiguredTargetValue(
      ConfiguredTarget configuredTarget) {
    // Transitive packages are not serialized.
    this(configuredTarget, null);
  }

  NonRuleConfiguredTargetValue(
      ConfiguredTarget configuredTarget, @Nullable NestedSet<Package> transitivePackages) {
    this.configuredTarget = Preconditions.checkNotNull(configuredTarget);
    this.transitivePackages = transitivePackages;
  }

  @Nullable // May be null after clearing.
  @Override
  public ConfiguredTarget getConfiguredTarget() {
    return configuredTarget;
  }

  @Nullable
  @Override
  public NestedSet<Package> getTransitivePackages() {
    return transitivePackages;
  }

  @Override
  public void clear(boolean clearEverything) {
    if (clearEverything) {
      configuredTarget = null;
    }
    transitivePackages = null;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("configuredTarget", configuredTarget).toString();
  }
}
