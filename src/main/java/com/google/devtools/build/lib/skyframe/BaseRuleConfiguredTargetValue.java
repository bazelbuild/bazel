// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.analysis.config.CommonOptions.EMPTY_OPTIONS;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Package;
import javax.annotation.Nullable;

/** Common base class for configured target values for rules and non-rules. */
abstract class BaseRuleConfiguredTargetValue<T extends ConfiguredTarget>
    implements ConfiguredTargetValue {
  // This variable is non-final because it may be clear()ed to save memory. It is null only after
  // clear(true) is called.
  @Nullable private T configuredTarget;

  // May be null after clearing; because transitive packages are not tracked; or after
  // deserialization.
  @Nullable private transient NestedSet<Package> transitivePackages;

  BaseRuleConfiguredTargetValue(
      T configuredTarget, @Nullable NestedSet<Package> transitivePackages) {
    this.configuredTarget = Preconditions.checkNotNull(configuredTarget);
    this.transitivePackages = transitivePackages;
  }

  @Nullable // May be null after clearing.
  @Override
  public T getConfiguredTarget() {
    return configuredTarget;
  }

  @Nullable
  @Override
  public NestedSet<Package> getTransitivePackages() {
    return transitivePackages;
  }

  @Override
  public void clear(boolean clearEverything) {
    // Snapshot this because this method is sometimes called from multiple threads. We don't need
    // to actually synchronize: both threads will have the same logic and so there's no risk of
    // a race condition.
    T ct = this.configuredTarget;
    if (ct != null
        && ct.getConfigurationKey() != null
        && ct.getConfigurationChecksum().equals(EMPTY_OPTIONS.checksum())) {
      // Keep these to avoid the need to re-create them later, they are dependencies of the empty
      // configuration key and will never change.
      return;
    }
    if (clearEverything) {
      this.configuredTarget = null;
    }
    this.transitivePackages = null;
  }
}
