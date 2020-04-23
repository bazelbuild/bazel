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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * {@link SkyKey} implementation used for {@link ToolchainResolutionFunction} to produce {@link
 * UnloadedToolchainContextImpl} instances.
 */
@AutoValue
public abstract class UnloadedToolchainContextKey implements SkyKey {

  /** Returns a new {@link Builder}. */
  public static Builder key() {
    return new AutoValue_UnloadedToolchainContextKey.Builder()
        .requiredToolchainTypeLabels(ImmutableSet.of())
        .execConstraintLabels(ImmutableSet.of())
        .shouldSanityCheckConfiguration(false);
  }

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.TOOLCHAIN_RESOLUTION;
  }

  abstract BuildConfigurationValue.Key configurationKey();

  abstract ImmutableSet<Label> requiredToolchainTypeLabels();

  abstract ImmutableSet<Label> execConstraintLabels();

  abstract boolean shouldSanityCheckConfiguration();

  /** Builder for {@link UnloadedToolchainContextKey}. */
  @AutoValue.Builder
  public interface Builder {
    Builder configurationKey(BuildConfigurationValue.Key key);

    Builder requiredToolchainTypeLabels(ImmutableSet<Label> requiredToolchainTypeLabels);

    Builder requiredToolchainTypeLabels(Label... requiredToolchainTypeLabels);

    Builder execConstraintLabels(ImmutableSet<Label> execConstraintLabels);

    Builder execConstraintLabels(Label... execConstraintLabels);

    Builder shouldSanityCheckConfiguration(boolean shouldSanityCheckConfiguration);

    UnloadedToolchainContextKey build();
  }
}
