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
package com.google.devtools.build.lib.skyframe.toolchains;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Optional;

/**
 * {@link SkyKey} implementation used for {@link ToolchainResolutionFunction} to produce {@link
 * UnloadedToolchainContextImpl} instances.
 */
@AutoValue
public abstract class ToolchainContextKey implements SkyKey {

  private static final SkyKeyInterner<ToolchainContextKey> interner = SkyKey.newInterner();

  /** Returns a new {@link Builder}. */
  public static Builder key() {
    return new AutoValue_ToolchainContextKey.Builder()
        .toolchainTypes(ImmutableSet.of())
        .execConstraintLabels(ImmutableSet.of())
        .debugTarget(false);
  }

  @Override
  public final SkyFunctionName functionName() {
    return SkyFunctions.TOOLCHAIN_RESOLUTION;
  }

  @Override
  public final SkyKeyInterner<?> getSkyKeyInterner() {
    return interner;
  }

  public abstract BuildConfigurationKey configurationKey();

  abstract ImmutableSet<ToolchainTypeRequirement> toolchainTypes();

  abstract ImmutableSet<Label> execConstraintLabels();

  abstract Optional<Label> forceExecutionPlatform();

  public abstract boolean debugTarget();

  /** Builder for {@link ToolchainContextKey}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder configurationKey(BuildConfigurationKey key);

    public abstract Builder toolchainTypes(ImmutableSet<ToolchainTypeRequirement> toolchainTypes);

    public abstract Builder toolchainTypes(ToolchainTypeRequirement... toolchainTypes);

    public abstract Builder execConstraintLabels(ImmutableSet<Label> execConstraintLabels);

    public abstract Builder execConstraintLabels(Label... execConstraintLabels);

    public abstract Builder debugTarget(boolean flag);

    public abstract Builder forceExecutionPlatform(Label execPlatform);

    public final ToolchainContextKey build() {
      return interner.intern(autoBuild());
    }

    abstract ToolchainContextKey autoBuild();
  }
}
