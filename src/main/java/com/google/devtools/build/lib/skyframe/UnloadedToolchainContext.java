// Copyright 2019 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Set;

/**
 * Represents the state of toolchain resolution once the specific required toolchains have been
 * determined, but before the toolchain dependencies have been resolved.
 */
@AutoValue
public abstract class UnloadedToolchainContext implements ToolchainContext, SkyValue {

  /** Returns a new {@link UnloadedToolchainContextKey.Builder}. */
  public static UnloadedToolchainContextKey.Builder key() {
    return new AutoValue_UnloadedToolchainContext_UnloadedToolchainContextKey.Builder()
        .requiredToolchainTypeLabels(ImmutableSet.of())
        .execConstraintLabels(ImmutableSet.of())
        .shouldSanityCheckConfiguration(false);
  }

  /** {@link SkyKey} implementation used for {@link ToolchainResolutionFunction}. */
  @AutoValue
  public abstract static class UnloadedToolchainContextKey implements SkyKey {

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

  public static Builder builder() {
    return new AutoValue_UnloadedToolchainContext.Builder();
  }

  /** Builder class to help create the {@link UnloadedToolchainContext}. */
  @AutoValue.Builder
  public interface Builder {
    /** Sets the selected execution platform that these toolchains use. */
    Builder setExecutionPlatform(PlatformInfo executionPlatform);

    /** Sets the target platform that these toolchains generate output for. */
    Builder setTargetPlatform(PlatformInfo targetPlatform);

    /** Sets the toolchain types that were requested. */
    Builder setRequiredToolchainTypes(Set<ToolchainTypeInfo> requiredToolchainTypes);

    /**
     * Maps from the actual toolchain type to the resolved toolchain implementation that should be
     * used.
     */
    Builder setToolchainTypeToResolved(
        ImmutableBiMap<ToolchainTypeInfo, Label> toolchainTypeToResolved);

    /**
     * Maps from the actual requested {@link Label} to the discovered {@link ToolchainTypeInfo}.
     *
     * <p>Note that the key may be different from {@link ToolchainTypeInfo#typeLabel()} if the
     * requested {@link Label} is an {@code alias}.
     */
    Builder setRequestedLabelToToolchainType(
        ImmutableMap<Label, ToolchainTypeInfo> requestedLabelToToolchainType);

    UnloadedToolchainContext build();
  }

  /** The map of toolchain type to resolved toolchain to be used. */
  public abstract ImmutableBiMap<ToolchainTypeInfo, Label> toolchainTypeToResolved();

  /**
   * Maps from the actual requested {@link Label} to the discovered {@link ToolchainTypeInfo}.
   *
   * <p>Note that the key may be different from {@link ToolchainTypeInfo#typeLabel()} if the
   * requested {@link Label} is an {@code alias}. In this case, there will be two {@link Label
   * labels} for the same {@link ToolchainTypeInfo}.
   */
  public abstract ImmutableMap<Label, ToolchainTypeInfo> requestedLabelToToolchainType();

  @Override
  public ImmutableSet<Label> resolvedToolchainLabels() {
    return toolchainTypeToResolved().values();
  }
}
