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
package com.google.devtools.build.lib.skyframe.toolchains;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Set;

/**
 * Represents the state of toolchain resolution once the specific required toolchains have been
 * determined, but before the toolchain dependencies have been resolved.
 */
@AutoValue
public abstract class UnloadedToolchainContextImpl implements SkyValue, UnloadedToolchainContext {

  public static Builder builder(ToolchainContextKey key) {
    return new AutoValue_UnloadedToolchainContextImpl.Builder()
        .setKey(key)
        .setToolchainTypes(ImmutableSet.of())
        .setRequestedLabelToToolchainType(ImmutableMap.of())
        .setToolchainTypeToResolved(ImmutableSetMultimap.of())
        .setErrorData(null);
  }

  /** Builder class to help create the {@link UnloadedToolchainContextImpl}. */
  @AutoValue.Builder
  public interface Builder {
    /** Sets the key that identifies this context. */
    Builder setKey(ToolchainContextKey key);

    /** Sets the selected execution platform that these toolchains use. */
    Builder setExecutionPlatform(PlatformInfo executionPlatform);

    /** Sets the target platform that these toolchains generate output for. */
    Builder setTargetPlatform(PlatformInfo targetPlatform);

    /** Sets the toolchain types that were requested. */
    Builder setToolchainTypes(Set<ToolchainTypeRequirement> toolchainTypes);

    /**
     * Maps from the actual toolchain type to the resolved toolchain implementation that should be
     * used.
     */
    Builder setToolchainTypeToResolved(
        ImmutableSetMultimap<ToolchainTypeInfo, Label> toolchainTypeToResolved);

    /**
     * Maps from the actual requested {@link Label} to the discovered {@link ToolchainTypeInfo}.
     *
     * <p>Note that the key may be different from {@link ToolchainTypeInfo#typeLabel()} if the
     * requested {@link Label} is an {@code alias}.
     */
    Builder setRequestedLabelToToolchainType(
        ImmutableMap<Label, ToolchainTypeInfo> requestedLabelToToolchainType);

    /** Stores an exception that occurred during resolution of this toolchain. */
    Builder setErrorData(NoMatchingPlatformData errorData);

    UnloadedToolchainContextImpl build();
  }

  @Override
  public ImmutableSet<Label> resolvedToolchainLabels() {
    return toolchainTypeToResolved().inverse().keySet();
  }

  protected abstract Builder toBuilder();
}
