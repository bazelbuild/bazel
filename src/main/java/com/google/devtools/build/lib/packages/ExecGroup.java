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

package com.google.devtools.build.lib.packages;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.starlarkbuildapi.ExecGroupApi;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import javax.annotation.Nullable;

/** Resolves the appropriate toolchains for the given parameters. */
@AutoValue
public abstract class ExecGroup implements ExecGroupApi {

  // This is intentionally a string that would fail {@code Identifier.isValid} so that
  // users can't create a group with the same name.
  public static final String DEFAULT_EXEC_GROUP_NAME = "default-exec-group";

  /** Returns a builder for a new ExecGroup. */
  public static Builder builder() {
    return new AutoValue_ExecGroup.Builder()
        .toolchainTypes(ImmutableSet.of())
        .execCompatibleWith(ImmutableSet.of());
  }

  /** Create an exec group that inherits from the default exec group. */
  public static ExecGroup copyFromDefault() {
    return builder().copyFrom(DEFAULT_EXEC_GROUP_NAME).build();
  }

  /** Returns the required toolchain types for this exec group. */
  public ImmutableSet<ToolchainTypeRequirement> toolchainTypes() {
    return ImmutableSet.copyOf(toolchainTypesMap().values());
  }

  @Nullable
  public ToolchainTypeRequirement toolchainType(Label label) {
    return toolchainTypesMap().get(label);
  }

  /** Returns the underlying map from label to ToolchainTypeRequirement. */
  public abstract ImmutableMap<Label, ToolchainTypeRequirement> toolchainTypesMap();

  /** Returns the execution constraints for this exec group. */
  public abstract ImmutableSet<Label> execCompatibleWith();

  /** Returns the name of another exec group in the same rule to copy data from. */
  @Nullable
  public abstract String copyFrom();

  /** Creates a new exec group that inherits from the given group and this group. */
  public ExecGroup inheritFrom(ExecGroup other) {
    Builder builder = builder().copyFrom(null);
    builder.toolchainTypesMapBuilder().putAll(this.toolchainTypesMap());
    builder.toolchainTypesMapBuilder().putAll(other.toolchainTypesMap());

    builder.execCompatibleWith(
        new ImmutableSet.Builder<Label>()
            .addAll(this.execCompatibleWith())
            .addAll(other.execCompatibleWith())
            .build());

    return builder.build();
  }

  /** A builder interface to create ExecGroup instances. */
  @AutoValue.Builder
  public interface Builder {

    /** Sets the toolchain type requirements. */
    @CanIgnoreReturnValue
    default Builder toolchainTypes(ImmutableSet<ToolchainTypeRequirement> toolchainTypes) {
      toolchainTypes.forEach(this::addToolchainType);
      return this;
    }

    ImmutableMap.Builder<Label, ToolchainTypeRequirement> toolchainTypesMapBuilder();

    @CanIgnoreReturnValue
    default Builder addToolchainType(ToolchainTypeRequirement toolchainTypeRequirement) {
      this.toolchainTypesMapBuilder()
          .put(toolchainTypeRequirement.toolchainType(), toolchainTypeRequirement);
      return this;
    }

    /** Sets the execution constraints. */
    Builder execCompatibleWith(ImmutableSet<Label> execCompatibleWith);

    /** Sets the name of another exec group in the same rule to copy from. */
    Builder copyFrom(@Nullable String copyfrom);

    /** Returns the new ExecGroup instance. */
    ExecGroup build();
  }
}
