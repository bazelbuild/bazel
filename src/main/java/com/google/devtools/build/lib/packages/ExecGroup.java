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

import static com.google.common.base.Preconditions.checkArgument;
import static java.util.Objects.requireNonNull;

import com.google.auto.value.AutoBuilder;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.ExecGroupApi;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Identifier;

/**
 * Resolves the appropriate toolchains for the given parameters.
 *
 * @param toolchainTypesMap Returns the underlying map from label to ToolchainTypeRequirement.
 * @param execCompatibleWith Returns the execution constraints for this exec group.
 * @param copyFromDefault Whether this exec group should copy the data from the default exec group
 *     in the same rule.
 */
@AutoCodec
public record ExecGroup(
    ImmutableMap<Label, ToolchainTypeRequirement> toolchainTypesMap,
    ImmutableSet<Label> execCompatibleWith,
    boolean copyFromDefault)
    implements ExecGroupApi {
  public ExecGroup {
    requireNonNull(toolchainTypesMap, "toolchainTypesMap");
    requireNonNull(execCompatibleWith, "execCompatibleWith");
    checkArgument(
        !copyFromDefault || (toolchainTypesMap.isEmpty() && execCompatibleWith.isEmpty()));
  }

  // This is intentionally a string that would fail {@code Identifier.isValid} so that
  // users can't create a group with the same name.
  public static final String DEFAULT_EXEC_GROUP_NAME = "default-exec-group";

  /** An exec group that copies all data from the default exec group. */
  public static final ExecGroup COPY_FROM_DEFAULT = builder().copyFromDefault(true).build();

  /** Returns a builder for a new ExecGroup. */
  public static Builder builder() {
    return new AutoBuilder_ExecGroup_Builder()
        .copyFromDefault(false)
        .toolchainTypes(ImmutableSet.of())
        .execCompatibleWith(ImmutableSet.of());
  }

  /** Returns true if the given exec group is an automatic exec group. */
  public static boolean isAutomatic(String execGroupName) {
    return !Identifier.isValid(execGroupName) && !execGroupName.equals(DEFAULT_EXEC_GROUP_NAME);
  }

  /** Returns the required toolchain types for this exec group. */
  public ImmutableSet<ToolchainTypeRequirement> toolchainTypes() {
    return ImmutableSet.copyOf(toolchainTypesMap().values());
  }

  @Nullable
  public ToolchainTypeRequirement toolchainType(Label label) {
    return toolchainTypesMap().get(label);
  }

  public Builder toBuilder() {
    return new AutoBuilder_ExecGroup_Builder(this);
  }

  /**
   * Prepares the input exec groups.
   *
   * <p>Adds auto exec groups when {@code useAutoExecGroups} is true.
   */
  public static ImmutableMap<String, ExecGroup> process(
      ImmutableMap<String, ExecGroup> execGroups,
      ImmutableSet<Label> defaultExecWith,
      ImmutableMultimap<String, Label> execGroupExecWith,
      ImmutableSet<ToolchainTypeRequirement> defaultToolchainTypes,
      boolean useAutoExecGroups) {
    var processedGroups =
        ImmutableMap.<String, ExecGroup>builderWithExpectedSize(
            useAutoExecGroups
                ? (execGroups.size() + defaultToolchainTypes.size())
                : execGroups.size());
    for (Map.Entry<String, ExecGroup> entry : execGroups.entrySet()) {
      String name = entry.getKey();
      ExecGroup execGroup = entry.getValue();

      if (execGroup.copyFromDefault()) {
        execGroup =
            ExecGroup.builder()
                .execCompatibleWith(defaultExecWith)
                .toolchainTypes(defaultToolchainTypes)
                .build();
      }
      ImmutableCollection<Label> extraExecWith = execGroupExecWith.get(name);
      if (!extraExecWith.isEmpty()) {
        execGroup =
            execGroup.toBuilder()
                .execCompatibleWith(
                    ImmutableSet.<Label>builder()
                        .addAll(execGroup.execCompatibleWith())
                        .addAll(extraExecWith)
                        .build())
                .build();
      }

      processedGroups.put(name, execGroup);
    }

    if (useAutoExecGroups) {
      // Creates one exec group for each toolchain (automatic exec groups).
      for (ToolchainTypeRequirement toolchainType : defaultToolchainTypes) {
        ImmutableSet<Label> execCompatibleWith = defaultExecWith;
        ImmutableCollection<Label> extraExecWith =
            execGroupExecWith.get(toolchainType.toolchainType().getUnambiguousCanonicalForm());
        if (!extraExecWith.isEmpty()) {
          execCompatibleWith =
              ImmutableSet.<Label>builder().addAll(defaultExecWith).addAll(extraExecWith).build();
        }
        processedGroups.put(
            toolchainType.toolchainType().toString(),
            ExecGroup.builder()
                .addToolchainType(toolchainType)
                .execCompatibleWith(execCompatibleWith)
                .build());
      }
    }
    return processedGroups.buildOrThrow();
  }

  /** A builder interface to create ExecGroup instances. */
  @AutoBuilder
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

    /** Do not call, internal usage only. */
    Builder copyFromDefault(boolean copyFromDefault);

    /** Returns the new ExecGroup instance. */
    ExecGroup build();
  }
}
