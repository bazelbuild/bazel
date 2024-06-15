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

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.auto.value.AutoBuilder;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.skyframe.SkyKey;
import javax.annotation.Nullable;

/**
 * An immutable record of the state of Skyfocus. This is recorded as a member in {@link
 * SkyframeExecutor}.
 *
 * @param enabled If true, Skyfocus may run at the end of the build, depending on the state of the
 *     graph and working set conditions.
 * @param forcedRerun If true, Skyfocus will always run at the end of the build, regardless of the
 *     state of working set or the graph.
 * @param focusedTargetLabels The set of targets focused in this server instance
 * @param workingSet Files/dirs representing the working set. Can be empty, specified by the command
 *     line flag, or automatically derived. Although the working set is represented as {@link
 *     FileStateKey}, the presence of a directory path's {@code FileStateKey} is sufficient to
 *     represent the corresponding directory listing state node.
 * @param verificationSet The set of files/dirs that are not in the working set, but is in the
 *     transitive closure of focusedTargetLabels.
 * @param options The latest instance of {@link SkyfocusOptions}.
 * @param buildConfiguration The latest top level build configuration.
 */
public record SkyfocusState(
    boolean enabled,
    boolean forcedRerun,
    ImmutableSet<Label> focusedTargetLabels,
    WorkingSetType workingSetType,
    ImmutableSet<FileStateKey> workingSet,
    ImmutableSet<SkyKey> verificationSet,
    @Nullable SkyfocusOptions options,
    @Nullable BuildConfigurationValue buildConfiguration) {

  /**
   * Builder for the {@code SkyfocusState} record.
   *
   * <p>This must reflect all parameters in the record constructor.
   */
  @AutoBuilder
  public interface Builder {
    Builder enabled(boolean enable);

    Builder forcedRerun(boolean forcedRerun);

    Builder focusedTargetLabels(ImmutableSet<Label> focusedTargetLabels);

    Builder workingSetType(WorkingSetType workingSetType);

    Builder workingSet(ImmutableSet<FileStateKey> workingSet);

    Builder verificationSet(ImmutableSet<SkyKey> verificationSet);

    Builder options(@Nullable SkyfocusOptions options);

    Builder buildConfiguration(@Nullable BuildConfigurationValue buildConfiguration);

    SkyfocusState build();
  }

  public Builder toBuilder() {
    return new AutoBuilder_SkyfocusState_Builder(this);
  }

  /** Describes how the working set was constructed. */
  public enum WorkingSetType {
    /** Automatically derived by the source state and the command line (e.g. focused targets) */
    DERIVED,

    /** The value of --experimental_working_set. Will override derived sets if used. */
    USER_DEFINED
  }

  /** The canonical state to completely disable Skyfocus in the build. */
  public static final SkyfocusState DISABLED =
      new SkyfocusState(
          false,
          false,
          ImmutableSet.of(),
          WorkingSetType.DERIVED,
          ImmutableSet.of(),
          ImmutableSet.of(),
          null,
          null);

  public ImmutableSet<String> workingSetStrings() {
    return workingSet.stream()
        .map(fsk -> fsk.argument().getRootRelativePath().toString())
        .collect(toImmutableSet());
  }

}
