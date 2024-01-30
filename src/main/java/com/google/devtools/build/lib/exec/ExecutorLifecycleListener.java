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
package com.google.devtools.build.lib.exec;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.skyframe.EphemeralCheckIfOutputConsumed;
import com.google.devtools.build.lib.util.AbruptExitException;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Type that can get informed about executor lifecycle events.
 *
 * <p>Notifications occur in this order:
 *
 * <ol>
 *   <li>{@link #executorCreated}
 *   <li>{@link #executionPhaseStarting}
 *   <li>{@link #executionPhaseEnding}
 * </ol>
 */
public interface ExecutorLifecycleListener {

  /** Handles executor creation. */
  void executorCreated() throws AbruptExitException;

  /**
   * Handles the start of the execution phase.
   *
   * @param actionGraph actions as calculated in the analysis phase. Null in Skymeld mode.
   * @param topLevelArtifacts supplies all output artifacts from top-level targets and aspects. Null
   *     in skymeld mode.
   * @param ephemeralCheckIfOutputConsumed tests whether an artifact is consumed in this build.
   */
  void executionPhaseStarting(
      @Nullable ActionGraph actionGraph,
      @Nullable Supplier<ImmutableSet<Artifact>> topLevelArtifacts,
      @Nullable EphemeralCheckIfOutputConsumed ephemeralCheckIfOutputConsumed)
      throws AbruptExitException, InterruptedException;

  /** Handles the end of the execution phase. */
  void executionPhaseEnding();
}
