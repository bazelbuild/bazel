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

import com.google.common.base.Supplier;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.analysis.ArtifactsToOwnerLabels;
import com.google.devtools.build.lib.util.AbruptExitException;

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
   * @param actionGraph actions as calcuated in the analysis phase
   * @param topLevelArtifactsToOwnerLabels supplier of all output artifacts from top-level targets
   *     and aspects, mapped to their owners
   */
  void executionPhaseStarting(
      ActionGraph actionGraph, Supplier<ArtifactsToOwnerLabels> topLevelArtifactsToOwnerLabels)
      throws AbruptExitException, InterruptedException;

  /** Handles the end of the execution phase. */
  void executionPhaseEnding();
}
