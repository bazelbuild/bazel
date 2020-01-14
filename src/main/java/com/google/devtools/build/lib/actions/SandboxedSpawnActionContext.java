// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableList;
import javax.annotation.Nullable;

/**
 * A context that allows execution of {@link Spawn} instances similar to {@link SpawnActionContext},
 * but with the additional restriction that during execution the {@link Spawn} must not be allowed
 * to modify the current execution root of the build. Instead, the {@link Spawn} should be executed
 * in a sandbox or on a remote system and its output files only be moved to the execution root.
 */
public interface SandboxedSpawnActionContext extends SpawnActionContext {

  /** Lambda interface to stop other instances of the same spawn before writing outputs. */
  @FunctionalInterface
  interface StopConcurrentSpawns {
    /**
     * Stops other instances of the same spawn before writing outputs.
     *
     * <p>This should be called once by each of the concurrent spawns to ensure that the others are
     * stopped, thus preventing conflicts when writing to the output tree.
     */
    void stop() throws InterruptedException;
  }

  /**
   * Executes the given spawn.
   *
   * <p>When the {@link SpawnActionContext} is about to write output files into the execroot, it
   * first asks any other concurrent instances of this same spawn (handled by other spawn runners
   * when dynamic scheduling is enabled) to stop by invoking the {@code stopConcurrentSpawns}
   * lambda.
   *
   * @return a List of {@link SpawnResult}s containing metadata about the Spawn's execution. This
   *     will typically contain one element, but could contain no elements if spawn execution did
   *     not complete, or contain multiple elements if multiple sub-spawns were executed
   */
  ImmutableList<SpawnResult> exec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      // TODO(jmmv): Inject an empty lambda instead of allowing this to be null. Need to find a way
      // to deal with non-null implying speculation in e.g. AbstractSpawnStrategy (if it matters).
      @Nullable StopConcurrentSpawns stopConcurrentSpawns)
      throws ExecException, InterruptedException;
}
