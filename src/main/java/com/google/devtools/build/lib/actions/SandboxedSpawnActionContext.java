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

import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A context that allows execution of {@link Spawn} instances similar to {@link SpawnActionContext},
 * but with the additional restriction, that during execution the {@link Spawn} must not be allowed
 * to modify the current execution root of the build. Instead, the {@link Spawn} should be executed
 * in a sandbox or on a remote system and its output files only be moved to the execution root, if
 * the implementation is able to {@code compareAndSet} the {@link AtomicReference} that is passed to
 * the {@link #exec} method to its own class object (e.g. LinuxSandboxedStrategy.class).
 *
 * <p>If the {@code compareAndSet} fails, the Spawn strategy should abandon the output of its
 * execution and throw an {@link InterruptedException} from its {@code exec} method.
 */
public interface SandboxedSpawnActionContext extends SpawnActionContext {

  /**
   * Executes the given spawn.
   *
   * <p>When the {@link SpawnActionContext} is about to move the output files of the spawn out of
   * the sandbox into the execroot, it has to first verify that the {@link AtomicReference} is still
   * null or already set to a value uniquely identifying the current {@link SpawnActionContext}
   * (e.g. the class object of the strategy). This is to ensure that in case multiple {@link
   * SandboxedSpawnActionContext} instances are processing the {@link Spawn} in parallel that only
   * one strategy actually generates the output files.
   *
   * <p>If the {@link AtomicReference} is not null (thus {@code #compareAndSet} fails) and not set
   * to the unique reference of the strategy, the {@link SandboxedSpawnActionContext} should abandon
   * all results and raise {@link InterruptedException}.
   *
   * @return a Set of {@link SpawnResult}s containing metadata about the Spawn's execution. This
   *   will typically contain one element, but could contain no elements if spawn execution did not
   *   complete, or contain multiple elements if multiple sub-spawns were executed
   */
  Set<SpawnResult> exec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws ExecException, InterruptedException;
}
