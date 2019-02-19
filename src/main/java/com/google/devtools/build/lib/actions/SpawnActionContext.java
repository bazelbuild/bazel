// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.common.collect.Iterables;
import java.util.List;

/**
 * A context that allows execution of {@link Spawn} instances.
 */
@ActionContextMarker(name = "spawn")
public interface SpawnActionContext extends ActionContext {

  /** Executes the given spawn and returns metadata about the execution. */
  List<SpawnResult> exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException;

  /**
   * Executes the given spawn, possibly asynchronously, and returns a FutureSpawn to represent the
   * execution, which can be listened to / registered with Skyframe.
   */
  default FutureSpawn execMaybeAsync(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    SpawnResult result = Iterables.getOnlyElement(exec(spawn, actionExecutionContext));
    return FutureSpawn.immediate(result);
  }
}
