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


/**
 * A context that allows execution of {@link Spawn} instances.
 */
@ActionContextMarker(name = "spawn")
public interface SpawnActionContext extends Executor.ActionContext {

  /** Executes the given spawn. */
  void exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException;

  /**
   * This implements a tri-state mode. There are three possible cases: (1) implementations of this
   * class can unconditionally execute spawns locally, (2) they can follow whatever is set for the
   * corresponding spawn (see {@link Spawn#isRemotable}), or (3) they can unconditionally execute
   * spawns remotely, i.e., force remote execution.
   *
   * <p>Passing the spawns remotable flag to this method returns whether the spawn will actually be
   * executed remotely.
   */
  boolean isRemotable(String mnemonic, boolean remotable);
}
