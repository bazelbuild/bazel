// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.actions.Executor.ActionContext;

/**
 * An object that provides execution strategies to {@link BlazeExecutor}.
 *
 * <p>For more information, see {@link ActionContextConsumer}.
 */
public abstract class ActionContextProvider {
  /**
   * Returns the execution strategies that are provided by this object.
   *
   * <p>These may or may not actually end up in the executor depending on the command line options
   * and other factors influencing how the executor is set up.
   */
  public abstract Iterable<ActionContext> getActionContexts();

  /**
   * Called when the executor is constructed. The parameter contains all the contexts that were
   * selected for this execution phase.
   */
  public void executorCreated(Iterable<ActionContext> usedContexts) throws ExecutorInitException {}

  /**
   * Called when the execution phase is started.
   */
  public void executionPhaseStarting(
      ActionInputFileCache actionInputFileCache,
      ActionGraph actionGraph,
      Iterable<Artifact> topLevelArtifacts)
      throws ExecutorInitException, InterruptedException {}

  /**
   * Called when the execution phase is finished.
   */
  public void executionPhaseEnding() {}
}
