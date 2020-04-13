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
package com.google.devtools.build.lib.analysis.actions;

import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.SpawnContinuation;

/**
 * The action context for {@link AbstractFileWriteAction} instances (technically instances of
 * subclasses).
 */
public interface FileWriteActionContext extends ActionContext {

  /**
   * Writes the output created by the {@link DeterministicWriter} to the sole output of the given
   * action.
   */
  SpawnContinuation beginWriteOutputToFile(
      AbstractAction action,
      ActionExecutionContext actionExecutionContext,
      DeterministicWriter deterministicWriter,
      boolean makeExecutable,
      boolean isRemotable)
      throws InterruptedException;
}
