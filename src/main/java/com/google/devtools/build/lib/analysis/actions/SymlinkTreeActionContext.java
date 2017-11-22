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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.SpawnResult;
import java.util.List;

/**
 * Action context for symlink tree actions (an action that creates a tree of symlinks).
 */
public interface SymlinkTreeActionContext extends ActionContext {

  /**
   * Creates the symlink tree.
   *
   * @return a list of SpawnResults created during symlink creation, if any
   */
  List<SpawnResult> createSymlinks(
      SymlinkTreeAction action,
      ActionExecutionContext actionExecutionContext,
      ImmutableMap<String, String> shellEnvironment,
      boolean enableRunfiles)
      throws ActionExecutionException, InterruptedException;
}
