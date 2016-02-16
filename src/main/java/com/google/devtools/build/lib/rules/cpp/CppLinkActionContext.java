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
package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.ActionContextMarker;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.ResourceSet;

/**
 * Context for executing {@link CppLinkAction}s.
 */
@ActionContextMarker(name = "C++ link")
public interface CppLinkActionContext extends ActionContext {

  /**
   * Returns the estimated resource consumption of the action.
   */
  ResourceSet estimateResourceConsumption(CppLinkAction action);

  /**
   * Executes the specified action.
   */
  void exec(CppLinkAction action,
      ActionExecutionContext actionExecutionContext)
      throws ExecException, ActionExecutionException, InterruptedException;
}
