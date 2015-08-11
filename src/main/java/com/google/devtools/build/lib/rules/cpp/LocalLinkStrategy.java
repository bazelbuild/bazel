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

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;

import java.util.List;

/**
 * A link strategy that runs the linking step on the local host.
 *
 * <p>The set of input files necessary to successfully complete the link is the middleman-expanded
 * set of the action's dependency inputs (which includes crosstool and libc dependencies, as
 * defined by {@link com.google.devtools.build.lib.rules.cpp.CppHelper#getCrosstoolInputsForLink
 * CppHelper.getCrosstoolInputsForLink}).
 */
@ExecutionStrategy(contextType = CppLinkActionContext.class, name = { "local" })
public final class LocalLinkStrategy extends LinkStrategy {

  public LocalLinkStrategy() {
  }

  @Override
  public void exec(CppLinkAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, ActionExecutionException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    List<String> argv = action.getCommandLine();
    executor.getSpawnActionContext(action.getMnemonic()).exec(
        new BaseSpawn.Local(argv, action.getEnvironment(), action),
        actionExecutionContext);
  }

  @Override
  public String linkStrategyName() {
    return "local";
  }

  @Override
  public ResourceSet estimateResourceConsumption(CppLinkAction action) {
    return action.estimateResourceConsumptionLocal();
  }
}
