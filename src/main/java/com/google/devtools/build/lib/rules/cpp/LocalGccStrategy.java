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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.common.options.OptionsClassProvider;

import java.util.Collection;

/**
 * Run gcc locally by delegating to spawn.
 */
@ExecutionStrategy(name = { "local" },
          contextType = CppCompileActionContext.class)
public class LocalGccStrategy implements CppCompileActionContext {
  private static final Reply CANNED_REPLY = new Reply() {
    @Override
    public String getContents() {
      throw new IllegalStateException("Remotely computed data requested for local action");
    }
  };

  public LocalGccStrategy(OptionsClassProvider options) {
  }

  @Override
  public String strategyLocality() {
    return "local";
  }

  @Override
  public boolean needsIncludeScanning() {
    return false;
  }

  @Override
  public CppCompileActionContext.Reply execWithReply(
      CppCompileAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    actionExecutionContext.getExecutor().getSpawnActionContext(action.getMnemonic())
        .exec(new BaseSpawn.Local(action.getArgv(), action.getEnvironment(), action),
            actionExecutionContext);
    return null;
  }

  @Override
  public ResourceSet estimateResourceConsumption(CppCompileAction action) {
    return action.estimateResourceConsumptionLocal();
  }

  @Override
  public Collection<String> getScannedIncludeFiles(
      CppCompileAction action, ActionExecutionContext actionExecutionContext) {
    return ImmutableList.of();
  }

  @Override
  public Reply getReplyFromException(ExecException e, CppCompileAction action) {
    return CANNED_REPLY;
  }
}
