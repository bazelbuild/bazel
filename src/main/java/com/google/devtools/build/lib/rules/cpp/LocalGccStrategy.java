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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.ResourceSet;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Run gcc locally by delegating to spawn.
 */
@ExecutionStrategy(name = { "local" },
          contextType = CppCompileActionContext.class)
public class LocalGccStrategy implements CppCompileActionContext {
  private static final Reply CANNED_REPLY = new Reply() {
    @Override
    public byte[] getContents() {
      throw new IllegalStateException("Remotely computed data requested for local action");
    }
  };

  @Override
  public String strategyLocality() {
    return "local";
  }

  public static void updateEnv(CppCompileAction action, Map<String, String> env) {
    // We cannot locally execute an action that does not expect to output a .d file, since we would
    // have no way to tell what files that it included were used during compilation.
    env.put("INTERCEPT_LOCALLY_EXECUTABLE", action.getDotdFile().artifact() == null ? "0" : "1");
  }

  @Override
  public boolean needsIncludeScanning() {
    return false;
  }

  @Override
  public Collection<Artifact> findAdditionalInputs(CppCompileAction action,
      ActionExecutionContext actionExecutionContext) throws ExecException, InterruptedException {
    return null;
  }

  @Override
  public CppCompileActionContext.Reply execWithReply(
      CppCompileAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    Map<String, String> env = new HashMap<>();
    env.putAll(action.getEnvironment());
    updateEnv(action, env);
    actionExecutionContext.getExecutor().getSpawnActionContext(action.getMnemonic())
        .exec(new BaseSpawn.Local(action.getArgv(), env, action),
            actionExecutionContext);
    return null;
  }

  @Override
  public ResourceSet estimateResourceConsumption(CppCompileAction action) {
    return action.estimateResourceConsumptionLocal();
  }

  @Override
  public Collection<Artifact> getScannedIncludeFiles(
      CppCompileAction action, ActionExecutionContext actionExecutionContext) {
    return ImmutableList.of();
  }

  @Override
  public Reply getReplyFromException(ExecException e, CppCompileAction action) {
    return CANNED_REPLY;
  }
}
