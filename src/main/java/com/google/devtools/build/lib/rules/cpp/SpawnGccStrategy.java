// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;

import java.util.Collection;

/**
 * A cpp strategy that simply passes everything through to the default spawn action strategy.
 */
@ExecutionStrategy(
  contextType = CppCompileActionContext.class,
  name = {"spawn"}
)
public class SpawnGccStrategy implements CppCompileActionContext {

  @Override
  public boolean needsIncludeScanning() {
    return false;
  }

  @Override
  public Collection<Artifact> findAdditionalInputs(
      CppCompileAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    return null;
  }

  @Override
  public CppCompileActionContext.Reply execWithReply(
      CppCompileAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    SpawnActionContext spawnActionContext = executor.getSpawnActionContext(action.getMnemonic());
    Spawn spawn =
        new BaseSpawn(
            action.getArgv(),
            action.getEnvironment(),
            ImmutableMap.<String, String>of(),
            action,
            estimateResourceConsumption(action));
    spawnActionContext.exec(spawn, actionExecutionContext);
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
    return null;
  }
}
