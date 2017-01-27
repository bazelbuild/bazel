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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
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
 * A context for C++ compilation that calls into a {@link SpawnActionContext}.
 */
@ExecutionStrategy(
  contextType = CppCompileActionContext.class,
  name = {"spawn"}
)
public class SpawnGccStrategy implements CppCompileActionContext {

  /**
   * A {@link Spawn} that wraps a {@link CppCompileAction} and adds its
   * {@code additionalInputs} (potential files included) to its inputs.
   */
  private static class GccSpawn extends BaseSpawn {
    private final Iterable<? extends ActionInput> inputs;

    public GccSpawn(CppCompileAction action, ResourceSet resources) {
      super(action.getArgv(), action.getEnvironment(), action.getExecutionInfo(), action,
          resources);
      this.inputs = Iterables.concat(action.getInputs(), action.getAdditionalInputs());
    }

    @Override
    public Iterable<? extends ActionInput> getInputFiles() {
      return ImmutableSet.copyOf(inputs);
    }
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
    Spawn spawn = new GccSpawn(action, estimateResourceConsumption(action));
    spawnActionContext.exec(spawn, actionExecutionContext);
    return null;
  }

  @Override
  public ResourceSet estimateResourceConsumption(CppCompileAction action) {
    return action.estimateResourceConsumptionLocal();
  }

  @Override
  public Reply getReplyFromException(ExecException e, CppCompileAction action) {
    return null;
  }
}
