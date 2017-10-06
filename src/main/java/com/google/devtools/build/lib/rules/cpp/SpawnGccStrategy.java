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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.UserExecException;
import java.util.Set;

/**
 * A context for C++ compilation that calls into a {@link SpawnActionContext}.
 */
@ExecutionStrategy(
  contextType = CppCompileActionContext.class,
  name = {"spawn"}
)
public class SpawnGccStrategy implements CppCompileActionContext {
  @Override
  public Iterable<Artifact> findAdditionalInputs(
      CppCompileAction action,
      ActionExecutionContext actionExecutionContext,
      IncludeProcessing includeProcessing)
      throws ExecException, InterruptedException {
    return null;
  }

  @Override
  public CppCompileActionResult execWithReply(
      CppCompileAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    if (action.getDotdFile() != null && action.getDotdFile().artifact() == null) {
      throw new UserExecException("cannot execute remotely or locally: "
          + action.getPrimaryInput().getExecPathString());
    }
    Iterable<Artifact> inputs = Iterables.concat(action.getInputs(), action.getAdditionalInputs());
    Spawn spawn = new SimpleSpawn(
        action,
        ImmutableList.copyOf(action.getArgv()),
        ImmutableMap.copyOf(action.getEnvironment()),
        ImmutableMap.copyOf(action.getExecutionInfo()),
        EmptyRunfilesSupplier.INSTANCE,
        ImmutableList.<Artifact>copyOf(inputs),
        /*tools=*/ImmutableList.<Artifact>of(),
        /*filesetManifests=*/ImmutableList.<Artifact>of(),
        action.getOutputs().asList(),
        action.estimateResourceConsumptionLocal());

    Set<SpawnResult> spawnResults =
        actionExecutionContext
            .getSpawnActionContext(action.getMnemonic())
            .exec(spawn, actionExecutionContext);
    return CppCompileActionResult.builder().setSpawnResults(spawnResults).build();
  }
}
