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
import java.util.List;

/** A context for C++ compilation that calls into a {@link SpawnActionContext}. */
@ExecutionStrategy(
    contextType = CppCompileActionContext.class,
    name = {"spawn"})
public class SpawnGccStrategy implements CppCompileActionContext {
  @Override
  public CppCompileActionResult execWithReply(
      CppCompileAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {

    Iterable<Artifact> inputs =
        Iterables.concat(
            /**
             * Intentionally not adding {@link CppCompileAction#inputsForInvalidation}, those are
             * not needed for execution.
             */
            action.getMandatoryInputs(), action.getAdditionalInputs());
    Spawn spawn =
        new SimpleSpawn(
            action,
            ImmutableList.copyOf(action.getArguments()),
            ImmutableMap.copyOf(action.getEnvironment(actionExecutionContext.getClientEnv())),
            ImmutableMap.copyOf(action.getExecutionInfo()),
            EmptyRunfilesSupplier.INSTANCE,
            ImmutableMap.of(),
            ImmutableList.copyOf(inputs),
            /* tools= */ ImmutableList.of(),
            action.getOutputs().asList(),
            action.estimateResourceConsumptionLocal());

    if (action.getDotdFile() != null) {
      // .d file scanning happens locally even if the compiler was run remotely. We thus need
      // to ensure that the .d file is staged on the local fileystem.
      actionExecutionContext =
          actionExecutionContext.withRequiredLocalOutputs(
              ImmutableList.of(action.getDotdFile().artifact()));
    }

    List<SpawnResult> spawnResults =
        actionExecutionContext
            .getContext(SpawnActionContext.class)
            .exec(spawn, actionExecutionContext);
    return CppCompileActionResult.builder().setSpawnResults(spawnResults).build();
  }
}
