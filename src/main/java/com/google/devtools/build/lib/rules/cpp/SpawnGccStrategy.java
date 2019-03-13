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
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

/**
 * A context for C++ compilation that calls into a {@link SpawnActionContext}.
 */
@ExecutionStrategy(
  contextType = CppCompileActionContext.class,
  name = {"spawn"}
)
public class SpawnGccStrategy implements CppCompileActionContext {
  private static class InMemoryFile implements CppCompileActionContext.Reply {
    private final byte[] contents;

    InMemoryFile(byte[] contents) {
      this.contents = contents;
    }

    @Override
    public byte[] getContents() {
      return contents;
    }
  }

  @Override
  public CppCompileActionResult execWithReply(
      CppCompileAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    // Intentionally not adding {@link CppCompileAction#inputsForInvalidation}, those are not needed
    // for execution.
    ImmutableList<ActionInput> inputs =
        new ImmutableList.Builder<ActionInput>()
            .addAll(action.getMandatoryInputs())
            .addAll(action.getAdditionalInputs())
            .build();
    action.clearAdditionalInputs();

    ImmutableMap<String, String> executionInfo = action.getExecutionInfo();
    ImmutableList.Builder<ActionInput> outputs =
        ImmutableList.builderWithExpectedSize(action.getOutputs().size() + 1);
    outputs.addAll(action.getOutputs());
    if (action.getDotdFile() != null && action.useInMemoryDotdFiles()) {
      outputs.add(action.getDotdFile());
      /*
       * CppCompileAction does dotd file scanning locally inside the Bazel process and thus
       * requires the dotd file contents to be available locally. In remote execution, we
       * generally don't want to stage all remote outputs on the local file system and thus
       * we need to tell the remote strategy (if any) to at least make the .d file available
       * in-memory. We can do that via
       * {@link ExecutionRequirements.REMOTE_EXECUTION_INLINE_OUTPUTS}.
       */
      executionInfo =
          ImmutableMap.<String, String>builderWithExpectedSize(executionInfo.size() + 1)
              .putAll(executionInfo)
              .put(
                  ExecutionRequirements.REMOTE_EXECUTION_INLINE_OUTPUTS,
                  action.getDotdFile().getExecPathString())
              .build();
    }

    Spawn spawn =
        new SimpleSpawn(
            action,
            ImmutableList.copyOf(action.getArguments()),
            action.getEnvironment(actionExecutionContext.getClientEnv()),
            executionInfo,
            inputs,
            outputs.build(),
            action.estimateResourceConsumptionLocal());

    SpawnActionContext context = actionExecutionContext.getContext(SpawnActionContext.class);
    List<SpawnResult> spawnResults = context.exec(spawn, actionExecutionContext);
    // The SpawnActionContext guarantees that the first list entry is the successful one.
    SpawnResult spawnResult = spawnResults.get(0);

    CppCompileActionResult.Builder cppCompileActionResultBuilder =
        CppCompileActionResult.builder().setSpawnResults(spawnResults);

    if (action.getDotdFile() != null) {
      InputStream in = spawnResult.getInMemoryOutput(action.getDotdFile());
      if (in != null) {
        byte[] contents;
        try {
          contents = ByteStreams.toByteArray(in);
        } catch (IOException e) {
          throw new EnvironmentalExecException("Reading in-memory .d file failed", e);
        }
        cppCompileActionResultBuilder.setContextReply(new InMemoryFile(contents));
      }
    }
    return cppCompileActionResultBuilder.build();
  }
}
