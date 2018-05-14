// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel;

import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ActionContextConsumer;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.SpawnActionContextMaps;
import com.google.devtools.build.lib.exec.SpawnLogContext;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.AsynchronousFileOutputStream;
import java.io.IOException;

/**
 * Module providing on-demand spawn logging.
 */
public final class SpawnLogModule extends BlazeModule {

  private static final class SpawnLogContextConsumer implements ActionContextConsumer {
    @Override
    public void populate(SpawnActionContextMaps.Builder builder) {
      builder.strategyByContextMap().put(SpawnLogContext.class, "");
    }
  }

  private SpawnLogContext spawnLogContext;

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    env.getEventBus().register(this);
    ExecutionOptions executionOptions = env.getOptions().getOptions(ExecutionOptions.class);
    if (executionOptions != null
        && executionOptions.executionLogFile != null
        && !executionOptions.executionLogFile.isEmpty()) {
      try {
        spawnLogContext = new SpawnLogContext(
            env.getExecRoot(),
            new AsynchronousFileOutputStream(executionOptions.executionLogFile));
      } catch (IOException e) {
        env.getReporter().handle(Event.error(e.getMessage()));
        env.getBlazeModuleEnvironment().exit(new AbruptExitException(ExitCode.COMMAND_LINE_ERROR));
      }
      builder.addActionContext(spawnLogContext);
      builder.addActionContextConsumer(new SpawnLogContextConsumer());
    } else {
      spawnLogContext = null;
    }
  }

  @Override
  public void afterCommand() {
    if (spawnLogContext != null) {
      try {
        spawnLogContext.close();
      } catch (IOException e) {
        throw new RuntimeException(e);
      } finally {
        spawnLogContext = null;
      }
    }
  }
}
