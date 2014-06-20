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
package com.google.devtools.build.lib.standalone;

import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.KillableObserver;
import com.google.devtools.build.lib.shell.TimeoutKillableObserver;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.io.FileOutErr;

import java.io.File;

/**
 * Strategy that uses subprocessing to execute a process.
 */
@ExecutionStrategy(name = { "standalone" }, contextType = SpawnActionContext.class)
public class LocalSpawnStrategy implements SpawnActionContext {
  /**
   * Executes the given {@code spawn}.
   */
  @Override
  public void exec(Spawn spawn,
      ActionExecutionContext actionExecutionContext)
      throws ExecException {
    Executor executor = actionExecutionContext.getExecutor();
    executor.reportSubcommand(Label.print(spawn.getOwner().getLabel()),
        spawn.asShellCommand(executor.getExecRoot()));
    String[] args = spawn.getArguments().toArray(new String[]{});
    Command cmd = new Command(args, spawn.getEnvironment(),
        new File(executor.getExecRoot().getPathString()));

    // TODO(bazel-team): figure out how to support test timeouts.
    double timeoutSecs = -1.0;
    KillableObserver observer =
        (timeoutSecs > 0) ? new TimeoutKillableObserver((long) (timeoutSecs * 1000.0)) :
        Command.NO_OBSERVER;

    FileOutErr outErr = actionExecutionContext.getFileOutErr();
    try {
      cmd.execute(
          /* stdin */ new byte[]{},
          observer,
          outErr.getOutputStream(),
          outErr.getErrorStream(),
          /*killSubprocessOnInterrupt*/ true);
    } catch (CommandException e) {
      throw new UserExecException(String.format("Command.execute(%s): ", (Object[]) args), e);
    }
  }

  @Override
  public String strategyLocality(String mnemonic, boolean remotable) {
    return "standalone";
  }

  @Override
  public boolean isRemotable(String mnemonic, boolean remotable) {
    return false;
  }
}
