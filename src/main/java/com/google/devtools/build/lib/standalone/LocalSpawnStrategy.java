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
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Strategy that uses subprocessing to execute a process.
 */
@ExecutionStrategy(name = { "standalone" }, contextType = SpawnActionContext.class)
public class LocalSpawnStrategy implements SpawnActionContext {
  private final boolean verboseFailures;

  private final Path processWrapper;

  public LocalSpawnStrategy(Path execRoot, boolean verboseFailures) {
    this.verboseFailures = verboseFailures;
    this.processWrapper = execRoot.getRelative(
        "_bin/process-wrapper" + OsUtils.executableExtension());
  }

  /**
   * Executes the given {@code spawn}.
   */
  @Override
  public void exec(Spawn spawn,
      ActionExecutionContext actionExecutionContext)
      throws ExecException {
    Executor executor = actionExecutionContext.getExecutor();
    if (executor.reportsSubcommands()) {
      executor.reportSubcommand(Label.print(spawn.getOwner().getLabel()),
          spawn.asShellCommand(executor.getExecRoot()));
    }

    // We must wrap the subprocess with process-wrapper to kill the process tree.
    // All actions therefore depend on the process-wrapper file. Since it's embedded,
    // we don't bother with declaring it as an input.
    List<String> args = new ArrayList<>();
    if (OS.getCurrent() != OS.WINDOWS) {
      // TODO(bazel-team): process-wrapper seems to work on Windows, but requires
      // additional setup as it is an msys2 binary, so it needs msys2 DLLs on %PATH%.
      // Disable it for now to make the setup easier and to avoid further PATH hacks.
      // Ideally we should have a native implementation of process-wrapper for Windows.
      args.add(processWrapper.getPathString());
      args.add("-1"); /* timeout */
      args.add("0");  /* kill delay. */

      // TODO(bazel-team): use process-wrapper redirection so we don't have to
      // pass test logs through the Java heap.
      args.add("-");  /* stdout. */
      args.add("-");  /* stderr. */
    }
    args.addAll(spawn.getArguments());

    String cwd = executor.getExecRoot().getPathString();
    Command cmd = new Command(args.toArray(new String[]{}), spawn.getEnvironment(), new File(cwd));

    FileOutErr outErr = actionExecutionContext.getFileOutErr();
    try {
      cmd.execute(
          /* stdin */ new byte[]{},
          Command.NO_OBSERVER,
          outErr.getOutputStream(),
          outErr.getErrorStream(),
          /*killSubprocessOnInterrupt*/ true);
    } catch (CommandException e) {
      String message = CommandFailureUtils.describeCommandFailure(
          verboseFailures, spawn.getArguments(), spawn.getEnvironment(), cwd);
      throw new UserExecException(String.format("%s: %s", message, e));
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
