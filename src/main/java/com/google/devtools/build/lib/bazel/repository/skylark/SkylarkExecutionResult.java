// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.repository.skylark;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.shell.BadExitStatusException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;

import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * A Skylark structure
 */
@SkylarkModule(
  name = "exec_result",
  doc =
      "A structure storing result of repository_ctx.execute() method. It contains the standard"
          + " output stream content, the standard error stream content and the execution return"
          + " code."
)
public class SkylarkExecutionResult {
  private final int returnCode;
  private final String stdout;
  private final String stderr;

  private SkylarkExecutionResult(int returnCode, String stdout, String stderr) {
    this.returnCode = returnCode;
    this.stdout = stdout;
    this.stderr = stderr;
  }

  private SkylarkExecutionResult(CommandResult result) {
    // TODO(dmarting): if a lot of data is sent to stdout, this will use all the memory and
    // Bazel will crash. Maybe we should use custom output streams that throw an appropriate
    // exception when reaching a specific size.
    this.stdout = new String(result.getStdout(), StandardCharsets.UTF_8);
    this.stderr = new String(result.getStderr(), StandardCharsets.UTF_8);
    this.returnCode = result.getTerminationStatus().getExitCode();
  }

  @SkylarkCallable(
    name = "return_code",
    structField = true,
    doc = "The return code returned after the execution of the program."
  )
  public int returnCode() {
    return returnCode;
  }

  @SkylarkCallable(
    name = "stdout",
    structField = true,
    doc = "The content of the standard output returned by the execution."
  )
  public String stdout() {
    return stdout;
  }

  @SkylarkCallable(
    name = "stderr",
    structField = true,
    doc = "The content of the standard error output returned by the execution."
  )
  public String stderr() {
    return stderr;
  }

  /**
   * Executes a command given by a list of arguments and returns a SkylarkExecutionResult with
   * the output of the command.
   */
  static SkylarkExecutionResult execute(List<Object> args, long timeout) throws EvalException {
    try {
      String[] argsArray = new String[args.size()];
      for (int i = 0; i < args.size(); i++) {
        // We might have skylark path, do conversion.
        Object arg = args.get(i);
        if (!(arg instanceof String || arg instanceof SkylarkPath)) {
          throw new EvalException(
              Location.BUILTIN, "Argument " + i + " of execute is neither a path nor a string.");
        }
        argsArray[i] = arg.toString();
      }
      CommandResult result = new Command(argsArray).execute(new byte[] {}, timeout, false);
      return new SkylarkExecutionResult(result);
    } catch (BadExitStatusException e) {
      return new SkylarkExecutionResult(e.getResult());
    } catch (CommandException e) {
      return new SkylarkExecutionResult(256, "", e.getMessage());
    }
  }
}
