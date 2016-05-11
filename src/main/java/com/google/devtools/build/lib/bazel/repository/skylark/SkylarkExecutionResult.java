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
import com.google.devtools.build.lib.util.Preconditions;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * A structure callable from Skylark that stores the result of repository_ctx.execute() method. It
 * contains the standard output stream content, the standard error stream content and the execution
 * return code.
 */
@SkylarkModule(
  name = "exec_result",
  doc =
      "A structure storing result of repository_ctx.execute() method. It contains the standard"
          + " output stream content, the standard error stream content and the execution return"
          + " code."
)
final class SkylarkExecutionResult {
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
    doc = "The return code returned after the execution of the program. 256 if an error happened"
        + " while executing the command."
  )
  public int getReturnCode() {
    return returnCode;
  }

  @SkylarkCallable(
    name = "stdout",
    structField = true,
    doc = "The content of the standard output returned by the execution."
  )
  public String getStdout() {
    return stdout;
  }

  @SkylarkCallable(
    name = "stderr",
    structField = true,
    doc = "The content of the standard error output returned by the execution."
  )
  public String getStderr() {
    return stderr;
  }

  /**
   * Returns a Builder that can be used to execute a command and build an execution result.
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * A Builder class to build a {@link SkylarkExecutionResult} object by executing a command.
   */
  static final class Builder {

    private final List<String> args = new ArrayList<>();
    private long timeout = -1;
    private boolean executed = false;

    /**
     * Adds arguments to the list of arguments to pass to the command. The first argument is
     * expected to be the binary to execute. The subsequent arguments are the arguments passed
     * to the binary.
     *
     * <p>Each argument can be either a string or a {@link SkylarkPath}, passing another argument
     * will fail when executing the command.
     */
    Builder addArguments(Iterable<Object> args) throws EvalException {
      for (Object arg : args) {
        // We might have skylark path, do conversion.
        if (!(arg instanceof String || arg instanceof SkylarkPath)) {
          throw new EvalException(
              Location.BUILTIN,
              "Argument " + this.args.size() + " of execute is neither a path nor a string.");
        }
        this.args.add(arg.toString());
      }
      return this;
    }

    /**
     * Sets the timeout, in milliseconds, after which the executed command will be terminated.
     */
    Builder setTimeout(long timeout) {
      Preconditions.checkArgument(timeout > 0, "Timeout must be a positive number.");
      this.timeout = timeout;
      return this;
    }

    /**
     * Execute the command specified by {@link #addArguments(Iterable)}.
     */
    SkylarkExecutionResult execute() throws EvalException {
      Preconditions.checkArgument(timeout > 0, "Timeout must be set prior to calling execute().");
      Preconditions.checkArgument(!args.isEmpty(), "No command specified.");
      Preconditions.checkState(!executed, "Command was already executed, cannot re-use builder.");
      executed = true;

      try {
        String[] argsArray = new String[args.size()];
        for (int i = 0; i < args.size(); i++) {
          argsArray[i] = args.get(i);
        }
        CommandResult result = new Command(argsArray).execute(new byte[]{}, timeout, false);
        return new SkylarkExecutionResult(result);
      } catch (BadExitStatusException e) {
        return new SkylarkExecutionResult(e.getResult());
      } catch (CommandException e) {
        // 256 is outside of the standard range for exit code on Unixes. We are not guaranteed that
        // on all system it would be outside of the standard range.
        return new SkylarkExecutionResult(256, "", e.getMessage());
      }
    }
  }
}
