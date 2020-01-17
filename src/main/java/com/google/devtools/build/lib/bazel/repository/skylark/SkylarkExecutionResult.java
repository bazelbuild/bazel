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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.BadExitStatusException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.shell.TerminationStatus;
import com.google.devtools.build.lib.skylarkbuildapi.repository.SkylarkExecutionResultApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.io.DelegatingOutErr;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A structure callable from Skylark that stores the result of repository_ctx.execute() method. It
 * contains the standard output stream content, the standard error stream content and the execution
 * return code.
 */
@Immutable
final class SkylarkExecutionResult implements SkylarkExecutionResultApi {
  private final int returnCode;
  private final String stdout;
  private final String stderr;

  SkylarkExecutionResult(int returnCode, String stdout, String stderr) {
    this.returnCode = returnCode;
    this.stdout = stdout;
    this.stderr = stderr;
  }

  @Override
  public int getReturnCode() {
    return returnCode;
  }

  @Override
  public String getStdout() {
    return stdout;
  }

  @Override
  public String getStderr() {
    return stderr;
  }

  /**
   * Returns a Builder that can be used to execute a command and build an execution result.
   *
   * @param environment pass through the list of environment variables from the client to be passed
   * to the execution environment.
   */
  public static Builder builder(Map<String, String> environment) {
    return new Builder(environment);
  }

  /**
   * A Builder class to build a {@link SkylarkExecutionResult} object by executing a command.
   */
  static final class Builder {

    private final List<String> args = new ArrayList<>();
    private File directory = null;
    private final Map<String, String> envBuilder = Maps.newLinkedHashMap();
    private long timeout = -1;
    private boolean executed = false;
    private boolean quiet;

    private Builder(Map<String, String> environment) {
      envBuilder.putAll(environment);
    }

    /**
     * Adds arguments to the list of arguments to pass to the command. The first argument is
     * expected to be the binary to execute. The subsequent arguments are the arguments passed to
     * the binary.
     *
     * <p>Each argument can be either a string or a {@link SkylarkPath}, passing another argument
     * will fail when executing the command.
     */
    Builder addArguments(Iterable<?> args) throws EvalException {
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
     * Set the path to the directory to execute the result process. This method must be called
     * before calling {@link #execute()}.
     */
    Builder setDirectory(File path) throws EvalException {
      this.directory = path;
      return this;
    }

    /**
     * Add an environment variables to be added to the list of environment variables. For all
     * key <code>k</code> of <code>variables</code>, the resulting process will have the variable
     * <code>k=variables.get(k)</code> defined.
     */
    Builder addEnvironmentVariables(Map<String, String> variables) {
      this.envBuilder.putAll(variables);
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

    Builder setQuiet(boolean quiet) {
      this.quiet = quiet;
      return this;
    }

    private static String toString(ByteArrayOutputStream stream) {
      try {
        return new String(stream.toByteArray(), UTF_8);
      } catch (IllegalStateException e) {
        return "";
      }
    }

    /** Execute the command specified by {@link #addArguments(Iterable)}. */
    SkylarkExecutionResult execute() throws EvalException, InterruptedException {
      Preconditions.checkArgument(timeout > 0, "Timeout must be set prior to calling execute().");
      Preconditions.checkArgument(!args.isEmpty(), "No command specified.");
      Preconditions.checkState(!executed, "Command was already executed, cannot re-use builder.");
      Preconditions.checkNotNull(directory, "Directory must be set before calling execute().");
      executed = true;

      DelegatingOutErr delegator = new DelegatingOutErr();
      RecordingOutErr recorder = new RecordingOutErr();
      // TODO(dmarting): if a lot of data is sent to stdout, this will use all the memory and
      // Bazel will crash. Maybe we should use custom output streams that throw an appropriate
      // exception when reaching a specific size.
      delegator.addSink(recorder);
      if (!quiet) {
        delegator.addSink(OutErr.create(System.err, System.err));
      }
      try {
        String[] argsArray = new String[args.size()];
        for (int i = 0; i < args.size(); i++) {
          argsArray[i] = args.get(i);
        }
        Command command = new Command(argsArray, envBuilder, directory, Duration.ofMillis(timeout));
        CommandResult result =
            command.execute(delegator.getOutputStream(), delegator.getErrorStream());
        return new SkylarkExecutionResult(
            result.getTerminationStatus().getExitCode(),
            recorder.outAsLatin1(),
            recorder.errAsLatin1());
      } catch (BadExitStatusException e) {
        return new SkylarkExecutionResult(
            e.getResult().getTerminationStatus().getExitCode(), recorder.outAsLatin1(),
            recorder.errAsLatin1());
      } catch (AbnormalTerminationException e) {
        TerminationStatus status = e.getResult().getTerminationStatus();
        if (status.timedOut()) {
          // Signal a timeout by an exit code outside the normal range
          return new SkylarkExecutionResult(256, "", e.getMessage());
        } else if (status.exited()) {
          return new SkylarkExecutionResult(
              status.getExitCode(),
              toString(e.getResult().getStdoutStream()),
              toString(e.getResult().getStderrStream()));
        } else if (status.getTerminatingSignal() == 15) {
          // We have a bit of a problem here: we cannot distingusih between the case where
          // the SIGTERM was sent by something that the calling rule wants to legitimately handle,
          // and the case where it was sent by bazel to abort the build, e.g., because something
          // else failed.
          //
          // We just assume the latter to correctly handle aborts, accepting that rule authors have
          // to write their rules without relying on the ability to handle termination by signal 15.
          throw new InterruptedException();
        } else {
          return new SkylarkExecutionResult(
              status.getRawExitCode(),
              toString(e.getResult().getStdoutStream()),
              toString(e.getResult().getStderrStream()));
        }
      } catch (CommandException e) {
        // 256 is outside of the standard range for exit code on Unixes. We are not guaranteed that
        // on all system it would be outside of the standard range.
        return new SkylarkExecutionResult(256, "", e.getMessage());
      }
    }
  }
}
