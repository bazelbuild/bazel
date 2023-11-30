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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.bazel.execlog.StableSort;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.exec.SpawnLogContext;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.io.AsynchronousMessageOutputStream;
import com.google.devtools.build.lib.util.io.MessageOutputStream;
import com.google.devtools.build.lib.util.io.MessageOutputStreamWrapper.BinaryOutputStreamWrapper;
import com.google.devtools.build.lib.util.io.MessageOutputStreamWrapper.JsonOutputStreamWrapper;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.InputStream;
import javax.annotation.Nullable;

/** Module providing on-demand spawn logging. */
public final class SpawnLogModule extends BlazeModule {
  @Nullable private SpawnLogContext spawnLogContext;

  /** Output path for the raw output stream. */
  @Nullable private Path rawOutputPath;

  /** Output stream to write directly into during execution. */
  @Nullable private MessageOutputStream<SpawnExec> rawOutputStream;

  /**
   * Output stream to convert the raw output into after the execution is done.
   *
   * <p>We open the stream at the beginning of the command so that any errors (e.g., unwritable
   * location) are surfaced before execution begins.
   */
  @Nullable private MessageOutputStream<SpawnExec> convertedOutputStream;

  private CommandEnvironment env;

  private void clear() {
    spawnLogContext = null;
    rawOutputPath = null;
    rawOutputStream = null;
    convertedOutputStream = null;
    env = null;
  }

  private void initOutputs(CommandEnvironment env) throws IOException {
    clear();

    ExecutionOptions executionOptions = env.getOptions().getOptions(ExecutionOptions.class);
    if (executionOptions == null
        || (executionOptions.executionLogBinaryFile == null
            && executionOptions.executionLogJsonFile == null)) {
      // No logging requested.
      return;
    }

    if (executionOptions.executionLogBinaryFile != null
        && executionOptions.executionLogJsonFile != null) {
      String message =
          "Must specify at most one of --execution_log_json_file and --execution_log_binary_file";
      env.getBlazeModuleEnvironment()
          .exit(
              new AbruptExitException(
                  DetailedExitCode.of(
                      FailureDetail.newBuilder()
                          .setMessage(message)
                          .setExecutionOptions(
                              FailureDetails.ExecutionOptions.newBuilder()
                                  .setCode(
                                      FailureDetails.ExecutionOptions.Code
                                          .MULTIPLE_EXECUTION_LOG_FORMATS))
                          .build())));
      return;
    }

    this.env = env;

    Path workingDirectory = env.getWorkingDirectory();
    Path outputBase = env.getOutputBase();

    // Set up the raw output stream.
    // This stream performs the writes in a separate thread to avoid blocking execution.
    // If the unsorted binary format was requested, use the respective output path to avoid a
    // pointless conversion at the end. Otherwise, use a temporary path.
    if (executionOptions.executionLogBinaryFile != null && !executionOptions.executionLogSort) {
      rawOutputPath = workingDirectory.getRelative(executionOptions.executionLogBinaryFile);
    } else {
      rawOutputPath = outputBase.getRelative("execution.log");
    }
    rawOutputStream = new AsynchronousMessageOutputStream<>(rawOutputPath);

    // Set up the binary output stream, if distinct from the raw output stream.
    if (executionOptions.executionLogBinaryFile != null && executionOptions.executionLogSort) {
      convertedOutputStream =
          new BinaryOutputStreamWrapper<>(
              workingDirectory
                  .getRelative(executionOptions.executionLogBinaryFile)
                  .getOutputStream());
    }

    // Set up the text output stream.
    if (executionOptions.executionLogJsonFile != null) {
      convertedOutputStream =
          new JsonOutputStreamWrapper<>(
              workingDirectory
                  .getRelative(executionOptions.executionLogJsonFile)
                  .getOutputStream());
    }

    spawnLogContext =
        new SpawnLogContext(
            env.getExecRoot().asFragment(),
            rawOutputStream,
            env.getOptions().getOptions(ExecutionOptions.class),
            env.getOptions().getOptions(RemoteOptions.class),
            env.getRuntime().getFileSystem().getDigestFunction(),
            env.getXattrProvider());
  }

  @Override
  public void registerActionContexts(
      ModuleActionContextRegistry.Builder registryBuilder,
      CommandEnvironment env,
      BuildRequest buildRequest) {
    if (spawnLogContext != null) {
      // TODO(b/63987502): Pretty sure the "spawn-log" commandline identifier is never used as there
      // is no other SpawnLogContext to distinguish from.
      registryBuilder.register(SpawnLogContext.class, spawnLogContext, "spawn-log");
      registryBuilder.restrictTo(SpawnLogContext.class, "");
    }
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    env.getEventBus().register(this);

    try {
      initOutputs(env);
    } catch (IOException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      env.getBlazeModuleEnvironment()
          .exit(
              new AbruptExitException(
                  createDetailedExitCode(
                      "Error initializing execution log",
                      Code.EXECUTION_LOG_INITIALIZATION_FAILURE)));
    }
  }

  @Override
  public void afterCommand() throws AbruptExitException {
    if (spawnLogContext == null) {
      // No logging requested.
      return;
    }

    checkNotNull(rawOutputPath);

    boolean done = false;
    try {
      spawnLogContext.close();
      if (convertedOutputStream != null) {
        InputStream in = rawOutputPath.getInputStream();
        if (spawnLogContext.shouldSort()) {
          StableSort.stableSort(in, convertedOutputStream);
        } else {
          while (in.available() > 0) {
            SpawnExec ex = SpawnExec.parseDelimitedFrom(in);
            convertedOutputStream.write(ex);
          }
        }
        convertedOutputStream.close();
      }
      done = true;
    } catch (IOException e) {
      String message = e.getMessage() == null ? "Error writing execution log" : e.getMessage();
      throw new AbruptExitException(
          createDetailedExitCode(message, Code.EXECUTION_LOG_WRITE_FAILURE), e);
    } finally {
      if (convertedOutputStream != null) {
        if (!done) {
          env.getReporter()
              .handle(
                  Event.warn(
                      "Execution log might not have been populated. Raw execution log is at "
                          + rawOutputPath));
        } else {
          try {
            rawOutputPath.delete();
          } catch (IOException e) {
            // Intentionally ignored.
          }
        }
      }
      clear();
    }
  }

  private static DetailedExitCode createDetailedExitCode(String message, Code detailedCode) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setExecution(Execution.newBuilder().setCode(detailedCode))
            .build());
  }
}
