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

import com.google.common.primitives.Booleans;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.CompactSpawnLogContext;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.ExpandedSpawnLogContext;
import com.google.devtools.build.lib.exec.ExpandedSpawnLogContext.Encoding;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
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
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import javax.annotation.Nullable;

/** Module providing on-demand spawn logging. */
public final class SpawnLogModule extends BlazeModule {
  @Nullable private SpawnLogContext spawnLogContext;

  private void clear() {
    spawnLogContext = null;
  }

  private void initOutputs(CommandEnvironment env) throws IOException {
    clear();

    ExecutionOptions executionOptions = env.getOptions().getOptions(ExecutionOptions.class);
    if (executionOptions == null) {
      return;
    }

    int numFormats =
        Booleans.countTrue(
            executionOptions.executionLogCompactFile != null,
            executionOptions.executionLogBinaryFile != null,
            executionOptions.executionLogJsonFile != null);

    if (numFormats == 0) {
      // No logging requested.
      return;
    }

    if (numFormats > 1) {
      String message =
          "Must specify at most one of --execution_log_binary_file, --execution_log_json_file and"
              + " --experimental_execution_log_compact_file";
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

    Path workingDirectory = env.getWorkingDirectory();
    Path outputBase = env.getOutputBase();

    if (executionOptions.executionLogCompactFile != null) {
      spawnLogContext =
          new CompactSpawnLogContext(
              workingDirectory.getRelative(executionOptions.executionLogCompactFile),
              env.getExecRoot().asFragment(),
              env.getOptions().getOptions(RemoteOptions.class),
              env.getRuntime().getFileSystem().getDigestFunction(),
              env.getXattrProvider());
    } else {
      Path outputPath = null;
      Encoding encoding = null;

      if (executionOptions.executionLogBinaryFile != null) {
        encoding = Encoding.BINARY;
        outputPath = workingDirectory.getRelative(executionOptions.executionLogBinaryFile);
      } else if (executionOptions.executionLogJsonFile != null) {
        encoding = Encoding.JSON;
        outputPath = workingDirectory.getRelative(executionOptions.executionLogJsonFile);
      }

      // Use a well-known temporary path to avoid accumulation of potentially large files in /tmp
      // due to abnormally terminated invocations (e.g., when running out of memory).
      Path tempPath = outputBase.getRelative("execution.log");

      spawnLogContext =
          new ExpandedSpawnLogContext(
              checkNotNull(outputPath),
              tempPath,
              checkNotNull(encoding),
              /* sorted= */ executionOptions.executionLogSort,
              env.getExecRoot().asFragment(),
              env.getOptions().getOptions(RemoteOptions.class),
              env.getRuntime().getFileSystem().getDigestFunction(),
              env.getXattrProvider());
    }
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

    try {
      spawnLogContext.close();
    } catch (IOException e) {
      String message = e.getMessage() == null ? "Error writing execution log" : e.getMessage();
      throw new AbruptExitException(
          createDetailedExitCode(message, Code.EXECUTION_LOG_WRITE_FAILURE), e);
    } finally {
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
