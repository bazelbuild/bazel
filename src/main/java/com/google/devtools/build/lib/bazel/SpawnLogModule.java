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

import com.google.devtools.build.lib.bazel.execlog.StableSort;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.exec.SpawnLogContext;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.AsynchronousFileOutputStream;
import com.google.devtools.build.lib.util.io.MessageOutputStreamWrapper.BinaryOutputStreamWrapper;
import com.google.devtools.build.lib.util.io.MessageOutputStreamWrapper.JsonOutputStreamWrapper;
import com.google.devtools.build.lib.util.io.MessageOutputStreamWrapper.MessageOutputStreamCollection;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

/**
 * Module providing on-demand spawn logging.
 */
public final class SpawnLogModule extends BlazeModule {
  /**
   * SpawnLogContext will log to a temporary file as the execution is being performed. rawOutput is
   * the path to that temporary file.
   */
  private SpawnLogContext spawnLogContext;

  private Path rawOutput;

  /**
   * After the execution is done, the temporary file contents will be sorted and logged as the user
   * requested, to binary and/or json files. We will open the streams at the beginning of the
   * command so that any errors (e.g., unwritable location) will be surfaced before the execution
   * begins.
   */
  private MessageOutputStreamCollection outputStreams;

  private CommandEnvironment env;

  private void clear() {
    spawnLogContext = null;
    outputStreams = new MessageOutputStreamCollection();
    rawOutput = null;
    env = null;
  }

  private void initOutputs(CommandEnvironment env) throws IOException {
    clear();
    this.env = env;

    ExecutionOptions executionOptions = env.getOptions().getOptions(ExecutionOptions.class);
    if (executionOptions == null) {
      return;
    }
    FileSystem fileSystem = env.getRuntime().getFileSystem();
    Path workingDirectory = env.getWorkingDirectory();

    if (executionOptions.executionLogBinaryFile != null
        && !executionOptions.executionLogBinaryFile.isEmpty()) {
      outputStreams.addStream(
          new BinaryOutputStreamWrapper(
              workingDirectory
                  .getRelative(executionOptions.executionLogBinaryFile)
                  .getOutputStream()));
    }

    if (executionOptions.executionLogJsonFile != null
        && !executionOptions.executionLogJsonFile.isEmpty()) {
      outputStreams.addStream(
          new JsonOutputStreamWrapper(
              workingDirectory
                  .getRelative(executionOptions.executionLogJsonFile)
                  .getOutputStream()));
    }

    AsynchronousFileOutputStream outStream = null;
    if (executionOptions.executionLogFile != null && !executionOptions.executionLogFile.isEmpty()) {
      rawOutput = workingDirectory.getRelative(executionOptions.executionLogFile);
      outStream =
          new AsynchronousFileOutputStream(
              workingDirectory.getRelative(executionOptions.executionLogFile));
    } else if (!outputStreams.isEmpty()) {
      // Execution log requested but raw log file not specified
      File file = File.createTempFile("exec", ".log");
      rawOutput = fileSystem.getPath(file.getAbsolutePath());
      outStream = new AsynchronousFileOutputStream(rawOutput);
    }

    if (outStream == null) {
      // No logging needed
      clear();
      return;
    }

    spawnLogContext =
        new SpawnLogContext(
            env.getExecRoot(), outStream, env.getOptions().getOptions(RemoteOptions.class));
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
                  "Error initializing execution log", ExitCode.COMMAND_LINE_ERROR, e));
    }
  }

  @Override
  public void afterCommand() throws AbruptExitException {
    boolean done = false;
    if (spawnLogContext != null) {
      try {
        spawnLogContext.close();
        if (!outputStreams.isEmpty()) {
          InputStream in = rawOutput.getInputStream();
          StableSort.stableSort(in, outputStreams);
          outputStreams.close();
        }
        done = true;
      } catch (IOException e) {
        throw new AbruptExitException(ExitCode.LOCAL_ENVIRONMENTAL_ERROR, e);
      } finally {
        if (!done && !outputStreams.isEmpty()) {
          env.getReporter()
              .handle(
                  Event.warn(
                      "Execution log might not have been populated. Raw execution log is at "
                          + rawOutput));
        }
        clear();
      }
    }
  }
}
