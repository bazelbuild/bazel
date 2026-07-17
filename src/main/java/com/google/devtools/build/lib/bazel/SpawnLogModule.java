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

import com.google.common.eventbus.Subscribe;
import com.google.common.primitives.Booleans;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader.UploadContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.CompactSpawnLogContext;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.ExpandedSpawnLogContext;
import com.google.devtools.build.lib.exec.ExpandedSpawnLogContext.Encoding;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.exec.SpawnLogContext;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
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
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/** Module providing on-demand spawn logging. */
public final class SpawnLogModule extends BlazeModule {
  private static final int OUTPUT_BUFFER_SIZE = 100 * 1024;
  private static final String EXEC_LOG_COMPACT_FILENAME = "execution_log.binpb.zst";
  private static final String EXEC_LOG_BINARY_FILENAME = "execution_log.binpb";
  private static final String EXEC_LOG_JSON_FILENAME = "execution_log.json";

  @Nullable private SpawnLogContext spawnLogContext;
  @Nullable private Path outputPath;
  @Nullable private ListenableFuture<String> uriFuture;
  @Nullable private String logName;

  @Nullable private AbruptExitException abruptExit = null;

  private void clear() {
    spawnLogContext = null;
    outputPath = null;
    uriFuture = null;
    logName = null;
    abruptExit = null;
  }

  private void initOutputs(CommandEnvironment env) throws IOException {
    clear();
    try {
      ExecutionOptions executionOptions = env.getOptions().getOptions(ExecutionOptions.class);
      if (executionOptions == null) {
        return;
      }

      BuildEventProtocolOptions bepOptions =
          checkNotNull(env.getOptions().getOptions(BuildEventProtocolOptions.class));

      int numFormats =
          Booleans.countTrue(
              executionOptions.getExecutionLogCompactFile() != null,
              executionOptions.getExecutionLogBinaryFile() != null,
              executionOptions.getExecutionLogJsonFile() != null);

      if (numFormats == 0) {
        // No logging requested.
        return;
      }

      if (numFormats > 1) {
        String message =
            "Must specify at most one of --execution_log_binary_file, --execution_log_json_file and"
                + " --execution_log_compact_file";
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

      Path outputBase = env.getOutputBase();
      Predicate<Spawn> logSpawnPredicate =
          spawn -> executionOptions.getExecutionLogMnemonicFilter().test(spawn.getMnemonic());

      BufferedOutputStream outputStream = null;
      String displayName = null;
      PathFragment logPath = null;
      if (executionOptions.getExecutionLogCompactFile() != null) {
        logName = EXEC_LOG_COMPACT_FILENAME;
        logPath = executionOptions.getExecutionLogCompactFile();
      } else if (executionOptions.getExecutionLogBinaryFile() != null) {
        logName = EXEC_LOG_BINARY_FILENAME;
        logPath = executionOptions.getExecutionLogBinaryFile();
      } else {
        logName = EXEC_LOG_JSON_FILENAME;
        logPath = executionOptions.getExecutionLogJsonFile();
      }
      checkNotNull(logPath);

      if (!logPath.isEmpty()) {
        // Log path is specified, write to local file.
        outputPath = getAbsolutePath(logPath, env);
        outputStream = new BufferedOutputStream(outputPath.getOutputStream(), OUTPUT_BUFFER_SIZE);
        displayName = outputPath.toString();
      } else if (bepOptions.getStreamingLogFileUploads()) {
        // Path is empty but streaming is enabled.
        BuildEventArtifactUploader uploader =
            env.getRuntime()
                .getBuildEventArtifactUploaderFactoryMap()
                .select(bepOptions.getBuildEventUploadStrategy())
                .create(env);
        UploadContext uploadContext = uploader.startUpload(LocalFileType.LOG, null);
        outputStream =
            new BufferedOutputStream(uploadContext.getOutputStream(), OUTPUT_BUFFER_SIZE);
        uriFuture = uploadContext.uriFuture();
        displayName = logName + "-stream";
      } else {
        // Path is empty but streaming is not enabled. Disable logging.
        env.getBlazeModuleEnvironment()
            .exit(
                new AbruptExitException(
                    DetailedExitCode.of(
                        FailureDetail.newBuilder()
                            .setMessage(
                                "--execution_log_{compact,binary,json}_file is empty, but"
                                    + " --experimental_stream_log_file_uploads is not enabled."
                                    + " Execution log will not be uploaded to the BEP.")
                            .setExecutionOptions(
                                FailureDetails.ExecutionOptions.newBuilder()
                                    .setCode(
                                        FailureDetails.ExecutionOptions.Code
                                            .EXECUTION_LOG_STREAMING_DISABLED))
                            .build())));
        return;
      }

      if (outputStream == null) {
        // Null output stream from UploadContext - disable logging.
        env.getReporter()
            .handle(
                Event.warn(
                    "Execution log streaming is not enabled. Execution log will not be"
                        + " generated."));
        return;
      }

      checkNotNull(displayName);

      if (executionOptions.getExecutionLogCompactFile() != null) {
        spawnLogContext =
            new CompactSpawnLogContext(
                outputStream,
                displayName,
                env.getExecRoot().asFragment(),
                env.getWorkspaceName(),
                env.getOptions()
                    .getOptions(BuildLanguageOptions.class)
                    .getExperimentalSiblingRepositoryLayout(),
                env.getOptions().getOptions(RemoteOptions.class),
                env.getRuntime().getFileSystem().getDigestFunction(),
                env.getXattrProvider(),
                env.getCommandId(),
                env.getReporter(),
                logSpawnPredicate);
      } else {
        boolean binaryElseJson = executionOptions.getExecutionLogBinaryFile() != null;
        // Use a well-known temporary path to avoid accumulation of potentially large files in /tmp
        // due to abnormally terminated invocations (e.g., when running out of memory).
        Path tempPath = outputBase.getRelative(logName);

        spawnLogContext =
            new ExpandedSpawnLogContext(
                outputStream,
                displayName,
                outputPath,
                tempPath,
                binaryElseJson ? Encoding.BINARY : Encoding.JSON,
                /* sorted= */ executionOptions.getExecutionLogSort(),
                env.getExecRoot().asFragment(),
                env.getOptions().getOptions(RemoteOptions.class),
                env.getRuntime().getFileSystem().getDigestFunction(),
                env.getXattrProvider(),
                uriFuture != null,
                logSpawnPredicate);
      }
    } catch (InterruptedException e) {
      env.getReporter()
          .handle(Event.error("Error while setting up the execution log: " + e.getMessage()));
    }
  }

  /**
   * If the given path is absolute path, leave it as it is. If the given path is a relative path, it
   * is relative to the current working directory. If the given path starts with '%workspace%, it is
   * relative to the workspace root, which is the output of `bazel info workspace`.
   *
   * @return Absolute Path
   */
  private Path getAbsolutePath(PathFragment path, CommandEnvironment env) {
    String pathString = path.getPathString();
    if (env.getWorkspace() != null) {
      pathString = pathString.replace("%workspace%", env.getWorkspace().getPathString());
    }
    if (!PathFragment.isAbsolute(pathString)) {
      return env.getWorkingDirectory().getRelative(pathString);
    }

    return env.getRuntime().getFileSystem().getPath(pathString);
  }

  @Override
  public void registerActionContexts(
      ModuleActionContextRegistry.Builder registryBuilder,
      CommandEnvironment env,
      BuildRequest buildRequest) {
    if (spawnLogContext != null) {
      registryBuilder.register(SpawnLogContext.class, spawnLogContext);
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
                      String.format("Error initializing execution log: %s", e.getMessage()),
                      Code.EXECUTION_LOG_INITIALIZATION_FAILURE)));
    }
  }

  @Subscribe
  public void buildComplete(BuildCompleteEvent event) {
    // The log must be finalized in buildComplete() instead of afterCommand(), because it's our
    // last chance to publish it to the build event protocol.

    if (spawnLogContext == null) {
      // No logging requested.
      clear();
      return;
    }

    try {
      spawnLogContext.close();
      if (spawnLogContext.shouldPublish()) {
        checkNotNull(logName);
        if (uriFuture != null) {
          event.getResult().getBuildToolLogCollection().addUriFuture(logName, uriFuture);
        } else {
          event.getResult().getBuildToolLogCollection().addLocalFile(logName, outputPath);
        }
      }
    } catch (IOException e) {
      abruptExit =
          new AbruptExitException(
              createDetailedExitCode(
                  String.format("Error writing execution log: %s", e.getMessage()),
                  Code.EXECUTION_LOG_WRITE_FAILURE),
              e);
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

  @Override
  public void afterCommand() throws AbruptExitException {
    if (abruptExit != null) {
      throw abruptExit;
    }
  }
}
