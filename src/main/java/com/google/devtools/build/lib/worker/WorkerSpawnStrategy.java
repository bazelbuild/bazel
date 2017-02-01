// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Charsets;
import com.google.common.base.MoreObjects;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.sandbox.SpawnHelpers;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.protobuf.ByteString;
import java.io.ByteArrayOutputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;

/**
 * A spawn action context that launches Spawns the first time they are used in a persistent mode and
 * then shards work over all the processes.
 */
@ExecutionStrategy(
  name = {"worker"},
  contextType = SpawnActionContext.class
)
public final class WorkerSpawnStrategy implements SandboxedSpawnActionContext {

  /**
   * An input stream filter that records the first X bytes read from its wrapped stream.
   *
   * <p>The number bytes to record can be set via {@link #startRecording(int)}}, which also discards
   * any already recorded data. The recorded data can be retrieved via {@link
   * #getRecordedDataAsString(Charset)}.
   */
  private static final class RecordingInputStream extends FilterInputStream {
    private static final Pattern NON_PRINTABLE_CHARS =
        Pattern.compile("[^\\p{Print}\\t\\r\\n]", Pattern.UNICODE_CHARACTER_CLASS);

    private ByteArrayOutputStream recordedData;
    private int maxRecordedSize;

    protected RecordingInputStream(InputStream in) {
      super(in);
    }

    /**
     * Returns the maximum number of bytes that can still be recorded in our buffer (but not more
     * than {@code size}).
     */
    private int getRecordableBytes(int size) {
      if (recordedData == null) {
        return 0;
      }
      return Math.min(maxRecordedSize - recordedData.size(), size);
    }

    @Override
    public int read() throws IOException {
      int bytesRead = super.read();
      if (getRecordableBytes(bytesRead) > 0) {
        recordedData.write(bytesRead);
      }
      return bytesRead;
    }

    @Override
    public int read(byte[] b) throws IOException {
      int bytesRead = super.read(b);
      int recordableBytes = getRecordableBytes(bytesRead);
      if (recordableBytes > 0) {
        recordedData.write(b, 0, recordableBytes);
      }
      return bytesRead;
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
      int bytesRead = super.read(b, off, len);
      int recordableBytes = getRecordableBytes(bytesRead);
      if (recordableBytes > 0) {
        recordedData.write(b, off, recordableBytes);
      }
      return bytesRead;
    }

    public void startRecording(int maxSize) {
      recordedData = new ByteArrayOutputStream(maxSize);
      maxRecordedSize = maxSize;
    }

    /**
     * Reads whatever remaining data is available on the input stream if we still have space left in
     * the recording buffer, in order to maximize the usefulness of the recorded data for the
     * caller.
     */
    public void readRemaining() {
      try {
        byte[] dummy = new byte[getRecordableBytes(available())];
        read(dummy);
      } catch (IOException e) {
        // Ignore.
      }
    }

    /**
     * Returns the recorded data as a string, where non-printable characters are replaced with a '?'
     * symbol.
     */
    public String getRecordedDataAsString(Charset charsetName) throws UnsupportedEncodingException {
      String recordedString = recordedData.toString(charsetName.name());
      return NON_PRINTABLE_CHARS.matcher(recordedString).replaceAll("?").trim();
    }
  }

  public static final String ERROR_MESSAGE_PREFIX =
      "Worker strategy cannot execute this %s action, ";
  public static final String REASON_NO_FLAGFILE =
      "because the last argument does not contain a @flagfile";
  public static final String REASON_NO_TOOLS = "because the action has no tools";
  public static final String REASON_NO_EXECUTION_INFO =
      "because the action's execution info does not contain 'supports-workers=1'";

  private final WorkerPool workers;
  private final Path execRoot;
  private final boolean verboseFailures;
  private final int maxRetries;
  private final Multimap<String, String> extraFlags;
  private final boolean workerVerbose;

  public WorkerSpawnStrategy(
      BlazeDirectories blazeDirs,
      WorkerPool workers,
      boolean verboseFailures,
      int maxRetries,
      boolean workerVerbose,
      Multimap<String, String> extraFlags) {
    Preconditions.checkNotNull(workers);
    this.workers = Preconditions.checkNotNull(workers);
    this.execRoot = blazeDirs.getExecRoot();
    this.verboseFailures = verboseFailures;
    this.maxRetries = maxRetries;
    this.workerVerbose = workerVerbose;
    this.extraFlags = extraFlags;
  }

  @Override
  public void exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    exec(spawn, actionExecutionContext, null);
  }

  @Override
  public void exec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws ExecException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    EventHandler eventHandler = executor.getEventHandler();
    StandaloneSpawnStrategy standaloneStrategy =
        Preconditions.checkNotNull(executor.getContext(StandaloneSpawnStrategy.class));

    if (executor.reportsSubcommands()) {
      executor.reportSubcommand(spawn);
    }

    if (!spawn.getExecutionInfo().containsKey("supports-workers")
        || !spawn.getExecutionInfo().get("supports-workers").equals("1")) {
      eventHandler.handle(
          Event.warn(
              String.format(ERROR_MESSAGE_PREFIX + REASON_NO_EXECUTION_INFO, spawn.getMnemonic())));
      standaloneStrategy.exec(spawn, actionExecutionContext);
      return;
    }

    // We assume that the spawn to be executed always gets a @flagfile argument, which contains the
    // flags related to the work itself (as opposed to start-up options for the executed tool).
    // Thus, we can extract the last element from its args (which will be the @flagfile), expand it
    // and put that into the WorkRequest instead.
    if (!Iterables.getLast(spawn.getArguments()).startsWith("@")) {
      throw new UserExecException(
          String.format(ERROR_MESSAGE_PREFIX + REASON_NO_FLAGFILE, spawn.getMnemonic()));
    }

    if (Iterables.isEmpty(spawn.getToolFiles())) {
      throw new UserExecException(
          String.format(ERROR_MESSAGE_PREFIX + REASON_NO_TOOLS, spawn.getMnemonic()));
    }

    executor
        .getEventBus()
        .post(ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), "worker"));

    FileOutErr outErr = actionExecutionContext.getFileOutErr();

    ImmutableList<String> args =
        ImmutableList.<String>builder()
            .addAll(spawn.getArguments().subList(0, spawn.getArguments().size() - 1))
            .add("--persistent_worker")
            .addAll(
                MoreObjects.firstNonNull(
                    extraFlags.get(spawn.getMnemonic()), ImmutableList.<String>of()))
            .build();
    ImmutableMap<String, String> env = spawn.getEnvironment();

    try {
      ActionInputFileCache inputFileCache = actionExecutionContext.getActionInputFileCache();

      HashCode workerFilesHash = WorkerFilesHash.getWorkerFilesHash(
          spawn.getToolFiles(), actionExecutionContext);
      Map<PathFragment, Path> inputFiles =
          new SpawnHelpers(execRoot).getMounts(spawn, actionExecutionContext);
      Set<PathFragment> outputFiles = SandboxHelpers.getOutputFiles(spawn);
      WorkerKey key =
          new WorkerKey(
              args,
              env,
              execRoot,
              spawn.getMnemonic(),
              workerFilesHash,
              inputFiles,
              outputFiles,
              writeOutputFiles != null);

      WorkRequest.Builder requestBuilder = WorkRequest.newBuilder();
      expandArgument(requestBuilder, Iterables.getLast(spawn.getArguments()));

      List<ActionInput> inputs =
          ActionInputHelper.expandArtifacts(
              spawn.getInputFiles(), actionExecutionContext.getArtifactExpander());

      for (ActionInput input : inputs) {
        byte[] digestBytes = inputFileCache.getDigest(input);
        ByteString digest;
        if (digestBytes == null) {
          digest = ByteString.EMPTY;
        } else {
          digest = ByteString.copyFromUtf8(HashCode.fromBytes(digestBytes).toString());
        }

        requestBuilder
            .addInputsBuilder()
            .setPath(input.getExecPathString())
            .setDigest(digest)
            .build();
      }

      WorkResponse response =
          execInWorker(eventHandler, key, requestBuilder.build(), maxRetries, writeOutputFiles);

      outErr.getErrorStream().write(response.getOutputBytes().toByteArray());

      if (response.getExitCode() != 0) {
        throw new UserExecException(
            String.format(
                "Worker process sent response with exit code: %d.", response.getExitCode()));
      }
    } catch (IOException e) {
      String message =
          CommandFailureUtils.describeCommandFailure(
              verboseFailures, spawn.getArguments(), env, execRoot.getPathString());
      throw new UserExecException(message, e);
    }
  }

  /**
   * Recursively expands arguments by replacing @filename args with the contents of the referenced
   * files. The @ itself can be escaped with @@.
   *
   * @param requestBuilder the WorkRequest.Builder that the arguments should be added to.
   * @param arg the argument to expand.
   * @throws java.io.IOException if one of the files containing options cannot be read.
   */
  private void expandArgument(WorkRequest.Builder requestBuilder, String arg) throws IOException {
    if (arg.startsWith("@") && !arg.startsWith("@@")) {
      for (String line : Files.readAllLines(
          Paths.get(execRoot.getRelative(arg.substring(1)).getPathString()), UTF_8)) {
        if (line.length() > 0) {
          expandArgument(requestBuilder, line);
        }
      }
    } else {
      requestBuilder.addArguments(arg);
    }
  }

  private WorkResponse execInWorker(
      EventHandler eventHandler,
      WorkerKey key,
      WorkRequest request,
      int retriesLeft,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws IOException, InterruptedException, UserExecException {
    Worker worker = null;
    WorkResponse response = null;

    try {
      worker = workers.borrowObject(key);
      worker.prepareExecution(key);

      request.writeDelimitedTo(worker.getOutputStream());
      worker.getOutputStream().flush();

      RecordingInputStream recordingStream = new RecordingInputStream(worker.getInputStream());
      recordingStream.startRecording(4096);
      try {
        response = WorkResponse.parseDelimitedFrom(recordingStream);
      } catch (IOException e2) {
        // If protobuf couldn't parse the response, try to print whatever the failing worker wrote
        // to stdout - it's probably a stack trace or some kind of error message that will help the
        // user figure out why the compiler is failing.
        recordingStream.readRemaining();
        String data = recordingStream.getRecordedDataAsString(Charsets.UTF_8);
        eventHandler.handle(
            Event.warn("Worker process returned an unparseable WorkResponse:\n" + data));
        throw e2;
      }

      if (writeOutputFiles != null
          && !writeOutputFiles.compareAndSet(null, WorkerSpawnStrategy.class)) {
        throw new InterruptedException();
      }

      worker.finishExecution(key);

      if (response == null) {
        throw new UserExecException(
            "Worker process did not return a WorkResponse. This is probably caused by a "
                + "bug in the worker, writing unexpected other data to stdout.");
      }
    } catch (IOException e) {
      if (worker != null) {
        workers.invalidateObject(key, worker);
        worker = null;
      }

      if (retriesLeft > 0) {
        // The worker process failed, but we still have some retries left. Let's retry with a fresh
        // worker.
        if (workerVerbose) {
          eventHandler.handle(
              Event.warn(
                  key.getMnemonic()
                      + " worker failed ("
                      + Throwables.getStackTraceAsString(e)
                      + "), invalidating and retrying with new worker..."));
        } else {
          eventHandler.handle(
              Event.warn(
                  key.getMnemonic()
                      + " worker failed, invalidating and retrying with new worker..."));
        }
        return execInWorker(eventHandler, key, request, retriesLeft - 1, writeOutputFiles);
      } else {
        throw e;
      }
    } finally {
      if (worker != null) {
        workers.returnObject(key, worker);
      }
    }
    return response;
  }

  @Override
  public String toString() {
    return "worker";
  }

  @Override
  public boolean willExecuteRemotely(boolean remotable) {
    return false;
  }

  @Override
  public boolean shouldPropagateExecException() {
    return false;
  }
}
