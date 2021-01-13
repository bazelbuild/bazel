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

import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Interface to a worker process running as a single child process.
 *
 * <p>A worker process must follow this protocol to be usable via this class: The worker process is
 * spawned on demand. The worker process is free to exit whenever necessary, as new instances will
 * be relaunched automatically. Communication happens via the WorkerProtocol protobuf, sent to and
 * received from the worker process via stdin / stdout.
 *
 * <p>Other code in Blaze can talk to the worker process via input / output streams provided by this
 * class.
 */
class SingleplexWorker extends Worker {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** The execution root of the worker. */
  protected final Path workDir;
  /**
   * Stream for recording the WorkResponse as it's read, so that it can be printed in the case of
   * parsing failures.
   */
  @Nullable private RecordingInputStream recordingInputStream;
  /** The implementation of the worker protocol (JSON or Proto). */
  @Nullable private WorkerProtocolImpl workerProtocol;

  private Subprocess process;
  /** True if we deliberately destroyed this process. */
  private boolean wasDestroyed;

  private Thread shutdownHook;

  SingleplexWorker(WorkerKey workerKey, int workerId, final Path workDir, Path logFile) {
    super(workerKey, workerId, logFile);
    this.workDir = workDir;

    final SingleplexWorker self = this;
    this.shutdownHook =
        new Thread(
            () -> {
              // Not sure why this is needed. philwo@ added it without explanation.
              self.shutdownHook = null;
              self.destroy();
            });
    Runtime.getRuntime().addShutdownHook(shutdownHook);
  }

  Subprocess createProcess() throws IOException {
    ImmutableList<String> args = workerKey.getArgs();
    File executable = new File(args.get(0));
    if (!executable.isAbsolute() && executable.getParent() != null) {
      List<String> newArgs = new ArrayList<>(args);
      newArgs.set(0, new File(workDir.getPathFile(), newArgs.get(0)).getAbsolutePath());
      args = ImmutableList.copyOf(newArgs);
    }
    SubprocessBuilder processBuilder = new SubprocessBuilder();
    processBuilder.setArgv(args);
    processBuilder.setWorkingDirectory(workDir.getPathFile());
    processBuilder.setStderr(logFile.getPathFile());
    processBuilder.setEnv(workerKey.getEnv());
    return processBuilder.start();
  }

  @Override
  public void prepareExecution(
      SandboxInputs inputFiles, SandboxOutputs outputs, Set<PathFragment> workerFiles)
      throws IOException {
    if (process == null) {
      process = createProcess();
      recordingInputStream = new RecordingInputStream(process.getInputStream());
    }
    if (workerProtocol == null) {
      switch (workerKey.getProtocolFormat()) {
        case JSON:
          workerProtocol = new JsonWorkerProtocol(process.getOutputStream(), recordingInputStream);
          break;
        case PROTO:
          workerProtocol = new ProtoWorkerProtocol(process.getOutputStream(), recordingInputStream);
          break;
      }
    }
  }

  @Override
  void putRequest(WorkRequest request) throws IOException {
    workerProtocol.putRequest(request);
  }

  @Override
  WorkResponse getResponse(int requestId) throws IOException {
    recordingInputStream.startRecording(4096);
    return workerProtocol.getResponse();
  }

  @Override
  public void finishExecution(Path execRoot) throws IOException {}

  @Override
  void destroy() {
    if (shutdownHook != null) {
      Runtime.getRuntime().removeShutdownHook(shutdownHook);
    }
    if (workerProtocol != null) {
      try {
        workerProtocol.close();
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("Caught IOException while closing worker protocol.");
      }
      workerProtocol = null;
    }
    if (process != null) {
      wasDestroyed = true;
      process.destroyAndWait();
    }
  }

  /** Returns true if this process is dead but we didn't deliberately kill it. */
  @Override
  boolean diedUnexpectedly() {
    return process != null && !wasDestroyed && !process.isAlive();
  }

  @Override
  public Optional<Integer> getExitValue() {
    return process != null && !process.isAlive()
        ? Optional.of(process.exitValue())
        : Optional.empty();
  }

  @Override
  String getRecordingStreamMessage() {
    recordingInputStream.readRemaining();
    return recordingInputStream.getRecordedDataAsString();
  }

  @Override
  public String toString() {
    return workerKey.getMnemonic() + " worker #" + workerId;
  }
}
