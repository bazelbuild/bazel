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
import com.google.common.hash.HashCode;
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
import java.util.SortedMap;

/**
 * Interface to a worker process running as a child process.
 *
 * <p>A worker process must follow this protocol to be usable via this class: The worker process is
 * spawned on demand. The worker process is free to exit whenever necessary, as new instances will
 * be relaunched automatically. Communication happens via the WorkerProtocol protobuf, sent to and
 * received from the worker process via stdin / stdout.
 *
 * <p>Other code in Blaze can talk to the worker process via input / output streams provided by this
 * class.
 */
class Worker {
  /** An unique identifier of the work process. */
  protected final WorkerKey workerKey;
  /** An unique ID of the worker. It will be used in WorkRequest and WorkResponse as well. */
  protected final int workerId;
  /** The execution root of the worker. */
  protected final Path workDir;
  /** The path of the log file. */
  protected final Path logFile;
  /** Stream for reading the WorkResponse. */
  protected RecordingInputStream recordingStream;

  private Subprocess process;
  private Thread shutdownHook;
  /** True if we deliberately destroyed this process. */
  private boolean wasDestroyed;

  Worker(WorkerKey workerKey, int workerId, final Path workDir, Path logFile) {
    this.workerKey = workerKey;
    this.workerId = workerId;
    this.workDir = workDir;
    this.logFile = logFile;

    final Worker self = this;
    this.shutdownHook =
        new Thread(
            () -> {
              try {
                self.shutdownHook = null;
                self.destroy();
              } catch (IOException e) {
                // We can't do anything here.
              }
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

  void destroy() throws IOException {
    if (shutdownHook != null) {
      Runtime.getRuntime().removeShutdownHook(shutdownHook);
    }
    if (process != null) {
      wasDestroyed = true;
      process.destroyAndWait();
    }
  }

  /**
   * Returns a unique id for this worker. This is used to distinguish different worker processes in
   * logs and messages.
   */
  int getWorkerId() {
    return this.workerId;
  }

  HashCode getWorkerFilesCombinedHash() {
    return workerKey.getWorkerFilesCombinedHash();
  }

  SortedMap<PathFragment, HashCode> getWorkerFilesWithHashes() {
    return workerKey.getWorkerFilesWithHashes();
  }

  boolean isAlive() {
    // This is horrible, but Process.isAlive() is only available from Java 8 on and this is the
    // best we can do prior to that.
    return !process.finished();
  }

  /** Returns true if this process is dead but we didn't deliberately kill it. */
  boolean diedUnexpectedly() {
    return process != null && !wasDestroyed && !process.isAlive();
  }

  /** Returns the exit value of this worker's process, if it has exited. */
  public Optional<Integer> getExitValue() {
    return process != null && !process.isAlive()
        ? Optional.of(process.exitValue())
        : Optional.empty();
  }

  void putRequest(WorkRequest request) throws IOException {
    request.writeDelimitedTo(process.getOutputStream());
    process.getOutputStream().flush();
  }

  WorkResponse getResponse() throws IOException {
    recordingStream = new RecordingInputStream(process.getInputStream());
    recordingStream.startRecording(4096);
    // response can be null when the worker has already closed stdout at this point and thus
    // the InputStream is at EOF.
    return WorkResponse.parseDelimitedFrom(recordingStream);
  }

  String getRecordingStreamMessage() {
    recordingStream.readRemaining();
    return recordingStream.getRecordedDataAsString();
  }

  public void prepareExecution(
      SandboxInputs inputFiles, SandboxOutputs outputs, Set<PathFragment> workerFiles)
      throws IOException {
    if (process == null) {
      process = createProcess();
    }
  }

  public void finishExecution(Path execRoot) throws IOException {}

  public Path getLogFile() {
    return logFile;
  }
}
