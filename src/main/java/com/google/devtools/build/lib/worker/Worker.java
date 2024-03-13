// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.sandbox.cgroups.VirtualCGroup;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.IOException;
import java.util.Optional;
import java.util.Set;
import java.util.SortedMap;

/**
 * An abstract superclass for persistent workers. Workers execute actions in long-running processes
 * that can handle multiple actions.
 */
public abstract class Worker {

  /** An unique identifier of the work process. */
  protected final WorkerKey workerKey;

  /** An unique ID of the worker. It will be used in WorkRequest and WorkResponse as well. */
  protected final int workerId;

  /** The path of the log file for this worker. */
  protected final Path logFile;

  public Worker(WorkerKey workerKey, int workerId, Path logFile, WorkerProcessStatus status) {
    this.workerKey = workerKey;
    this.workerId = workerId;
    this.logFile = logFile;
    this.status = status;
  }

  protected final WorkerProcessStatus status;

  /**
   * Returns a unique id for this worker. This is used to distinguish different worker processes in
   * logs and messages.
   */
  public int getWorkerId() {
    return this.workerId;
  }

  /** Returns the path of the log file for this worker. */
  public Path getLogFile() {
    return logFile;
  }

  /** Returns the worker key of this worker */
  public WorkerKey getWorkerKey() {
    return workerKey;
  }

  public WorkerProcessStatus getStatus() {
    return status;
  }

  abstract public VirtualCGroup getCgroup();

  HashCode getWorkerFilesCombinedHash() {
    return workerKey.getWorkerFilesCombinedHash();
  }

  SortedMap<PathFragment, byte[]> getWorkerFilesWithDigests() {
    return workerKey.getWorkerFilesWithDigests();
  }

  /** Returns true if this worker is sandboxed. */
  public abstract boolean isSandboxed();

  /**
   * Sets the reporter this {@code Worker} should report anomalous events to, or clears it. We
   * expect the reporter to be cleared at end of build.
   */
  void setReporter(EventHandler reporter) {}

  /**
   * Performs the necessary steps to prepare for execution. Once this is done, the worker should be
   * able to receive a WorkRequest without further setup.
   */
  public abstract void prepareExecution(
      SandboxInputs inputFiles, SandboxOutputs outputs, Set<PathFragment> workerFiles)
      throws IOException, InterruptedException, UserExecException;

  /**
   * Sends a WorkRequest to the worker.
   *
   * @param request The request to send.
   * @throws IOException If there was a problem doing I/O, or this thread was interrupted at a time
   *     where some or all of the expected I/O has been done.
   */
  abstract void putRequest(WorkRequest request) throws IOException;

  /**
   * Waits to receive a response from the worker. This method should return as soon as a response
   * has been received, moving of files and cleanup should wait until finishExecution().
   *
   * @param requestId ID of the request to retrieve a response for.
   * @return The WorkResponse received.
   * @throws IOException If there was a problem doing I/O.
   * @throws InterruptedException If this thread was interrupted, which can also happen during IO.
   */
  abstract WorkResponse getResponse(int requestId) throws IOException, InterruptedException;

  /**
   * Does whatever cleanup may be required after execution is done.
   *
   * @param execRoot The global execRoot, where outputs must go.
   * @param outputs The expected outputs.
   */
  public void finishExecution(Path execRoot, SandboxOutputs outputs) throws IOException {
    status.maybeUpdateStatus(WorkerProcessStatus.Status.ALIVE);
  }

  /**
   * Destroys this worker. Once this has been called, we assume it's safe to clean up related
   * directories.
   */
  abstract void destroy();

  /** Returns true if this worker is dead but we didn't deliberately kill it. */
  abstract boolean diedUnexpectedly();

  /** Returns the exit value of this worker's process, if it has exited. */
  public abstract Optional<Integer> getExitValue();

  /**
   * Returns the last message received on the InputStream, if an unparseable message has been
   * received.
   */
  abstract String getRecordingStreamMessage();

  /** Returns process id pf worker, if process started. Otherwise returns -1. */
  abstract long getProcessId();
}
