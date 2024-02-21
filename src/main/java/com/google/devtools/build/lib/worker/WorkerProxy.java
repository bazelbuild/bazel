// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.worker;

import com.google.common.flogger.GoogleLogger;
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

/** A proxy that talks to the multiplexer */
class WorkerProxy extends Worker {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  protected final WorkerMultiplexer workerMultiplexer;

  /** The execution root of the worker. This is the CWD of the worker process. */
  protected final Path workDir;

  WorkerProxy(
      WorkerKey workerKey,
      int workerId,
      Path logFile,
      WorkerMultiplexer workerMultiplexer,
      Path workDir) {
    // Worker proxies of the same multiplexer share a WorkerProcessStatus.
    super(workerKey, workerId, logFile, workerMultiplexer.getStatus());
    this.workDir = workDir;
    this.workerMultiplexer = workerMultiplexer;
  }

  @Override
  public Optional<VirtualCGroup> getCgroup() {
    // WorkerProxy does not have a cgroup at the momemnt. Consider adding it to the
    // multiplexer and returning it here?
    return Optional.empty();
  }

  @Override
  public boolean isSandboxed() {
    return false;
  }

  @Override
  void setReporter(EventHandler reporter) {
    // We might have created this multiplexer after setting the reporter for existing multiplexers
    workerMultiplexer.setReporter(reporter);
  }

  @Override
  public void prepareExecution(
      SandboxInputs inputFiles, SandboxOutputs outputs, Set<PathFragment> workerFiles)
      throws IOException, InterruptedException {
    workerMultiplexer.createProcess(workDir);
  }

  @Override
  synchronized void destroy() {
    try {
      WorkerMultiplexerManager.removeInstance(workerKey);
    } catch (UserExecException e) {
      logger.atWarning().withCause(e).log("Exception");
    }
  }

  /** Send the WorkRequest to multiplexer. */
  @Override
  protected void putRequest(WorkRequest request) throws IOException {
    workerMultiplexer.putRequest(request);
  }

  /** Wait for WorkResponse from multiplexer. */
  @Override
  WorkResponse getResponse(int requestId) throws InterruptedException, IOException {
    return workerMultiplexer.getResponse(requestId);
  }

  @Override
  boolean diedUnexpectedly() {
    return workerMultiplexer.diedUnexpectedly();
  }

  @Override
  public Optional<Integer> getExitValue() {
    return workerMultiplexer.getExitValue();
  }

  @Override
  String getRecordingStreamMessage() {
    return workerMultiplexer.getRecordingStreamMessage();
  }

  @Override
  public String toString() {
    return workerKey.getMnemonic() + " proxy worker #" + workerId;
  }

  @Override
  public long getProcessId() {
    return workerMultiplexer.getProcessId();
  }
}
