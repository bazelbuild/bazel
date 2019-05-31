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

import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A proxy that talks to the multiplexer
 */
final class WorkerProxy extends Worker {
  private static final Logger logger = Logger.getLogger(WorkerProxy.class.getName());
  private ByteArrayOutputStream request;
  private WorkerMultiplexer workerMultiplexer;
  private Thread shutdownHook;

  WorkerProxy(WorkerKey workerKey, int workerId, Path workDir, Path logFile, WorkerMultiplexer workerMultiplexer) {
    super(workerKey, workerId, workDir, logFile);
    request = new ByteArrayOutputStream();
    this.workerMultiplexer = workerMultiplexer;

    final WorkerProxy self = this;
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

  @Override
  void createProcess() throws IOException {
    workerMultiplexer.createProcess(workerKey, workDir, logFile);
  }

  @Override
  boolean isAlive() {
    return workerMultiplexer.isProcessAlive();
  }

  @Override
  public void prepareExecution(
      Map<PathFragment, Path> inputFiles, SandboxHelpers.SandboxOutputs outputs, Set<PathFragment> workerFiles)
      throws IOException {
    createProcess();
  }

  @Override
  synchronized void destroy() throws IOException {
    if (shutdownHook != null) {
      Runtime.getRuntime().removeShutdownHook(shutdownHook);
    }
    try {
      WorkerMultiplexerManager.removeInstance(workerKey.hashCode());
    } catch (InterruptedException e) {
      logger.warning("InterruptedException was caught while destroying multiplexer. "
          + "It could because the multiplexer was interrupted.");
    }
  }

  /** Send the WorkRequest to multiplexer. */
  @Override
  void putRequest(WorkRequest request) throws IOException {
    try {
      workerMultiplexer.resetResponseChecker(workerId);
      workerMultiplexer.putRequest(request);
    } catch (InterruptedException e) {
      /**
       * We can't throw InterruptedException to WorkerSpawnRunner because of the principle of override.
       * InterruptedException will happen when Bazel is waiting for semaphore but user terminates the
       * process, so we do nothing here.
       */
      logger.warning("InterruptedException was caught while sending worker request. "
          + "It could because the multiplexer was interrupted.");
    }
  }

  /** Wait for WorkResponse from multiplexer. */
  @Override
  WorkResponse getResponse() throws IOException {
    try {
      recordingStream = new RecordingInputStream(workerMultiplexer.getResponse(workerId));
      recordingStream.startRecording(4096);
    } catch (InterruptedException e) {
      /**
       * We can't throw InterruptedException to WorkerSpawnRunner because of the principle of override.
       * InterruptedException will happen when Bazel is waiting for semaphore but user terminates the
       * process, so we do nothing here.
       */
      logger.warning("InterruptedException was caught while waiting for work response. "
          + "It could because the multiplexer was interrupted.");
    }
    // response can be null when the worker has already closed stdout at this point and thus
    // the InputStream is at EOF.
    return WorkResponse.parseDelimitedFrom(recordingStream);
  }
}
