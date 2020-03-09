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

package com.google.devtools.build.lib.worker;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Semaphore;
import java.util.logging.Logger;

/** An intermediate worker that sends request and receives response from the worker process. */
public class WorkerMultiplexer extends Thread {
  private static final Logger logger = Logger.getLogger(WorkerMultiplexer.class.getName());
  /**
   * WorkerMultiplexer is running as a thread on its own. When worker process returns the
   * WorkResponse, it is stored in this map and wait for WorkerProxy to retrieve the response.
   */
  private Map<Integer, InputStream> workerProcessResponse;
  /** A semaphore to protect workerProcessResponse object. */
  private Semaphore semWorkerProcessResponse;
  /**
   * After sending the WorkRequest, WorkerProxy will wait on a semaphore to be released.
   * WorkerMultiplexer is responsible to release the corresponding semaphore in order to signal
   * WorkerProxy that the WorkerResponse has been received.
   */
  private Map<Integer, Semaphore> responseChecker;
  /** A semaphore to protect responseChecker object. */
  private Semaphore semResponseChecker;
  /** The worker process that this WorkerMultiplexer should be talking to. */
  private Subprocess process;
  /**
   * If one of the worker processes returns unparseable response, discard all the responses from
   * other worker processes.
   */
  private boolean isUnparseable;
  /** InputStream from worker process. */
  private RecordingInputStream recordingStream;
  /** If worker process returns null, return null to WorkerProxy and discard all the responses. */
  private boolean isNull;
  /** A flag to stop multiplexer thread. */
  private boolean isInterrupted;

  WorkerMultiplexer() {
    semWorkerProcessResponse = new Semaphore(1);
    semResponseChecker = new Semaphore(1);
    responseChecker = new HashMap<>();
    workerProcessResponse = new HashMap<>();
    isUnparseable = false;
    isNull = false;
    isInterrupted = false;
  }

  /** Only start one worker process for each WorkerMultiplexer, if it hasn't. */
  public synchronized void createProcess(WorkerKey workerKey, Path workDir, Path logFile)
      throws IOException {
    if (this.process == null) {
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
      this.process = processBuilder.start();
    }
    if (!this.isAlive()) {
      this.start();
    }
  }

  public synchronized void destroyMultiplexer() {
    if (this.process != null) {
      destroyProcess(this.process);
    }
    isInterrupted = true;
  }

  private void destroyProcess(Subprocess process) {
    boolean wasInterrupted = false;
    try {
      process.destroy();
      while (true) {
        try {
          process.waitFor();
          return;
        } catch (InterruptedException ie) {
          wasInterrupted = true;
        }
      }
    } finally {
      // Read this for detailed explanation: http://www.ibm.com/developerworks/library/j-jtp05236/
      if (wasInterrupted) {
        Thread.currentThread().interrupt(); // preserve interrupted status
      }
    }
  }

  public boolean isProcessAlive() {
    return !this.process.finished();
  }

  /** Send the WorkRequest to worker process. */
  public synchronized void putRequest(WorkRequest request) throws IOException {
    request.writeDelimitedTo(process.getOutputStream());
    process.getOutputStream().flush();
  }

  /** Wait on a semaphore for the WorkResponse returned from worker process. */
  public InputStream getResponse(Integer workerId) throws IOException, InterruptedException {
    semResponseChecker.acquire();
    Semaphore waitForResponse = responseChecker.get(workerId);
    semResponseChecker.release();

    if (waitForResponse == null) {
      /**
       * If multiplexer is interrupted when WorkerProxy is trying to send request, request is not
       * sent so no need to wait for response.
       */
      return null;
    }

    // The semaphore will throw InterruptedException when the multiplexer is terminated.
    waitForResponse.acquire();

    if (isNull) {
      return null;
    }

    if (isUnparseable) {
      recordingStream.readRemaining();
      throw new IOException(recordingStream.getRecordedDataAsString());
    }

    semWorkerProcessResponse.acquire();
    InputStream response = workerProcessResponse.get(workerId);
    semWorkerProcessResponse.release();
    return response;
  }

  /** Reset the semaphore map before sending request to worker process. */
  public void resetResponseChecker(Integer workerId) throws InterruptedException {
    semResponseChecker.acquire();
    responseChecker.put(workerId, new Semaphore(0));
    semResponseChecker.release();
  }

  /**
   * When it gets a WorkResponse from worker process, put that WorkResponse in workerProcessResponse
   * and signal responseChecker.
   */
  private void waitResponse() throws InterruptedException, IOException {
    recordingStream = new RecordingInputStream(process.getInputStream());
    recordingStream.startRecording(4096);
    WorkResponse parsedResponse = WorkResponse.parseDelimitedFrom(recordingStream);

    if (parsedResponse == null) {
      isNull = true;
      releaseAllSemaphores();
      return;
    }

    Integer workerId = parsedResponse.getRequestId();
    ByteArrayOutputStream tempOs = new ByteArrayOutputStream();
    parsedResponse.writeDelimitedTo(tempOs);

    semWorkerProcessResponse.acquire();
    workerProcessResponse.put(workerId, new ByteArrayInputStream(tempOs.toByteArray()));
    semWorkerProcessResponse.release();

    semResponseChecker.acquire();
    responseChecker.get(workerId).release();
    semResponseChecker.release();
  }

  /** A multiplexer thread that listens to the WorkResponse from worker process. */
  @Override
  public void run() {
    while (!isInterrupted) {
      try {
        waitResponse();
      } catch (IOException e) {
        isUnparseable = true;
        releaseAllSemaphores();
        logger.warning(
            "IOException was caught while waiting for worker response. "
                + "It could because the worker returned unparseable response.");
      } catch (InterruptedException e) {
        logger.warning(
            "InterruptedException was caught while waiting for worker response. "
                + "It could because the multiplexer was interrupted.");
      }
    }
    logger.warning(
        "Multiplexer thread has been terminated. It could because the memory is running low on"
            + " your machine. There may be other reasons.");
  }

  /** Release all the semaphores */
  private void releaseAllSemaphores() {
    try {
      semResponseChecker.acquire();
      for (Integer workerId : responseChecker.keySet()) {
        responseChecker.get(workerId).release();
      }
    } catch (InterruptedException e) {
      // Do nothing
    } finally {
      semResponseChecker.release();
    }
  }
}
