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

import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.devtools.build.lib.vfs.Path;
import java.io.File;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Semaphore;

/** An intermediate worker that sends request and receives response from the worker processes. */
public class WorkerMultiplexer extends Thread {
  /**
   * WorkerMultiplexer is running as a thread on its own. When worker process
   * returns the WorkResponse, it is stored in this map and wait for
   * WorkerProxy to retrieve the response.
   */
  private Map<Integer, InputStream> workerProcessResponse;
  /** A semaphore to protect workerProcessResponse object. */
  private Semaphore semWorkerProcessResponse;
  /**
   * After sending the WorkRequest, WorkerProxy will wait on a semaphore to be
   * released. WorkerMultiplexer is responsible to release the corresponding
   * semaphore in order to signal WorkerProxy that the WorkerResponse has been
   * received.
   */
  private Map<Integer, Semaphore> responseChecker;
  /** A semaphore to protect responseChecker object. */
  private Semaphore semResponseChecker;
  /** The worker process that this WorkerMultiplexer should be talking to. */
  private Subprocess process;

  WorkerMultiplexer() {
    semWorkerProcessResponse = new Semaphore(1);
    semResponseChecker = new Semaphore(1);
    responseChecker = new HashMap<>();
    workerProcessResponse = new HashMap<>();

    final WorkerMultiplexer self = this;
  }

  /** Only start one worker process for each WorkerMultiplexer, if it hasn't. */
  public synchronized void createProcess(WorkerKey workerKey, Path workDir, Path logFile) throws IOException {
    if (this.process == null) {
      List<String> args = workerKey.getArgs();
      File executable = new File(args.get(0));
      if (!executable.isAbsolute() && executable.getParent() != null) {
        args = new ArrayList<>(args);
        args.set(0, new File(workDir.getPathFile(), args.get(0)).getAbsolutePath());
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

  /** Pass the WorkRequest to worker process. */
  public synchronized void putRequest(byte[] request) throws IOException {
    OutputStream stdin = process.getOutputStream();
    stdin.write(request);
    stdin.flush();
  }

  /** A WorkerProxy waits on a semaphore for the WorkResponse returned from worker process. */
  public InputStream getResponse(Integer workerId) throws InterruptedException {
    semResponseChecker.acquire();
    Semaphore waitForResponse = responseChecker.get(workerId);
    semResponseChecker.release();

    // The semaphore will throw InterruptedException when the process is terminated.
    waitForResponse.acquire();

    semWorkerProcessResponse.acquire();
    InputStream response = workerProcessResponse.get(workerId);
    semWorkerProcessResponse.release();
    return response;
  }

  /** Reset the map that indicates if the WorkResponses have been returned. */
  public void resetResponseChecker(Integer workerId) throws InterruptedException {
    semResponseChecker.acquire();
    responseChecker.put(workerId, new Semaphore(0));
    semResponseChecker.release();
  }

  /**
   * When it gets a WorkResponse from worker process, put that WorkResponse in
   * workerProcessResponse and signal responseChecker.
   */
  private void waitResponse() throws InterruptedException, IOException {
    InputStream stdout = process.getInputStream();
    WorkResponse parsedResponse = WorkResponse.parseDelimitedFrom(stdout);

    if (parsedResponse == null) return;

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

  /** A multiplexer thread that listens to the WorkResponses from worker process. */
  public void run() {
    while (!this.interrupted()) {
      try {
        waitResponse();
      } catch (Exception e) {
        // We can't do anything here.
      }
    }
  }
}
