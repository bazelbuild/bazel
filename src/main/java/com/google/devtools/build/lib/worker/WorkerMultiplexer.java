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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.shell.SubprocessFactory;
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
import java.util.Optional;
import java.util.concurrent.Semaphore;

/**
 * An intermediate worker that sends requests and receives responses from the worker processes.
 * There is at most one of these per {@code WorkerKey}, corresponding to one worker process. {@code
 * WorkerMultiplexer} objects run in separate long-lived threads. {@code WorkerProxy} objects call
 * into them to send requests. When a worker process returns a {@code WorkResponse}, {@code
 * WorkerMultiplexer} wakes up the relevant {@code WorkerProxy} to retrieve the response.
 */
public class WorkerMultiplexer extends Thread {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  /**
   * A map of {@code WorkResponse}s received from the worker process. They are stored in this map
   * until the corresponding {@code WorkerProxy} picks them up.
   */
  private final Map<Integer, InputStream> workerProcessResponse;
  /** A semaphore to protect {@code workerProcessResponse} object. */
  private final Semaphore semWorkerProcessResponse;
  /**
   * A map of semaphores corresponding to {@code WorkRequest}s. After sending the {@code
   * WorkRequest}, {@code WorkerProxy} will wait on a semaphore to be released. {@code
   * WorkerMultiplexer} is responsible for releasing the corresponding semaphore in order to signal
   * {@code WorkerProxy} that the {@code WorkerResponse} has been received.
   */
  private final Map<Integer, Semaphore> responseChecker;
  /** A semaphore to protect responseChecker object. */
  private final Semaphore semResponseChecker;
  /** The worker process that this WorkerMultiplexer should be talking to. */
  private Subprocess process;
  /**
   * Set to true if one of the worker processes returns an unparseable response. We then discard all
   * the responses from other worker processes and abort.
   */
  private boolean isUnparseable;
  /** InputStream from the worker process. */
  private RecordingInputStream recordingStream;
  /**
   * True if we have received EOF on the stream from the worker process. We then stop processing,
   * and all workers still waiting for responses will fail.
   */
  private boolean isWorkerStreamClosed;
  /** True if the multiplexer thread has been interrupted. */
  private boolean isInterrupted;
  /**
   * The log file of the actual running worker process. It is shared between all WorkerProxy
   * instances for this multiplexer.
   */
  private final Path logFile;

  /** For testing only, allow a way to fake subprocesses. */
  private SubprocessFactory subprocessFactory;

  WorkerMultiplexer(Path logFile) {
    semWorkerProcessResponse = new Semaphore(1);
    semResponseChecker = new Semaphore(1);
    responseChecker = new HashMap<>();
    workerProcessResponse = new HashMap<>();
    isUnparseable = false;
    isWorkerStreamClosed = false;
    isInterrupted = false;
    this.logFile = logFile;
  }

  /**
   * Creates a worker process corresponding to this {@code WorkerMultiplexer}, if it doesn't already
   * exist. Also makes sure this {@code WorkerMultiplexer} runs as a separate thread.
   */
  public synchronized void createProcess(WorkerKey workerKey, Path workDir) throws IOException {
    // The process may have died in the meanwhile (e.g. between builds).
    if (this.process == null || !this.process.isAlive()) {
      ImmutableList<String> args = workerKey.getArgs();
      File executable = new File(args.get(0));
      if (!executable.isAbsolute() && executable.getParent() != null) {
        List<String> newArgs = new ArrayList<>(args);
        newArgs.set(0, new File(workDir.getPathFile(), newArgs.get(0)).getAbsolutePath());
        args = ImmutableList.copyOf(newArgs);
      }
      SubprocessBuilder processBuilder =
          subprocessFactory != null
              ? new SubprocessBuilder(subprocessFactory)
              : new SubprocessBuilder();
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

  /**
   * Returns the path of the log file shared by all multiplex workers using this process. May be
   * null if the process has not started yet.
   */
  public Path getLogFile() {
    return logFile;
  }

  /**
   * Signals this object to destroy itself, including the worker process. The object might not be
   * fully destroyed at the end of this call, but will terminate soon.
   */
  public synchronized void destroyMultiplexer() {
    if (this.process != null) {
      destroyProcess(this.process);
      this.process = null;
    }
    isInterrupted = true;
  }

  /** Destroys the worker subprocess. This might block forever if the subprocess refuses to die. */
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

  /**
   * Sends the WorkRequest to worker process. This method is called on the thread of a {@code
   * WorkerProxy}.
   */
  public synchronized void putRequest(WorkRequest request) throws IOException {
    request.writeDelimitedTo(process.getOutputStream());
    process.getOutputStream().flush();
  }

  /**
   * Waits on a semaphore for the {@code WorkResponse} returned from worker process. This method is
   * called on the thread of a {@code WorkerProxy}.
   */
  public InputStream getResponse(Integer requestId) throws IOException, InterruptedException {
    try {
      semResponseChecker.acquire();
      Semaphore waitForResponse = responseChecker.get(requestId);
      semResponseChecker.release();

      if (waitForResponse == null) {
        // If the multiplexer is interrupted when a {@code WorkerProxy} is trying to send a request,
        // the request is not sent, so there is no need to wait for a response.
        return null;
      }

      // Wait for the multiplexer to get our response and release this semaphore. The semaphore will
      // throw {@code InterruptedException} when the multiplexer is terminated.
      waitForResponse.acquire();

      if (isWorkerStreamClosed) {
        return null;
      }

      if (isUnparseable) {
        recordingStream.readRemaining();
        throw new IOException(recordingStream.getRecordedDataAsString());
      }

      semWorkerProcessResponse.acquire();
      InputStream response = workerProcessResponse.get(requestId);
      semWorkerProcessResponse.release();
      return response;
    } finally {
      // TODO(b/151767359): Make sure these also get cleared if a worker gets
      semResponseChecker.acquire();
      responseChecker.remove(requestId);
      semResponseChecker.release();
      semWorkerProcessResponse.acquire();
      workerProcessResponse.remove(requestId);
      semWorkerProcessResponse.release();
    }
  }

  /**
   * Resets the semaphore map for {@code requestId} before sending a request to the worker process.
   * This method is called on the thread of a {@code WorkerProxy}.
   */
  void resetResponseChecker(Integer requestId) throws InterruptedException {
    semResponseChecker.acquire();
    responseChecker.put(requestId, new Semaphore(0));
    semResponseChecker.release();
  }

  /**
   * Waits to read a {@code WorkResponse} from worker process, put that {@code WorkResponse} in
   * {@code workerProcessResponse} and release the semaphore for the {@code WorkerProxy}.
   */
  private void waitResponse() throws InterruptedException, IOException {
    Subprocess p = this.process;
    if (p == null || !p.isAlive()) {
      // Avoid busy-wait for a new process.
      Thread.sleep(1);
      return;
    }
    recordingStream = new RecordingInputStream(p.getInputStream());
    recordingStream.startRecording(4096);
    WorkResponse parsedResponse = WorkResponse.parseDelimitedFrom(recordingStream);

    // A null parsedResponse can only happen if the input stream is closed.
    if (parsedResponse == null) {
      isWorkerStreamClosed = true;
      releaseAllSemaphores();
      return;
    }

    int requestId = parsedResponse.getRequestId();
    ByteArrayOutputStream tempOs = new ByteArrayOutputStream();
    parsedResponse.writeDelimitedTo(tempOs);

    semWorkerProcessResponse.acquire();
    workerProcessResponse.put(requestId, new ByteArrayInputStream(tempOs.toByteArray()));
    semWorkerProcessResponse.release();

    // TODO(b/151767359): When allowing cancellation, remove responses that have no matching
    // entry in responseChecker.
    semResponseChecker.acquire();
    responseChecker.get(requestId).release();
    semResponseChecker.release();
  }

  /** The multiplexer thread that listens to the WorkResponse from worker process. */
  @Override
  public void run() {
    while (!isInterrupted) {
      try {
        waitResponse();
      } catch (IOException e) {
        isUnparseable = true;
        releaseAllSemaphores();
        logger.atWarning().withCause(e).log(
            "IOException was caught while waiting for worker response. "
                + "It could because the worker returned unparseable response.");
      } catch (InterruptedException e) {
        logger.atWarning().withCause(e).log(
            "InterruptedException was caught while waiting for worker response. "
                + "It could because the multiplexer was interrupted.");
      }
    }
    logger.atWarning().log(
        "Multiplexer thread has been terminated. It could because the memory is running low on"
            + " your machine. There may be other reasons.");
  }

  /** Release all the semaphores */
  private void releaseAllSemaphores() {
    try {
      semResponseChecker.acquire();
      for (Integer requestId : responseChecker.keySet()) {
        responseChecker.get(requestId).release();
      }
    } catch (InterruptedException e) {
      // Do nothing
    } finally {
      semResponseChecker.release();
    }
  }

  String getRecordingStreamMessage() {
    return recordingStream.getRecordedDataAsString();
  }

  /** Returns true if this process has died for other reasons than a call to {@code #destroy()}. */
  boolean diedUnexpectedly() {
    Subprocess p = this.process; // Protects against this.process getting null.
    return p != null && !p.isAlive() && !isInterrupted;
  }

  /** Returns the exit value of multiplexer's process, if it has exited. */
  Optional<Integer> getExitValue() {
    Subprocess p = this.process; // Protects against this.process getting null.
    return p != null && !p.isAlive() ? Optional.of(p.exitValue()) : Optional.empty();
  }

  /** For testing only, to verify that maps are cleared after responses are reaped. */
  @VisibleForTesting
  boolean noOutstandingRequests() {
    return responseChecker.isEmpty() && workerProcessResponse.isEmpty();
  }

  @VisibleForTesting
  void setProcessFactory(SubprocessFactory factory) {
    subprocessFactory = factory;
  }
}
