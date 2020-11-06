// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.List;
import java.util.function.BiFunction;

/**
 * A helper class that handles WorkRequests
 * (https://docs.bazel.build/versions/master/persistent-workers.html), including multiplex workers
 * (https://docs.bazel.build/versions/master/multiplex-worker.html).
 */
public class WorkRequestHandler implements AutoCloseable {

  /** Contains the logic for reading {@link WorkRequest}s and writing {@link WorkResponse}s. */
  public interface WorkerMessageProcessor {
    /** Reads the next incoming request from this worker's stdin. */
    public WorkRequest readWorkRequest() throws IOException;

    /**
     * Writes the provided {@link WorkResponse} to this worker's stdout. This function is also
     * responsible for flushing the stdout.
     */
    public void writeWorkResponse(WorkResponse workResponse) throws IOException;

    /** Clean up. */
    public void close() throws IOException;
  }

  /** The function to be called after each {@link WorkRequest} is read. */
  private final BiFunction<List<String>, PrintWriter, Integer> callback;

  /** This worker's stderr. */
  private final PrintStream stderr;

  private final WorkerMessageProcessor messageProcessor;

  /**
   * Creates a {@code WorkRequestHandler} that will call {@code callback} for each WorkRequest
   * received. The first argument to {@code callback} is the set of command-line arguments, the
   * second is where all error messages and similar should be written to. The callback should return
   * an exit code indicating success (0) or failure (nonzero).
   */
  public WorkRequestHandler(
      BiFunction<List<String>, PrintWriter, Integer> callback,
      PrintStream stderr,
      WorkerMessageProcessor messageProcessor) {
    this.callback = callback;
    this.stderr = stderr;
    this.messageProcessor = messageProcessor;
  }

  /**
   * Runs an infinite loop of reading {@link WorkRequest} from {@code in}, running the callback,
   * then writing the corresponding {@link WorkResponse} to {@code out}. If there is an error
   * reading or writing the requests or responses, it writes an error message on {@code err} and
   * returns. If {@code in} reaches EOF, it also returns.
   */
  public void processRequests() throws IOException {
    while (true) {
      WorkRequest request = messageProcessor.readWorkRequest();
        if (request == null) {
          break;
        }
        if (request.getRequestId() != 0) {
        Thread t = createResponseThread(request);
          t.start();
        } else {
        respondToRequest(request);
      }
    }
  }

  /** Creates a new {@link Thread} to process a multiplex request. */
  public Thread createResponseThread(WorkRequest request) {
    Thread currentThread = Thread.currentThread();
    return new Thread(
        () -> {
          try {
            respondToRequest(request);
          } catch (IOException e) {
            e.printStackTrace(stderr);
            // In case of error, shut down the entire worker.
            currentThread.interrupt();
          }
        },
        "multiplex-request-" + request.getRequestId());
  }

  /** Handles and responds to the given {@link WorkRequest}. */
  @VisibleForTesting
  void respondToRequest(WorkRequest request) throws IOException {
    try (StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw)) {
      int exitCode;
      try {
        exitCode = callback.apply(request.getArgumentsList(), pw);
      } catch (RuntimeException e) {
        e.printStackTrace(pw);
        exitCode = 1;
      }
      pw.flush();
      WorkResponse workResponse =
          WorkResponse.newBuilder()
              .setOutput(sw.toString())
              .setExitCode(exitCode)
              .setRequestId(request.getRequestId())
              .build();
      synchronized (this) {
        messageProcessor.writeWorkResponse(workResponse);
      }
    }
  }

  @Override
  public void close() throws IOException {
    messageProcessor.close();
  }
}
