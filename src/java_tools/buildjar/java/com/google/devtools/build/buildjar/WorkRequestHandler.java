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
package com.google.devtools.build.buildjar;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.IOException;
import java.io.InputStream;
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
public class WorkRequestHandler {

  private final BiFunction<List<String>, PrintWriter, Integer> callback;

  /**
   * Creates a {@code WorkRequestHandler} that will call {@code callback} for each WorkRequest
   * received. The first argument to {@code callback} is the set of command-line arguments, the
   * second is where all error messages and similar should be written to.
   */
  public WorkRequestHandler(BiFunction<List<String>, PrintWriter, Integer> callback) {
    this.callback = callback;
  }

  /**
   * Runs an infinite loop of reading {@code WorkRequest} from {@code in}, running the callback,
   * then writing the corresponding {@code WorkResponse} to {@code out}. If there is an error
   * reading or writing the requests or responses, it writes an error message on {@code err} and
   * returns. If {@code in} reaches EOF, it also returns.
   *
   * @return 0 if we reached EOF, 1 if there was an error.
   */
  public int processRequests(InputStream in, PrintStream out, PrintStream err) {
    while (true) {
      try {
        WorkRequest request = WorkRequest.parseDelimitedFrom(in);

        if (request == null) {
          break;
        }

        if (request.getRequestId() != 0) {
          Thread t = createResponseThread(request, out, err);
          t.start();
        } else {
          respondToRequest(request, out);
        }
      } catch (IOException e) {
        e.printStackTrace(err);
        return 1;
      }
    }
    return 0;
  }

  /** Creates a new {@code Thread} to process a multiplex request. */
  public Thread createResponseThread(WorkRequest request, PrintStream out, PrintStream err) {
    Thread currentThread = Thread.currentThread();
    return new Thread(
        () -> {
          try {
            respondToRequest(request, out);
          } catch (IOException e) {
            e.printStackTrace(err);
            // In case of error, shut down the entire worker.
            currentThread.interrupt();
          }
        },
        "multiplex-request-" + request.getRequestId());
  }

  /** Responds to {@code request}, writing the {@code WorkResponse} proto to {@code out}. */
  @VisibleForTesting
  void respondToRequest(WorkRequest request, PrintStream out) throws IOException {
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
        workResponse.writeDelimitedTo(out);
      }
    }
    out.flush();

    // This would be a tempting place to suggest a GC, but it causes a 10% performance hit.
  }
}
