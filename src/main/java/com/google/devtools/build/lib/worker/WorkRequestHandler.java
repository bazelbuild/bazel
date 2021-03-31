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
import com.sun.management.OperatingSystemMXBean;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.management.ManagementFactory;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
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

  final WorkerMessageProcessor messageProcessor;

  private final CpuTimeBasedGcScheduler gcScheduler;

  /**
   * Creates a {@code WorkRequestHandler} that will call {@code callback} for each WorkRequest
   * received.
   *
   * @param callback Callback method for executing a single WorkRequest in a thread. The first
   *     argument to {@code callback} is the set of command-line arguments, the second is where all
   *     error messages and other user-oriented messages should be written to. The callback must
   *     return an exit code indicating success (zero) or failure (nonzero).
   * @param stderr Stream that log messages should be written to, typically the process' stderr.
   * @param messageProcessor Object responsible for parsing {@code WorkRequest}s from the server and
   *     writing {@code WorkResponses} to the server.
   */
  public WorkRequestHandler(
      BiFunction<List<String>, PrintWriter, Integer> callback,
      PrintStream stderr,
      WorkerMessageProcessor messageProcessor) {
    this(callback, stderr, messageProcessor, Duration.ZERO);
  }

  /**
   * Creates a {@code WorkRequestHandler} that will call {@code callback} for each WorkRequest
   * received.
   *
   * @param callback Callback method for executing a single WorkRequest in a thread. The first
   *     argument to {@code callback} is the set of command-line arguments, the second is where all
   *     error messages and other user-oriented messages should be written to. The callback must
   *     return an exit code indicating success (zero) or failure (nonzero).
   * @param stderr Stream that log messages should be written to, typically the process' stderr.
   * @param messageProcessor Object responsible for parsing {@code WorkRequest}s from the server and
   *     writing {@code WorkResponses} to the server.
   * @param cpuUsageBeforeGc The minimum amount of CPU time between explicit garbage collection
   *     calls.
   */
  public WorkRequestHandler(
      BiFunction<List<String>, PrintWriter, Integer> callback,
      PrintStream stderr,
      WorkerMessageProcessor messageProcessor,
      Duration cpuUsageBeforeGc) {
    this.callback = callback;
    this.stderr = stderr;
    this.messageProcessor = messageProcessor;
    this.gcScheduler = new CpuTimeBasedGcScheduler(cpuUsageBeforeGc);
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
      gcScheduler.maybePerformGc();
    }
  }

  @Override
  public void close() throws IOException {
    messageProcessor.close();
  }

  /**
   * Class that performs GC occasionally, based on how much CPU time has passed. This strikes a
   * compromise between blindly doing GC after e.g. every request, which takes too much CPU, and not
   * doing explicit GC at all, which causes poor garbage collection in some cases.
   */
  private static class CpuTimeBasedGcScheduler {
    /**
     * After this much CPU time has elapsed, we may force a GC run. Set to {@link Duration#ZERO} to
     * disable.
     */
    private final Duration cpuUsageBeforeGc;

    /** The total process CPU time at the last GC run (or from the start of the worker). */
    private final AtomicReference<Duration> cpuTimeAtLastGc;

    /** Used to get the CPU time used by this process. */
    private static final OperatingSystemMXBean bean =
        (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();

    /**
     * Creates a new {@link CpuTimeBasedGcScheduler} that may perform GC after {@code
     * cpuUsageBeforeGc} amount of CPU time has been used.
     */
    public CpuTimeBasedGcScheduler(Duration cpuUsageBeforeGc) {
      this.cpuUsageBeforeGc = cpuUsageBeforeGc;
      this.cpuTimeAtLastGc = new AtomicReference<>(getCpuTime());
    }

    private Duration getCpuTime() {
      return !cpuUsageBeforeGc.isZero()
          ? Duration.ofNanos(bean.getProcessCpuTime())
          : Duration.ZERO;
    }

    /** Call occasionally to perform a GC if enough CPU time has been used. */
    private void maybePerformGc() {
      if (!cpuUsageBeforeGc.isZero()) {
        Duration currentCpuTime = getCpuTime();
        Duration lastCpuTime = cpuTimeAtLastGc.get();
        // Do GC when enough CPU time has been used, but only if nobody else beat us to it.
        if (currentCpuTime.minus(lastCpuTime).compareTo(cpuUsageBeforeGc) > 0
            && cpuTimeAtLastGc.compareAndSet(lastCpuTime, currentCpuTime)) {
          System.gc();
          // Avoid counting GC CPU time against CPU time before next GC.
          cpuTimeAtLastGc.compareAndSet(currentCpuTime, getCpuTime());
        }
      }
    }
  }
}
