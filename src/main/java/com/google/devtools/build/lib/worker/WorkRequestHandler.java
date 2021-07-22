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
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;

/**
 * A helper class that handles WorkRequests
 * (https://docs.bazel.build/versions/main/persistent-workers.html), including multiplex workers
 * (https://docs.bazel.build/versions/main/multiplex-worker.html).
 */
public class WorkRequestHandler implements AutoCloseable {
  /** Contains the logic for reading {@link WorkRequest}s and writing {@link WorkResponse}s. */
  public interface WorkerMessageProcessor {
    /** Reads the next incoming request from this worker's stdin. */
    WorkRequest readWorkRequest() throws IOException;

    /**
     * Writes the provided {@link WorkResponse} to this worker's stdout. This function is also
     * responsible for flushing the stdout.
     */
    void writeWorkResponse(WorkResponse workResponse) throws IOException;

    /** Clean up. */
    void close() throws IOException;
  }

  /** Holds information necessary to properly handle a request, especially for cancellation. */
  static class RequestInfo {
    /** The thread handling the request. */
    final Thread thread;
    /** If true, we have received a cancel request for this request. */
    private boolean cancelled;
    /**
     * The builder for the response to this request. Since only one response must be sent per
     * request, this builder must be accessed through takeBuilder(), which zeroes this field and
     * returns the builder.
     */
    private WorkResponse.Builder responseBuilder = WorkResponse.newBuilder();

    RequestInfo(Thread thread) {
      this.thread = thread;
    }

    /** Sets whether this request has been cancelled. */
    void setCancelled() {
      cancelled = true;
    }

    /** Returns true if this request has been cancelled. */
    boolean isCancelled() {
      return cancelled;
    }

    /**
     * Returns the response builder. If called more than once on the same instance, subsequent calls
     * will return {@code null}.
     */
    synchronized Optional<WorkResponse.Builder> takeBuilder() {
      WorkResponse.Builder b = responseBuilder;
      responseBuilder = null;
      return Optional.ofNullable(b);
    }

    /**
     * Adds {@code s} as output to when the response eventually gets built. Does nothing if the
     * response has already been taken. There is no guarantee that the response hasn't already been
     * taken, making this call a no-op. This may be called multiple times. No delimiters are added
     * between strings from multiple calls.
     */
    synchronized void addOutput(String s) {
      if (responseBuilder != null) {
        responseBuilder.setOutput(responseBuilder.getOutput() + s);
      }
    }
  }

  /** Requests that are currently being processed. Visible for testing. */
  final ConcurrentMap<Integer, RequestInfo> activeRequests = new ConcurrentHashMap<>();

  /** The function to be called after each {@link WorkRequest} is read. */
  private final WorkRequestCallback callback;

  /** This worker's stderr. */
  private final PrintStream stderr;

  final WorkerMessageProcessor messageProcessor;

  private final BiConsumer<Integer, Thread> cancelCallback;

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
  @Deprecated
  public WorkRequestHandler(
      BiFunction<List<String>, PrintWriter, Integer> callback,
      PrintStream stderr,
      WorkerMessageProcessor messageProcessor) {
    this(callback, stderr, messageProcessor, Duration.ZERO, null);
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
   *     calls. Pass Duration.ZERO to not do explicit garbage collection.
   * @deprecated Use WorkRequestHandlerBuilder instead.
   */
  @Deprecated()
  public WorkRequestHandler(
      BiFunction<List<String>, PrintWriter, Integer> callback,
      PrintStream stderr,
      WorkerMessageProcessor messageProcessor,
      Duration cpuUsageBeforeGc) {
    this(callback, stderr, messageProcessor, cpuUsageBeforeGc, null);
  }

  /**
   * Creates a {@code WorkRequestHandler} that will call {@code callback} for each WorkRequest
   * received. Only used for the Builder.
   *
   * @deprecated Use WorkRequestHandlerBuilder instead.
   */
  @Deprecated
  private WorkRequestHandler(
      BiFunction<List<String>, PrintWriter, Integer> callback,
      PrintStream stderr,
      WorkerMessageProcessor messageProcessor,
      Duration cpuUsageBeforeGc,
      BiConsumer<Integer, Thread> cancelCallback) {
    this(
        new WorkRequestCallback((request, pw) -> callback.apply(request.getArgumentsList(), pw)),
        stderr,
        messageProcessor,
        cpuUsageBeforeGc,
        cancelCallback);
  }

  /**
   * Creates a {@code WorkRequestHandler} that will call {@code callback} for each WorkRequest
   * received. Only used for the Builder.
   *
   * @param callback WorkRequestCallback object with Callback method for executing a single
   *     WorkRequest in a thread. The first argument to {@code callback} is the WorkRequest, the
   *     second is where all error messages and other user-oriented messages should be written to.
   *     The callback must return an exit code indicating success (zero) or failure (nonzero).
   */
  private WorkRequestHandler(
      WorkRequestCallback callback,
      PrintStream stderr,
      WorkerMessageProcessor messageProcessor,
      Duration cpuUsageBeforeGc,
      BiConsumer<Integer, Thread> cancelCallback) {
    this.callback = callback;
    this.stderr = stderr;
    this.messageProcessor = messageProcessor;
    this.gcScheduler = new CpuTimeBasedGcScheduler(cpuUsageBeforeGc);
    this.cancelCallback = cancelCallback;
  }

  /** A wrapper class for the callback BiFunction */
  public static class WorkRequestCallback {

    /**
     * Callback method for executing a single WorkRequest in a thread. The first argument to {@code
     * callback} is the WorkRequest, the second is where all error messages and other user-oriented
     * messages should be written to. The callback must return an exit code indicating success
     * (zero) or failure (nonzero).
     */
    private final BiFunction<WorkRequest, PrintWriter, Integer> callback;

    public WorkRequestCallback(BiFunction<WorkRequest, PrintWriter, Integer> callback) {
      this.callback = callback;
    }

    public Integer apply(WorkRequest workRequest, PrintWriter printWriter) {
      return callback.apply(workRequest, printWriter);
    }
  }

  /** Builder class for WorkRequestHandler. Required parameters are passed to the constructor. */
  public static class WorkRequestHandlerBuilder {
    private final WorkRequestCallback callback;
    private final PrintStream stderr;
    private final WorkerMessageProcessor messageProcessor;
    private Duration cpuUsageBeforeGc = Duration.ZERO;
    private BiConsumer<Integer, Thread> cancelCallback;

    /**
     * Creates a {@code WorkRequestHandlerBuilder}.
     *
     * @param callback Callback method for executing a single WorkRequest in a thread. The first
     *     argument to {@code callback} is the set of command-line arguments, the second is where
     *     all error messages and other user-oriented messages should be written to. The callback
     *     must return an exit code indicating success (zero) or failure (nonzero).
     * @param stderr Stream that log messages should be written to, typically the process' stderr.
     * @param messageProcessor Object responsible for parsing {@code WorkRequest}s from the server
     *     and writing {@code WorkResponses} to the server.
     * @deprecated use WorkRequestHandlerBuilder with WorkRequestCallback instead
     */
    @Deprecated
    public WorkRequestHandlerBuilder(
        BiFunction<List<String>, PrintWriter, Integer> callback,
        PrintStream stderr,
        WorkerMessageProcessor messageProcessor) {
      this(
          new WorkRequestCallback((request, pw) -> callback.apply(request.getArgumentsList(), pw)),
          stderr,
          messageProcessor);
    }

    /**
     * Creates a {@code WorkRequestHandlerBuilder}.
     *
     * @param callback WorkRequestCallback object with Callback method for executing a single
     *     WorkRequest in a thread. The first argument to {@code callback} is the WorkRequest, the
     *     second is where all error messages and other user-oriented messages should be written to.
     *     The callback must return an exit code indicating success (zero) or failure (nonzero).
     * @param stderr Stream that log messages should be written to, typically the process' stderr.
     * @param messageProcessor Object responsible for parsing {@code WorkRequest}s from the server
     *     and writing {@code WorkResponses} to the server.
     */
    public WorkRequestHandlerBuilder(
        WorkRequestCallback callback, PrintStream stderr, WorkerMessageProcessor messageProcessor) {
      this.callback = callback;
      this.stderr = stderr;
      this.messageProcessor = messageProcessor;
    }

    /**
     * Sets the minimum amount of CPU time between explicit garbage collection calls. Pass
     * Duration.ZERO to not do explicit garbage collection (the default).
     */
    public WorkRequestHandlerBuilder setCpuUsageBeforeGc(Duration cpuUsageBeforeGc) {
      this.cpuUsageBeforeGc = cpuUsageBeforeGc;
      return this;
    }

    /**
     * Sets a callback will be called when a cancellation message has been received. The callback
     * will be call with the request ID and the thread executing the request.
     */
    public WorkRequestHandlerBuilder setCancelCallback(BiConsumer<Integer, Thread> cancelCallback) {
      this.cancelCallback = cancelCallback;
      return this;
    }

    /** Returns a WorkRequestHandler instance with the values in this Builder. */
    public WorkRequestHandler build() {
      return new WorkRequestHandler(
          callback, stderr, messageProcessor, cpuUsageBeforeGc, cancelCallback);
    }
  }

  /**
   * Runs an infinite loop of reading {@link WorkRequest} from {@code in}, running the callback,
   * then writing the corresponding {@link WorkResponse} to {@code out}. If there is an error
   * reading or writing the requests or responses, it writes an error message on {@code err} and
   * returns. If {@code in} reaches EOF, it also returns.
   */
  public void processRequests() throws IOException {
    try {
      while (true) {
        WorkRequest request = messageProcessor.readWorkRequest();
        if (request == null) {
          break;
        }
        if (request.getCancel()) {
          respondToCancelRequest(request);
        } else {
          startResponseThread(request);
        }
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      stderr.println("InterruptedException processing requests.");
    }
  }

  /** Starts a thread for the given request. */
  void startResponseThread(WorkRequest request) throws InterruptedException {
    Thread currentThread = Thread.currentThread();
    String threadName =
        request.getRequestId() > 0
            ? "multiplex-request-" + request.getRequestId()
            : "singleplex-request";
    // TODO(larsrc): See if this can be handled with a queue instead, without introducing more
    // race conditions.
    if (request.getRequestId() == 0) {
      while (activeRequests.containsKey(request.getRequestId())) {
        // b/194051480: Previous singleplex requests can still be in activeRequests for a bit after
        // the response has been sent. We need to wait for them to vanish.
        Thread.sleep(1);
      }
    }
    Thread t =
        new Thread(
            () -> {
              RequestInfo requestInfo = activeRequests.get(request.getRequestId());
              if (requestInfo == null) {
                // Already cancelled
                return;
              }
              try {
                respondToRequest(request, requestInfo);
              } catch (IOException e) {
                e.printStackTrace(stderr);
                // In case of error, shut down the entire worker.
                currentThread.interrupt();
              } finally {
                activeRequests.remove(request.getRequestId());
              }
            },
            threadName);
    activeRequests.put(request.getRequestId(), new RequestInfo(t));
    t.start();
  }

  /** Handles and responds to the given {@link WorkRequest}. */
  @VisibleForTesting
  void respondToRequest(WorkRequest request, RequestInfo requestInfo) throws IOException {
    try (StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw)) {
      int exitCode;
      try {
        exitCode = callback.apply(request, pw);
      } catch (RuntimeException e) {
        e.printStackTrace(pw);
        exitCode = 1;
      }
      pw.flush();
      Optional<WorkResponse.Builder> optBuilder = requestInfo.takeBuilder();
      if (optBuilder.isPresent()) {
        WorkResponse.Builder builder = optBuilder.get();
        builder.setRequestId(request.getRequestId());
        if (requestInfo.isCancelled()) {
          builder.setWasCancelled(true);
        } else {
          builder.setOutput(builder.getOutput() + sw).setExitCode(exitCode);
        }
        WorkResponse response = builder.build();
        synchronized (this) {
          messageProcessor.writeWorkResponse(response);
        }
      }
      gcScheduler.maybePerformGc();
    }
  }

  /**
   * Handles cancelling an existing request, including sending a response if that is not done by the
   * time {@code cancelCallback.accept} returns.
   */
  void respondToCancelRequest(WorkRequest request) throws IOException {
    // Theoretically, we could have gotten two singleplex requests, and we can't tell those apart.
    // However, that's a violation of the protocol, so we don't try to handle it (not least because
    // handling it would be quite error-prone).
    RequestInfo ri = activeRequests.remove(request.getRequestId());

    if (ri == null) {
      return;
    }
    if (cancelCallback == null) {
      ri.setCancelled();
      // This is either an error on the server side or a version mismatch between the server setup
      // and the binary. It's better to wait for the regular work to finish instead of breaking the
      // build, but we should inform the user about the bad setup.
      ri.addOutput(
          String.format(
              "Cancellation request received for worker request %d, but this worker does not"
                  + " support cancellation.\n",
              request.getRequestId()));
    } else {
      if (ri.thread.isAlive() && !ri.isCancelled()) {
        ri.setCancelled();
        cancelCallback.accept(request.getRequestId(), ri.thread);
        Optional<WorkResponse.Builder> builder = ri.takeBuilder();
        if (builder.isPresent()) {
          WorkResponse response =
              builder.get().setWasCancelled(true).setRequestId(request.getRequestId()).build();
          synchronized (this) {
            messageProcessor.writeWorkResponse(response);
          }
        }
      }
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
