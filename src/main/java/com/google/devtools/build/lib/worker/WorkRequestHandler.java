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
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.sun.management.OperatingSystemMXBean;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;

/**
 * A helper class that handles WorkRequests (https://bazel.build/docs/persistent-workers), including
 * multiplex workers (https://bazel.build/docs/multiplex-worker).
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
    private final AtomicBoolean cancelled = new AtomicBoolean(false);
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
      cancelled.set(true);
    }

    /** Returns true if this request has been cancelled. */
    boolean isCancelled() {
      return cancelled.get();
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
  /**
   * A scheduler that runs garbage collection after a certain amount of CPU time has passed. In our
   * experience, explicit GC reclaims much more than implicit GC. This scheduler helps make sure
   * very busy workers don't grow ridiculously large.
   */
  private final CpuTimeBasedGcScheduler gcScheduler;
  /**
   * A scheduler that runs garbage collection after a certain amount of time without any activity.
   * In our experience, explicit GC reclaims much more than implicit GC. This scheduler helps make
   * sure workers don't hang on to excessive memory after they are done working.
   */
  private final IdleGcScheduler idleGcScheduler;

  /**
   * If set, this worker will stop handling requests and shut itself down. This can happen if
   * something throws an {@link Error}.
   */
  private final AtomicBoolean shutdownWorker = new AtomicBoolean(false);

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
        cancelCallback,
        Duration.ZERO);
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
      BiConsumer<Integer, Thread> cancelCallback,
      Duration idleTimeBeforeGc) {
    this.callback = callback;
    this.stderr = stderr;
    this.messageProcessor = messageProcessor;
    this.gcScheduler = new CpuTimeBasedGcScheduler(cpuUsageBeforeGc);
    this.cancelCallback = cancelCallback;
    this.idleGcScheduler = new IdleGcScheduler(idleTimeBeforeGc);
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

    public Integer apply(WorkRequest workRequest, PrintWriter printWriter)
        throws InterruptedException {
      Integer result = callback.apply(workRequest, printWriter);
      if (Thread.interrupted()) {
        throw new InterruptedException("Work request interrupted: " + workRequest.getRequestId());
      }
      return result;
    }
  }

  /** Builder class for WorkRequestHandler. Required parameters are passed to the constructor. */
  public static class WorkRequestHandlerBuilder {
    private final WorkRequestCallback callback;
    private final PrintStream stderr;
    private final WorkerMessageProcessor messageProcessor;
    private Duration cpuUsageBeforeGc = Duration.ZERO;
    private BiConsumer<Integer, Thread> cancelCallback;
    private Duration idleTimeBeforeGc = Duration.ZERO;

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
    @CanIgnoreReturnValue
    public WorkRequestHandlerBuilder setCpuUsageBeforeGc(Duration cpuUsageBeforeGc) {
      this.cpuUsageBeforeGc = cpuUsageBeforeGc;
      return this;
    }

    /**
     * Sets a callback will be called when a cancellation message has been received. The callback
     * will be call with the request ID and the thread executing the request.
     */
    @CanIgnoreReturnValue
    public WorkRequestHandlerBuilder setCancelCallback(BiConsumer<Integer, Thread> cancelCallback) {
      this.cancelCallback = cancelCallback;
      return this;
    }

    /** Sets the time without any work that should elapse before forcing a GC. */
    @CanIgnoreReturnValue
    public WorkRequestHandlerBuilder setIdleTimeBeforeGc(Duration idleTimeBeforeGc) {
      this.idleTimeBeforeGc = idleTimeBeforeGc;
      return this;
    }

    /** Returns a WorkRequestHandler instance with the values in this Builder. */
    public WorkRequestHandler build() {
      return new WorkRequestHandler(
          callback, stderr, messageProcessor, cpuUsageBeforeGc, cancelCallback, idleTimeBeforeGc);
    }
  }

  /**
   * Runs an infinite loop of reading {@link WorkRequest} from {@code in}, running the callback,
   * then writing the corresponding {@link WorkResponse} to {@code out}. If there is an error
   * reading or writing the requests or responses, it writes an error message on {@code err} and
   * returns. If {@code in} reaches EOF, it also returns.
   *
   * <p>This function also wraps the system streams in a {@link WorkerIO} instance that prevents the
   * underlying tool from writing to {@link System#out} or reading from {@link System#in}, which
   * would corrupt the worker worker protocol. When the while loop exits, the original system
   * streams will be swapped back into {@link System}.
   */
  public void processRequests() throws IOException {
    // Wrap the system streams into a WorkerIO instance to prevent unexpected reads and writes on
    // stdin/stdout.
    WorkerIO workerIO = WorkerIO.capture();

    try {
      while (!shutdownWorker.get()) {
        WorkRequest request = messageProcessor.readWorkRequest();
        idleGcScheduler.markActivity(true);
        if (request == null) {
          break;
        }
        if (request.getCancel()) {
          respondToCancelRequest(request);
        } else {
          startResponseThread(workerIO, request);
        }
      }
    } catch (IOException e) {
      stderr.println("Error reading next WorkRequest: " + e);
      e.printStackTrace(stderr);
    } finally {
      idleGcScheduler.stop();
      // TODO(b/220878242): Give the outstanding requests a chance to send a "shutdown" response,
      // but also try to kill stuck threads. For now, we just interrupt the remaining threads.
      // We considered doing System.exit here, but that is hard to test and would deny the callers
      // of this method a chance to clean up. Instead, we initiate the cleanup of our resources here
      // and the caller can decide whether to wait for an orderly shutdown or now.
      for (RequestInfo ri : activeRequests.values()) {
        if (ri.thread.isAlive()) {
          try {
            ri.thread.interrupt();
          } catch (RuntimeException e) {
            // If we can't interrupt, we can't do much else.
          }
        }
      }

      try {
        // Unwrap the system streams placing the original streams back
        workerIO.close();
      } catch (Exception e) {
        stderr.println(e.getMessage());
      }
    }
  }

  /** Starts a thread for the given request. */
  void startResponseThread(WorkerIO workerIO, WorkRequest request) {
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
        try {
          Thread.sleep(1);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          return;
        }
      }
    }
    Thread t =
        new Thread(
            () -> {
              RequestInfo requestInfo = activeRequests.get(request.getRequestId());
              if (requestInfo == null) {
                // Already cancelled
                idleGcScheduler.markActivity(!activeRequests.isEmpty());
                return;
              }
              try {
                respondToRequest(workerIO, request, requestInfo);
              } catch (IOException e) {
                // IOExceptions here means a problem talking to the server, so we must shut down.
                if (!shutdownWorker.compareAndSet(false, true)) {
                  stderr.println("Error communicating with server, shutting down worker.");
                  e.printStackTrace(stderr);
                  currentThread.interrupt();
                }
              } finally {
                activeRequests.remove(request.getRequestId());
                idleGcScheduler.markActivity(!activeRequests.isEmpty());
              }
            },
            threadName);
    t.setUncaughtExceptionHandler(
        (t1, e) -> {
          // Shut down the worker in case of severe issues. We don't handle RuntimeException here,
          // as those are not serious enough to merit shutting down the worker.
          if (e instanceof Error && shutdownWorker.compareAndSet(false, true)) {
            stderr.println("Error thrown by worker thread, shutting down worker.");
            e.printStackTrace(stderr);
            currentThread.interrupt();
            idleGcScheduler.stop();
            System.exit(1);
          }
        });
    RequestInfo previous = activeRequests.putIfAbsent(request.getRequestId(), new RequestInfo(t));
    if (previous != null) {
      // Kill worker since this shouldn't happen: server didn't follow the worker protocol
      throw new IllegalStateException("Request still active: " + request.getRequestId());
    }
    t.start();
  }

  /**
   * Handles and responds to the given {@link WorkRequest}.
   *
   * @throws IOException if there is an error talking to the server. Errors from calling the {@link
   *     #callback} are reported with exit code 1.
   */
  @VisibleForTesting
  void respondToRequest(WorkerIO workerIO, WorkRequest request, RequestInfo requestInfo)
      throws IOException {
    int exitCode;
    StringWriter sw = new StringWriter();
    try (PrintWriter pw = new PrintWriter(sw)) {
      try {
        exitCode = callback.apply(request, pw);
      } catch (InterruptedException e) {
        exitCode = 1;
      } catch (RuntimeException e) {
        e.printStackTrace(pw);
        exitCode = 1;
      }

      try {
        // Read out the captured string for the final WorkResponse output
        String captured = workerIO.readCapturedAsUtf8String().trim();
        if (!captured.isEmpty()) {
          pw.write(captured);
        }
      } catch (IOException e) {
        stderr.println(e.getMessage());
      }
    }
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

  /**
   * Marks the given request as cancelled and uses {@link #cancelCallback} to request cancellation.
   *
   * <p>For simplicity, and to avoid blocking in {@link #cancelCallback}, response to cancellation
   * is still handled by {@link #respondToRequest} once the canceled request aborts (or finishes).
   */
  void respondToCancelRequest(WorkRequest request) {
    // Theoretically, we could have gotten two singleplex requests, and we can't tell those apart.
    // However, that's a violation of the protocol, so we don't try to handle it (not least because
    // handling it would be quite error-prone).
    RequestInfo ri = activeRequests.get(request.getRequestId());

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
        Thread t =
            new Thread(
                // Response will be sent from request thread once request handler returns.
                // We can ignore any exceptions in cancel callback since it's best effort.
                () -> cancelCallback.accept(request.getRequestId(), ri.thread));
        t.start();
      }
    }
  }

  @Override
  public void close() throws IOException {
    messageProcessor.close();
  }

  /** Schedules GC when the worker has been idle for a while */
  private static class IdleGcScheduler {
    private Instant lastActivity = Instant.EPOCH;
    private Instant lastGc = Instant.EPOCH;
    /** Minimum duration from the end of activity until we perform an idle GC. */
    private final Duration idleTimeBeforeGc;

    private final ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(1);
    private ScheduledFuture<?> futureGc = null;

    /**
     * Creates a new scheduler.
     *
     * @param idleTimeBeforeGc The time from the last activity until attempting GC.
     */
    public IdleGcScheduler(Duration idleTimeBeforeGc) {
      this.idleTimeBeforeGc = idleTimeBeforeGc;
    }

    synchronized void start() {
      if (!idleTimeBeforeGc.isZero()) {
        futureGc =
            executor.schedule(this::maybeDoGc, idleTimeBeforeGc.toMillis(), TimeUnit.MILLISECONDS);
      }
    }

    /**
     * Should be called whenever there is some sort of activity starting or ending. Better to call
     * too often.
     */
    synchronized void markActivity(boolean anythingActive) {
      lastActivity = Instant.now();
      if (futureGc != null) {
        futureGc.cancel(false);
        futureGc = null;
      }
      if (!anythingActive) {
        start();
      }
    }

    private void maybeDoGc() {
      if (lastGc.isBefore(lastActivity)
          && lastActivity.isBefore(Instant.now().minus(idleTimeBeforeGc))) {
        System.gc();
        lastGc = Instant.now();
      } else {
        start();
      }
    }

    synchronized void stop() {
      if (futureGc != null) {
        futureGc.cancel(false);
        futureGc = null;
      }
      executor.shutdown();
    }
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

  /**
   * Class that wraps the standard {@link System#in}, {@link System#out}, and {@link System#err}
   * with our own ByteArrayOutputStream that allows {@link WorkRequestHandler} to safely capture
   * outputs that can't be directly captured by the PrintStream associated with the work request.
   *
   * <p>This is most useful when integrating JVM tools that write exceptions and logs directly to
   * {@link System#out} and {@link System#err}, which would corrupt the persistent worker protocol.
   * We also redirect {@link System#in}, just in case a tool should attempt to read it.
   *
   * <p>WorkerIO implements {@link AutoCloseable} and will swap the original streams back into
   * {@link System} once close has been called.
   */
  public static class WorkerIO implements AutoCloseable {
    private final InputStream originalInputStream;
    private final PrintStream originalOutputStream;
    private final PrintStream originalErrorStream;
    private final ByteArrayOutputStream capturedStream;
    private final AutoCloseable restore;

    /**
     * Creates a new {@link WorkerIO} that allows {@link WorkRequestHandler} to capture standard
     * output and error streams that can't be directly captured by the PrintStream associated with
     * the work request.
     */
    @VisibleForTesting
    WorkerIO(
        InputStream originalInputStream,
        PrintStream originalOutputStream,
        PrintStream originalErrorStream,
        ByteArrayOutputStream capturedStream,
        AutoCloseable restore) {
      this.originalInputStream = originalInputStream;
      this.originalOutputStream = originalOutputStream;
      this.originalErrorStream = originalErrorStream;
      this.capturedStream = capturedStream;
      this.restore = restore;
    }

    /** Wraps the standard System streams and WorkerIO instance */
    public static WorkerIO capture() {
      // Save the original streams
      InputStream originalInputStream = System.in;
      PrintStream originalOutputStream = System.out;
      PrintStream originalErrorStream = System.err;

      // Replace the original streams with our own instances
      ByteArrayOutputStream capturedStream = new ByteArrayOutputStream();
      PrintStream outputBuffer = new PrintStream(capturedStream, true);
      ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(new byte[0]);
      System.setIn(byteArrayInputStream);
      System.setOut(outputBuffer);
      System.setErr(outputBuffer);

      return new WorkerIO(
          originalInputStream,
          originalOutputStream,
          originalErrorStream,
          capturedStream,
          () -> {
            System.setIn(originalInputStream);
            System.setOut(originalOutputStream);
            System.setErr(originalErrorStream);
            outputBuffer.close();
            byteArrayInputStream.close();
          });
    }

    /** Returns the original input stream most commonly provided by {@link System#in} */
    @VisibleForTesting
    InputStream getOriginalInputStream() {
      return originalInputStream;
    }

    /** Returns the original output stream most commonly provided by {@link System#out} */
    @VisibleForTesting
    PrintStream getOriginalOutputStream() {
      return originalOutputStream;
    }

    /** Returns the original error stream most commonly provided by {@link System#err} */
    @VisibleForTesting
    PrintStream getOriginalErrorStream() {
      return originalErrorStream;
    }

    /** Returns the captured outputs as a UTF-8 string */
    @VisibleForTesting
    String readCapturedAsUtf8String() throws IOException {
      capturedStream.flush();
      String captureOutput = capturedStream.toString(StandardCharsets.UTF_8);
      capturedStream.reset();
      return captureOutput;
    }

    @Override
    public void close() throws Exception {
      restore.close();
    }
  }
}
