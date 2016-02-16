// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler;

import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * A convenient way to actively get access to timing information (e.g. for logging and/or
 * profiling purposes) with minimal boilerplate. The lack of boilerplate comes at a performance
 * cost; do not use {@link AutoProfiler} on performance critical code.
 *
 * <p>The intended usage is:
 *
 * <pre>
 * {@code
 * try (AutoProfiler p = AutoProfiler.logged("<description of your code>")) {
 *   // Your code here.
 * }
 * }
 * </pre>
 *
 * <p>but if the try-with-resources pattern is too cumbersome, you can also do
 *
 * <pre>
 * {@code
 * AutoProfiler p = AutoProfiler.logged("<description of your code>");
 * // Your code here.
 * long elapsedTimeNanos = p.completeAndGetElapsedTimeNanos();
 * }
 * </pre>
 *
 * <p>An {@link AutoProfiler} can also automatically talk to the active {@link Profiler} instance:
 *
 * <pre>
 * {@code
 * try (AutoProfiler p = AutoProfiler.profiled("<description of your code>")) {
 *   // Your code here.
 * }
 * }
 * </pre>
 */
public class AutoProfiler implements AutoCloseable {
  private static final AtomicReference<Clock> CLOCK_REF = new AtomicReference<>(null);
  private static final AtomicReference<LoggingElapsedTimeReceiverFactory>
      LOGGING_ELAPSED_TIME_RECEIVER_FACTORY_REF = new AtomicReference<>(
          LoggingElapsedTimeReceiver.FACTORY);

  private final ElapsedTimeReceiver elapsedTimeReceiver;
  private final long startTimeNanos;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  private AutoProfiler(ElapsedTimeReceiver elapsedTimeReceiver, long startTimeNanos) {
    this.elapsedTimeReceiver = elapsedTimeReceiver;
    this.startTimeNanos = startTimeNanos;
  }

  /** A opaque receiver of elapsed time information. */
  public interface ElapsedTimeReceiver {
    /**
     * Receives the elapsed time of the lifetime of an {@link AutoProfiler} instance.
     *
     * <p>Note that System#nanoTime isn't guaranteed to be non-decreasing, so implementations should
     * check for non-positive {@code elapsedTimeNanos} if they care about this sort of thing.
     */
    void accept(long elapsedTimeNanos);
  }

  /**
   * Sets the clock to use. May be called before trying to call any of the other static methods;
   * otherwise {@link BlazeClock#instance} will be used.
   *
   * <p>Calling this more than once (e.g. after calling any of the other static methods) results in
   * unspecified behavior.
   */
  public static void setClock(Clock clock) {
    CLOCK_REF.set(clock);
  }

  private static long nanoTime() {
    Clock clock = CLOCK_REF.get();
    return clock != null ? clock.nanoTime() : BlazeClock.instance().nanoTime();
  }

  /**
   * Factory to use for creating {@link ElapsedTimeReceiver} instances for servicing
   * {@link #logged} et al.
   */
  public interface LoggingElapsedTimeReceiverFactory {
    ElapsedTimeReceiver create(
        String taskDescription, Logger logger, long minTimeForLogging, TimeUnit timeUnit);
  }

  /**
   * Inject a custom {@link LoggingElapsedTimeReceiverFactory} for servicing {@link #logged} et al.
   *
   * <p>Use this if you want custom logging behavior.
   */
  public static void setLoggingElapsedTimeReceiverFactory(
      LoggingElapsedTimeReceiverFactory factory) {
    LOGGING_ELAPSED_TIME_RECEIVER_FACTORY_REF.set(factory);
  }

  /**
   * Returns an {@link AutoProfiler} that, when closed, logs the elapsed time in milliseconds to
   * the given {@link Logger}.
   *
   * <p>The returned {@link AutoProfiler} is thread-safe.
   */
  public static AutoProfiler logged(String taskDescription, Logger logger) {
    return logged(taskDescription, logger, TimeUnit.MILLISECONDS);
  }

  /**
   * Returns an {@link AutoProfiler} that, when closed, logs the elapsed time in the given
   * {@link TimeUnit} to the given {@link Logger}.
   *
   * <p>The returned {@link AutoProfiler} is thread-safe.
   */
  public static AutoProfiler logged(String taskDescription, Logger logger, TimeUnit timeUnit) {
    return create(
        LOGGING_ELAPSED_TIME_RECEIVER_FACTORY_REF
            .get()
            .create(taskDescription, logger, 0L, timeUnit));
  }

  /**
   * Returns an {@link AutoProfiler} that, when closed, logs the elapsed time in milliseconds to the
   * given {@link Logger} if it is greater than {@code minTimeForLoggingInMilliseconds}.
   *
   * <p>The returned {@link AutoProfiler} is thread-safe.
   */
  public static AutoProfiler logged(
      String taskDescription, Logger logger, long minTimeForLoggingInMilliseconds) {
    return create(
        LOGGING_ELAPSED_TIME_RECEIVER_FACTORY_REF
            .get()
            .create(
                taskDescription, logger, minTimeForLoggingInMilliseconds, TimeUnit.MILLISECONDS));
  }

  /**
   * Returns an {@link AutoProfiler} that, when closed, records the elapsed time using
   * {@link Profiler}.
   *
   * <p>The returned {@link AutoProfiler} is thread-safe.
   */
  public static AutoProfiler profiled(Object object, ProfilerTask profilerTaskType) {
    return create(new ProfilingElapsedTimeReceiver(object, profilerTaskType));
  }

  /**
   * Returns an {@link AutoProfiler} that, when closed, records the elapsed time using
   * {@link Profiler} and also logs it (in milliseconds) to the given {@link Logger}.
   *
   * <p>The returned {@link AutoProfiler} is thread-safe.
   */
  public static AutoProfiler profiledAndLogged(String taskDescription,
      ProfilerTask profilerTaskType, Logger logger) {
    ElapsedTimeReceiver profilingReceiver =
        new ProfilingElapsedTimeReceiver(taskDescription, profilerTaskType);
    ElapsedTimeReceiver loggingReceiver =
        LOGGING_ELAPSED_TIME_RECEIVER_FACTORY_REF
            .get()
            .create(taskDescription, logger, 0L, TimeUnit.MILLISECONDS);
    return create(new SequencedElapsedTimeReceiver(profilingReceiver, loggingReceiver));
  }

  /**
   * Returns an {@link AutoProfiler} that doesn't do anything when closed and so is only useful for
   * {@link #completeAndGetElapsedTimeNanos()}.
   */
  public static AutoProfiler timed() {
    return create(NULL_RECEIVER);
  }

  /**
   * Returns an {@link AutoProfiler} that, when closed, invokes the given
   * {@link ElapsedTimeReceiver}.
   *
   * <p>The returned {@link AutoProfiler} is as thread-safe as the given
   * {@link ElapsedTimeReceiver} is.
   */
  public static AutoProfiler create(ElapsedTimeReceiver elapsedTimeReceiver) {
    return new AutoProfiler(elapsedTimeReceiver, nanoTime());
  }

  /**
   * Manually completes the profiling (useful to trigger the underlying action on completion).
   *
   * <p>At most one of {@link #complete}, {@link #completeAndGetElapsedTimeNanos} and {@link #close}
   * may be called.
   */
  public void complete() {
    close();
  }

  /**
   * Manually completes the profiling and returns the elapsed time in nanoseconds.
   *
   * <p>At most one of {@link #complete}, {@link #completeAndGetElapsedTimeNanos} and {@link #close}
   * may be called.
   */
  public long completeAndGetElapsedTimeNanos() {
    long elapsedTimeNanos = nanoTime() - startTimeNanos;
    Preconditions.checkState(closed.compareAndSet(false, true));
    elapsedTimeReceiver.accept(elapsedTimeNanos);
    return elapsedTimeNanos;
  }

  /**
   * Automatically completes the profiling.
   *
   * <p>At most one of {@link #complete}, {@link #completeAndGetElapsedTimeNanos} and {@link #close}
   * may be called.
   */
  @Override
  public void close() {
    long elapsedTimeNanos = nanoTime() - startTimeNanos;
    Preconditions.checkState(closed.compareAndSet(false, true));
    elapsedTimeReceiver.accept(elapsedTimeNanos);
  }

  private static final ElapsedTimeReceiver NULL_RECEIVER = new ElapsedTimeReceiver() {
    @Override
    public void accept(long elapsedTimeNanos) {
    }
  };

  private static class SequencedElapsedTimeReceiver implements ElapsedTimeReceiver {
    private final ElapsedTimeReceiver firstReceiver;
    private final ElapsedTimeReceiver secondReceiver;

    private SequencedElapsedTimeReceiver(ElapsedTimeReceiver firstReceiver,
        ElapsedTimeReceiver secondReceiver) {
      this.firstReceiver = firstReceiver;
      this.secondReceiver = secondReceiver;
    }

    @Override
    public void accept(long elapsedTimeNanos) {
      firstReceiver.accept(elapsedTimeNanos);
      secondReceiver.accept(elapsedTimeNanos);
    }
  }

  /**
   * {@link ElapsedTimeReceiver} that logs a message in {@link #accept} if the elapsed time is at
   * least a given threshold.
   */
  public static class LoggingElapsedTimeReceiver implements ElapsedTimeReceiver {
    private static final LoggingElapsedTimeReceiverFactory FACTORY =
        new LoggingElapsedTimeReceiverFactory() {
          @Override
          public ElapsedTimeReceiver create(
              String taskDescription, Logger logger, long minTimeForLogging, TimeUnit timeUnit) {
            return new LoggingElapsedTimeReceiver(
                taskDescription, logger, minTimeForLogging, timeUnit);
          }
        };

    private final String taskDescription;
    protected final Logger logger;
    private final TimeUnit timeUnit;
    private final long minNanosForLogging;

    protected LoggingElapsedTimeReceiver(
        String taskDescription, Logger logger, long minTimeForLogging, TimeUnit timeUnit) {
      this.taskDescription = taskDescription;
      this.logger = logger;
      this.timeUnit = timeUnit;
      this.minNanosForLogging = timeUnit.toNanos(minTimeForLogging);
    }

    /**
     * Returns a message about the amount of time spent doing a task if {@code elapsedTimeNanos} is
     * at least {@link #minNanosForLogging} and null otherwise.
     */
    @Nullable
    protected String loggingMessage(long elapsedTimeNanos) {
      if (elapsedTimeNanos >= minNanosForLogging) {
        return String.format(
            "Spent %d %s doing %s",
            timeUnit.convert(elapsedTimeNanos, TimeUnit.NANOSECONDS),
            timeUnit.toString().toLowerCase(),
            taskDescription);
      } else {
        return null;
      }
    }

    @Override
    public void accept(long elapsedTimeNanos) {
      String logMessage = loggingMessage(elapsedTimeNanos);
      if (logMessage != null) {
        logger.info(logMessage);
      }
    }
  }

  private static class ProfilingElapsedTimeReceiver implements ElapsedTimeReceiver {
    private final long startTimeNanos;
    private final Object object;
    private final ProfilerTask profilerTaskType;

    private ProfilingElapsedTimeReceiver(Object object, ProfilerTask profilerTaskType) {
      this.startTimeNanos = Profiler.nanoTimeMaybe();
      this.object = object;
      this.profilerTaskType = profilerTaskType;
    }

    @Override
    public void accept(long elapsedTimeNanos) {
      if (elapsedTimeNanos > 0) {
        Profiler.instance().logSimpleTaskDuration(startTimeNanos, elapsedTimeNanos,
            profilerTaskType, object);
      }
    }
  }
}

