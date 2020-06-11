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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A convenient way to actively get access to timing information (e.g. for logging and/or profiling
 * purposes) with minimal boilerplate. The lack of boilerplate comes at a performance cost; do not
 * use {@link AutoProfiler} on performance critical code.
 *
 * <p>The intended usage is:
 *
 * <pre>{@code
 * try (AutoProfiler p = GoogleAutoProfilerUtils.logged("<description of your code>")) {
 *   // Your code here.
 * }
 * }</pre>
 *
 * <p>but if the try-with-resources pattern is too cumbersome, you can also do
 *
 * <pre>{@code
 * AutoProfiler p = GoogleAutoProfilerUtils.logged("<description of your code>");
 * // Your code here.
 * long elapsedTimeNanos = p.completeAndGetElapsedTimeNanos();
 * }</pre>
 *
 * <p>An {@link AutoProfiler} can also automatically talk to the active {@link Profiler} instance:
 *
 * <pre>{@code
 * try (AutoProfiler p = AutoProfiler.profiled("<description of your code>")) {
 *   // Your code here.
 * }
 * }</pre>
 */
public class AutoProfiler implements SilentCloseable {
  private static final AtomicReference<Clock> CLOCK_REF = new AtomicReference<>(null);

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
   * Returns an {@link AutoProfiler} that, when closed, records the elapsed time using {@link
   * Profiler}.
   *
   * <p>The returned {@link AutoProfiler} is thread-safe.
   */
  public static AutoProfiler profiled(String description, ProfilerTask profilerTaskType) {
    return create(new ProfilingElapsedTimeReceiver(description, profilerTaskType));
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

  static class ProfilingElapsedTimeReceiver implements ElapsedTimeReceiver {
    private final long startTimeNanos;
    private final String description;
    private final ProfilerTask profilerTaskType;

    ProfilingElapsedTimeReceiver(String description, ProfilerTask profilerTaskType) {
      this.startTimeNanos = Profiler.nanoTimeMaybe();
      this.description = description;
      this.profilerTaskType = profilerTaskType;
    }

    @Override
    public void accept(long elapsedTimeNanos) {
      if (elapsedTimeNanos > 0) {
        Profiler.instance()
            .logSimpleTaskDuration(
                startTimeNanos, Duration.ofNanos(elapsedTimeNanos), profilerTaskType, description);
      }
    }
  }
}
