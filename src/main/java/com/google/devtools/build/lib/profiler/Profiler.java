// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.clock.Clock;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Static accessor for the {@link TraceProfilerService}.
 *
 * <p>This class provides a global access point to the trace profiler so it doesn't have to be
 * threaded through most of the codebase. Usage typically looks like:
 *
 * <p><code>
 * try (SilentCloseable c = Profiler.instance().profile("my task")) {
 *   // code to be profiled
 * }
 * </code>
 *
 * <p>It's also possible to save the {@code Profiler.instance()} return value in a variable and
 * re-use it later.
 *
 * <p>The purpose of this class is let both the LC and SC use the trace profiler without both of
 * them depending on the full implementation at compile time. At runtime, the symbolic references to
 * {@link Profiler} on both sides must link against the SC version. Any future additions to the
 * profiler API should mirror the existing methods: a delegating implementation falling back to a
 * no-op, with the actual implementation in {@link TraceProfilerServiceImpl}.
 */
@SuppressWarnings("GoodTime") // This code is very performance sensitive.
public final class Profiler implements TraceProfilerService {
  private static final Profiler instance = new Profiler();

  @Nullable private static volatile TraceProfilerService traceProfilerService;

  private static final SilentCloseable NOP_CLOSEABLE = () -> {};
  private static final TimeSeries NOP_TIME_SERIES = new NoOpTimeSeries();
  private static final AsyncProfiler NOP_ASYNC_PROFILER = new NoOpAsyncProfiler();

  private Profiler() {}

  /**
   * Returns the singleton {@link Profiler} instance, which is valid for the entire lifetime of the
   * server.
   *
   * <p>With the exception of the {@link #start} method, the singleton instance provides a no-op
   * implementation of {@link TraceProfilerService} until {@link #setTraceProfilerService} is
   * called, after which it forwards all instance method calls to the implementation thus installed.
   * Calling {@link #start} before {@link #setTraceProfilerService} will throw an exception.
   *
   * <p>With this arrangement, {@link Profiler} methods other than {@link #start} may be called
   * liberally anywhere in the codebase, even if {@link #setTraceProfilerService} is called after
   * the {@link Profiler} singleton has already been retrieved, or if it is never called (as might
   * be the case in a test or a non-Bazel binary incorporating parts of the Bazel codebase).
   */
  public static Profiler instance() {
    return instance;
  }

  /**
   * Installs the {@link TraceProfilerService}. In a production context, this is expected to be
   * called exactly once during server startup.
   *
   * <p>From this point onwards, methods called on the singleton {@link Profiler} instance will be
   * forwarded to this {@link TraceProfilerService}.
   */
  public static void setTraceProfilerService(TraceProfilerService traceProfilerService) {
    checkState(
        Profiler.traceProfilerService == null,
        "setTraceProfilerService must not be called multiple times");
    Profiler.traceProfilerService = traceProfilerService;
  }

  /**
   * Same as {@link #setTraceProfilerService}, except that it may be called more than once for
   * testing purposes.
   */
  @VisibleForTesting
  public static void setTraceProfilerServiceForTesting(TraceProfilerService traceProfilerService) {
    Profiler.traceProfilerService = traceProfilerService;
  }

  @Override
  public long nanoTimeMaybe() {
    if (traceProfilerService != null) {
      return traceProfilerService.nanoTimeMaybe();
    }
    return -1;
  }

  @Override
  public boolean isActive() {
    if (traceProfilerService != null) {
      return traceProfilerService.isActive();
    }
    return false;
  }

  @Override
  public SilentCloseable profile(ProfilerTask type, String description) {
    if (traceProfilerService != null) {
      return traceProfilerService.profile(type, description);
    }
    return NOP_CLOSEABLE;
  }

  @Override
  public SilentCloseable profile(ProfilerTask type, Supplier<String> description) {
    if (traceProfilerService != null) {
      return traceProfilerService.profile(type, description);
    }
    return NOP_CLOSEABLE;
  }

  @Override
  public SilentCloseable profile(String description) {
    if (traceProfilerService != null) {
      return traceProfilerService.profile(description);
    }
    return NOP_CLOSEABLE;
  }

  @Override
  public void logSimpleTask(long startTimeNanos, ProfilerTask type, String description) {
    if (traceProfilerService != null) {
      traceProfilerService.logSimpleTask(startTimeNanos, type, description);
    }
  }

  @Override
  public void logSimpleTask(
      long startTimeNanos, long stopTimeNanos, ProfilerTask type, String description) {
    if (traceProfilerService != null) {
      traceProfilerService.logSimpleTask(startTimeNanos, stopTimeNanos, type, description);
    }
  }

  @Override
  public void logSimpleTaskDuration(
      long startTimeNanos, Duration duration, ProfilerTask type, String description) {
    if (traceProfilerService != null) {
      traceProfilerService.logSimpleTaskDuration(startTimeNanos, duration, type, description);
    }
  }

  @Override
  public void logEventAtTime(long atTimeNanos, ProfilerTask type, String description) {
    if (traceProfilerService != null) {
      traceProfilerService.logEventAtTime(atTimeNanos, type, description);
    }
  }

  @Override
  public void logEvent(ProfilerTask type, String description) {
    if (traceProfilerService != null) {
      traceProfilerService.logEvent(type, description);
    }
  }

  @Override
  public void start(
      Set<ProfilerTask> profiledTasks,
      OutputStream stream,
      Format format,
      String outputBase,
      UUID buildID,
      boolean recordAllDurations,
      Clock clock,
      long execStartTimeNanos,
      boolean slimProfile,
      boolean includePrimaryOutput,
      boolean includeTargetLabel,
      boolean includeConfiguration,
      boolean collectTaskHistograms)
      throws IOException {
    if (traceProfilerService == null) {
      throw new IllegalStateException("cannot call start before setTraceProfilerService");
    }
    traceProfilerService.start(
        profiledTasks,
        stream,
        format,
        outputBase,
        buildID,
        recordAllDurations,
        clock,
        execStartTimeNanos,
        /* slimProfile= */ slimProfile,
        /* includePrimaryOutput= */ includePrimaryOutput,
        /* includeTargetLabel= */ includeTargetLabel,
        /* includeConfiguration= */ includeConfiguration,
        /* collectTaskHistograms= */ collectTaskHistograms);
  }

  @Override
  public void stop() throws IOException {
    if (traceProfilerService != null) {
      traceProfilerService.stop();
    }
  }

  @Override
  public void clear() {
    if (traceProfilerService != null) {
      traceProfilerService.clear();
    }
  }

  @Override
  public List<StatRecorder> getTasksHistograms() {
    if (traceProfilerService != null) {
      return traceProfilerService.getTasksHistograms();
    }
    return ImmutableList.of();
  }

  @Override
  public Iterable<SlowTask> getSlowestTasks() {
    if (traceProfilerService != null) {
      return traceProfilerService.getSlowestTasks();
    }
    return ImmutableList.of();
  }

  @Override
  public boolean isProfiling(ProfilerTask type) {
    if (traceProfilerService != null) {
      return traceProfilerService.isProfiling(type);
    }
    return false;
  }

  @Override
  public void markPhase(ProfilePhase phase) throws InterruptedException {
    if (traceProfilerService != null) {
      traceProfilerService.markPhase(phase);
    }
  }

  @Override
  public SilentCloseable profileAction(
      ProfilerTask type,
      String mnemonic,
      String description,
      String primaryOutput,
      String targetLabel,
      String configuration) {
    if (traceProfilerService != null) {
      return traceProfilerService.profileAction(
          type, mnemonic, description, primaryOutput, targetLabel, configuration);
    }
    return NOP_CLOSEABLE;
  }

  @Override
  public void completeTask(long startTimeNanos, ProfilerTask type, String description) {
    if (traceProfilerService != null) {
      traceProfilerService.completeTask(startTimeNanos, type, description);
    }
  }

  @Override
  public void registerCounterSeriesCollector(CounterSeriesCollector collector) {
    if (traceProfilerService != null) {
      traceProfilerService.registerCounterSeriesCollector(collector);
    }
  }

  @Override
  public void unregisterCounterSeriesCollector(CounterSeriesCollector collector) {
    if (traceProfilerService != null) {
      traceProfilerService.unregisterCounterSeriesCollector(collector);
    }
  }

  @Override
  public void logCounters(
      Map<CounterSeriesTask, double[]> counterSeriesMap,
      Duration profileStart,
      Duration bucketDuration) {
    if (traceProfilerService != null) {
      traceProfilerService.logCounters(counterSeriesMap, profileStart, bucketDuration);
    }
  }

  @Override
  public Duration getProfileElapsedTime() {
    if (traceProfilerService != null) {
      return traceProfilerService.getProfileElapsedTime();
    }
    return Duration.ZERO;
  }

  @Override
  public Duration getServerProcessCpuTime() {
    if (traceProfilerService != null) {
      return traceProfilerService.getServerProcessCpuTime();
    }
    return Duration.ZERO;
  }

  @Override
  @CanIgnoreReturnValue
  public <T> ListenableFuture<T> profileFuture(
      ListenableFuture<T> future, String prefix, ProfilerTask type, String description) {
    if (traceProfilerService != null) {
      return traceProfilerService.profileFuture(future, prefix, type, description);
    }
    return future;
  }

  @Override
  public AsyncProfiler profileAsync(String prefix, String description) {
    if (traceProfilerService != null) {
      return traceProfilerService.profileAsync(prefix, description);
    }
    return NOP_ASYNC_PROFILER;
  }

  @Override
  public TimeSeries createTimeSeries(Duration startTime, Duration bucketDuration) {
    if (traceProfilerService != null) {
      return traceProfilerService.createTimeSeries(startTime, bucketDuration);
    }
    return NOP_TIME_SERIES;
  }

  private static class NoOpTimeSeries implements TimeSeries {
    @Override
    public void addRange(Duration startTime, Duration endTime) {}

    @Override
    public void addRange(Duration rangeStart, Duration rangeEnd, double value) {}

    @Override
    public double[] toDoubleArray(int len) {
      return new double[len];
    }
  }

  private static class NoOpAsyncProfiler implements AsyncProfiler {
    @Override
    public SilentCloseable profile(ProfilerTask type, String description) {
      return NOP_CLOSEABLE;
    }

    @Override
    public SilentCloseable profile(String description) {
      return NOP_CLOSEABLE;
    }

    @Override
    public <T> ListenableFuture<T> profileFuture(ListenableFuture<T> future, String description) {
      return future;
    }

    @Override
    @CanIgnoreReturnValue
    public <T> ListenableFuture<T> profileFuture(
        ListenableFuture<T> future, ProfilerTask type, String description) {
      return future;
    }

    @Override
    public Runnable profileCallback(Runnable runnable, String description) {
      return runnable;
    }

    @Override
    public Runnable profileCallback(Runnable runnable, ProfilerTask type, String description) {
      return runnable;
    }

    @Override
    public <T> Consumer<T> profileCallback(Consumer<T> consumer, String description) {
      return consumer;
    }

    @Override
    public <T> Consumer<T> profileCallback(
        Consumer<T> consumer, ProfilerTask type, String description) {
      return consumer;
    }

    @Override
    public void close() {}
  }
}
