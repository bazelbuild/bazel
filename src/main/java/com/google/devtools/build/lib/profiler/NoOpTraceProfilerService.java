// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.clock.Clock;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.OutputStream;
import java.time.Duration;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Supplier;

/** A profiler that does nothing and crashes if started. DO NOT USE IN PRODUCTION. */
public final class NoOpTraceProfilerService implements TraceProfilerService {
  private static final SilentCloseable NOP_CLOSEABLE = () -> {};
  private static final TimeSeries NOP_TIME_SERIES = new NoOpTimeSeries();
  private static final AsyncProfiler NOP_ASYNC_PROFILER = new NoOpAsyncProfiler();

  @Override
  public long nanoTimeMaybe() {
    return -1;
  }

  @Override
  public boolean isActive() {
    return false;
  }

  @Override
  public SilentCloseable profile(ProfilerTask type, String description) {
    return NOP_CLOSEABLE;
  }

  @Override
  public SilentCloseable profile(ProfilerTask type, Supplier<String> description) {
    return NOP_CLOSEABLE;
  }

  @Override
  public SilentCloseable profile(String description) {
    return NOP_CLOSEABLE;
  }

  @Override
  public void logSimpleTask(long startTimeNanos, ProfilerTask type, String description) {}

  @Override
  public void logSimpleTask(
      long startTimeNanos, long stopTimeNanos, ProfilerTask type, String description) {}

  @Override
  public void logSimpleTaskDuration(
      long startTimeNanos, Duration duration, ProfilerTask type, String description) {}

  @Override
  public void logEventAtTime(long atTimeNanos, ProfilerTask type, String description) {}

  @Override
  public void logEvent(ProfilerTask type, String description) {}

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
      boolean collectTaskHistograms,
      LocalResourceCollector localResourceCollector) {
    throw new IllegalStateException(
        "Set a real TraceProfilerService if you want to use the profiler. See"
            + " Profiler#instance().");
  }

  @Override
  public void stop() {}

  @Override
  public void clear() {}

  @Override
  public ImmutableList<StatRecorder> getTasksHistograms() {
    return ImmutableList.of();
  }

  @Override
  public ImmutableList<SlowTask> getSlowestTasks() {
    return ImmutableList.of();
  }

  @Override
  public boolean isProfiling(ProfilerTask type) {
    return false;
  }

  @Override
  public void markPhase(ProfilePhase phase) {}

  @Override
  public SilentCloseable profileAction(
      ProfilerTask type,
      String mnemonic,
      String description,
      String primaryOutput,
      String targetLabel,
      String configuration) {
    return NOP_CLOSEABLE;
  }

  @Override
  public void completeTask(long startTimeNanos, ProfilerTask type, String description) {}

  @Override
  public void registerCounterSeriesCollector(CounterSeriesCollector collector) {}

  @Override
  public void logCounters(
      Map<CounterSeriesTask, double[]> counterSeriesMap,
      Duration profileStart,
      Duration bucketDuration) {}

  @Override
  public Duration getProfileElapsedTime() {
    return Duration.ZERO;
  }

  @Override
  public Duration getServerProcessCpuTime() {
    return Duration.ZERO;
  }

  @Override
  @CanIgnoreReturnValue
  public <T> ListenableFuture<T> profileFuture(
      ListenableFuture<T> future, String prefix, ProfilerTask type, String description) {
    return future;
  }

  @Override
  public AsyncProfiler profileAsync(String prefix, String description) {
    return NOP_ASYNC_PROFILER;
  }

  @Override
  public TimeSeries createTimeSeries(Duration startTime, Duration bucketDuration) {
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
