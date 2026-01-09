// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.time.Duration;
import java.util.function.Function;
import java.util.function.ToLongFunction;
import javax.annotation.Nullable;

/** Metrics aggregated per execution kind. */
@SuppressWarnings("GoodTime") // Use ints instead of Durations to improve build time (cl/505728570)
public final class AggregatedSpawnMetrics {

  public static final AggregatedSpawnMetrics EMPTY =
      new AggregatedSpawnMetrics(null, null, null, null);

  // Note that we are using fields instead of e.g. a map of SpawnMetrics to avoid the map overhead.
  // While this results in a bit more boilerplate code, it is worth the memory savings.
  @Nullable private final SpawnMetrics remoteMetrics;
  @Nullable private final SpawnMetrics localMetrics;
  @Nullable private final SpawnMetrics workerMetrics;
  @Nullable private final SpawnMetrics otherMetrics;

  private AggregatedSpawnMetrics(
      @Nullable SpawnMetrics remoteMetrics,
      @Nullable SpawnMetrics localMetrics,
      @Nullable SpawnMetrics workerMetrics,
      @Nullable SpawnMetrics otherMetrics) {
    this.remoteMetrics = remoteMetrics;
    this.localMetrics = localMetrics;
    this.workerMetrics = workerMetrics;
    this.otherMetrics = otherMetrics;
  }

  /**
   * Returns all present {@link SpawnMetrics}.
   *
   * <p>There will be at most one {@link SpawnMetrics} object per {@link SpawnMetrics.ExecKind}.
   */
  public ImmutableCollection<SpawnMetrics> getAllMetrics() {
    ImmutableList.Builder<SpawnMetrics> metrics = ImmutableList.builder();
    if (remoteMetrics != null) {
      metrics.add(remoteMetrics);
    }
    if (localMetrics != null) {
      metrics.add(localMetrics);
    }
    if (workerMetrics != null) {
      metrics.add(workerMetrics);
    }
    if (otherMetrics != null) {
      metrics.add(otherMetrics);
    }
    return metrics.build();
  }

  /**
   * Returns {@link SpawnMetrics} for the provided execution kind.
   *
   * <p>This will never return {@code null}, but the {@link SpawnMetrics} can be empty.
   */
  public SpawnMetrics getMetrics(SpawnMetrics.ExecKind kind) {
    SpawnMetrics result =
        switch (kind) {
          case REMOTE -> remoteMetrics;
          case LOCAL -> localMetrics;
          case WORKER -> workerMetrics;
          case OTHER -> otherMetrics;
        };
    return result != null ? result : SpawnMetrics.Builder.forExec(kind).build();
  }

  /**
   * Returns {@link SpawnMetrics} for the remote execution.
   *
   * @see #getMetrics(SpawnMetrics.ExecKind)
   */
  public SpawnMetrics getRemoteMetrics() {
    return getMetrics(SpawnMetrics.ExecKind.REMOTE);
  }

  /**
   * Returns a new {@link AggregatedSpawnMetrics} that incorporates the provided metrics by summing
   * the duration ones and taking the maximum for the non-duration ones.
   */
  public AggregatedSpawnMetrics sumDurationsMaxOther(SpawnMetrics other) {
    SpawnMetrics.ExecKind kind = other.execKind();
    SpawnMetrics existing = getMetrics(kind);
    SpawnMetrics.Builder builder =
        SpawnMetrics.Builder.forExec(kind)
            .addDurations(existing)
            .addDurations(other)
            .maxNonDurations(existing)
            .maxNonDurations(other);

    SpawnMetrics newMetric = builder.build();

    SpawnMetrics newRemoteMetrics = remoteMetrics;
    SpawnMetrics newLocalMetrics = localMetrics;
    SpawnMetrics newWorkerMetrics = workerMetrics;
    SpawnMetrics newOtherMetrics = otherMetrics;

    switch (kind) {
      case REMOTE -> newRemoteMetrics = newMetric;
      case LOCAL -> newLocalMetrics = newMetric;
      case WORKER -> newWorkerMetrics = newMetric;
      case OTHER -> newOtherMetrics = newMetric;
    }

    return new AggregatedSpawnMetrics(
        newRemoteMetrics, newLocalMetrics, newWorkerMetrics, newOtherMetrics);
  }

  /**
   * Returns the total duration across all execution kinds.
   *
   * <p>Example: {@code getTotalDuration(SpawnMetrics::queueTime)} will give the total queue time
   * across all execution kinds.
   */
  public int getTotalDuration(Function<SpawnMetrics, Integer> extract) {
    int result = 0;
    if (remoteMetrics != null) {
      result += extract.apply(remoteMetrics);
    }
    if (localMetrics != null) {
      result += extract.apply(localMetrics);
    }
    if (workerMetrics != null) {
      result += extract.apply(workerMetrics);
    }
    if (otherMetrics != null) {
      result += extract.apply(otherMetrics);
    }
    return result;
  }

  /**
   * Returns the maximum value of a non-duration metric across all execution kinds.
   *
   * <p>Example: {@code getMaxNonDuration(0, SpawnMetrics::inputFiles)} returns the maximum number
   * of input files across all the execution kinds.
   */
  public long getMaxNonDuration(long initialValue, ToLongFunction<SpawnMetrics> extract) {
    long result = initialValue;
    if (remoteMetrics != null) {
      result = Long.max(result, extract.applyAsLong(remoteMetrics));
    }
    if (localMetrics != null) {
      result = Long.max(result, extract.applyAsLong(localMetrics));
    }
    if (workerMetrics != null) {
      result = Long.max(result, extract.applyAsLong(workerMetrics));
    }
    if (otherMetrics != null) {
      result = Long.max(result, extract.applyAsLong(otherMetrics));
    }
    return result;
  }

  public String toString(Duration total, boolean summary) {
    // For now keep compatibility with the old output and only report the remote execution.
    // TODO(michalt): Change this once the local and worker executions populate more metrics.
    return SpawnMetrics.ExecKind.REMOTE
        + " "
        + getRemoteMetrics().toString((int) total.toMillis(), summary);
  }

  /** Builder for {@link AggregatedSpawnMetrics}. */
  public static class Builder {
    @Nullable private SpawnMetrics.Builder remoteMetricsBuilder;
    @Nullable private SpawnMetrics.Builder localMetricsBuilder;
    @Nullable private SpawnMetrics.Builder workerMetricsBuilder;
    @Nullable private SpawnMetrics.Builder otherMetricsBuilder;

    public AggregatedSpawnMetrics build() {
      return new AggregatedSpawnMetrics(
          remoteMetricsBuilder != null ? remoteMetricsBuilder.build() : null,
          localMetricsBuilder != null ? localMetricsBuilder.build() : null,
          workerMetricsBuilder != null ? workerMetricsBuilder.build() : null,
          otherMetricsBuilder != null ? otherMetricsBuilder.build() : null);
    }

    @CanIgnoreReturnValue
    public Builder addDurations(SpawnMetrics metrics) {
      getBuilder(metrics.execKind()).addDurations(metrics);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addDurations(AggregatedSpawnMetrics aggregated) {
      aggregated.getAllMetrics().forEach(this::addDurations);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addNonDurations(SpawnMetrics metrics) {
      getBuilder(metrics.execKind()).addNonDurations(metrics);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addNonDurations(AggregatedSpawnMetrics aggregated) {
      aggregated.getAllMetrics().forEach(this::addNonDurations);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder maxNonDurations(SpawnMetrics metrics) {
      getBuilder(metrics.execKind()).maxNonDurations(metrics);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder maxNonDurations(AggregatedSpawnMetrics aggregated) {
      aggregated.getAllMetrics().forEach(this::maxNonDurations);
      return this;
    }

    private SpawnMetrics.Builder getBuilder(SpawnMetrics.ExecKind kind) {
      switch (kind) {
        case REMOTE -> {
          if (remoteMetricsBuilder == null) {
            remoteMetricsBuilder = SpawnMetrics.Builder.forRemoteExec();
          }
          return remoteMetricsBuilder;
        }
        case LOCAL -> {
          if (localMetricsBuilder == null) {
            localMetricsBuilder = SpawnMetrics.Builder.forLocalExec();
          }
          return localMetricsBuilder;
        }
        case WORKER -> {
          if (workerMetricsBuilder == null) {
            workerMetricsBuilder = SpawnMetrics.Builder.forWorkerExec();
          }
          return workerMetricsBuilder;
        }
        case OTHER -> {
          if (otherMetricsBuilder == null) {
            otherMetricsBuilder = SpawnMetrics.Builder.forOtherExec();
          }
          return otherMetricsBuilder;
        }
      }
      throw new IllegalArgumentException("Unknown ExecKind: " + kind);
    }
  }
}
