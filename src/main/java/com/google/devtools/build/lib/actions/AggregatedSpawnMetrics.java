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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import java.time.Duration;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.ToLongFunction;
import javax.annotation.Nullable;

/** Metrics aggregated per execution kind. */
public final class AggregatedSpawnMetrics {

  public static final AggregatedSpawnMetrics EMPTY = new AggregatedSpawnMetrics(ImmutableMap.of());

  /** Static holder for lazy initialization. */
  private static class EmptyMetrics {
    /**
     * Map with empty {@link SpawnMetrics}.
     *
     * <p>This is useful for {@link #getMetrics(SpawnMetrics.ExecKind)} where we need to return an
     * empty {@link SpawnMetrics} with the correct {@link SpawnMetrics.ExecKind}.
     */
    private static final ImmutableMap<SpawnMetrics.ExecKind, SpawnMetrics> INSTANCE =
        createEmptyMetrics();

    private static final ImmutableMap<SpawnMetrics.ExecKind, SpawnMetrics> createEmptyMetrics() {
      EnumMap<SpawnMetrics.ExecKind, SpawnMetrics> map = new EnumMap<>(SpawnMetrics.ExecKind.class);
      for (SpawnMetrics.ExecKind kind : SpawnMetrics.ExecKind.values()) {
        map.put(kind, new SpawnMetrics.Builder().setExecKind(kind).build());
      }
      return Maps.immutableEnumMap(map);
    }
  }

  private final ImmutableMap<SpawnMetrics.ExecKind, SpawnMetrics> metricsMap;

  private AggregatedSpawnMetrics(ImmutableMap<SpawnMetrics.ExecKind, SpawnMetrics> metricsMap) {
    this.metricsMap = metricsMap;
  }

  /**
   * Returns all present {@link SpawnMetrics}.
   *
   * <p>There will be at most one {@link SpawnMetrics} object per {@link SpawnMetrics.ExecKind}.
   */
  public ImmutableCollection<SpawnMetrics> getAllMetrics() {
    return metricsMap.values();
  }

  /**
   * Returns {@link SpawnMetrics} for the provided execution kind.
   *
   * <p>This will never return {@code null}, but the {@link SpawnMetrics} can be empty.
   */
  public SpawnMetrics getMetrics(SpawnMetrics.ExecKind kind) {
    @Nullable SpawnMetrics metrics = metricsMap.get(kind);
    if (metrics == null) {
      return EmptyMetrics.INSTANCE.get(kind);
    }
    return metrics;
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
   * them with the existing ones (if any).
   */
  public AggregatedSpawnMetrics sumAllMetrics(SpawnMetrics other) {
    SpawnMetrics existing = getMetrics(other.execKind());
    SpawnMetrics.Builder builder =
        new SpawnMetrics.Builder()
            .setExecKind(other.execKind())
            .addDurations(existing)
            .addDurations(other)
            .addNonDurations(existing)
            .addNonDurations(other);

    EnumMap<SpawnMetrics.ExecKind, SpawnMetrics> map = new EnumMap<>(SpawnMetrics.ExecKind.class);
    map.putAll(metricsMap);
    map.put(other.execKind(), builder.build());
    return new AggregatedSpawnMetrics(Maps.immutableEnumMap(map));
  }

  /**
   * Returns a new {@link AggregatedSpawnMetrics} that incorporates the provided metrics by summing
   * the duration ones and taking the maximum for the non-duration ones.
   */
  public AggregatedSpawnMetrics sumDurationsMaxOther(SpawnMetrics other) {
    SpawnMetrics existing = getMetrics(other.execKind());
    SpawnMetrics.Builder builder =
        new SpawnMetrics.Builder()
            .setExecKind(other.execKind())
            .addDurations(existing)
            .addDurations(other)
            .maxNonDurations(existing)
            .maxNonDurations(other);

    EnumMap<SpawnMetrics.ExecKind, SpawnMetrics> map = new EnumMap<>(SpawnMetrics.ExecKind.class);
    map.putAll(metricsMap);
    map.put(other.execKind(), builder.build());
    return new AggregatedSpawnMetrics(Maps.immutableEnumMap(map));
  }

  /**
   * Returns the total duration across all execution kinds.
   *
   * <p>Example: {@code getTotalDuration(SpawnMetrics::queueTime)} will give the total queue time
   * across all execution kinds.
   */
  public Duration getTotalDuration(Function<SpawnMetrics, Duration> extract) {
    Duration result = Duration.ZERO;
    for (SpawnMetrics metric : metricsMap.values()) {
      result = result.plus(extract.apply(metric));
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
    for (SpawnMetrics metric : metricsMap.values()) {
      result = Long.max(result, extract.applyAsLong(metric));
    }
    return result;
  }

  public String toString(Duration total, boolean summary) {
    // For now keep compatibility with the old output and only report the remote execution.
    // TODO(michalt): Change this once the local and worker executions populate more metrics.
    return SpawnMetrics.ExecKind.REMOTE + " " + getRemoteMetrics().toString(total, summary);
  }

  /** Builder for {@link AggregatedSpawnMetrics}. */
  public static class Builder {

    private final EnumMap<SpawnMetrics.ExecKind, SpawnMetrics.Builder> builderMap =
        new EnumMap<>(SpawnMetrics.ExecKind.class);

    public AggregatedSpawnMetrics build() {
      EnumMap<SpawnMetrics.ExecKind, SpawnMetrics> map = new EnumMap<>(SpawnMetrics.ExecKind.class);
      for (Map.Entry<SpawnMetrics.ExecKind, SpawnMetrics.Builder> entry : builderMap.entrySet()) {
        map.put(entry.getKey(), entry.getValue().build());
      }
      return new AggregatedSpawnMetrics(Maps.immutableEnumMap(map));
    }

    public Builder addDurations(SpawnMetrics metrics) {
      getBuilder(metrics.execKind()).addDurations(metrics);
      return this;
    }

    public Builder addDurations(AggregatedSpawnMetrics aggregated) {
      aggregated.getAllMetrics().forEach(this::addDurations);
      return this;
    }

    public Builder addNonDurations(SpawnMetrics metrics) {
      getBuilder(metrics.execKind()).addNonDurations(metrics);
      return this;
    }

    public Builder addNonDurations(AggregatedSpawnMetrics aggregated) {
      aggregated.getAllMetrics().forEach(this::addNonDurations);
      return this;
    }

    public Builder maxNonDurations(SpawnMetrics metrics) {
      getBuilder(metrics.execKind()).maxNonDurations(metrics);
      return this;
    }

    public Builder maxNonDurations(AggregatedSpawnMetrics aggregated) {
      aggregated.getAllMetrics().forEach(this::maxNonDurations);
      return this;
    }

    private SpawnMetrics.Builder getBuilder(SpawnMetrics.ExecKind kind) {
      return builderMap.computeIfAbsent(
          kind, (SpawnMetrics.ExecKind k) -> new SpawnMetrics.Builder().setExecKind(k));
    }
  }
}
