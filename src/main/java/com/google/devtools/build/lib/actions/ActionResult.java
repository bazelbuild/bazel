// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import java.time.Duration;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;

/** Holds the result(s) of an action's execution. */
@AutoValue
public abstract class ActionResult {

  /** An empty ActionResult used by Actions that don't have any metadata to return. */
  public static final ActionResult EMPTY = ActionResult.create(ImmutableList.of());

  /** Returns the SpawnResults for the action. */
  public abstract ImmutableList<SpawnResult> spawnResults();

  /** Returns a builder that can be used to construct a {@link ActionResult} object. */
  public static Builder builder() {
    return new AutoValue_ActionResult.Builder();
  }

  /**
   * Returns the cumulative time taken by a series of {@link SpawnResult}s.
   *
   * @param getSpawnResultExecutionTime a selector that returns either the wall, user or system time
   *     for each {@link SpawnResult} being considered
   * @return the cumulative time, or empty if no spawn results contained this time
   */
  private Optional<Duration> getCumulativeTime(
      Function<SpawnResult, Optional<Duration>> getSpawnResultExecutionTime) {
    Long totalMillis = null;
    for (SpawnResult spawnResult : spawnResults()) {
      Optional<Duration> executionTime = getSpawnResultExecutionTime.apply(spawnResult);
      if (executionTime.isPresent()) {
        if (totalMillis == null) {
          totalMillis = executionTime.get().toMillis();
        } else {
          totalMillis += executionTime.get().toMillis();
        }
      }
    }
    if (totalMillis == null) {
      return Optional.empty();
    } else {
      return Optional.of(Duration.ofMillis(totalMillis));
    }
  }

  /**
   * Returns the cumulative total of long values taken from a series of {@link SpawnResult}s.
   *
   * @param getSpawnResultLongValue a selector that returns a long value for each {@link
   *     SpawnResult} being considered
   * @return the total, or empty if no spawn results contained this long value
   */
  private Optional<Long> getCumulativeLong(
      Function<SpawnResult, Optional<Long>> getSpawnResultLongValue) {
    Long longTotal = null;
    for (SpawnResult spawnResult : spawnResults()) {
      Optional<Long> longValue = getSpawnResultLongValue.apply(spawnResult);
      if (longValue.isPresent()) {
        if (longTotal == null) {
          longTotal = longValue.get();
        } else {
          longTotal += longValue.get();
        }
      }
    }
    if (longTotal == null) {
      return Optional.empty();
    } else {
      return Optional.of(longTotal);
    }
  }

  /**
   * Returns the cumulative command execution wall time for the {@link Action}.
   *
   * @return the cumulative measurement, or empty in case of execution errors or when the
   *     measurement is not implemented for the current platform
   */
  public Optional<Duration> cumulativeCommandExecutionWallTime() {
    return getCumulativeTime(spawnResult -> spawnResult.getWallTime());
  }

  /**
   * Returns the cumulative command execution user time for the {@link Action}.
   *
   * @return the cumulative measurement, or empty in case of execution errors or when the
   *     measurement is not implemented for the current platform
   */
  public Optional<Duration> cumulativeCommandExecutionUserTime() {
    return getCumulativeTime(spawnResult -> spawnResult.getUserTime());
  }

  /**
   * Returns the cumulative command execution system time for the {@link Action}.
   *
   * @return the cumulative measurement, or empty in case of execution errors or when the
   *     measurement is not implemented for the current platform
   */
  public Optional<Duration> cumulativeCommandExecutionSystemTime() {
    return getCumulativeTime(spawnResult -> spawnResult.getSystemTime());
  }

  /**
   * Returns the cumulative number of block input operations for the {@link Action}.
   *
   * @return the cumulative measurement, or empty in case of execution errors or when the
   *     measurement is not implemented for the current platform
   */
  public Optional<Long> cumulativeCommandExecutionBlockInputOperations() {
    return getCumulativeLong(spawnResult -> spawnResult.getNumBlockInputOperations());
  }

  /**
   * Returns the cumulative number of block output operations for the {@link Action}.
   *
   * @return the cumulative measurement, or empty in case of execution errors or when the
   *     measurement is not implemented for the current platform
   */
  public Optional<Long> cumulativeCommandExecutionBlockOutputOperations() {
    return getCumulativeLong(spawnResult -> spawnResult.getNumBlockOutputOperations());
  }

  /**
   * Returns the cumulative number of involuntary context switches for the {@link Action}.
   *
   * @return the cumulative measurement, or empty in case of execution errors or when the
   *     measurement is not implemented for the current platform
   */
  public Optional<Long> cumulativeCommandExecutionInvoluntaryContextSwitches() {
    return getCumulativeLong(spawnResult -> spawnResult.getNumInvoluntaryContextSwitches());
  }

  /**
   * Indicates whether all {@link Spawn}s executed locally or not.
   *
   * @return true if all spawns of action executed locally
   */
  public boolean locallyExecuted() {
    boolean locallyExecuted = true;
    for (SpawnResult spawnResult : spawnResults()) {
      locallyExecuted &= !spawnResult.wasRemote();
    }
    return locallyExecuted;
  }

  /**
   * Returns the cumulative command execution CPU time for the {@link Action}.
   *
   * @return the cumulative measurement, or empty in case of execution errors or when the
   *     measurement is not implemented for the current platform
   */
  public Optional<Duration> cumulativeCommandExecutionCpuTime() {
    Optional<Duration> userTime = cumulativeCommandExecutionUserTime();
    Optional<Duration> systemTime = cumulativeCommandExecutionSystemTime();

    if (!userTime.isPresent() && !systemTime.isPresent()) {
      return Optional.empty();
    } else if (userTime.isPresent() && !systemTime.isPresent()) {
      return userTime;
    } else if (!userTime.isPresent() && systemTime.isPresent()) {
      return systemTime;
    } else {
      checkState(userTime.isPresent() && systemTime.isPresent());
      return Optional.of(userTime.get().plus(systemTime.get()));
    }
  }

  /** Creates an ActionResult given a list of SpawnResults. */
  public static ActionResult create(List<SpawnResult> spawnResults) {
    if (spawnResults == null) {
      return EMPTY;
    } else {
      return builder().setSpawnResults(ImmutableList.copyOf(spawnResults)).build();
    }
  }

  /** Builder for a {@link ActionResult} instance, which is immutable once built. */
  @AutoValue.Builder
  public abstract static class Builder {

    /** Sets the SpawnResults for the action. */
    public abstract Builder setSpawnResults(ImmutableList<SpawnResult> spawnResults);

    /** Builds and returns an ActionResult object. */
    public abstract ActionResult build();
  }
}
