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
package com.google.devtools.build.lib.profiler.statistics;

import com.google.devtools.build.lib.profiler.ProfilePhase;
import java.time.Duration;
import java.util.EnumMap;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Extracts and keeps summary statistics from all {@link ProfilePhase}s for formatting to various
 * outputs.
 */
public final class PhaseSummaryStatistics implements Iterable<ProfilePhase> {

  private long totalDurationNanos;
  private final EnumMap<ProfilePhase, Long> durations;

  public PhaseSummaryStatistics() {
    durations = new EnumMap<>(ProfilePhase.class);
    totalDurationNanos = 0;
  }

  /** Add a single profile phase. */
  public void addProfilePhase(ProfilePhase phase, Duration duration) {
    totalDurationNanos += duration.toNanos();
    durations.put(phase, duration.toNanos());
  }

  /** @return whether the given {@link ProfilePhase} was executed */
  private boolean contains(ProfilePhase phase) {
    return durations.containsKey(phase);
  }

  /**
   * @return the execution duration of a given {@link ProfilePhase}
   * @throws NoSuchElementException if the given {@link ProfilePhase} was not executed
   */
  public long getDurationNanos(ProfilePhase phase) {
    checkContains(phase);
    return durations.get(phase);
  }

  /**
   * @return The duration of the phase relative to the sum of all phase durations
   * @throws NoSuchElementException if the given {@link ProfilePhase} was not executed
   */
  public double getRelativeDuration(ProfilePhase phase) {
    checkContains(phase);
    return (double) getDurationNanos(phase) / totalDurationNanos;
  }

  public long getTotalDuration() {
    return totalDurationNanos;
  }

  @Override
  public Iterator<ProfilePhase> iterator() {
    return durations.keySet().iterator();
  }

  private void checkContains(ProfilePhase phase) {
    if (!contains(phase)) {
      throw new NoSuchElementException("Phase " + phase + " was not executed");
    }
  }
}

