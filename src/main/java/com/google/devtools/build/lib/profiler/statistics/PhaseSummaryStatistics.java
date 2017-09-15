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
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo;
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

  public PhaseSummaryStatistics(ProfileInfo info) {
    this();
    addProfileInfo(info);
  }

  /**
   * Add a summary of the {@link ProfilePhase}s durations from a {@link ProfileInfo}.
   */
  public void addProfileInfo(ProfileInfo info) {
    for (ProfilePhase phase : ProfilePhase.values()) {
      ProfileInfo.Task phaseTask = info.getPhaseTask(phase);
      if (phaseTask != null) {
        long phaseDuration = info.getPhaseDuration(phaseTask);
        totalDurationNanos += phaseDuration;
        durations.put(phase, phaseDuration);
      }
    }
  }

  /**
   * @return whether the given {@link ProfilePhase} was executed
   */
  public boolean contains(ProfilePhase phase) {
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

  /**
   * Converts {@link #getRelativeDuration(ProfilePhase)} to a percentage string
   * @return formatted percentage string ("%.2f%%") or "N/A" when totalNanos is 0.
   * @throws NoSuchElementException if the given {@link ProfilePhase} was not executed
   */
  public String getPrettyPercentage(ProfilePhase phase) {
    checkContains(phase);
    if (totalDurationNanos == 0) {
      // Return "not available" string if total is 0 and result is undefined.
      return "N/A";
    }
    return String.format("%.2f%%", getRelativeDuration(phase) * 100);
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

