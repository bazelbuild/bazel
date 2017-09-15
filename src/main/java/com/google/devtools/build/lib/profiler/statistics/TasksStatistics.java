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

import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.Task;
import com.google.devtools.build.lib.util.LongArrayList;
import java.util.List;

/**
 * Data container to aggregate execution time statistics of multiple tasks grouped by some name.
 */
public class TasksStatistics {
  public final String name;
  public final int count;
  public final long minNanos;
  public final long maxNanos;
  public final double medianNanos;
  /** Standard deviation of the execution time in milliseconds since computation in nanoseconds can
   * overflow.
   */
  public final double standardDeviationMillis;
  public final long totalNanos;

  public TasksStatistics(
      String name,
      int count,
      long minNanos,
      long maxNanos,
      double medianNanos,
      double standardDeviationMillis,
      long totalNanos) {
    this.name = name;
    this.count = count;
    this.minNanos = minNanos;
    this.maxNanos = maxNanos;
    this.medianNanos = medianNanos;
    this.standardDeviationMillis = standardDeviationMillis;
    this.totalNanos = totalNanos;
  }

  public double minimumMillis() {
    return toMilliSeconds(minNanos);
  }

  public double maximumMillis() {
    return toMilliSeconds(maxNanos);
  }

  public double meanNanos() {
    return totalNanos / count;
  }

  public double meanMillis() {
    return toMilliSeconds(meanNanos());
  }

  public double medianMillis() {
    return toMilliSeconds(medianNanos);
  }

  public double totalMillis() {
    return toMilliSeconds(totalNanos);
  }

  @Override
  public String toString() {
    return String.format(
        "%s %d %.3f %.3f %.3f %.3f %.3f %.3f",
        name,
        count,
        minimumMillis(),
        meanMillis(),
        medianMillis(),
        maximumMillis(),
        standardDeviationMillis,
        totalMillis());
  }

  /**
   * @return The set of statistics grouped in this class, computed from a list of {@link Task}s.
   */
  public static TasksStatistics create(String name, List<Task> tasks) {
    LongArrayList durations = new LongArrayList(tasks.size());
    for (Task task : tasks) {
      durations.add(task.durationNanos);
    }
    return create(name, durations);
  }

  /**
   * @return The set of statistics grouped in this class, computed from a list of durations
   */
  public static TasksStatistics create(String name, LongArrayList durations) {
    durations.sort();
    int count = durations.size();
    long min = durations.get(0);
    long max = durations.get(count - 1);

    int midIndex = count / 2;
    double median =
        count % 2 == 0
            ? (durations.get(midIndex) + durations.get(midIndex - 1)) / 2.0
            : durations.get(midIndex);

    // Compute standard deviation with a shift to avoid catastrophic cancellation
    // and also do it in milliseconds, as in nanoseconds it overflows
    long sum = 0L;
    double sumOfSquaredShiftedMillis = 0L;
    final long shift = min;

    for (int index = 0; index < count; index++) {
      sum += durations.get(index);
      double taskDurationShiftMillis = toMilliSeconds(durations.get(index) - shift);
      sumOfSquaredShiftedMillis += taskDurationShiftMillis * taskDurationShiftMillis;
    }
    double sumShiftedMillis = toMilliSeconds(sum - count * shift);

    double standardDeviation =
        Math.sqrt(
            (sumOfSquaredShiftedMillis - (sumShiftedMillis * sumShiftedMillis) / count) / count);

    return new TasksStatistics(name, count, min, max, median, standardDeviation, sum);
  }

  static double toMilliSeconds(double nanoseconds) {
    return nanoseconds / 1000000.0;
  }
}
