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

import com.google.devtools.build.lib.profiler.ProfileInfo;
import com.google.devtools.build.lib.profiler.ProfileInfo.Task;

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
  public final long selfNanos;

  public TasksStatistics(
      String name,
      int count,
      long minNanos,
      long maxNanos,
      double medianNanos,
      double standardDeviationMillis,
      long totalNanos,
      long selfNanos) {
    this.name = name;
    this.count = count;
    this.minNanos = minNanos;
    this.maxNanos = maxNanos;
    this.medianNanos = medianNanos;
    this.standardDeviationMillis = standardDeviationMillis;
    this.totalNanos = totalNanos;
    this.selfNanos = selfNanos;
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

  public double selfMillis() {
    return toMilliSeconds(selfNanos);
  }

  public double selfMeanNanos() {
    return selfNanos / count;
  }

  public double selfMeanMillis() {
    return toMilliSeconds(selfMeanNanos());
  }

  /**
   * @return The set of statistics grouped in this class, computed from a list of {@link Task}s.
   */
  public static TasksStatistics create(String name, List<Task> tasks) {
    tasks = ProfileInfo.TASK_DURATION_ORDERING.immutableSortedCopy(tasks);
    int count = tasks.size();
    long min = tasks.get(0).durationNanos;
    long max = tasks.get(count - 1).durationNanos;

    int midIndex = count / 2;
    double median =
        tasks.size() % 2 == 0
            ? (tasks.get(midIndex).durationNanos + tasks.get(midIndex - 1).durationNanos) / 2.0
            : tasks.get(midIndex).durationNanos;

    // Compute standard deviation with a shift to avoid catastrophic cancellation
    // and also do it in milliseconds, as in nanoseconds it overflows
    long sum = 0L;
    long self = 0L;
    double sumOfSquaredShiftedMillis = 0L;
    final long shift = min;

    for (Task task : tasks) {
      sum += task.durationNanos;
      self += task.durationNanos - task.getInheritedDuration();
      double taskDurationShiftMillis = toMilliSeconds(task.durationNanos - shift);
      sumOfSquaredShiftedMillis += taskDurationShiftMillis * taskDurationShiftMillis;
    }
    double sumShiftedMillis = toMilliSeconds(sum - count * shift);

    double standardDeviation =
        Math.sqrt(
            (sumOfSquaredShiftedMillis - (sumShiftedMillis * sumShiftedMillis) / count) / count);

    return new TasksStatistics(name, count, min, max, median, standardDeviation, sum, self);
  }

  static double toMilliSeconds(double nanoseconds) {
    return nanoseconds / 1000000.0;
  }
}
