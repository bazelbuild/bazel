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
package com.google.devtools.build.lib.profiler.output;

import com.google.devtools.build.lib.actions.MiddlemanAction;
import com.google.devtools.build.lib.profiler.ProfileInfo.CriticalPathEntry;
import com.google.devtools.build.lib.profiler.statistics.CriticalPathStatistics;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.TimeUtilities;

import java.io.PrintStream;

/**
 * Generate textual output from {@link CriticalPathStatistics}.
 */
//TODO(bazel-team): Also print remote vs build stats recorded by Logging.CriticalPathStats
public final class CriticalPathText extends TextPrinter {

  private final CriticalPathStatistics criticalPathStats;
  private long executionTime;

  public CriticalPathText(
      PrintStream out, CriticalPathStatistics critPathStats, long executionTime) {
    super(out);
    this.criticalPathStats = critPathStats;
    this.executionTime = executionTime;
  }

  public void printTimingBreakdown() {
    CriticalPathEntry totalPath = criticalPathStats.getTotalPath();
    CriticalPathEntry optimalPath = criticalPathStats.getOptimalPath();
    if (totalPath != null) {
      if (!totalPath.isComponent()) {
        printCriticalPathTimingBreakdown(totalPath, optimalPath);
      }
    } else {
      lnPrint("Critical path not available because no action graph was generated.");
    }
  }

  void printCriticalPathTimingBreakdown(
      CriticalPathEntry totalPath, CriticalPathEntry optimalPath) {
    lnPrint(totalPath.task.type);

    lnPrintf(
        TWO_COLUMN_FORMAT,
        "Worker thread scheduling delays",
        TimeUtilities.prettyTime(criticalPathStats.getWorkerWaitTime()));
    lnPrintf(
        TWO_COLUMN_FORMAT,
        "Main thread scheduling delays",
        TimeUtilities.prettyTime(criticalPathStats.getMainThreadWaitTime()));

    printLn();
    lnPrint("Critical path time:");

    long totalTime = totalPath.cumulativeDuration;
    lnPrintf(
        "%-37s %10s (%s of execution time)",
        "Actual time",
        TimeUtilities.prettyTime(totalTime),
        prettyPercentage((double) totalTime / executionTime));

    long optimalTime = optimalPath.cumulativeDuration;
    lnPrintf(
        "%-37s %10s (%s of execution time)",
        "Time excluding scheduling delays",
        TimeUtilities.prettyTime(optimalTime),
        prettyPercentage((double) optimalTime / executionTime));

    printLn();
    // Artificial critical path if we ignore all the time spent in all tasks,
    // except time directly attributed to the ACTION tasks.
    lnPrint("Time related to:");

    for (Pair<String, Double> relativePathDuration : criticalPathStats) {
      lnPrintf(
          TWO_COLUMN_FORMAT,
          relativePathDuration.first,
          prettyPercentage(relativePathDuration.second));
    }
  }

  /**
   * Print total and optimal critical paths if available.
   */
  public void printCriticalPaths() {
    CriticalPathEntry totalPath = criticalPathStats.getTotalPath();
    printCriticalPath("Critical path", totalPath);
    // In critical path components we do not record scheduling delay data so it does not make
    // sense to differentiate it.
    if (!totalPath.isComponent()) {
      printCriticalPath(
          "Critical path excluding scheduling delays", criticalPathStats.getOptimalPath());
    }
  }

  private void printCriticalPath(String title, CriticalPathEntry path) {
    lnPrintf("%s (%s):", title, TimeUtilities.prettyTime(path.cumulativeDuration));

    boolean isComponent = path.isComponent();
    if (isComponent) {
      lnPrintf("%6s %11s %8s   %s", "Id", "Time", "Percentage", "Description");
    } else {
      lnPrintf("%6s %11s %8s %8s   %s", "Id", "Time", "Share", "Critical", "Description");
    }

    long totalPathTime = path.cumulativeDuration;
    int middlemanCount = 0;
    long middlemanDuration = 0L;
    long middlemanCritTime = 0L;

    for (; path != null; path = path.next) {
      if (path.task.id < 0) {
        // Ignore fake actions.
        continue;
      } else if (path.task.getDescription().startsWith(MiddlemanAction.MIDDLEMAN_MNEMONIC + " ")
          || path.task.getDescription().startsWith("TargetCompletionMiddleman")) {
        // Aggregate middleman actions.
        middlemanCount++;
        middlemanDuration += path.duration;
        middlemanCritTime += path.getCriticalTime();
      } else {
        String desc = path.task.getDescription().replace(':', ' ');
        if (isComponent) {
          lnPrintf(
              "%6d %11s %8s   %s",
              path.task.id,
              TimeUtilities.prettyTime(path.duration),
              prettyPercentage((double) path.duration / totalPathTime),
              desc);
        } else {
          lnPrintf(
              "%6d %11s %8s %8s   %s",
              path.task.id,
              TimeUtilities.prettyTime(path.duration),
              prettyPercentage((double) path.duration / totalPathTime),
              prettyPercentage((double) path.getCriticalTime() / totalPathTime),
              desc);
        }
      }
    }
    if (middlemanCount > 0) {
      if (isComponent) {
        lnPrintf(
            "       %11s %8s   [%d middleman actions]",
            TimeUtilities.prettyTime(middlemanDuration),
            prettyPercentage((double) middlemanDuration / totalPathTime),
            middlemanCount);
      } else {
        lnPrintf(
            "       %11s %8s %8s   [%d middleman actions]",
            TimeUtilities.prettyTime(middlemanDuration),
            prettyPercentage((double) middlemanDuration / totalPathTime),
            prettyPercentage((double) middlemanCritTime / totalPathTime),
            middlemanCount);
      }
    }
  }
}


