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

import com.google.common.base.Optional;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.statistics.CriticalPathStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseSummaryStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseVfsStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseVfsStatistics.Stat;
import com.google.devtools.build.lib.util.TimeUtilities;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.EnumMap;

import javax.annotation.Nullable;

/**
 * Output {@link PhaseSummaryStatistics}, {@link PhaseStatistics} and {@link PhaseVfsStatistics}
 * in text format.
 */
public final class PhaseText extends TextPrinter {

  private final PhaseSummaryStatistics phaseSummaryStats;
  private final EnumMap<ProfilePhase, PhaseStatistics> phaseStatistics;
  private final Optional<CriticalPathStatistics> criticalPathStatistics;
  private final int vfsStatsLimit;
  private final int missingActionsCount;

  /**
   * @param vfsStatsLimit maximum number of VFS statistics to print, or -1 for no limit.
   */
  public PhaseText(
      PrintStream out,
      PhaseSummaryStatistics phaseSummaryStats,
      EnumMap<ProfilePhase, PhaseStatistics> phaseStatistics,
      Optional<CriticalPathStatistics> critPathStats,
      int missingActionsCount,
      int vfsStatsLimit) {
    super(out);
    this.phaseSummaryStats = phaseSummaryStats;
    this.phaseStatistics = phaseStatistics;
    this.criticalPathStatistics = critPathStats;
    this.missingActionsCount = missingActionsCount;
    this.vfsStatsLimit = vfsStatsLimit;
  }

  public void print() {
    printPhaseSummaryStatistics();

    for (ProfilePhase phase :
        Arrays.asList(ProfilePhase.INIT, ProfilePhase.LOAD, ProfilePhase.ANALYZE)) {
      PhaseStatistics statistics = phaseStatistics.get(phase);
      if (statistics.wasExecuted()) {
        printPhaseStatistics(statistics);
      }
    }
    printExecutionPhaseStatistics();
  }

  /**
   * Print a table for the phase overview with runtime and runtime percentage per phase and total.
   */
  private void printPhaseSummaryStatistics() {
    print("\n=== PHASE SUMMARY INFORMATION ===\n");
    for (ProfilePhase phase : phaseSummaryStats) {
      long phaseDuration = phaseSummaryStats.getDurationNanos(phase);
      double relativeDuration = phaseSummaryStats.getRelativeDuration(phase);
      lnPrintf(
          THREE_COLUMN_FORMAT,
          "Total " + phase.nick + " phase time",
          TimeUtilities.prettyTime(phaseDuration),
          prettyPercentage(relativeDuration));
    }
    lnPrintf(
        THREE_COLUMN_FORMAT,
        "Total run time",
        TimeUtilities.prettyTime(phaseSummaryStats.getTotalDuration()),
        "100.00%");
    printLn();
  }

  /**
   * Prints all statistics from {@link PhaseStatistics} in text form.
   */
  private void printPhaseStatistics(PhaseStatistics stats) {
    lnPrintf("=== %s PHASE INFORMATION ===\n", stats.getProfilePhase().nick.toUpperCase());

    lnPrintf(
        TWO_COLUMN_FORMAT,
        "Total " + stats.getProfilePhase().nick + " phase time",
        TimeUtilities.prettyTime(stats.getPhaseDurationNanos()));
    printLn();

    if (!stats.isEmpty()) {
      printTimingDistribution(stats);
      printLn();
      printVfsStatistics(stats.getVfsStatistics());
    }
  }

  private void printExecutionPhaseStatistics() {
    PhaseStatistics prepPhase = phaseStatistics.get(ProfilePhase.PREPARE);
    PhaseStatistics execPhase = phaseStatistics.get(ProfilePhase.EXECUTE);
    PhaseStatistics finishPhase = phaseStatistics.get(ProfilePhase.FINISH);
    if (!execPhase.wasExecuted()) {
      return;
    }
    lnPrint("=== EXECUTION PHASE INFORMATION ===\n");

    long graphTime = execPhase.getTotalDurationNanos(ProfilerTask.ACTION_GRAPH);
    long execTime = execPhase.getPhaseDurationNanos() - graphTime;

    if (prepPhase.wasExecuted()) {
      lnPrintf(
          TWO_COLUMN_FORMAT,
          "Total preparation time",
          TimeUtilities.prettyTime(prepPhase.getPhaseDurationNanos()));
    }
    lnPrintf(
        TWO_COLUMN_FORMAT,
        "Total execution phase time",
        TimeUtilities.prettyTime(execPhase.getPhaseDurationNanos()));
    if (finishPhase.wasExecuted()) {
      lnPrintf(
          TWO_COLUMN_FORMAT,
          "Total time finalizing build",
          TimeUtilities.prettyTime(finishPhase.getPhaseDurationNanos()));
    }
    printLn();
    lnPrintf(
        TWO_COLUMN_FORMAT, "Action dependency map creation", TimeUtilities.prettyTime(graphTime));
    lnPrintf(TWO_COLUMN_FORMAT, "Actual execution time", TimeUtilities.prettyTime(execTime));

    CriticalPathText criticalPaths = null;
    if (criticalPathStatistics.isPresent()) {
      criticalPaths = new CriticalPathText(out, criticalPathStatistics.get(), execTime);
      criticalPaths.printTimingBreakdown();
      printLn();
    }

    printTimingDistribution(execPhase);
    printLn();

    if (criticalPathStatistics.isPresent()) {
      criticalPaths.printCriticalPaths();
      printLn();
    }

    if (missingActionsCount > 0) {
      lnPrint(missingActionsCount);
      print(
          " action(s) are present in the"
              + " action graph but missing instrumentation data. Most likely the profile file"
              + " has been created during a failed or aborted build.");
      printLn();
    }

    printVfsStatistics(execPhase.getVfsStatistics());
  }

  /**
   * Prints a table of task types and their relative total and average execution time as well as
   * how many tasks of each type there were
   */
  private void printTimingDistribution(PhaseStatistics stats) {
    lnPrint("Total time (across all threads) spent on:");
    lnPrintf("%18s %8s %8s %11s", "Type", "Total", "Count", "Average");
    for (ProfilerTask type : stats) {
      lnPrintf(
          "%18s %8s %8d %11s",
          type.toString(),
          prettyPercentage(stats.getTotalRelativeDuration(type)),
          stats.getCount(type),
          TimeUtilities.prettyTime(stats.getMeanDuration(type)));
    }
  }

  /**
   * Print the time spent on VFS operations on each path. Output is grouped by operation and
   * sorted by descending duration. If multiple of the same VFS operation were logged for the same
   * path, print the total duration.
   */
  private void printVfsStatistics(@Nullable PhaseVfsStatistics stats) {
    if (vfsStatsLimit == 0 || stats == null || stats.isEmpty()) {
      return;
    }

    lnPrint("VFS path statistics:");
    lnPrintf("%15s %10s %10s %s", "Type", "Frequency", "Duration", "Path");
    for (ProfilerTask type : stats) {
      int numPrinted = 0;
      for (Stat stat : stats.getSortedStatistics(type)) {
        if (vfsStatsLimit != -1 && numPrinted++ == vfsStatsLimit) {
          lnPrintf("... %d more ...", stats.getStatisticsCount(type) - vfsStatsLimit);
          break;
        }
        lnPrintf(
            "%15s %10d %10s %s",
            type.name(),
            stat.getCount(),
            TimeUtilities.prettyTime(stat.getDuration()),
            stat.path);
      }
    }
    printLn();
  }
}


