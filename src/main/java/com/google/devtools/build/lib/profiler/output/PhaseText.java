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
import com.google.devtools.build.lib.util.TimeUtilities;
import java.io.PrintStream;
import java.time.Duration;
import java.util.Arrays;
import java.util.EnumMap;

/** Output {@link PhaseSummaryStatistics} and {@link PhaseStatistics} in text format. */
public final class PhaseText extends TextPrinter {

  private final PhaseSummaryStatistics phaseSummaryStats;
  private final EnumMap<ProfilePhase, PhaseStatistics> phaseStatistics;
  private final Optional<CriticalPathStatistics> criticalPathStatistics;

  public PhaseText(
      PrintStream out,
      PhaseSummaryStatistics phaseSummaryStats,
      EnumMap<ProfilePhase, PhaseStatistics> phaseStatistics,
      Optional<CriticalPathStatistics> critPathStats) {
    super(out);
    this.phaseSummaryStats = phaseSummaryStats;
    this.phaseStatistics = phaseStatistics;
    this.criticalPathStatistics = critPathStats;
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
    lnPrint("=== PHASE SUMMARY INFORMATION ===\n");
    for (ProfilePhase phase : phaseSummaryStats) {
      long phaseDurationInMs =
          Duration.ofNanos(phaseSummaryStats.getDurationNanos(phase)).toMillis();
      double relativeDuration = phaseSummaryStats.getRelativeDuration(phase);
      lnPrintf(
          THREE_COLUMN_FORMAT,
          "Total " + phase.nick + " phase time",
          String.format("%.3f s", phaseDurationInMs / 1000.0),
          prettyPercentage(relativeDuration));
    }

    lnPrintf("------------------------------------------------");
    long totalDurationInMs = Duration.ofNanos(phaseSummaryStats.getTotalDuration()).toMillis();
    lnPrintf(
        THREE_COLUMN_FORMAT,
        "Total run time",
        String.format("%.3f s", totalDurationInMs / 1000.0),
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

    long execTime = execPhase.getPhaseDurationNanos();

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
    lnPrintf(TWO_COLUMN_FORMAT, "Actual execution time", TimeUtilities.prettyTime(execTime));

    CriticalPathText criticalPaths = null;
    if (criticalPathStatistics.isPresent()) {
      criticalPaths = new CriticalPathText(out, criticalPathStatistics.get());
      printLn();
    }

    printTimingDistribution(execPhase);
    printLn();

    if (criticalPathStatistics.isPresent()) {
      criticalPaths.printCriticalPaths();
      printLn();
    }
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
}

