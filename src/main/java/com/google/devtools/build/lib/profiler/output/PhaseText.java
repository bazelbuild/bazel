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

import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.statistics.CriticalPathStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseSummaryStatistics;
import java.io.PrintStream;
import java.time.Duration;

/** Output {@link PhaseSummaryStatistics} and {@link CriticalPathStatistics} in text format. */
public final class PhaseText extends TextPrinter {

  private final PhaseSummaryStatistics phaseSummaryStats;
  private final CriticalPathStatistics criticalPathStatistics;

  public PhaseText(
      PrintStream out,
      PhaseSummaryStatistics phaseSummaryStats,
      CriticalPathStatistics criticalPathStatistics) {
    super(out);
    this.phaseSummaryStats = phaseSummaryStats;
    this.criticalPathStatistics = criticalPathStatistics;
  }

  public void print() {
    printPhaseSummaryStatistics();

    CriticalPathText criticalPaths = new CriticalPathText(out, criticalPathStatistics);
    criticalPaths.printCriticalPaths();
    printLn();
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
}

