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
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.statistics.PhaseStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseSummaryStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseVfsStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseVfsStatistics.Stat;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.util.TimeUtilities;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.EnumMap;

/**
 * Output {@link PhaseSummaryStatistics}, {@link PhaseStatistics} and {@link PhaseVfsStatistics}
 * in HTML format.
 */
public final class PhaseHtml extends HtmlPrinter {

  private final PhaseSummaryStatistics phaseSummaryStats;
  private final EnumMap<ProfilePhase, PhaseStatistics> phaseStatistics;
  private final int vfsStatsLimit;

  /**
   * @param vfsStatsLimit maximum number of VFS statistics to print, or -1 for no limit.
   */
  public PhaseHtml(
      PrintStream out,
      PhaseSummaryStatistics phaseSummaryStats,
      EnumMap<ProfilePhase, PhaseStatistics> phaseStatistics,
      int vfsStatsLimit) {
    super(out);
    this.phaseSummaryStats = phaseSummaryStats;
    this.phaseStatistics = phaseStatistics;
    this.vfsStatsLimit = vfsStatsLimit;
  }

  /**
   * Output a style tag with all necessary CSS directives
   */
  public void printCss() {
    lnPrint("<style type=\"text/css\"><!--");
    down();
    lnPrint("div.phase-statistics {");
    lnPrint("  margin: 0 10;");
    lnPrint("  font-size: small;");
    lnPrint("  font-family: monospace;");
    lnPrint("  float: left;");
    lnPrint("}");
    lnPrint("table.phase-statistics {");
    lnPrint("  border: 0px; text-align: right;");
    lnPrint("}");
    lnPrint("table.phase-statistics td {");
    lnPrint("  padding: 0 5;");
    lnPrint("}");
    lnPrint("td.left {");
    lnPrint("  text-align: left;");
    lnPrint("}");
    lnPrint("td.center {");
    lnPrint("  text-align: center;");
    lnPrint("}");
    up();
    lnPrint("--></style>");
  }

  /**
   * Print tables from {@link #phaseSummaryStats} and {@link #phaseStatistics} side by side.
   */
  public void print() {
    printPhaseSummaryStatistics();

    for (ProfilePhase phase :
        Arrays.asList(ProfilePhase.INIT, ProfilePhase.LOAD, ProfilePhase.ANALYZE)) {
      PhaseStatistics statistics = phaseStatistics.get(phase);
      if (statistics == null || !statistics.wasExecuted()) {
        continue;
      }
      printPhaseStatistics(statistics);
    }
    lnElement("div", "style", "clear: both;");
  }

  /**
   * Print header and tables for a single phase.
   */
  private void printPhaseStatistics(PhaseStatistics phaseStat) {
    lnOpen("div", "class", "phase-statistics");
    lnElement(
        "h3",
        String.format(
            "%s Phase Information", StringUtil.capitalize(phaseStat.getProfilePhase().nick)));
    lnOpen("table", "class", "phase-statistics");
    lnOpen("tr");
    open("td", "class", "left", "colspan", "3");
    printf("Total %s phase time", phaseStat.getProfilePhase().nick);
    close(); // td
    element("td", TimeUtilities.prettyTime(phaseStat.getPhaseDurationNanos()));
    lnClose(); // tr

    if (!phaseStat.isEmpty()) {
      lnOpen("tr");
      element("td", "class", "left", "colspan", "4", "Total time (across all threads) spent on:");
      close(); // tr
      lnOpen("tr");
      element("td", "Type");
      element("td", "Total");
      element("td", "Count");
      element("td", "Average");
      close(); // tr
      for (ProfilerTask taskType : phaseStat) {
        lnOpen("tr", "class", "phase-task-statistics");
        element("td", taskType);
        element("td", prettyPercentage(phaseStat.getTotalRelativeDuration(taskType)));
        element("td", phaseStat.getCount(taskType));
        element("td", TimeUtilities.prettyTime(phaseStat.getMeanDuration(taskType)));
        close(); // tr
      }
    }
    lnClose(); // table
    printVfsStatistics(phaseStat.getVfsStatistics());
    lnClose(); // div
  }

  /**
   * Print the time spent on VFS operations on each path. Output is grouped by operation and sorted
   * by descending duration. If multiple of the same VFS operation were logged for the same path,
   * print the total duration.
   */
  private void printVfsStatistics(PhaseVfsStatistics stats) {
    if (vfsStatsLimit == 0 || stats.isEmpty()) {
      return;
    }

    lnElement("h4", "VFS path statistics:");

    lnOpen("table", "class", "phase-statistics");
    lnOpen("tr");
    element("td", "Type");
    element("td", "Frequency");
    element("td", "Duration");
    element("td", "class", "left", "Path");
    close(); // tr

    for (ProfilerTask type : stats) {
      int numPrinted = 0;
      for (Stat stat : stats.getSortedStatistics(type)) {
        lnOpen("tr");
        if (vfsStatsLimit != -1 && numPrinted++ == vfsStatsLimit) {
          open("td", "class", "center", "colspan", "4");
          printf("... %d more ...", stats.getStatisticsCount(type) - vfsStatsLimit);
          close();
          close(); // tr
          break;
        }
        element("td", type.name());
        element("td", stat.getCount());
        element("td", TimeUtilities.prettyTime(stat.getDuration()));
        element("td", "class", "left", stat.path);
        close(); // tr
      }
    }

    lnClose(); // table
  }

  /**
   * Print a table for the phase overview with runtime and runtime percentage per phase and total.
   */
  private void printPhaseSummaryStatistics() {
    lnOpen("div", "class", "phase-statistics");
    lnElement("h3", "Phase Summary Information");
    lnOpen("table", "class", "phase-statistics");
    for (ProfilePhase phase : phaseSummaryStats) {
      lnOpen("tr");
      lnOpen("td", "class", "left");
      printf("Total %s phase time", phase.nick);
      close();
      element("td", TimeUtilities.prettyTime(phaseSummaryStats.getDurationNanos(phase)));
      element("td", phaseSummaryStats.getPrettyPercentage(phase));
      lnClose(); // tr
    }
    lnOpen("tr");
    lnElement("td", "class", "left", "Total run time");
    element("td", TimeUtilities.prettyTime(phaseSummaryStats.getTotalDuration()));
    element("td", "100.00%");
    lnClose(); // tr
    lnClose(); // table
    lnClose(); // div
  }
}


