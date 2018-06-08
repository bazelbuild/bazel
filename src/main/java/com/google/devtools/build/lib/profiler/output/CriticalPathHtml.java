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

import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.CriticalPathEntry;
import com.google.devtools.build.lib.profiler.statistics.CriticalPathStatistics;
import com.google.devtools.build.lib.profiler.statistics.CriticalPathStatistics.MiddleManStatistics;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.TimeUtilities;
import java.io.PrintStream;

/**
 * Generate HTML output from {@link CriticalPathStatistics}.
 */
//TODO(bazel-team): Also print remote vs build stats recorded by Logging.CriticalPathStats
public final class CriticalPathHtml extends HtmlPrinter {

  private final CriticalPathStatistics criticalPathStats;
  private final long executionTime;

  public CriticalPathHtml(
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

  /**
   * Print table rows for timing statistics and per path timing percentages.
   */
  private void printCriticalPathTimingBreakdown(
      CriticalPathEntry totalPath, CriticalPathEntry optimalPath) {
    lnOpen("tr");
    element("td", "colspan", "4", totalPath.task.type);
    close();

    lnOpen("tr");
    element("td", "colspan", "3", "Main thread scheduling delays");
    element("td", TimeUtilities.prettyTime(criticalPathStats.getMainThreadWaitTime()));
    close(); // tr

    lnOpen("tr");
    element("td", "colspan", "4", "Critical path time:");
    close();

    long totalTime = totalPath.cumulativeDuration;
    lnOpen("tr");
    element("td", "Actual time");
    element("td", TimeUtilities.prettyTime(totalTime));
    element(
        "td",
        String.format(
            "(%s of execution time)", prettyPercentage((double) totalTime / executionTime)));
    close(); // tr

    long optimalTime = optimalPath.cumulativeDuration;
    element("td", "colspan", "2", "Time excluding scheduling delays");
    element("td", TimeUtilities.prettyTime(optimalTime));
    element(
        "td",
        String.format(
            "(%s of execution time)", prettyPercentage((double) optimalTime / executionTime)));
    close(); // tr

    // Artificial critical path if we ignore all the time spent in all tasks,
    // except time directly attributed to the ACTION tasks.
    lnElement("tr");
    lnOpen("tr");
    element("td", "colspan", "4", "Time related to:");
    close();

    for (Pair<String, Double> relativePathDuration : criticalPathStats) {
      lnOpen("tr");
      element("td", "colspan", "3", relativePathDuration.first);
      element("td", prettyPercentage(relativePathDuration.second));
      close();
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
    lnOpen("table");
    lnOpen("tr");
    element(
        "td",
        "colspan",
        "4",
        String.format("%s (%s):", title, TimeUtilities.prettyTime(path.cumulativeDuration)));
    close(); // tr

    lnOpen("tr");
    boolean pathIsComponent = path.isComponent();
    element("th", "Id");
    element("th", "Time");
    element("th", "Share");
    if (!pathIsComponent) {
      element("th", "Critical");
    }
    element("th", "Description");
    close(); // tr

    long totalPathTime = path.cumulativeDuration;

    for (CriticalPathEntry pathEntry : criticalPathStats.getMiddlemanFilteredPath(path)) {
      String desc = pathEntry.task.getDescription().replace(':', ' ');
      lnOpen("tr");
      element("td", pathEntry.task.id);
      element("td", "style", "text-align: right",
          TimeUtilities.prettyTime(pathEntry.duration).replace(" ", "&nbsp;"));
      element("td", prettyPercentage((double) pathEntry.duration / totalPathTime));
      if (!pathIsComponent) {
        element("td", prettyPercentage((double) pathEntry.getCriticalTime() / totalPathTime));
      }
      element("td", desc);
      close(); // tr
    }
    MiddleManStatistics middleMan = MiddleManStatistics.create(path);
    if (middleMan.count > 0) {
      lnOpen("tr");
      element("td");
      element("td", TimeUtilities.prettyTime(middleMan.duration));
      element("td", prettyPercentage((double) middleMan.duration / totalPathTime));
      if (!pathIsComponent) {
        element("td", prettyPercentage((double) middleMan.criticalTime / totalPathTime));
      }
      element("td", String.format("[%d middleman actions]", middleMan.count));
      close(); // tr
    }
    lnClose(); // table
  }
}


