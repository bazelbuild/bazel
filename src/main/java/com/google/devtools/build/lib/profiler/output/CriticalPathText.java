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
import com.google.devtools.build.lib.util.TimeUtilities;
import java.io.PrintStream;

/**
 * Generate textual output from {@link CriticalPathStatistics}.
 */
public final class CriticalPathText extends TextPrinter {
  private final CriticalPathStatistics criticalPathStats;

  public CriticalPathText(PrintStream out, CriticalPathStatistics critPathStats) {
    super(out);
    this.criticalPathStats = critPathStats;
  }

  /**
   * Print total and optimal critical paths if available.
   */
  public void printCriticalPaths() {
    CriticalPathEntry totalPath = criticalPathStats.getTotalPath();
    printCriticalPath("Critical path", totalPath);
  }

  private void printCriticalPath(String title, CriticalPathEntry path) {
    lnPrintf("%s (%s):", title, TimeUtilities.prettyTime(path.cumulativeDuration));
    lnPrintf("%6s %11s %8s   %s", "Id", "Time", "Percentage", "Description");

    long totalPathTime = path.cumulativeDuration;

    for (CriticalPathEntry pathEntry : criticalPathStats.getFilteredPath(path)) {
      String desc = pathEntry.task.getDescription().replace(':', ' ');
      lnPrintf(
          "%6d %11s %8s   %s",
          pathEntry.task.id,
          TimeUtilities.prettyTime(pathEntry.duration),
          prettyPercentage((double) pathEntry.duration / totalPathTime),
          desc);
    }
  }
}


