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
import com.google.devtools.build.lib.profiler.statistics.MultiProfileStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseStatistics;
import com.google.devtools.build.lib.util.TimeUtilities;
import com.google.devtools.build.lib.vfs.Path;

import java.io.PrintStream;
import java.util.EnumMap;

/**
 * Formats the per file phase statistics from {@link MultiProfileStatistics} into HTML tables.
 */
public final class MultiProfilePhaseHtml extends HtmlPrinter {

  private final MultiProfileStatistics statistics;

  public MultiProfilePhaseHtml(PrintStream out, MultiProfileStatistics statistics) {
    super(out);
    this.statistics = statistics;
  }

  /**
   * Prints CSS definitions and JavaScript code.
   */
  void printHtmlHead() {
    lnOpen("style", "type", "text/css", "<!--");
    lnPrint("div.profiles-table {");
    lnPrint("  width: 95%; margin: 0 auto; height: auto;");
    lnPrint("}");
    lnPrint("-->");
    close(); // style
  }

  /**
   * Prints the table data and JS for each phase and file.
   *
   * <p>Code must be added to the callback that is run when the Visualization library has loaded.
   */
  public void printVisualizationCallbackJs() {
    lnPrint("var multiData;");
    lnPrint("var statsDiv;");
    lnPrint("var profileTable;");
    for (ProfilePhase phase : statistics.getSummaryStatistics()) {
      lnPrintf("statsDiv = document.getElementById('profile_file_stats_%s');", phase.nick);
      lnPrint("multiData = new google.visualization.DataTable();");
      lnPrint("multiData.addColumn('string', 'File');");
      lnPrint("multiData.addColumn('number', 'total');");
      PhaseStatistics summaryPhaseStatistics = statistics.getSummaryPhaseStatistics(phase);
      for (ProfilerTask taskType : summaryPhaseStatistics) {
        lnPrintf("multiData.addColumn('number', '%s %%');", taskType.name());
      }
      lnPrint("multiData.addRows([");
      down();
      for (Path file : statistics) {
        EnumMap<ProfilePhase, PhaseStatistics> phases = statistics.getPhaseStatistics(file);
        PhaseStatistics phaseStatistics = phases.get(phase);
        lnPrintf("['%s', ", file);
        long phaseDuration = phaseStatistics.getPhaseDurationNanos();
        printf("{v:%d, f:'%s'}, ", phaseDuration, TimeUtilities.prettyTime(phaseDuration));
        for (ProfilerTask taskType : summaryPhaseStatistics) {
          if (phaseStatistics.wasExecuted(taskType)) {
            double relative = phaseStatistics.getTotalRelativeDuration(taskType);
            printf("{v:%.4f, f:'%.3f %%'}, ", relative, relative * 100);
          } else {
            print("0, ");
          }
        }
        print("],");
      }
      lnPrint("]);");
      up();
      lnPrint("profileTable = new google.visualization.Table(statsDiv);");
      lnPrint("profileTable.draw(multiData, {showRowNumber: true, width: '100%%'});");
    }
  }

  /**
   * Prints divs for the tables of statistics for profile files and their phases.
   */
  void printHtmlBody() {
    lnPrint("<a name='profile_file_stats'/>");
    lnElement("h3", "Profile File Statistics");
    lnOpen("div", "class", "profiles-tables", "id", "profile_file_stats");
    for (ProfilePhase phase : statistics.getSummaryStatistics()) {
      lnOpen("div");
      lnElement("h4", phase.nick);
      lnElement("div", "class", "profiles-table", "id", "profile_file_stats_" + phase.nick);
      lnClose();
    }
    lnClose(); // div
  }
}


