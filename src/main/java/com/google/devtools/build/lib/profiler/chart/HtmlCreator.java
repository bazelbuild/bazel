// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.profiler.chart;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.profiler.ProfileInfo;
import com.google.devtools.build.lib.profiler.ProfilePhaseStatistics;
import com.google.devtools.build.lib.vfs.Path;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;

/**
 * Creates an HTML page displaying the various statistics and charts generated
 * from the profile file.
 */
public final class HtmlCreator {

  private final PrintStream out;
  private final Chart chart;
  private final HtmlChartVisitor chartVisitor;
  private final Optional<SkylarkStatistics> skylarkStats;

  /**
   * Pre-formatted statistics for each phase of the profiled build.
   */
  private final List<ProfilePhaseStatistics> statistics;

  private HtmlCreator(
      PrintStream out,
      Chart chart,
      Optional<SkylarkStatistics> skylarkStats,
      int htmlPixelsPerSecond,
      List<ProfilePhaseStatistics> statistics) {
    this.out = out;
    this.chart = chart;
    chartVisitor = new HtmlChartVisitor(out, htmlPixelsPerSecond);
    this.skylarkStats = skylarkStats;
    this.statistics = statistics;
  }

  private void print() {
    htmlFrontMatter();
    chart.accept(chartVisitor);

    out.println("<h2>Statistics</h2>");
    printPhaseStatistics();

    if (skylarkStats.isPresent()) {
      skylarkStats.get().printHtmlBody();
    }
    htmlBackMatter();
  }

  private void htmlFrontMatter() {
    out.println("<html><head>");
    out.printf("<title>%s</title>", chart.getTitle());
    chartVisitor.printCss(chart.getSortedTypes());

    if (skylarkStats.isPresent()) {
      skylarkStats.get().printHtmlHead();
    }

    out.println("</head>");
    out.println("<body>");
  }

  private void htmlBackMatter() {
    out.println("</body>");
    out.println("</html>");
  }

  /**
   * Print a table from {@link #statistics} arranging the phases side by side.
   */
  private void printPhaseStatistics() {
    out.println("<table border=\"0\" width=\"100%\"><tr>");
    String statsSeparator = "";
    for (ProfilePhaseStatistics stat : statistics) {
      out.println(statsSeparator);
      out.println("<td valign=\"top\" style=\"margin: 0 10 0;\">");
      String title = stat.getTitle();
      if (!title.isEmpty()) {
        out.println(String.format("<h3>%s</h3>", title));
      }
      out.println("<pre>" + stat.getStatistics() + "</pre></td>");
      statsSeparator = "<td><div style=\"width:20px;\">&#160;</div></td>";
    }
    out.println("</tr></table>");
  }

  /**
   * Writes the HTML profiling information.
   * @param info
   * @param htmlFile
   * @param statistics
   * @param detailed
   * @param htmlPixelsPerSecond
   * @throws IOException
   */
  public static void createHtml(
      ProfileInfo info,
      Path htmlFile,
      List<ProfilePhaseStatistics> statistics,
      boolean detailed,
      int htmlPixelsPerSecond)
      throws IOException {
    try (PrintStream out = new PrintStream(new BufferedOutputStream(htmlFile.getOutputStream()))) {
      ChartCreator chartCreator;
      Optional<SkylarkStatistics> skylarkStats;
      if (detailed) {
        chartCreator = new DetailedChartCreator(info);
        skylarkStats = Optional.of(new SkylarkStatistics(out, info));
      } else {
        chartCreator = new AggregatingChartCreator(info);
        skylarkStats = Optional.absent();
      }
      Chart chart = chartCreator.create();
      new HtmlCreator(out, chart, skylarkStats, htmlPixelsPerSecond, statistics).print();
    }
  }
}
