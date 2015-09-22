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
package com.google.devtools.build.lib.profiler.output;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.profiler.ProfileInfo;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.chart.AggregatingChartCreator;
import com.google.devtools.build.lib.profiler.chart.Chart;
import com.google.devtools.build.lib.profiler.chart.ChartCreator;
import com.google.devtools.build.lib.profiler.chart.DetailedChartCreator;
import com.google.devtools.build.lib.profiler.chart.HtmlChartVisitor;
import com.google.devtools.build.lib.profiler.statistics.PhaseStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseSummaryStatistics;
import com.google.devtools.build.lib.profiler.statistics.SkylarkStatistics;
import com.google.devtools.build.lib.vfs.Path;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.EnumMap;

/**
 * Creates an HTML page displaying the various statistics and charts generated
 * from the profile file.
 */
public final class HtmlCreator extends HtmlPrinter {

  private final Chart chart;
  private final HtmlChartVisitor chartVisitor;
  private final Optional<SkylarkHtml> skylarkStats;
  private final String title;
  private final PhaseHtml phases;

  private HtmlCreator(
      PrintStream out,
      String title,
      Chart chart,
      Optional<SkylarkHtml> skylarkStats,
      int htmlPixelsPerSecond,
      PhaseHtml phases) {
    super(out);
    this.title = title;
    this.chart = chart;
    chartVisitor = new HtmlChartVisitor(out, htmlPixelsPerSecond);
    this.skylarkStats = skylarkStats;
    this.phases = phases;
  }

  private void print() {
    htmlFrontMatter();
    chart.accept(chartVisitor);

    element("a", "name", "Statistics");
    element("h2", "Statistics");
    phases.print();

    if (skylarkStats.isPresent()) {
      skylarkStats.get().printHtmlBody();
    }
    htmlBackMatter();
  }

  private void htmlFrontMatter() {
    lnOpen("html");
    lnOpen("head");
    lnElement("title", title);
    chartVisitor.printCss(chart.getSortedTypes());

    phases.printCss();

    if (skylarkStats.isPresent()) {
      skylarkStats.get().printHtmlHead();
    }

    lnClose();
    lnOpen("body");
    lnElement("h1", title);
  }

  private void htmlBackMatter() {
    lnClose();
    lnClose();
  }

  /**
   * Writes the HTML profiling information.
   *
   * @throws IOException
   */
  public static void create(
      ProfileInfo info,
      Path htmlFile,
      PhaseSummaryStatistics phaseSummaryStats,
      EnumMap<ProfilePhase, PhaseStatistics> statistics,
      boolean detailed,
      int htmlPixelsPerSecond,
      int vfsStatsLimit)
      throws IOException {
    try (PrintStream out = new PrintStream(new BufferedOutputStream(htmlFile.getOutputStream()))) {
      ChartCreator chartCreator;
      PhaseHtml phaseHtml = new PhaseHtml(out, phaseSummaryStats, statistics, vfsStatsLimit);
      Optional<SkylarkHtml> skylarkStats;
      if (detailed) {
        skylarkStats = Optional.of(new SkylarkHtml(out, new SkylarkStatistics(info)));
        chartCreator = new DetailedChartCreator(info);
      } else {
        chartCreator = new AggregatingChartCreator(info);
        skylarkStats = Optional.absent();
      }
      Chart chart = chartCreator.create();
      new HtmlCreator(out, info.comment, chart, skylarkStats, htmlPixelsPerSecond, phaseHtml)
          .print();
    }
  }
}
