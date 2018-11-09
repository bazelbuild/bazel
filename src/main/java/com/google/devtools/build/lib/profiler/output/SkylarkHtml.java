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

import com.google.common.base.Joiner;
import com.google.common.base.StandardSystemProperty;
import com.google.devtools.build.lib.profiler.statistics.SkylarkStatistics;
import com.google.devtools.build.lib.profiler.statistics.TasksStatistics;
import com.google.devtools.build.lib.util.LongArrayList;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Map;

/**
 * Formats {@link SkylarkStatistics} as HTML tables and histogram charts.
 */
public final class SkylarkHtml extends HtmlPrinter {

  /**
   * How many characters from the end of the location of a Skylark function to display.
   */
  private static final int NUM_LOCATION_CHARS_UNABBREVIATED = 40;

  private static final String JS_DATA_VAR = "starlarkData";
  private static final String JS_TABLE_VAR = JS_DATA_VAR + "Table";

  private final SkylarkStatistics stats;
  private final boolean printHistograms;

  public SkylarkHtml(PrintStream out, SkylarkStatistics stats) {
    this(out, stats, true);
  }

  public SkylarkHtml(PrintStream out, SkylarkStatistics stats, boolean printHistograms) {
    super(out);
    this.stats = stats;
    this.printHistograms = printHistograms;
  }

  /**
   * Prints all CSS definitions and JavaScript code. May be a large amount of output.
   */
  void printHtmlHead() {
    lnOpen("style", "type", "text/css", "<!--");
    lnPrint("div.starlark-histogram {");
    lnPrint("  width: 95%; margin: 0 auto; display: none;");
    lnPrint("}");
    lnPrint("div.starlark-chart {");
    lnPrint("  width: 100%; height: 200px; margin: 0 auto 2em;");
    lnPrint("}");
    lnPrint("div.starlark-table {");
    lnPrint("  width: 95%; margin: 0 auto;");
    lnPrint("}");
    lnPrint("-->");
    close(); // style

    lnOpen("script", "type", "text/javascript");
    lnPrintf("var %s = {};\n", JS_DATA_VAR);
    lnPrintf("var %s = {};\n", JS_TABLE_VAR);
    lnPrint("var histogramData;");

    if (printHistograms) {
      lnPrint("var options = {");
      down();
      lnPrint("isStacked: true,");
      lnPrint("legend: { position: 'none' },");
      lnPrint("hAxis: { },");
      lnPrint("histogram: { lastBucketPercentile: 5 },");
      lnPrint("vAxis: { title: '# calls', viewWindowMode: 'pretty', gridlines: { count: -1 } }");
      up();
      lnPrint("};");
    }

    lnPrint("function selectHandler(category) {");
    down();
    lnPrint("return function() {");
    down();
    printf("var selection = %s[category].getSelection();", JS_TABLE_VAR);
    lnPrint("if (selection.length < 1) return;");
    lnPrint("var item = selection[0];");
    lnPrintf("var loc = %s[category].getValue(item.row, 0);", JS_DATA_VAR);
    lnPrintf("var func = %s[category].getValue(item.row, 1);", JS_DATA_VAR);
    lnPrint("var histogramDiv = document.getElementById(category+'-histogram');");
    if (printHistograms) {
      lnPrint("var key = loc + '#' + func;");
      lnPrint("var histData = histogramData[category][key];");
      lnPrint("var fnOptions = JSON.parse(JSON.stringify(options));");
      lnPrint("fnOptions.title = loc + '#' + func;");
      lnPrint("var chartDiv = document.getElementById(category+'-chart');");
      lnPrint("var chart = new google.visualization.Histogram(chartDiv);");
      lnPrint("histogramDiv.style.display = 'block';");
      lnPrint("chart.draw(histData, fnOptions);");
    } else {
      lnPrint("var chartDiv = document.getElementById(category+'-chart');");
      lnPrint("chartDiv.innerHTML = '<h3>' + loc + '#' + func + '</h3>';");
      lnPrint("chartDiv.style.height = 'auto';");
      lnPrint("histogramDiv.style.display = 'block';");
    }
    up();
    lnPrint("}");
    up();
    lnPrint("};");

    lnClose(); // script
  }

  /**
   * Prints the data for the tables of Skylark function statistics and - if needed - the
   * histogram data.
   */
  void printVisualizationCallbackJs() {
    printStatsJs(
        stats.getUserFunctionStatistics(),
        stats.getUserFunctionSelfStatistics(),
        "user",
        stats.getUserTotalNanos());
    printStatsJs(
        stats.getCompiledUserFunctionStatistics(),
        stats.getCompiledUserFunctionSelfStatistics(),
        "compiled",
        stats.getCompiledUserTotalNanos());
    printStatsJs(
        stats.getBuiltinFunctionStatistics(),
        stats.getBuiltinFunctionSelfStatistics(),
        "builtin",
        stats.getBuiltinTotalNanos());

    if (printHistograms) {
      printHistogramData();

      lnPrint("document.querySelector('#user-close').onclick = function() {");
      lnPrint("  document.querySelector('#user-histogram').style.display = 'none';");
      lnPrint("};");
      lnPrint("document.querySelector('#compiled-close').onclick = function() {");
      lnPrint("  document.querySelector('#compiled-histogram').style.display = 'none';");
      lnPrint("};");
      lnPrint("document.querySelector('#builtin-close').onclick = function() {");
      lnPrint("  document.querySelector('#builtin-histogram').style.display = 'none';");
      lnPrint("};");
    }
  }

  private void printHistogramData() {
    lnPrint("histogramData = {");
    down();
    printHistogramData(stats.getBuiltinFunctionDurations(), "builtin");
    printHistogramData(stats.getUserFunctionDurations(), "user");
    printHistogramData(stats.getCompiledUserFunctionDurations(), "compiled");
    up();
    lnPrint("}");
  }

  private void printHistogramData(Map<String, LongArrayList> functionDurations, String category) {
    lnPrintf("'%s': {", category);
    down();
    for (Map.Entry<String, LongArrayList> entry : functionDurations.entrySet()) {
      String function = entry.getKey();
      LongArrayList durations = entry.getValue();
      lnPrintf("'%s': google.visualization.arrayToDataTable(", function);
      lnPrint("[['duration']");
      for (int index = 0; index < durations.size(); index++) {
        printf(",[%f]", durations.get(index) / 1000000.);
      }

      lnPrint("], false),");
    }
    up();
    lnPrint("},");
  }

  private void printStatsJs(
      Map<String, TasksStatistics> taskStatistics,
      Map<String, TasksStatistics> taskSelfStatistics,
      String category,
      long totalNanos) {
    String tmpVar = category + JS_DATA_VAR;
    lnPrintf("var statsDiv = document.getElementById('%s_function_stats');", category);
    if (taskStatistics.isEmpty()) {
      lnPrint(
          "statsDiv.innerHTML = '<i>No relevant function calls to display. Some minor"
              + " builtin functions may have been ignored because their names could not be used"
              + " as variables in JavaScript.</i>'");
    } else {
      lnPrintf("var %s = new google.visualization.DataTable();", tmpVar);
      lnPrintf("%s.addColumn('string', 'Location');", tmpVar);
      lnPrintf("%s.addColumn('string', 'Function');", tmpVar);
      lnPrintf("%s.addColumn('number', 'count');", tmpVar);
      lnPrintf("%s.addColumn('number', 'min');", tmpVar);
      lnPrintf("%s.addColumn('number', 'mean');", tmpVar);
      lnPrintf("%s.addColumn('number', 'mean self');", tmpVar);
      lnPrintf("%s.addColumn('number', 'median');", tmpVar);
      lnPrintf("%s.addColumn('number', 'median self');", tmpVar);
      lnPrintf("%s.addColumn('number', 'max');", tmpVar);
      lnPrintf("%s.addColumn('number', 'max self');", tmpVar);
      lnPrintf("%s.addColumn('number', 'std dev');", tmpVar);
      lnPrintf("%s.addColumn('number', 'self');", tmpVar);
      lnPrintf("%s.addColumn('number', 'self (%%)');", tmpVar);
      lnPrintf("%s.addColumn('number', 'total');", tmpVar);
      lnPrintf("%s.addColumn('number', 'relative (%%)');", tmpVar);
      lnPrintf("%s.addRows([", tmpVar);
      down();
      for (Map.Entry<String, TasksStatistics> entry : taskStatistics.entrySet()) {
        String function = entry.getKey();
        TasksStatistics stats = entry.getValue();
        TasksStatistics selfStats = taskSelfStatistics.get(function);
        double relativeTotal = (double) stats.totalNanos / totalNanos;
        double relativeSelf = (double) selfStats.totalNanos / stats.totalNanos;
        String[] split = stats.name.split("#");
        String location;
        String name;
        if (split.length > 1) {
          location = split[0];
          name = split[1];
        } else {
          location = "(unknown)";
          name = split[0];
        }
        lnPrintf("[{v:'%s', f:'%s'}, ", location, abbreviatePath(location));
        printf("'%s', ", name);
        printf("%d, ", stats.count);
        printf("%.3f, ", stats.minimumMillis());
        printf("%.3f, ", stats.meanMillis());
        printf("%.3f, ", selfStats.meanMillis());
        printf("%.3f, ", stats.medianMillis());
        printf("%.3f, ", selfStats.medianMillis());
        printf("%.3f, ", stats.maximumMillis());
        printf("%.3f, ", selfStats.maximumMillis());
        printf("%.3f, ", stats.standardDeviationMillis);
        printf("%.3f, ", selfStats.totalMillis());
        printf("{v:%.4f, f:'%.3f %%'}, ", relativeSelf, relativeSelf * 100);
        printf("%.3f,", stats.totalMillis());
        printf("{v:%.4f, f:'%.3f %%'},", relativeTotal, relativeTotal * 100);
        printf("],");
      }
      lnPrint("]);");
      up();
      lnPrintf("%s.%s = %s;", JS_DATA_VAR, category, tmpVar);
      lnPrintf("%s.%s = new google.visualization.Table(statsDiv);", JS_TABLE_VAR, category);
      lnPrintf(
          "google.visualization.events.addListener(%s.%s, 'select', selectHandler('%s'));",
          JS_TABLE_VAR,
          category,
          category);
      lnPrintf(
          "%s.%s.draw(%s.%s, {showRowNumber: true, width: '100%%', height: '100%%'});",
          JS_TABLE_VAR,
          category,
          JS_DATA_VAR,
          category);
    }
  }

  /**
   * Prints two sections for histograms and tables of statistics for user-defined and built-in
   * Skylark functions.
   */
  void printHtmlBody() {
    lnPrint("<a name='starlark_stats'/>");
    lnElement("h3", "Starlark Statistics");
    lnElement("p", "All duration columns in milliseconds, except where noted otherwise.");
    lnElement("h4", "User-Defined function execution time");
    lnOpen("div", "class", "starlark-histogram", "id", "user-histogram");
    lnElement("div", "class", "starlark-chart", "id", "user-chart");
    lnElement("button", "id", "user-close", "Hide");
    lnClose(); // div user-histogram
    lnElement("div", "class", "starlark-table", "id", "user_function_stats");

    lnElement("h4", "Compiled function execution time");
    lnOpen("div", "class", "starlark-histogram", "id", "compiled-histogram");
    lnElement("div", "class", "starlark-chart", "id", "compiled-chart");
    lnElement("button", "id", "user-close", "Hide");
    lnClose(); // div compiled-histogram
    lnElement("div", "class", "starlark-table", "id", "compiled_function_stats");

    lnElement("h4", "Builtin function execution time");
    lnOpen("div", "class", "starlark-histogram", "id", "builtin-histogram");
    lnElement("div", "class", "starlark-chart", "id", "builtin-chart");
    lnElement("button", "id", "builtin-close", "Hide");
    lnClose(); // div builtin-histogram
    lnElement("div", "class", "starlark-table", "id", "builtin_function_stats");
  }

  /**
   * Computes a string keeping the structure of the input but reducing the amount of characters on
   * elements at the front if necessary.
   *
   * <p>Reduces the length of function location strings by keeping at least the last element fully
   * intact and at most {@link #NUM_LOCATION_CHARS_UNABBREVIATED} from other
   * elements from the end. Elements before are abbreviated with their first two characters.
   *
   * <p>Example:
   * "//source/tree/with/very/descriptive/and/long/hierarchy/of/directories/longfilename.bzl:42"
   * becomes: "//so/tr/wi/ve/de/an/lo/hierarch/of/directories/longfilename.bzl:42"
   *
   * <p>There is no fixed length to the result as the last element is kept and the location may
   * have many elements.
   *
   * @param location Either a sequence of path elements separated by
   *     {@link StandardSystemProperty#FILE_SEPARATOR} and preceded by some root element
   *     (e.g. "/", "C:\") or path elements separated by "." and having no root element.
   */
  private String abbreviatePath(String location) {
    String[] elements;
    int lowestAbbreviateIndex;
    String root;
    String separator = StandardSystemProperty.FILE_SEPARATOR.value();
    if (location.contains(separator)) {
      elements = location.split(separator);
      // must take care to preserve file system roots (e.g. "/", "C:\"), keep separate
      lowestAbbreviateIndex = 1;
      root = location.substring(0, location.indexOf(separator) + 1);
    } else {
      // must be java class name for a builtin function
      elements = location.split("\\.");
      lowestAbbreviateIndex = 0;
      root = "";
      separator = ".";
    }

    String last = elements[elements.length - 1];
    int remaining = NUM_LOCATION_CHARS_UNABBREVIATED - last.length();
    // start from the next to last element of the location and add until "remaining" many
    // chars added, abbreviate rest with first 2 characters
    for (int index = elements.length - 2; index >= lowestAbbreviateIndex; index--) {
      String element = elements[index];
      if (remaining > 0) {
        int length = Math.min(remaining, element.length());
        element = element.substring(0, length);
        remaining -= length;
      } else {
        element = element.substring(0, Math.min(2, element.length()));
      }
      elements[index] = element;
    }
    return root + Joiner.on(separator).join(Arrays.asList(elements).subList(1, elements.length));
  }
}
