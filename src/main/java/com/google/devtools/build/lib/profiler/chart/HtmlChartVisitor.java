// Copyright 2014 The Bazel Authors. All rights reserved.
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

import java.io.PrintStream;
import java.util.List;
import java.util.Locale;

/**
 * {@link ChartVisitor} that builds HTML from the visited chart and prints it
 * out to the given {@link PrintStream}.
 */
public class HtmlChartVisitor implements ChartVisitor {

  /** The default width of a second in the chart. */
  private static final int DEFAULT_PIXEL_PER_SECOND = 50;

  /** The horizontal offset of second zero. */
  private static final int H_OFFSET = 40;

  /** The font size of the row labels. */
  private static final int ROW_LABEL_FONT_SIZE = 7;

  /** The height of a bar in pixels. */
  private static final int BAR_HEIGHT = 8;

  /** The space between twp bars in pixels. */
  private static final int BAR_SPACE = 2;

  /** The height of a row. */
  private static final int ROW_HEIGHT = BAR_HEIGHT + BAR_SPACE;

  /** The {@link PrintStream} to output the HTML to. */
  private final PrintStream out;

  /** The maxmimum stop time of any bar in the chart. */
  private long maxStop;

  /** The width of a second in the chart. */
  private final int pixelsPerSecond;

  /**
   * Creates the visitor, with a default width of a second of 50 pixels.
   *
   * @param out the {@link PrintStream} to output the HTML to
   */
  public HtmlChartVisitor(PrintStream out) {
    this(out, DEFAULT_PIXEL_PER_SECOND);
  }

  /**
   * Creates the visitor.
   *
   * @param out the {@link PrintStream} to output the HTML to
   * @param pixelsPerSecond The width of a second in the chart. (In pixels)
   */
  public HtmlChartVisitor(PrintStream out, int pixelsPerSecond) {
    this.out = out;
    this.pixelsPerSecond = pixelsPerSecond;
  }

  @Override
  public void visit(Chart chart) {
    maxStop = chart.getMaxStop();

    printContentBox();

    heading("Tasks", 2);
    out.println("<p>To get more information about a task point the mouse at one of the bars.</p>");

    out.printf(
        "<div style='position:relative; height: %dpx; margin: %dpx'>\n",
        chart.getRowCount() * ROW_HEIGHT, H_OFFSET + 10);
  }

  @Override
  public void visit(ChartColumn column) {
    int width = scale(column.getWidth());
    if (width == 0) {
      return;
    }
    int left = scale(column.getStart());
    int height = column.getRowCount() * ROW_HEIGHT;
    String style = chartTypeNameAsCSSClass(column.getType().getName());
    box(left, 0, width, height, style, column.getLabel(), 10);
  }

  @Override
  public void visit(ChartRow slot) {
    String style = slot.getIndex() % 2 == 0 ? "shade-even" : "shade-odd";
    int top = slot.getIndex() * ROW_HEIGHT;
    int width = scale(maxStop) + 1;

    label(-H_OFFSET, top, width + H_OFFSET, ROW_HEIGHT, ROW_LABEL_FONT_SIZE, slot.getId());
    box(0, top, width, ROW_HEIGHT, style, "", 0);
  }

  @Override
  public void visit(ChartBar bar) {
    int width = scale(bar.getWidth());
    if (width == 0) {
      return;
    }
    int left = scale(bar.getStart());
    int top = bar.getRow().getIndex() * ROW_HEIGHT;
    String style = chartTypeNameAsCSSClass(bar.getType().getName());
    if (bar.getHighlight()) {
      style += "-highlight";
    }
    box(left, top + 2, width, BAR_HEIGHT, style, bar.getLabel(), 20);
  }

  @Override
  public void visit(ChartLine chartLine) {
    int start = chartLine.getStartRow().getIndex() * ROW_HEIGHT;
    int stop = chartLine.getStopRow().getIndex() * ROW_HEIGHT;
    int time = scale(chartLine.getStartTime());

    if (start < stop) {
      verticalLine(time, start + 1, 1, (stop - start) + ROW_HEIGHT, Color.RED);
    } else {
      verticalLine(time, stop + 1, 1, (start - stop) + ROW_HEIGHT, Color.RED);
    }
  }

  @Override
  public void endVisit(Chart chart) {
    printTimeAxis(chart);
    out.println("</div>");

    heading("Legend", 2);
    printLegend(chart.getSortedTypes());
  }

  /** Converts the given value from the bar of the chart to pixels. */
  private int scale(long value) {
    return (int) (value / (1000000000L / pixelsPerSecond));
  }

  /**
   * Prints a box with links to the sections of the generated HTML document.
   */
  private void printContentBox() {
    out.println("<div style='position:fixed; top:1em; right:1em; z-index:50; padding: 1ex;"
        + "border:1px solid #888; background-color:#eee; width:100px'><h3>Content</h3>");
    out.println("<p style='text-align:left;font-size:small;margin:2px'>"
        + "<a href='#Tasks'>Tasks</a></p>");
    out.println("<p style='text-align:left;font-size:small;margin:2px'>"
        + "<a href='#Legend'>Legend</a></p>");
    out.println("<p style='text-align:left;font-size:small;margin:2px'>"
        + "<a href='#Statistics'>Statistics</a></p></div>");
  }

  /**
   * Prints the time axis of the chart and vertical lines for every second.
   */
  private void printTimeAxis(Chart chart) {
    int location = 0;
    int second = 0;
    int end = scale(chart.getMaxStop());
    while (location < end) {
      label(location + 4, -17, pixelsPerSecond, ROW_HEIGHT, 0, second + "s");
      verticalLine(location, -20, 1, chart.getRowCount() * ROW_HEIGHT + 20, Color.GRAY);
      location += pixelsPerSecond;
      second += 1;
    }
  }

  public void printCss(List<ChartBarType> types) {
    out.println("<style type=\"text/css\"><!--");
    out.println("body { font-family: Sans; }");
    out.printf("div.shade-even { position:absolute; border: 0px; background-color:#dddddd }\n");
    out.printf("div.shade-odd { position:absolute; border: 0px; background-color:#eeeeee }\n");
    for (ChartBarType type : types) {
      String name = chartTypeNameAsCSSClass(type.getName());
      String color = formatColor(type.getColor());

      out.printf(
          "div.%s-border { position:absolute; border:1px solid grey; background-color:%s }\n",
          name, color);
      out.printf(
          "div.%s-highlight { position:absolute; border:1px solid red; background-color:%s }\n",
          name, color);
      out.printf("div.%s { position:absolute; border:0px; margin:1px; background-color:%s }\n",
          name, color);
    }
    out.println("--></style>");
  }

  /**
   * Prints the legend for the chart at the current position in the document. The
   * legend is printed in columns of 10 rows each.
   *
   * @param types the list of {@link ChartBarType}s to print in the legend.
   */
  private void printLegend(List<ChartBarType> types) {
    final int boxHeight = 20;
    final int lineHeight = 25;
    final int entriesPerColumn = 10;
    final int legendWidth = 350;
    int legendHeight;
    if (types.size() / entriesPerColumn >= 1) {
      legendHeight = entriesPerColumn;
    } else {
      legendHeight = types.size() % entriesPerColumn;
    }

    out.printf("<div style='position:relative; height: %dpx;'>",
        (legendHeight + 1) * lineHeight);

    int left = -legendWidth;
    int top;
    int i = 0;
    for (ChartBarType type : types) {
      if (i % entriesPerColumn == 0) {
        left += legendWidth;
        i = 0;
      }
      top = lineHeight * i;
      String style = chartTypeNameAsCSSClass(type.getName()) + "-border";
      box(left, top, boxHeight, boxHeight, style, type.getName(), 0);
      label(left + lineHeight + 10, top, legendWidth - 10, boxHeight, 0, type.getName());
      i++;
    }
    out.println("</div>");
  }

  /**
   * Prints a head-line at the current position in the document.
   *
   * @param text the text to print
   * @param level the headline level
   */
  private void heading(String text, int level) {
    anchor(text);
    out.printf("<h%d >%s</h%d>\n", level, text, level);
  }

  /**
   * Prints a box with the given location, size, background color and border.
   *
   * @param x the x location of the top left corner of the box
   * @param y the y location of the top left corner of the box
   * @param width the width location of the box
   * @param height the height location of the box
   * @param style the CSS style class to use for the box
   * @param title the text displayed when the mouse hovers over the box
   */
  private void box(int x, int y, int width, int height, String style, String title, int zIndex) {
    out.printf("<div class=\"%s\" title=\"%s\" "
        + "style=\"left:%dpx; top:%dpx; width:%dpx; height:%dpx; z-index:%d\"></div>\n",
        style, title, x, y, width, height, zIndex);
  }

  /**
   * Prints a label with the given location, size, background color and border.
   *
   * @param x the x location of the top left corner of the box
   * @param y the y location of the top left corner of the box
   * @param width the width location of the box
   * @param height the height location of the box
   * @param fontSize the font size of text in the box, 0 for default
   * @param text the text displayed in the box
   */
  private void label(int x, int y, int width, int height, int fontSize, String text) {
    if (fontSize > 0) {
      out.printf("<div style=\"position:absolute; left:%dpx; top:%dpx; width:%dpx; "
          + "height:%dpx; font-size:%dpt\">%s</div>\n",
          x, y, width, height, fontSize, text);
    } else {
      out.printf("<div style=\"position:absolute; left:%dpx; top:%dpx; width:%dpx; "
          + "height:%dpx\">%s</div>\n",
          x, y, width, height, text);
    }
  }

  /**
   * Prints a vertical line of given width, height and color at the given
   * location.
   *
   * @param x the x location of the start point of the line
   * @param y the y location of the start point of the line
   * @param width the width of the line
   * @param length the length of the line
   * @param color the color of the line
   */
  private void verticalLine(int x, int y, int width, int length, Color color) {
    out.printf("<div style='position: absolute; left: %dpx; top: %dpx; width: %dpx; "
        + "height: %dpx; border-left: %dpx solid %s'" + "></div>\n",
        x, y, width, length, width, formatColor(color));
  }

  /**
   * Prints an HTML anchor with the given name,
   */
  private void anchor(String name) {
    out.println("<a name='" + name + "'/>");
  }

  /** Formats the given {@link Color} to a css style color string. */
  public static String formatColor(Color color) {
    int r = color.getRed();
    int g = color.getGreen();
    int b = color.getBlue();
    int a = color.getAlpha();

    // US Locale is used to ensure a dot as decimal separator
    return String.format(Locale.US, "rgba(%d,%d,%d,%f)", r, g, b, (a / 255.0));
  }

  /**
   * Transform the name into a form suitable as a css class.
   */
  private String chartTypeNameAsCSSClass(String name) {
    return name.replace(' ', '_');
  }
}
