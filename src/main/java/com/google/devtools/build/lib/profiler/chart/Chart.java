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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Data of a Gantt Chart to visualize the data of a profiled build.
 */
public class Chart {

  /** The type that is returned when an unknown type is looked up. */
  public static final ChartBarType UNKNOWN_TYPE = new ChartBarType("Unknown type", Color.RED);

  /** The rows of the chart. */
  private final Map<Long, ChartRow> rows = new HashMap<>();

  /** The columns on the chart. */
  private final List<ChartColumn> columns = new ArrayList<>();

  /** The lines on the chart. */
  private final List<ChartLine> lines = new ArrayList<>();

  /** The types of the bars in the chart. */
  private final Map<String, ChartBarType> types = new HashMap<>();

  /** The running index of the rows in the chart. */
  private int rowIndex = 0;

  /** The maximum stop value of any bar in the chart. */
  private long maxStop;

  /**
   * Adds a bar to a row of the chart. If a row with the given id already
   * exists, the bar is added to the row, otherwise a new row is created and the
   * bar is added to it.
   *
   * @param id the id of the row the new bar belongs to
   * @param start the start value of the bar
   * @param stop the stop value of the bar
   * @param type the type of the bar
   * @param highlight emphasize the bar
   * @param label the label of the bar
   */
  public void addBar(long id, long start, long stop, ChartBarType type, boolean highlight,
      String label) {
    ChartRow slot = addSlotIfAbsent(id);
    ChartBar bar = new ChartBar(slot, start, stop, type, highlight, label);
    slot.addBar(bar);
    maxStop = Math.max(maxStop, stop);
  }

  /**
   * Adds a bar to a row of the chart. If a row with the given id already
   * exists, the bar is added to the row, otherwise a new row is created and the
   * bar is added to it.
   *
   * @param id the id of the row the new bar belongs to
   * @param start the start value of the bar
   * @param stop the stop value of the bar
   * @param type the type of the bar
   * @param label the label of the bar
   */
  public void addBar(long id, long start, long stop, ChartBarType type, String label) {
    addBar(id, start, stop, type, false, label);
  }

  /**
   * Adds a vertical line to the chart.
   */
  public void addVerticalLine(long startId, long stopId, long pos) {
    ChartRow startSlot = addSlotIfAbsent(startId);
    ChartRow stopSlot = addSlotIfAbsent(stopId);
    ChartLine line = new ChartLine(startSlot, stopSlot, pos, pos);
    lines.add(line);
  }

  /**
   * Adds a column to the chart.
   *
   * @param start the start value of the bar
   * @param stop the stop value of the bar
   * @param type the type of the bar
   * @param label the label of the bar
   */
  public void addTimeRange(long start, long stop, ChartBarType type, String label) {
    ChartColumn column = new ChartColumn(start, stop, type, label);
    columns.add(column);
    maxStop = Math.max(maxStop, stop);
  }

  /**
   * Creates a new {@link ChartBarType} and adds it to the list of types of the
   * chart.
   *
   * @param name the name of the type
   * @param color the color of the chart
   * @return the newly created type
   */
  public ChartBarType createType(String name, Color color) {
    ChartBarType type = new ChartBarType(name, color);
    types.put(name, type);
    return type;
  }

  /**
   * Returns the type with the given name. If no type with the given name
   * exists, a type with name 'Unknown type' is added to the chart and returned.
   *
   * @param name the name of the type to look up
   */
  public ChartBarType lookUpType(String name) {
    ChartBarType type = types.get(name);
    if (type == null) {
      type = UNKNOWN_TYPE;
      types.put(type.getName(), type);
    }
    return type;
  }

  /**
   * Creates a new row with the given id if no row with this id existed.
   * Otherwise the existing row with the given id is returned.
   *
   * @param id the ID of the row
   * @return the existing row, if it was already present, the newly created one
   *         otherwise
   */
  private ChartRow addSlotIfAbsent(long id) {
    ChartRow slot = rows.computeIfAbsent(id, k -> new ChartRow(Long.toString(k), rowIndex++));
    return slot;
  }

  /**
   * Accepts a {@link ChartVisitor}. Calls {@link ChartVisitor#visit(Chart)},
   * delegates the visitor to the rows of the chart and calls
   * {@link ChartVisitor#endVisit(Chart)}.
   *
   * @param visitor the visitor to accept
   */
  public void accept(ChartVisitor visitor) {
    visitor.visit(this);
    for (ChartRow slot : rows.values()) {
      slot.accept(visitor);
    }
    int rowCount = getRowCount();
    for (ChartColumn column : columns) {
      column.setRowCount(rowCount);
      column.accept(visitor);
    }
    for (ChartLine line : lines) {
      line.accept(visitor);
    }
    visitor.endVisit(this);
  }

  /**
   * Returns the {@link ChartBarType}s, sorted by name.
   */
  public List<ChartBarType> getSortedTypes() {
    List<ChartBarType> list = new ArrayList<>(types.values());
    Collections.sort(list);
    return list;
  }

  /**
   * Returns the {@link ChartRow}s, sorted by their index.
   */
  public List<ChartRow> getSortedRows() {
    List<ChartRow> list = new ArrayList<>(rows.values());
    Collections.sort(list);
    return list;
  }

  public int getRowCount() {
    return rows.size();
  }

  public long getMaxStop() {
    return maxStop;
  }
}
