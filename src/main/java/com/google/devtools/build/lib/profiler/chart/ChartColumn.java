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

import com.google.common.base.Preconditions;

/**
 * A chart column. The column can be used to highlight a time-range.
 */
public class ChartColumn {

  /**
   * The start value of the bar. This value has no unit. The interpretation of
   * the value is up to the user of the class.
   */
  private final long start;

  /**
   * The stop value of the bar. This value has no unit. The interpretation of
   * the value is up to the user of the class.
   */
  private final long stop;

  /** The type of the bar. */
  private final ChartBarType type;

  /** The label of the bar. */
  private final String label;

  private int rowCount;

  /**
   * Creates a chart column.
   *
   * @param start the start value of the bar
   * @param stop the stop value of the bar
   * @param type the type of the bar
   * @param label the label of the bar
   */
  public ChartColumn(long start, long stop, ChartBarType type, String label) {
    Preconditions.checkNotNull(type);
    Preconditions.checkNotNull(label);
    this.start = start;
    this.stop = stop;
    this.type = type;
    this.label = label;
  }

  /**
   * Accepts a {@link ChartVisitor}. Calls {@link ChartVisitor#visit(ChartBar)}.
   *
   * @param visitor the visitor to accept
   */
  public void accept(ChartVisitor visitor) {
    visitor.visit(this);
  }

  public long getStart() {
    return start;
  }

  public long getWidth() {
    return stop - start;
  }

  public ChartBarType getType() {
    return type;
  }

  public String getLabel() {
    return label;
  }

  public int getRowCount() {
    return rowCount;
  }

  public void setRowCount(int rowCount) {
    this.rowCount = rowCount;
  }
}
