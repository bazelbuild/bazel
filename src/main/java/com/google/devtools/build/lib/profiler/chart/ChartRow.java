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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A row of a Gantt Chart. A chart row is identified by its id and has an index that
 * determines its location in the chart.
 */
public class ChartRow implements Comparable<ChartRow> {

  /** The unique id of this row. */
  private final String id;

  /** The index, i.e., the row number of the row in the chart. */
  private final int index;

  /** The list of bars in this row. */
  private final List<ChartBar> bars = new ArrayList<>();

  /**
   * Creates a chart row.
   *
   * @param id the unique id of this row
   * @param index the index, i.e., the row number, of the row in the chart
   */
  public ChartRow(String id, int index) {
    Preconditions.checkNotNull(id);
    this.id = id;
    this.index = index;
  }

  /**
   * Adds a bar to the chart row.
   *
   * @param bar the {@link ChartBar} to add
   */
  public void addBar(ChartBar bar) {
    bars.add(bar);
  }

  /**
   * Returns the bars of the row as an unmodifieable list.
   */
  public List<ChartBar> getBars() {
    return Collections.unmodifiableList(bars);
  }

  /**
   * Accepts a {@link ChartVisitor}. Calls {@link ChartVisitor#visit(ChartRow)}
   * and delegates the visitor to the bars of the chart row.
   *
   * @param visitor the visitor to accept
   */
  public void accept(ChartVisitor visitor) {
    visitor.visit(this);
    for (ChartBar bar : bars) {
      bar.accept(visitor);
    }
  }

  /**
   * {@inheritDoc}
   *
   * <p>Compares to rows by their index.
   */
  @Override
  public int compareTo(ChartRow other) {
    return index - other.index;
  }

  public int getIndex() {
    return index;
  }

  public String getId() {
    return id;
  }
}
