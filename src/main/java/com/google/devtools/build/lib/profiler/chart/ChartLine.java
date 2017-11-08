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
 * A chart line. Such lines can be used to connect boxes.
 */
public class ChartLine {
  private final ChartRow startRow;
  private final ChartRow stopRow;
  private final long startTime;
  /**
   * Creates a chart line.
   *
   * @param startRow the start row
   * @param stopRow the end row
   * @param startTime the start time
   * @param stopTime the end time
   */
  public ChartLine(ChartRow startRow, ChartRow stopRow, long startTime, long stopTime) {
    Preconditions.checkNotNull(startRow);
    Preconditions.checkNotNull(stopRow);
    this.startRow = startRow;
    this.stopRow = stopRow;
    this.startTime = startTime;
  }

  /**
   * Accepts a {@link ChartVisitor}. Calls {@link ChartVisitor#visit(ChartBar)}.
   *
   * @param visitor the visitor to accept
   */
  public void accept(ChartVisitor visitor) {
    visitor.visit(this);
  }

  public ChartRow getStartRow() {
    return startRow;
  }

  public ChartRow getStopRow() {
    return stopRow;
  }

  public long getStartTime() {
    return startTime;
  }
}
