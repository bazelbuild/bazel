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

/**
 * Visitor for {@link Chart} objects.
 */
public interface ChartVisitor {

  /**
   * Visits a {@link Chart} object before its children, i.e., rows and bars, are visited.
   *
   * @param chart the {@link Chart} to visit
   */
  void visit(Chart chart);

  /**
   * Visits a {@link ChartRow} object.
   *
   * @param chartRow the {@link ChartRow} to visit
   */
  void visit(ChartRow chartRow);

  /**
   * Visits a {@link ChartBar} object.
   *
   * @param chartBar the {@link ChartBar} to visit
   */
  void visit(ChartBar chartBar);

  /**
   * Visits a {@link ChartColumn} object.
   *
   * @param chartColumn the {@link ChartColumn} to visit
   */
  void visit(ChartColumn chartColumn);

  /**
   * Visits a {@link ChartLine} object.
   *
   * @param chartLine the {@link ChartLine} to visit
   */
  void visit(ChartLine chartLine);

  /**
   * Visits a {@link Chart} object after its children, i.e., rows and bars, are visited.
   *
   * @param chart the {@link Chart} to visit
   */
  void endVisit(Chart chart);
}
