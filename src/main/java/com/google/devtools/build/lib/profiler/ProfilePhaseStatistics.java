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

package com.google.devtools.build.lib.profiler;

/**
 * Hold pre-formatted statistics of a profiled execution phase.
 *
 * TODO(bazel-team): Change String statistics into StatisticsTable[], where StatisticsTable is an
 * Object with a title (can be null), header[columns] (can be null), data[rows][columns],
 * alignment[columns] (left/right).
 * The HtmlChartsVisitor can turn that into HTML tables, the text formatter can calculate the max
 * for each column and format the text accordingly.
 */
public class ProfilePhaseStatistics {
  private final String title;
  private final String statistics;

  public ProfilePhaseStatistics (String title, String statistics) {
    this.title = title;
    this.statistics = statistics;
  }

  public String getTitle(){
    return title;
  }

  public String getStatistics(){
    return statistics;
  }
}
