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

import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo;

/**
 * Provides some common functions for {@link ChartCreator}s.
 */
public final class CommonChartCreator {

  static void createCommonChartItems(Chart chart, ProfileInfo info) {
    createTypes(chart);

    // add common info
    for (ProfilePhase phase : ProfilePhase.values()) {
      addColumn(chart,info,phase);
    }
  }

  private static void addColumn(Chart chart, ProfileInfo info, ProfilePhase phase) {
    ProfileInfo.Task task = info.getPhaseTask(phase);
    if (task != null) {
      String label = task.type.description + ": " + task.getDescription();
      ChartBarType type = chart.lookUpType(task.getDescription());
      long stop = task.startTime - info.getMinTaskStartTime() + info.getPhaseDuration(task);
      chart.addTimeRange(task.startTime - info.getMinTaskStartTime(), stop, type, label);
    }
  }

  /**
   * Creates the {@link ChartBarType}s and adds them to the chart.
   */
  private static void createTypes(Chart chart) {
    for (ProfilePhase phase : ProfilePhase.values()) {
      chart.createType(phase.description, new Color(phase.color, true));
    }
  }
}
