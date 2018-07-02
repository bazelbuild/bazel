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

import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.CriticalPathEntry;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.Task;
import java.util.EnumSet;

/**
 * Implementation of {@link ChartCreator} that creates Gantt Charts that contain
 * bars for all tasks in the profile.
 */
public class DetailedChartCreator implements ChartCreator {

  /** The data of the profiled build. */
  private final ProfileInfo info;

  /**
   * Creates the chart creator.
   *
   * @param info the data of the profiled build
   */
  public DetailedChartCreator(ProfileInfo info) {
    this.info = info;
  }

  @Override
  public Chart create() {
    Chart chart = new Chart();
    CommonChartCreator.createCommonChartItems(chart, info);
    createTypes(chart);

    // calculate the critical path
    EnumSet<ProfilerTask> typeFilter = EnumSet.noneOf(ProfilerTask.class);
    CriticalPathEntry criticalPath = info.getCriticalPath(typeFilter);
    info.analyzeCriticalPath(typeFilter, criticalPath);

    for (Task task : info.allTasksById) {
      String label = task.type.description + ": " + task.getDescription();
      ChartBarType type = chart.lookUpType(task.type.description);
      long stop = task.startTime - info.getMinTaskStartTime() + task.durationNanos;
      CriticalPathEntry entry = null;

      // for top level tasks, check if they are on the critical path
      if (task.parentId == 0 && criticalPath != null) {
        entry = info.getNextCriticalPathEntryForTask(criticalPath, task);
        // find next top-level entry
        if (entry != null) {
          CriticalPathEntry nextEntry = entry.next;
          while (nextEntry != null && nextEntry.task.parentId != 0) {
            nextEntry = nextEntry.next;
          }
          if (nextEntry != null) {
            // time is start and not stop as we traverse the critical back backwards
            chart.addVerticalLine(task.threadId, nextEntry.task.threadId,
                task.startTime - info.getMinTaskStartTime());
          }
        }
      }

      chart.addBar(task.threadId, task.startTime - info.getMinTaskStartTime(), stop, type,
          entry != null, label);
    }

    return chart;
  }

  /**
   * Creates a {@link ChartBarType} for every known {@link ProfilerTask} and
   * adds it to the chart.
   *
   * @param chart the chart to add the types to
   */
  private void createTypes(Chart chart) {
    for (ProfilerTask task : ProfilerTask.values()) {
      chart.createType(task.description, new Color(task.color));
    }
  }
}
