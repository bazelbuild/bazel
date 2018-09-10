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
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.Task;
import java.util.EnumSet;
import java.util.Set;

/**
 * Implementation of {@link ChartCreator} that creates Gantt Charts that try to
 * minimize the number of bars while preserving as much information about the
 * execution of actions as possible.
 *
 * <p>Profiler tasks are categorized into four categories:
 * <ul>
 * <li>Actions: Actions executed.
 * <li>Blaze Internal: This category contains internal blaze tasks, like loading
 * packages, saving the action cache etc.
 * <li>Locks: Contains tasks that indicate that a thread is waiting for
 * resources.
 * <li>VFS: Contains tasks that access the file system.
 * </ul>
 */
public class AggregatingChartCreator implements ChartCreator {

  /** The tasks in the 'actions' category. */
  private static final Set<ProfilerTask> ACTION_TASKS = EnumSet.of(ProfilerTask.ACTION);

  /** The tasks in the 'blaze internal' category. */
  private static final Set<ProfilerTask> BLAZE_TASKS =
      EnumSet.of(
          ProfilerTask.CREATE_PACKAGE,
          ProfilerTask.INFO,
          ProfilerTask.UNKNOWN);

  /** The tasks in the 'locks' category. */
  private static final Set<ProfilerTask> LOCK_TASKS =
      EnumSet.of(ProfilerTask.ACTION_LOCK, ProfilerTask.WAIT);

  /** The tasks in the 'VFS' category. */
  private static final Set<ProfilerTask> VFS_TASKS =
      EnumSet.of(
          ProfilerTask.VFS_STAT,
          ProfilerTask.VFS_DIR,
          ProfilerTask.VFS_READLINK,
          ProfilerTask.VFS_MD5,
          ProfilerTask.VFS_DELETE,
          ProfilerTask.VFS_OPEN,
          ProfilerTask.VFS_READ,
          ProfilerTask.VFS_WRITE,
          ProfilerTask.VFS_GLOB,
          ProfilerTask.VFS_XATTR);

  /** The data of the profiled build. */
  private final ProfileInfo info;

  /** If true, VFS related information is added to the chart. */
  private final boolean showVFS;

  /** The type for bars of category 'blaze internal'. */
  private ChartBarType blazeType;

  /** The type for bars of category 'actions'. */
  private ChartBarType actionType;

  /** The type for bars of category 'locks'. */
  private ChartBarType lockType;

  /** The type for bars of category 'VFS'. */
  private ChartBarType vfsType;

  /**
   * Creates the chart creator. The created {@link ChartCreator} does not add
   * VFS related data to the generated chart.
   *
   * @param info the data of the profiled build
   */
  public AggregatingChartCreator(ProfileInfo info) {
    this(info, false);
  }

  /**
   * Creates the chart creator.
   *
   * @param info the data of the profiled build
   * @param showVFS if true, VFS related information is added to the chart
   */
  public AggregatingChartCreator(ProfileInfo info, boolean showVFS) {
    this.info = info;
    this.showVFS = showVFS;
  }

  @Override
  public Chart create() {
    Chart chart = new Chart();
    CommonChartCreator.createCommonChartItems(chart, info);
    createTypes(chart);

    for (ProfileInfo.Task task : info.allTasksById) {
      if (ACTION_TASKS.contains(task.type)) {
        createBar(chart, info.getMinTaskStartTime(), task, actionType);
      } else if (LOCK_TASKS.contains(task.type)) {
        createBar(chart, info.getMinTaskStartTime(), task, lockType);
      } else if (BLAZE_TASKS.contains(task.type)) {
        createBar(chart, info.getMinTaskStartTime(), task, blazeType);
      } else if (showVFS && VFS_TASKS.contains(task.type)) {
        createBar(chart, info.getMinTaskStartTime(), task, vfsType);
      }
    }

    return chart;
  }

  /**
   * Creates a bar and adds it to the chart.
   *
   * @param chart the chart to add the types to
   * @param task the profiler task from which the bar is created
   * @param type the type of the bar
   */
  private void createBar(Chart chart, long minTaskStartTime, Task task, ChartBarType type) {
    String label = task.type.description + ": " + task.getDescription();
    chart.addBar(task.threadId,
        task.startTime - minTaskStartTime,
        task.startTime - minTaskStartTime + task.durationNanos, type, label);
  }

  /**
   * Creates the {@link ChartBarType}s and adds them to the chart.
   *
   * @param chart the chart to add the types to
   */
  private void createTypes(Chart chart) {
    actionType = chart.createType("Action processing", new Color(0x000099));
    blazeType = chart.createType("Bazel internal processing", new Color(0x999999));
    lockType = chart.createType("Waiting for resources", new Color(0x990000));
    if (showVFS) {
      vfsType = chart.createType("File system access", new Color(0x009900));
    }
  }
}
