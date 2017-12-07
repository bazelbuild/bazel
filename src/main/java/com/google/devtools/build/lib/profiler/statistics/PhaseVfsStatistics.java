// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler.statistics;

import com.google.common.collect.ComparisonChain;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;
import com.google.common.collect.Tables;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.Task;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SortedSet;

/**
 * Compute and store statistics of all {@link ProfilerTask}s that begin with VFS_ in sorted order.
 */
public final class PhaseVfsStatistics implements Iterable<ProfilerTask> {

  /**
   * Duration, count and path for sorting by duration first and count in case of tie. Path for
   * easy returning of a {@link SortedSet}.
   */
  public static final class Stat implements Comparable<Stat> {
    private long duration;
    private long count;
    public final String path;

    public Stat(String path) {
      this.path = path;
    }

    public Stat(Stat other) {
      this.duration = other.duration;
      this.count = other.count;
      this.path = other.path;
    }

    public long getDuration() {
      return duration;
    }

    public long getCount() {
      return count;
    }

    private void add(Stat other) {
      this.duration += other.duration;
      this.count += other.count;
    }

    private void add(long duration) {
      this.duration += duration;
      this.count++;
    }

    /**
     * Order first by duration, then count, then path
     */
    @Override
    public int compareTo(Stat o) {
      return ComparisonChain.start()
          .compare(duration, o.duration)
          .compare(count, o.count)
          .compare(path, o.path)
          .result();
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof Stat) {
        return compareTo((Stat) obj) == 0;
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(duration, count, path);
    }
  }

  private final ProfilePhase phase;
  private final Table<ProfilerTask, String, Stat> statistics;

  public PhaseVfsStatistics(ProfilePhase phase) {
    this.phase = phase;
    this.statistics =
        Tables.newCustomTable(
            new EnumMap<ProfilerTask, Map<String, Stat>>(ProfilerTask.class), HashMap::new);
  }

  public PhaseVfsStatistics(final String workSpaceName, ProfilePhase phase, ProfileInfo info) {
    this(phase);
    addProfileInfo(workSpaceName, info);
  }

  /**
   * Accumulate statistics from another {@link ProfileInfo} in this object.
   */
  public void addProfileInfo(final String workSpaceName, ProfileInfo info) {
    Task phaseTask = info.getPhaseTask(phase);
    if (phaseTask == null) {
      return;
    }
    collectVfsEntries(workSpaceName, info.getTasksForPhase(phaseTask));
  }

  public ProfilePhase getProfilePhase() {
    return phase;
  }

  public boolean isEmpty() {
    return statistics.isEmpty();
  }

  /**
   * Builds a new {@link ImmutableSortedSet} of the path statistics for the given
   * {@link ProfilerTask}.
   *
   * <p>{@link Stat}s are sorted by their natural order.
   */
  public ImmutableSortedSet<Stat> getSortedStatistics(ProfilerTask taskType) {
    return ImmutableSortedSet.copyOf(statistics.row(taskType).values());
  }

  public int getStatisticsCount(ProfilerTask taskType) {
    return statistics.row(taskType).size();
  }

  @Override
  public Iterator<ProfilerTask> iterator() {
    return statistics.rowKeySet().iterator();
  }

  /**
   * Add statistics from another PhaseVfsStatistics aggregation to this one.
   */
  public void add(PhaseVfsStatistics other) {
    for (Cell<ProfilerTask, String, Stat> cell : other.statistics.cellSet()) {
      Stat stat = statistics.get(cell.getRowKey(), cell.getColumnKey());
      if (stat == null) {
        stat = new Stat(cell.getValue());
        statistics.put(cell.getRowKey(), stat.path, stat);
      } else {
        stat.add(cell.getValue());
      }
    }
  }

  /**
   * Add the VFS operations from the list of tasks to the {@link #statistics} table
   */
  private void collectVfsEntries(String workSpaceName, List<Task> taskList) {
    for (Task task : taskList) {
      collectVfsEntries(workSpaceName, Arrays.asList(task.subtasks));
      if (!task.type.name().startsWith("VFS_")) {
        continue;
      }

      String path = pathMapping(workSpaceName, task.getDescription());

      Stat stat = statistics.get(task.type, path);
      if (stat == null) {
        stat = new Stat(path);
        statistics.put(task.type, path, stat);
      }

      stat.add(task.durationNanos);
    }
  }

  private String pathMapping(String workSpaceName, String input) {
    if (workSpaceName.isEmpty()) {
      return input;
    } else {
      return input.substring(input.lastIndexOf("/" + workSpaceName) + 1);
    }
  }
}

