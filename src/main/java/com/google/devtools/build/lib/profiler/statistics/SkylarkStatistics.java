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

import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.Task;
import com.google.devtools.build.lib.util.LongArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Extracts the execution times of user-defined and built-in Skylark functions and computes
 * statistics.
 *
 * <p>The statistics are separated for the total duration taken for a function call and the
 * "self duration" taken only within the function itself, but not within any subtask of the
 * corresponding {@link Task}.
 */
public final class SkylarkStatistics {

  private final Map<String, LongArrayList> userFunctionDurations;
  private final Map<String, LongArrayList> userCompiledDurations;
  private final Map<String, LongArrayList> builtinFunctionDurations;

  /**
   * Self duration is the time taken just within a function itself, but not other subtasks of it.
   */
  private final Map<String, LongArrayList> userFunctionSelfDurations;
  private final Map<String, LongArrayList> userCompiledSelfDurations;
  private final Map<String, LongArrayList> builtinFunctionSelfDurations;
  private long userTotalNanos;
  private long userCompiledTotalNanos;
  private long builtinTotalNanos;

  public SkylarkStatistics() {
    userFunctionDurations = new HashMap<>();
    userCompiledDurations = new HashMap<>();
    builtinFunctionDurations = new HashMap<>();
    userFunctionSelfDurations = new HashMap<>();
    userCompiledSelfDurations = new HashMap<>();
    builtinFunctionSelfDurations = new HashMap<>();
  }

  public SkylarkStatistics(ProfileInfo info) {
    this();
    addProfileInfo(info);
  }

  /**
   * Adds Skylark function task durations from a {@link ProfileInfo} file.
   */
  public void addProfileInfo(ProfileInfo info) {
    computeStatistics(
        info.getSkylarkUserFunctionTasks(),
        info.getCompiledSkylarkUserFunctionTasks(),
        info.getSkylarkBuiltinFunctionTasks());
  }

  /**
   * @return the total time taken by all calls to built-in Skylark functions
   */
  public long getBuiltinTotalNanos() {
    return builtinTotalNanos;
  }

  /**
   * @return the total time taken by all calls to user-defined Skylark functions
   */
  public long getCompiledUserTotalNanos() {
    return userCompiledTotalNanos;
  }

  /**
   * @return the total time taken by all calls to user-defined Skylark functions
   */
  public long getUserTotalNanos() {
    return userTotalNanos;
  }

  /**
   * @return The execution durations of all calls to built-in Skylark functions.
   */
  public Map<String, LongArrayList> getBuiltinFunctionDurations() {
    return builtinFunctionDurations;
  }

  /**
   * return The execution durations of all calls to built-in functions excluding the durations of
   * all subtasks.
   */
  public Map<String, LongArrayList> getBuiltinFunctionSelfDurations() {
    return builtinFunctionSelfDurations;
  }

  /**
   * Builds and returns the {@link TasksStatistics} for the durations of each built-in function.
   * The return value is not cached and will be recomputed on another call.
   */
  public Map<String, TasksStatistics> getBuiltinFunctionStatistics() {
    return buildTasksStatistics(builtinFunctionDurations);
  }

  /**
   * Builds and returns the {@link TasksStatistics} for the self-times of each built-in function.
   * The return value is not cached and will be recomputed on another call.
   */
  public Map<String, TasksStatistics> getBuiltinFunctionSelfStatistics() {
    return buildTasksStatistics(builtinFunctionSelfDurations);
  }

  /**
   * @return The execution durations of all calls to user-defined Skylark functions.
   */
  public Map<String, LongArrayList> getCompiledUserFunctionDurations() {
    return userCompiledDurations;
  }

  /**
   * return The execution durations of all calls to user-defined functions excluding the durations
   * of all subtasks.
   */
  public Map<String, LongArrayList> getCompiledUserFunctionSelfDurations() {
    return userCompiledSelfDurations;
  }

  /**
   * Builds and returns the {@link TasksStatistics} for the durations of each user-defined
   * function. The return value is not cached and will be recomputed on another call.
   */
  public Map<String, TasksStatistics> getCompiledUserFunctionStatistics() {
    return buildTasksStatistics(userCompiledDurations);
  }

  /**
   * Builds and returns the {@link TasksStatistics} for the self-times of each user-defined
   * function. The return value is not cached and will be recomputed on another call.
   */
  public Map<String, TasksStatistics> getCompiledUserFunctionSelfStatistics() {
    return buildTasksStatistics(userCompiledSelfDurations);
  }

  /**
   * @return The execution durations of all calls to user-defined Skylark functions.
   */
  public Map<String, LongArrayList> getUserFunctionDurations() {
    return userFunctionDurations;
  }

  /**
   * return The execution durations of all calls to user-defined functions excluding the durations
   * of all subtasks.
   */
  public Map<String, LongArrayList> getUserFunctionSelfDurations() {
    return userFunctionSelfDurations;
  }

  /**
   * Builds and returns the {@link TasksStatistics} for the durations of each user-defined
   * function. The return value is not cached and will be recomputed on another call.
   */
  public Map<String, TasksStatistics> getUserFunctionStatistics() {
    return buildTasksStatistics(userFunctionDurations);
  }

  /**
   * Builds and returns the {@link TasksStatistics} for the self-times of each user-defined
   * function. The return value is not cached and will be recomputed on another call.
   */
  public Map<String, TasksStatistics> getUserFunctionSelfStatistics() {
    return buildTasksStatistics(userFunctionSelfDurations);
  }

  /**
   * For each Skylark function get the list of durations and self durations from the task maps.
   */
  private void computeStatistics(
      Multimap<String, Task> userTasks,
      Multimap<String, Task> userCompiledTasks,
      Multimap<String, Task> builtinTasks) {
    userTotalNanos += addDurations(userTasks, userFunctionDurations, userFunctionSelfDurations);
    userCompiledTotalNanos +=
        addDurations(userCompiledTasks, userCompiledDurations, userCompiledSelfDurations);
    builtinTotalNanos +=
        addDurations(builtinTasks, builtinFunctionDurations, builtinFunctionSelfDurations);
  }

  /**
   * Add all new durations to previously collected durations for all functions mapped to tasks.
   * @return The sum of the execution times of all {@link Task} values in the map.
   */
  private static long addDurations(
      Multimap<String, Task> functionTasks,
      Map<String, LongArrayList> durationsMap,
      Map<String, LongArrayList> selfDurationsMap) {
    long totalTime = 0;
    for (Map.Entry<String, Collection<Task>> entry : functionTasks.asMap().entrySet()) {
      String function = entry.getKey();
      Collection<Task> tasks = entry.getValue();
      LongArrayList durations;
      LongArrayList selfDurations;
      if (durationsMap.containsKey(function)) {
        durations = durationsMap.get(function);
        selfDurations = selfDurationsMap.get(function);
      } else {
        durations = new LongArrayList(tasks.size());
        selfDurations = new LongArrayList(tasks.size());
        durationsMap.put(function, durations);
        selfDurationsMap.put(function, selfDurations);
      }
      totalTime += addDurations(tasks, durations, selfDurations);
    }
    return totalTime;
  }

  /**
   * Add all durations and self-times of the given function to the maps.
   * @return The sum of the execution times of all {@link Task} values in the collection.
   */
  private static long addDurations(
      Collection<Task> tasks, LongArrayList durations, LongArrayList selfDurations) {
    long totalTime = 0;
    durations.ensureCapacity(durations.size() + tasks.size());
    selfDurations.ensureCapacity(selfDurations.size() + tasks.size());
    for (Task task : tasks) {
      durations.add(task.durationNanos);
      selfDurations.add(task.durationNanos - task.getInheritedDuration());
      totalTime += task.durationNanos;
    }
    return totalTime;
  }

  /**
   * Build a Map of {@link TasksStatistics} from the given duration maps.
   */
  private static Map<String, TasksStatistics> buildTasksStatistics(
      final Map<String, LongArrayList> durationsMap) {
    return Maps.transformEntries(durationsMap, TasksStatistics::create);
  }
}
