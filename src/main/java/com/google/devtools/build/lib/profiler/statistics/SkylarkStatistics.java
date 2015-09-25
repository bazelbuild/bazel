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

import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimaps;
import com.google.devtools.build.lib.profiler.ProfileInfo;
import com.google.devtools.build.lib.profiler.ProfileInfo.Task;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

/**
 * Computes various statistics for Skylark and built-in function usage and prints it to a given
 * {@link PrintStream}.
 */
public final class SkylarkStatistics {

  private final ListMultimap<String, Task> userFunctionTasks;
  private final ListMultimap<String, Task> builtinFunctionTasks;
  private final List<TasksStatistics> userFunctionStats;
  private final List<TasksStatistics> builtinFunctionStats;
  private long userTotalNanos;
  private long builtinTotalNanos;

  public SkylarkStatistics(ProfileInfo info) {
    userFunctionTasks = info.getSkylarkUserFunctionTasks();
    builtinFunctionTasks = info.getSkylarkBuiltinFunctionTasks();
    userFunctionStats = new ArrayList<>();
    builtinFunctionStats = new ArrayList<>();
    computeStatistics();
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
  public long getUserTotalNanos() {
    return userTotalNanos;
  }

  /**
   * @return a map from user-defined function descriptions of the form file:line#function to all
   *    corresponding {@link com.google.devtools.build.lib.profiler.ProfileInfo.Task}s.
   */
  public ListMultimap<String, Task> getUserFunctionTasks() {
    return userFunctionTasks;
  }

  /**
   * @return a map from built-in function descriptions of the form package.class#method to all
   *    corresponding {@link com.google.devtools.build.lib.profiler.ProfileInfo.Task}s.
   */
  public ListMultimap<String, Task> getBuiltinFunctionTasks() {
    return builtinFunctionTasks;
  }

  public List<TasksStatistics> getBuiltinFunctionStats() {
    return builtinFunctionStats;
  }

  public List<TasksStatistics> getUserFunctionStats() {
    return userFunctionStats;
  }

  /**
   * For each Skylark function compute a {@link TasksStatistics} object from the execution times of
   * all corresponding {@link Task}s from either {@link #userFunctionTasks} or
   * {@link #builtinFunctionTasks}. Fills fields {@link #userFunctionStats} and
   * {@link #builtinFunctionStats}.
   */
  private void computeStatistics() {
    userTotalNanos = computeStatistics(userFunctionTasks, userFunctionStats);
    builtinTotalNanos = computeStatistics(builtinFunctionTasks, builtinFunctionStats);
  }

  /**
   * For each Skylark function compute a {@link TasksStatistics} object from the execution times of
   * all corresponding {@link Task}s and add it to the list.
   * @param tasks Map from function name to all corresponding tasks.
   * @param stats The list to which {@link TasksStatistics} are to be added.
   * @return The sum of the execution times of all {@link Task} values in the map.
   */
  private static long computeStatistics(
      ListMultimap<String, Task> tasks, List<TasksStatistics> stats) {
    long total = 0L;
    for (Entry<String, List<Task>> entry : Multimaps.asMap(tasks).entrySet()) {
      TasksStatistics functionStats = TasksStatistics.create(entry.getKey(), entry.getValue());
      stats.add(functionStats);
      total += functionStats.totalNanos;
    }
    return total;
  }
}
