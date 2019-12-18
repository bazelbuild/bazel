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

import com.google.common.base.Predicate;
import java.util.EnumSet;

/**
 * All possible types of profiler tasks. Each type also defines description and
 * minimum duration in nanoseconds for it to be recorded as separate event and
 * not just be aggregated into the parent event.
 */
public enum ProfilerTask {
  /* WARNING:
   * Add new Tasks at the end (before Unknown) to not break the profiles that people have created!
   * The profile file format uses the ordinal() of this enumeration to identify the task.
   */
  PHASE("build phase marker", 0x336699),
  ACTION("action processing", 0x666699),
  __ACTION_BUILDER("parallel builder completion queue", 0xCC3399), // unused
  __ACTION_SUBMIT("execution queue submission", 0xCC3399), // unused
  ACTION_CHECK("action dependency checking", 10000000, 0x999933, 0),
  __ACTION_EXECUTE("action execution", 0x99CCFF), // unused
  ACTION_LOCK("action resource lock", 10000000, 0xCC9933, 0),
  ACTION_RELEASE("action resource release", 10000000, 0x006666, 0),
  __ACTION_GRAPH("action graph dependency", 0x3399FF), // unused
  ACTION_UPDATE("update action information", 10000000, 0x993300, 0),
  ACTION_COMPLETE("complete action execution", 0xCCCC99),
  INFO("general information", 0x000066),
  __EXCEPTION("exception", 0xFFCC66), // unused
  CREATE_PACKAGE("package creation", 0x6699CC),
  __PACKAGE_VALIDITY_CHECK("package validity check", 0x336699), // unused
  __SPAWN("local process spawn", 0x663366), // unused
  REMOTE_EXECUTION("remote action execution", 0x9999CC),
  LOCAL_EXECUTION("local action execution", 0xCCCCCC),
  SCANNER("include scanner", 0x669999),
  // 30 is a good number because the slowest items are stored in a heap, with temporarily
  // one more element, and with 31 items, a heap becomes a complete binary tree
  LOCAL_PARSE("Local parse to prepare for remote execution", 50000000, 0x6699CC, 30),
  UPLOAD_TIME("Remote execution upload time", 50000000, 0x6699CC, 0),
  PROCESS_TIME("Remote execution process wall time", 50000000, 0xF999CC, 0),
  REMOTE_QUEUE("Remote execution queuing time", 50000000, 0xCC6600, 0),
  REMOTE_SETUP("Remote execution setup", 50000000, 0xA999CC, 0),
  FETCH("Remote execution file fetching", 50000000, 0xBB99CC, 0),
  VFS_STAT("VFS stat", 10000000, 0x9999FF, 30),
  VFS_DIR("VFS readdir", 10000000, 0x0066CC, 30),
  VFS_READLINK("VFS readlink", 10000000, 0x99CCCC, 30),
  // TODO(olaola): rename to VFS_DIGEST. This refers to all digest function computations.
  VFS_MD5("VFS md5", 10000000, 0x999999, 30),
  VFS_XATTR("VFS xattr", 10000000, 0x9999DD, 30),
  VFS_DELETE("VFS delete", 10000000, 0xFFCC00, 0),
  VFS_OPEN("VFS open", 10000000, 0x009999, 30),
  VFS_READ("VFS read", 10000000, 0x99CC33, 30),
  VFS_WRITE("VFS write", 10000000, 0xFF9900, 30),
  VFS_GLOB("globbing", -1, 0x999966, 30),
  VFS_VMFS_STAT("VMFS stat", 10000000, 0x9999FF, 0),
  VFS_VMFS_DIR("VMFS readdir", 10000000, 0x0066CC, 0),
  VFS_VMFS_READ("VMFS read", 10000000, 0x99CC33, 0),
  WAIT("thread wait", 5000000, 0x66CCCC, 0),
  __CONFIGURED_TARGET("configured target creation", 0x663300), // unused
  THREAD_NAME("thread name", 0x996600), // Do not use directly!
  __TEST("for testing only", 0x000000), // unused
  SKYFRAME_EVAL("skyframe evaluator", 0xCC9900),
  SKYFUNCTION("skyfunction", 0xCC6600),
  CRITICAL_PATH("critical path", 0x666699),
  CRITICAL_PATH_COMPONENT("critical path component", 0x666699),
  HANDLE_GC_NOTIFICATION("gc notification", 0x996633),
  LOCAL_CPU_USAGE("cpu counters", 0x000000),
  ACTION_COUNTS("action counters", 0x000000),
  __PROCESS_SCAN("process scan", 0x000000), // unused
  __LOOP_OUTPUT_ARTIFACTS("loop output artifacts"), // unused
  __LOCATE_RELATIVE("locate relative"), // unused
  __CONSTRUCT_INCLUDE_PATHS("construct include paths"), // unused
  __PARSE_AND_HINTS_RESULTS("parse and hints results"), // unused
  __PROCESS_RESULTS_AND_ENQUEUE("process results and enqueue"), // unused
  STARLARK_PARSER("Starlark Parser"),
  STARLARK_USER_FN("Starlark user function call", -0xCC0033),
  STARLARK_BUILTIN_FN("Starlark builtin function call", 0x990033),
  STARLARK_USER_COMPILED_FN("Starlark compiled user function call", 0xCC0033),
  ACTION_FS_STAGING("Staging per-action file system", 0x000000),
  REMOTE_CACHE_CHECK("remote action cache check", 0x9999CC),
  REMOTE_DOWNLOAD("remote output download", 0x9999CC),
  REMOTE_NETWORK("remote network", 0x9999CC),
  UNKNOWN("Unknown event",  0x339966);

  // Size of the ProfilerTask value space.
  public static final int TASK_COUNT = ProfilerTask.values().length;

  /** Human readable description for the task. */
  public final String description;
  /**
   * Threshold for skipping tasks in the profile in nanoseconds, unless --record_full_profiler_data
   * is used.
   */
  public final long minDuration;
  /** Default color of the task, when rendered in a chart. */
  public final int color;
  /** How many of the slowest instances to keep. If 0, no slowest instance calculation is done. */
  public final int slowestInstancesCount;
  /** True if the metric records VFS operations */
  private final boolean vfs;

  private ProfilerTask(String description, long minDuration, int color, int slowestInstanceCount) {
    this.description = description;
    this.minDuration = minDuration;
    this.color = color;
    this.slowestInstancesCount = slowestInstanceCount;
    this.vfs = this.name().startsWith("VFS");
  }

  private ProfilerTask(String description, int color) {
    this(description, /* minDuration= */ -1, color, /* slowestInstanceCount= */ 0);
  }

  private ProfilerTask(String description) {
    this(description, /* minDuration= */ -1, /* color= */ 0x000000, /* slowestInstanceCount= */ 0);
  }

  /** Whether the Profiler collects the slowest instances of this task. */
  public boolean collectsSlowestInstances() {
    return slowestInstancesCount > 0;
  }

  /**
   * Build a set containing all ProfilerTasks for which the given predicate is true.
   */
  public static EnumSet<ProfilerTask> allSatisfying(Predicate<ProfilerTask> predicate) {
    EnumSet<ProfilerTask> set = EnumSet.noneOf(ProfilerTask.class);
    for (ProfilerTask taskType : values()) {
      if (predicate.apply(taskType)) {
        set.add(taskType);
      }
    }
    return set;
  }

  public boolean isVfs() {
    return vfs;
  }

  public boolean isStarlark() {
    return description.startsWith("Starlark ");
  }
}
