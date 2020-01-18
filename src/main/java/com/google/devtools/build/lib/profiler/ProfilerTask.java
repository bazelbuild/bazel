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
 * All possible types of profiler tasks. Each type also defines description and
 * minimum duration in nanoseconds for it to be recorded as separate event and
 * not just be aggregated into the parent event.
 */
public enum ProfilerTask {
  /* WARNING:
   * Add new Tasks at the end (before Unknown) to not break the profiles that people have created!
   * The profile file format uses the ordinal() of this enumeration to identify the task.
   */
  PHASE("build phase marker"),
  ACTION("action processing"),
  __ACTION_BUILDER("parallel builder completion queue"), // unused
  __ACTION_SUBMIT("execution queue submission"), // unused
  ACTION_CHECK("action dependency checking", 10000000, 0),
  __ACTION_EXECUTE("action execution"), // unused
  ACTION_LOCK("action resource lock", 10000000, 0),
  ACTION_RELEASE("action resource release", 10000000, 0),
  __ACTION_GRAPH("action graph dependency"), // unused
  ACTION_UPDATE("update action information", 10000000, 0),
  ACTION_COMPLETE("complete action execution"),
  INFO("general information"),
  __EXCEPTION("exception"), // unused
  CREATE_PACKAGE("package creation"),
  __PACKAGE_VALIDITY_CHECK("package validity check"), // unused
  __SPAWN("local process spawn"), // unused
  REMOTE_EXECUTION("remote action execution"),
  LOCAL_EXECUTION("local action execution"),
  SCANNER("include scanner"),
  // 30 is a good number because the slowest items are stored in a heap, with temporarily
  // one more element, and with 31 items, a heap becomes a complete binary tree
  LOCAL_PARSE("Local parse to prepare for remote execution", 50000000, 30),
  UPLOAD_TIME("Remote execution upload time", 50000000, 0),
  PROCESS_TIME("Remote execution process wall time", 50000000, 0),
  REMOTE_QUEUE("Remote execution queuing time", 50000000, 0),
  REMOTE_SETUP("Remote execution setup", 50000000, 0),
  FETCH("Remote execution file fetching", 50000000, 0),
  VFS_STAT("VFS stat", 10000000, 30),
  VFS_DIR("VFS readdir", 10000000, 30),
  VFS_READLINK("VFS readlink", 10000000, 30),
  // TODO(olaola): rename to VFS_DIGEST. This refers to all digest function computations.
  VFS_MD5("VFS md5", 10000000, 30),
  VFS_XATTR("VFS xattr", 10000000, 30),
  VFS_DELETE("VFS delete", 10000000, 0),
  VFS_OPEN("VFS open", 10000000, 30),
  VFS_READ("VFS read", 10000000, 30),
  VFS_WRITE("VFS write", 10000000, 30),
  VFS_GLOB("globbing", -1, 30),
  VFS_VMFS_STAT("VMFS stat", 10000000, 0),
  VFS_VMFS_DIR("VMFS readdir", 10000000, 0),
  VFS_VMFS_READ("VMFS read", 10000000, 0),
  WAIT("thread wait", 5000000, 0),
  __CONFIGURED_TARGET("configured target creation"), // unused
  THREAD_NAME("thread name"), // Do not use directly!
  __TEST("for testing only"), // unused
  SKYFRAME_EVAL("skyframe evaluator"),
  SKYFUNCTION("skyfunction"),
  CRITICAL_PATH("critical path"),
  CRITICAL_PATH_COMPONENT("critical path component"),
  HANDLE_GC_NOTIFICATION("gc notification"),
  LOCAL_CPU_USAGE("cpu counters"),
  ACTION_COUNTS("action counters"),
  __PROCESS_SCAN("process scan"), // unused
  __LOOP_OUTPUT_ARTIFACTS("loop output artifacts"), // unused
  __LOCATE_RELATIVE("locate relative"), // unused
  __CONSTRUCT_INCLUDE_PATHS("construct include paths"), // unused
  __PARSE_AND_HINTS_RESULTS("parse and hints results"), // unused
  __PROCESS_RESULTS_AND_ENQUEUE("process results and enqueue"), // unused
  STARLARK_PARSER("Starlark Parser"),
  STARLARK_USER_FN("Starlark user function call"),
  STARLARK_BUILTIN_FN("Starlark builtin function call"),
  STARLARK_USER_COMPILED_FN("Starlark compiled user function call"),
  ACTION_FS_STAGING("Staging per-action file system"),
  REMOTE_CACHE_CHECK("remote action cache check"),
  REMOTE_DOWNLOAD("remote output download"),
  REMOTE_NETWORK("remote network"),
  UNKNOWN("Unknown event");

  // Size of the ProfilerTask value space.
  public static final int TASK_COUNT = ProfilerTask.values().length;

  /** Human readable description for the task. */
  public final String description;
  /**
   * Threshold for skipping tasks in the profile in nanoseconds, unless --record_full_profiler_data
   * is used.
   */
  public final long minDuration;
  /** How many of the slowest instances to keep. If 0, no slowest instance calculation is done. */
  public final int slowestInstancesCount;
  /** True if the metric records VFS operations */
  private final boolean vfs;

  private ProfilerTask(String description, long minDuration, int slowestInstanceCount) {
    this.description = description;
    this.minDuration = minDuration;
    this.slowestInstancesCount = slowestInstanceCount;
    this.vfs = this.name().startsWith("VFS");
  }

  private ProfilerTask(String description) {
    this(description, /* minDuration= */ -1, /* slowestInstanceCount= */ 0);
  }

  /** Whether the Profiler collects the slowest instances of this task. */
  public boolean collectsSlowestInstances() {
    return slowestInstancesCount > 0;
  }

  public boolean isVfs() {
    return vfs;
  }

  public boolean isStarlark() {
    return description.startsWith("Starlark ");
  }
}
