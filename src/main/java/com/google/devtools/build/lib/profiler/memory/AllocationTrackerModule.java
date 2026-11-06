// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.profiler.memory;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlazeService;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.UUID;
import javax.annotation.Nullable;
import net.starlark.java.eval.Debug;

/**
 * A {@link BlazeModule} that can be used to record interesting information about all allocations
 * done during every command on the current blaze server.
 *
 * <p>To enable tracking, you must pass:
 *
 * <ol>
 *   <li>--host_jvm_args=-javaagent:(path to Google's java agent jar)
 *       <ul>
 *         <li>For Bazel use <a
 *             href="https://github.com/bazelbuild/bazel/tree/master/third_party/allocation_instrumenter">java-allocation-instrumenter-3.3.4.jar</a>
 *       </ul>
 *   <li>--host_jvm_args=-DRULE_MEMORY_TRACKER=1
 * </ol>
 *
 * <p>The memory tracking information is accessible via blaze dump --rules and blaze dump
 * --skylark_memory=(path)
 */
public class AllocationTrackerModule extends BlazeModule {

  /** Sample allocations every N bytes for performance. */
  private static final int SAMPLE_SIZE = 256 * 1024;
  /**
   * Add some variance to how often we sample, to avoid sampling the same callstack all the time due
   * to overly regular allocation patterns.
   */
  private static final int VARIANCE = 100;

  private AllocationTrackerService allocationTrackerService;

  private boolean enabled;
  @Nullable private AllocationTracker tracker = null;

  @Override
  public void globalInit(
      OptionsParsingResult optionsParsingResult, Iterable<BlazeService> blazeServices) {
    for (BlazeService blazeService : blazeServices) {
      if (blazeService instanceof AllocationTrackerService allocationTrackerService) {
        this.allocationTrackerService = allocationTrackerService;
        break;
      }
    }
    checkNotNull(allocationTrackerService, "expected AllocationTrackerService to be available");
  }

  @Override
  public void blazeStartup(
      OptionsParsingResult startupOptions,
      BlazeVersionInfo versionInfo,
      UUID instanceId,
      FileSystem fileSystem,
      ServerDirectories directories,
      Clock clock) {
    enabled = isRequested();
    if (enabled && allocationTrackerService.isSupported()) {
      tracker = new AllocationTracker(SAMPLE_SIZE, VARIANCE);
      Debug.setThreadHook(tracker);
      CurrentRuleTracker.setEnabled(true);
      allocationTrackerService.installAllocationTracker(tracker);
    }
  }

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    if (enabled && allocationTrackerService.isSupported()) {
      builder.setAllocationTracker(tracker);
    }
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    if (!enabled && isRequested()) {
      env.getReporter()
          .handle(
              Event.error(
                  "Failed to enable memory tracking, ensure that you set"
                      + " --host_jvm_args=-javaagent:<path to"
                      + " java-allocation-instrumenter-3.3.4.jar>"));
    }
  }

  private static boolean isRequested() {
    String memoryTrackerProperty = System.getProperty("RULE_MEMORY_TRACKER");
    return memoryTrackerProperty != null && memoryTrackerProperty.equals("1");
  }
}
