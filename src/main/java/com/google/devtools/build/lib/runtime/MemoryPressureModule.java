// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.skyframe.HighWaterMarkLimiter;
import com.google.devtools.build.lib.util.AbruptExitException;

/**
 * A {@link BlazeModule} that installs a {@link MemoryPressureListener} that reacts to memory
 * pressure events.
 */
public final class MemoryPressureModule extends BlazeModule {
  private RetainedHeapLimiter retainedHeapLimiter;
  private MemoryPressureListener memoryPressureListener;

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    retainedHeapLimiter = RetainedHeapLimiter.create(runtime.getBugReporter());
    memoryPressureListener = MemoryPressureListener.create(retainedHeapLimiter);
  }

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    memoryPressureListener.setEventBus(env.getEventBus());

    CommonCommandOptions commonOptions = env.getOptions().getOptions(CommonCommandOptions.class);
    HighWaterMarkLimiter highWaterMarkLimiter =
        new HighWaterMarkLimiter(
            env.getSkyframeExecutor(),
            env.getSyscallCache(),
            commonOptions.skyframeHighWaterMarkMemoryThreshold);

    retainedHeapLimiter.setThreshold(commonOptions.oomMoreEagerlyThreshold);

    env.getEventBus().register(highWaterMarkLimiter);
  }

  @Override
  public void afterCommand() {
    memoryPressureListener.setEventBus(null);
  }
}
