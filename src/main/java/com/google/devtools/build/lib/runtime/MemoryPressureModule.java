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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.skyframe.HighWaterMarkLimiter;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.common.options.OptionsBase;

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

    MemoryPressureOptions options = env.getOptions().getOptions(MemoryPressureOptions.class);
    HighWaterMarkLimiter highWaterMarkLimiter =
        new HighWaterMarkLimiter(
            env.getSkyframeExecutor(),
            env.getSyscallCache(),
            options.skyframeHighWaterMarkMemoryThreshold,
            options.skyframeHighWaterMarkMinorGcDropsPerInvocation,
            options.skyframeHighWaterMarkFullGcDropsPerInvocation);

    retainedHeapLimiter.setOptions(options);

    env.getEventBus().register(highWaterMarkLimiter);
  }

  @Override
  public void afterCommand() {
    memoryPressureListener.setEventBus(null);
  }

  @Override
  public ImmutableList<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableList.of(MemoryPressureOptions.class);
  }
}
