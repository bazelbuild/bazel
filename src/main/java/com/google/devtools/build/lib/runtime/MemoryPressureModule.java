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
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.runtime.MemoryPressure.MemoryPressureStats;
import com.google.devtools.build.lib.skyframe.HighWaterMarkLimiter;
import com.google.devtools.common.options.OptionsBase;
import com.google.errorprone.annotations.Keep;

/**
 * A {@link BlazeModule} that installs a {@link MemoryPressureListener} that reacts to memory
 * pressure events.
 */
public final class MemoryPressureModule extends BlazeModule {
  private final MemoryPressureListener memoryPressureListener = MemoryPressureListener.create();
  private HighWaterMarkLimiter highWaterMarkLimiter;
  private EventBus eventBus;

  @Override
  public ImmutableList<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableList.of(MemoryPressureOptions.class);
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    eventBus = env.getEventBus();
    memoryPressureListener.setEventBus(eventBus);

    MemoryPressureOptions options = env.getOptions().getOptions(MemoryPressureOptions.class);
    highWaterMarkLimiter =
        new HighWaterMarkLimiter(env.getSkyframeExecutor(), env.getSyscallCache(), options);
    memoryPressureListener.setGcThrashingDetector(GcThrashingDetector.createForCommand(options));

    eventBus.register(this);
    eventBus.register(highWaterMarkLimiter);
  }

  @Override
  public void afterCommand() {
    postStats();
    memoryPressureListener.setEventBus(null);
    memoryPressureListener.setGcThrashingDetector(null);
    eventBus = null;
    highWaterMarkLimiter = null;
  }

  @Subscribe
  @Keep
  void onCrash(@SuppressWarnings("unused") CrashEvent event) {
    postStats();
  }

  private void postStats() {
    MemoryPressureStats.Builder stats = MemoryPressureStats.newBuilder();
    highWaterMarkLimiter.addStatsAndReset(stats);
    eventBus.post(stats.build());
  }
}
