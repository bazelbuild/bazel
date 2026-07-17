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

package com.google.devtools.build.lib.platform;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/** Detects suspension events. */
public final class SystemDiskSpaceModule extends BlazeModule {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private PlatformNativeDepsService service;

  @GuardedBy("this")
  @Nullable
  private Reporter reporter;

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    service = checkNotNull(runtime.getBlazeService(PlatformNativeDepsService.class));
    service.registerDiskSpaceJni(this::diskSpaceCallback);
  }

  @Override
  public synchronized void beforeCommand(CommandEnvironment env) {
    this.reporter = env.getReporter();
  }

  @Override
  public synchronized void afterCommand() {
    this.reporter = null;
  }

  private synchronized void diskSpaceCallback(int value) {
    SystemDiskSpaceEvent event = new SystemDiskSpaceEvent(value);
    if (reporter != null) {
      reporter.post(event);
    }
    logger.atInfo().log("%s", event.logString());
  }
}
