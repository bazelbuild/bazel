// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;

/** Prevents the computer from going to sleep while a Bazel command is running. */
public final class SleepPreventionModule extends BlazeModule {
  private PlatformNativeDepsService service;

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    service = checkNotNull(runtime.getBlazeService(PlatformNativeDepsService.class));
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    service.pushDisableSleep();
  }

  @Override
  public void afterCommand() {
    service.popDisableSleep();
  }
}
