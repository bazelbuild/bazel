// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.util.Map;
import java.util.UUID;

/**
 * Encapsulates the state needed for a single command. The environment is dropped when the current
 * command is done and all corresponding objects are garbage collected.
 */
public final class CommandEnvironment {
  private final BlazeRuntime runtime;
  private final EventBus eventBus;

  public CommandEnvironment(BlazeRuntime runtime, EventBus eventBus) {
    this.runtime = runtime;
    this.eventBus = eventBus;
  }

  public BlazeRuntime getRuntime() {
    return runtime;
  }

  public BlazeDirectories getDirectories() {
    return runtime.getDirectories();
  }

  /**
   * Returns the reporter for events.
   */
  public Reporter getReporter() {
    return runtime.getReporter();
  }

  public EventBus getEventBus() {
    return eventBus;
  }

  public Map<String, String> getClientEnv() {
    return runtime.getClientEnv();
  }

  public PackageManager getPackageManager() {
    return runtime.getPackageManager();
  }

  public BuildView getView() {
    return runtime.getView();
  }

  public UUID getCommandId() {
    return runtime.getCommandId();
  }

  public SkyframeExecutor getSkyframeExecutor() {
    return runtime.getSkyframeExecutor();
  }

  public Path getWorkingDirectory() {
    return runtime.getWorkingDirectory();
  }

  public ActionCache getPersistentActionCache() throws IOException {
    return runtime.getPersistentActionCache(getReporter());
  }
}
