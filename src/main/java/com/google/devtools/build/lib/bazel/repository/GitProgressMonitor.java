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

package com.google.devtools.build.lib.bazel.repository;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import org.eclipse.jgit.lib.ProgressMonitor;

/**
 * ProgressMonitor for reporting progress for Git repository rules.
 */
class GitProgressMonitor implements ProgressMonitor {
  private final String remote;
  private final String message;
  private final ExtendedEventHandler eventHandler;

  private String workTitle;
  private int totalWork;
  private int completedWork;
  private int unfinishedTasks;

  GitProgressMonitor(String remote, String message, ExtendedEventHandler eventHandler) {
    this.remote = remote;
    this.message = message;
    this.eventHandler = eventHandler;
  }

  @Override
  public void start(int totalTasks) {
    this.unfinishedTasks = totalTasks;
  }

  private void report() {
    String progress = workTitle + " (" + completedWork + " / " + totalWork + ")";
    eventHandler.handle(Event.progress(message + ": " + progress));
    eventHandler.post(new GitFetchProgress(remote, progress));
  }

  @Override
  public void beginTask(String title, int totalWork) {
    this.totalWork = totalWork;
    this.completedWork = 0;
    this.workTitle = title;
    report();
  }

  @Override
  public boolean isCancelled() { return false; }

  @Override
  public void update(int completed) {
    completedWork += completed;
    report();
  }

  @Override
  public void endTask() {
    unfinishedTasks--;
    // The number of tasks to do as reported on the start event is sometimes underestimated, so
    // report a finished event after each task that could be the last one.
    if (unfinishedTasks <= 0) {
      eventHandler.post(new GitFetchProgress(remote, "done", true));
    }
  }
}
