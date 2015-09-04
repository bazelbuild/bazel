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

package com.google.devtools.build.lib.bazel.repository;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;

import org.eclipse.jgit.lib.ProgressMonitor;

/**
 * ProgressMonitor for reporting progress for Git repository rules.
 */
class GitProgressMonitor implements ProgressMonitor {
  private String message;
  private Reporter reporter;
  private int totalTasks;
  private int currentTask;

  private String workTitle;
  private int totalWork;
  private int completedWork;

  GitProgressMonitor(String message, Reporter reporter) {
    this.message = message;
    this.reporter = reporter;
  }

  @Override
  public void start(int totalTasks) {
    this.totalTasks = totalTasks;
    this.currentTask = 0;
  }

  private void report() {
    reporter.handle(
        Event.progress("[" + currentTask + " / " + totalTasks + "] "
            + message + ": " + workTitle + " ("
            + completedWork + " / " + totalWork + ")"));
  }

  @Override
  public void beginTask(String title, int totalWork) {
    ++currentTask;
    // TODO(dzc): Remove this when jgit reports totalTasks correctly in start().
    if (currentTask > totalTasks) {
      totalTasks = currentTask;
    }
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
  public void endTask() { }
}
