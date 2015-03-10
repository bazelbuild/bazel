// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import com.google.common.collect.ImmutableMap;

import java.util.HashMap;
import java.util.Map;

/**
 * Event signaling the end of the execution phase. Contains statistics about the action cache,
 * the metadata cache and about last file save times.
 */
public class ExecutionFinishedEvent {

  private final int outputDirtyFiles;
  private final int outputModifiedFilesDuringPreviousBuild;
  /** The mtime of the most recently saved source file when the build starts. */
  private long lastFileSaveTimeInMillis;

  /**
   * The (filename, mtime) pairs of all files saved between the last build's
   * start time and the current build's start time. Only applies to builds
   * running with existing Blaze servers. Currently disabled.
   */
  private Map<String, Long> changedFileSaveTimes = new HashMap<>();

  public ExecutionFinishedEvent(Map<String, Long> changedFileSaveTimes,
      long lastFileSaveTimeInMillis, int outputDirtyFiles,
      int outputModifiedFilesDuringPreviousBuild) {
    this.outputDirtyFiles = outputDirtyFiles;
    this.outputModifiedFilesDuringPreviousBuild = outputModifiedFilesDuringPreviousBuild;
    this.changedFileSaveTimes = ImmutableMap.copyOf(changedFileSaveTimes);
    this.lastFileSaveTimeInMillis = lastFileSaveTimeInMillis;
  }

  public long getLastFileSaveTimeInMillis() {
    return lastFileSaveTimeInMillis;
  }

  public int getOutputDirtyFiles() {
    return outputDirtyFiles;
  }

  public int getOutputModifiedFilesDuringPreviousBuild() {
    return outputModifiedFilesDuringPreviousBuild;
  }

  public Map<String, Long> getChangedFileSaveTimes() {
    return changedFileSaveTimes;
  }
}
