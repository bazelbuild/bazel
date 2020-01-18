// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.profiler;

/**
 * Build phase markers. Used as a separators between different build phases.
 */
public enum ProfilePhase {
  LAUNCH("launch", "Launch Blaze"),
  INIT("init", "Initialize command"),
  LOAD("loading", "Load packages"),
  ANALYZE("analysis", "Analyze dependencies"),
  LICENSE("license checking", "Analyze licenses"),
  PREPARE("preparation", "Prepare for build"),
  EXECUTE("execution", "Build artifacts"),
  FINISH("finish", "Complete build"),
  UNKNOWN("unknown", "unknown");

  /** Short name for the phase */
  public final String nick;
  /** Human readable description for the phase. */
  public final String description;

  ProfilePhase(String nick, String description) {
    this.nick = nick;
    this.description = description;
  }

  public static ProfilePhase getPhaseFromDescription(String description) {
    for (ProfilePhase profilePhase : ProfilePhase.values()) {
      if (profilePhase.description.equals(description)) {
        return profilePhase;
      }
    }
    return UNKNOWN;
  }
}
