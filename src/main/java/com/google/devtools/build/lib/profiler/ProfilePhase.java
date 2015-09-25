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
  LAUNCH("launch", "Launch Blaze", 0x3F9FCF9F),                // 9C9
  INIT("init", "Initialize command", 0x3F9F9FCF),              // 99C
  LOAD("loading", "Load packages", 0x3FCFFFCF),                // CFC
  ANALYZE("analysis", "Analyze dependencies", 0x3FCFCFFF),     // CCF
  LICENSE("license checking", "Analyze licenses", 0x3FCFFFFF), // CFF
  PREPARE("preparation", "Prepare for build", 0x3FFFFFCF),     // FFC
  EXECUTE("execution", "Build artifacts", 0x3FFFCFCF),         // FCC
  FINISH("finish", "Complete build",0x3FFFCFFF);               // FCF

  /** Short name for the phase */
  public final String nick;
  /** Human readable description for the phase. */
  public final String description;
  /** Default color of the task, when rendered in a chart. */
  public final int color;

  ProfilePhase(String nick, String description, int color) {
    this.nick = nick;
    this.description = description;
    this.color = color;
  }
}
