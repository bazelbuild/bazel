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

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;

/** Build phase markers. Used as a separators between different build phases. */
@SkybridgeInterface
public final class ProfilePhase {
  public static final ProfilePhase LAUNCH = new ProfilePhase("launch", "Launch Blaze");
  public static final ProfilePhase INIT = new ProfilePhase("init", "Initialize command");
  public static final ProfilePhase TARGET_PATTERN_EVAL =
      new ProfilePhase("target pattern evaluation", "Evaluate target patterns");
  public static final ProfilePhase ANALYZE =
      new ProfilePhase("interleaved loading-and-analysis", "Load and analyze dependencies");
  public static final ProfilePhase ANALYZE_AND_EXECUTE =
      new ProfilePhase(
          "interleaved loading, analysis and execution",
          "Load, analyze dependencies and build artifacts");
  public static final ProfilePhase LICENSE =
      new ProfilePhase("license checking", "Analyze licenses");
  public static final ProfilePhase PREPARE = new ProfilePhase("preparation", "Prepare for build");
  public static final ProfilePhase EXECUTE = new ProfilePhase("execution", "Build artifacts");
  public static final ProfilePhase FINISH = new ProfilePhase("finish", "Complete build");
  public static final ProfilePhase UNKNOWN = new ProfilePhase("unknown", "unknown");

  /** Short name for the phase */
  public final String nick;
  /** Human readable description for the phase. */
  public final String description;

  private ProfilePhase(String nick, String description) {
    this.nick = nick;
    this.description = description;
  }

  @Override
  public String toString() {
    return nick;
  }
}
