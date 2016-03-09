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

package com.google.devtools.build.lib.runtime;

/**
 * Represents how far into the build a given target has gone.
 * Used primarily for master log status reporting and representation.
 */
public enum BuildPhase {
  PARSING("parsing-failed", false),
  LOADING("loading-failed", false),
  ANALYSIS("analysis-failed", false),
  TEST_FILTERING("test-filtered", true),
  TARGET_FILTERING("target-filtered", true),
  NOT_BUILT("not-built", false),
  NOT_ANALYZED("not-analyzed", false),
  EXECUTION("build-failed", false),
  BLAZE_HALTED("blaze-halted", false),
  COMPLETE("built", true),
  // We skip a target when a previous target has failed to build with --nokeep_going.
  BUILD_SKIPPED("build-skipped", false);

  private final String msg;
  private final boolean success;

  BuildPhase(String msg, boolean success) {
    this.msg = msg;
    this.success = success;
  }

  public String getMessage() {
    return msg;
  }

  public boolean getSuccess() {
    return success;
  }
}
