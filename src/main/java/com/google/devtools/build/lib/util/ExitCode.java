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

package com.google.devtools.build.lib.util;

/**
 *  <p>Anything marked FAILURE is generally from a problem with the source code
 *  under consideration.  In these cases, a re-run in an identical client should
 *  produce an identical return code all things being constant.
 *
 *  <p>Anything marked as an ERROR is generally a problem unrelated to the
 *  source code itself.  It is either something wrong with the user's command
 *  line or the user's machine or environment.
 *
 *  <p>Note that these exit codes should be kept consistent with the codes
 *  returned by Blaze's launcher in //devtools/blaze/main:blaze.cc
 */
public enum ExitCode {
  SUCCESS(0),
  BUILD_FAILURE(1),
  PARSING_FAILURE(1),
  COMMAND_LINE_ERROR(2),
  TESTS_FAILED(3),
  PARTIAL_ANALYSIS_FAILURE(3),
  NO_TESTS_FOUND(4),
  COVERAGE_REPORT_NOT_GENERATED(5),
  RUN_FAILURE(6),
  ANALYSIS_FAILURE(7),
  INTERRUPTED(8),
  REMOTE_ENVIRONMENTAL_ERROR(32),
  OOM_ERROR(33),
  RESERVED_1(34),
  RESERVED_2(35),
  LOCAL_ENVIRONMENTAL_ERROR(36),
  BLAZE_INTERNAL_ERROR(37),
  PUBLISH_ERROR(38),  // Errors publishing the Blaze results to message-queueing system.
  RESERVED(40);
  /*
    exit codes [50..60] and 253 are reserved for site specific wrappers to Bazel.
  */

  // Keep in sync with the enum.
  private final static int FIRST_INFRASTRUCTURE_FAILURE = 30;

  private final int numericExitCode;

  private ExitCode(int exitCode) {
    this.numericExitCode = exitCode;
  }

  public int getNumericExitCode() {
    return numericExitCode;
  }

  /**
   * Returns true if the current exit code represents a failure of Blaze infrastructure,
   * vs. a build failure.
   */
  public boolean isInfrastructureFailure() {
    return numericExitCode >= FIRST_INFRASTRUCTURE_FAILURE;
  }
}
