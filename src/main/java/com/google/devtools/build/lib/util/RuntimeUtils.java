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
package com.google.devtools.build.lib.util;

import com.google.common.base.Throwables;

import java.util.logging.Logger;

import javax.annotation.Nullable;

/** Utilities for interacting with the jvm. */
public class RuntimeUtils {
  private static final Logger LOG = Logger.getLogger(RuntimeUtils.class.getName());

  @Nullable private static volatile Throwable unprocessedThrowableInTest = null;

  private RuntimeUtils() {}

  /**
   * In tests, System.exit may be disabled. Thus, the main thread should call this method whenever
   * it is about to block on thread completion that might hang because of a failed halt below.
   */
  public static void maybePropagateUnprocessedThrowableIfInTest() {
    Throwables.propagateIfPossible(unprocessedThrowableInTest);
  }

  /** Terminates the jvm with the exit code {@link ExitCode#BLAZE_INTERNAL_ERROR}. */
  public static void halt(Throwable crash) {
    LOG.info("About to call System#exit. Halt requested due to crash. See stderr for details.");
    System.err.println("About to call System#exit. Halt requested due to crash");
    crash.printStackTrace();
    System.err.println("... called from");
    new Exception().printStackTrace();
    unprocessedThrowableInTest = crash;
    System.exit(ExitCode.BLAZE_INTERNAL_ERROR.getNumericExitCode());
  }
}
