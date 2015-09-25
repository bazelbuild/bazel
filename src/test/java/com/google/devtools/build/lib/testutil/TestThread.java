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

package com.google.devtools.build.lib.testutil;

import static com.google.common.truth.Truth.assertWithMessage;

/**
 * Test thread implementation that allows the use of assertions within
 * spawned threads.
 *
 * Main test method must call {@link TestThread#joinAndAssertState(long)}
 * for each spawned test thread.
 */
public abstract class TestThread extends Thread {
  Throwable testException = null;
  boolean isSucceeded = false;

  /**
   * Specific test thread implementation overrides this method.
   */
  abstract public void runTest() throws Exception;

  @Override public final void run() {
    try {
      runTest();
      isSucceeded = true;
    } catch (Exception | AssertionError e) {
      testException = e;
    }
  }

  /**
   * Joins test thread (waiting specified number of ms) and validates that
   * it has been completed successfully.
   */
  public void joinAndAssertState(long timeout) throws InterruptedException {
    join(timeout);
    Throwable exception = this.testException;
    if (isAlive()) {
      exception = new AssertionError (
          "Test thread " + getName() + " is still alive");
      exception.setStackTrace(getStackTrace());
    }
    if(exception != null) {
      AssertionError error = new AssertionError("Test thread " + getName() + " failed to execute");
      error.initCause(exception);
      throw error;
    }
    assertWithMessage("Test thread " + getName() + " has not run successfully").that(isSucceeded)
        .isTrue();
  }
}
