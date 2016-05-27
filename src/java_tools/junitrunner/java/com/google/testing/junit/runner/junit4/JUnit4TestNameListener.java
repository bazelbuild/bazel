// Copyright 2011 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.junit4;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import com.google.testing.junit.runner.util.TestNameProvider;

import org.junit.runner.Description;
import org.junit.runner.notification.RunListener;

/**
 * A listener to get the name of a JUnit4 test. 
 */
@Singleton
class JUnit4TestNameListener extends RunListener {
  private final ThreadLocal<Description> runningTest = new ThreadLocal<>();
  private final SettableCurrentRunningTest currentRunningTest;

  @Inject
  public JUnit4TestNameListener(SettableCurrentRunningTest currentRunningTest) {
    this.currentRunningTest = currentRunningTest;
  }

  @Override
  public void testRunStarted(Description description) throws Exception {
    currentRunningTest.setGlobalTestNameProvider(new TestNameProvider() {
      @Override
      public Description get() {
        return runningTest.get();
      }
    });
  }

  @Override
  public void testStarted(Description description) throws Exception {
    runningTest.set(description);
  }

  @Override
  public void testFinished(Description description) throws Exception {
    runningTest.set(null);
  }
}
