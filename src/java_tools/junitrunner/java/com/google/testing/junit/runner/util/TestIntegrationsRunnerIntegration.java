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

package com.google.testing.junit.runner.util;

import com.google.testing.junit.runner.util.TestIntegrationsExporter.Callback;
import javax.annotation.Nullable;

/** JUnit runner integration code for TestIntegration. */
public class TestIntegrationsRunnerIntegration {
  private static final ThreadLocal<Callback> callbackForThread =
      new ThreadLocal<Callback>() {
        @Override
        protected Callback initialValue() {
          return NoOpCallback.INSTANCE;
        }
      };

  /**
   * Sets the per-thread callback.
   *
   * @param callback Callback
   */
  public static Callback setTestCaseForThread(@Nullable Callback callback) {
    Callback previousCallback = callbackForThread.get();
    if (callback == null) {
      callbackForThread.remove();
    } else {
      callbackForThread.set(callback);
    }
    return previousCallback;
  }

  static Callback getCallbackForThread() {
    // TODO(bazel-team): This won't work if the test is running in a different thread from the test
    // runner.
    return callbackForThread.get();
  }

  private static class NoOpCallback implements Callback {
    private static final Callback INSTANCE = new NoOpCallback();

    @Override
    public void exportTestIntegration(TestIntegration testIntegration) {}
  }
}
