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

import com.google.testing.junit.runner.util.TestPropertyExporter.Callback;

import javax.annotation.Nullable;

/**
 * JUnit runner integration code for test properties. Most code should not
 * use this, and should instead use {@link TestPropertyExporter}.
 */
public class TestPropertyRunnerIntegration {
  private static ThreadLocal<Callback> callbackForThread
      = new ThreadLocal<Callback>() {
        @Override
        protected TestPropertyExporter.Callback initialValue() {
          return NoOpCallback.INSTANCE;
        }
  };

  /**
   * Sets the per-thread callback.
   *
   * @param callback Callback
   */
  public static TestPropertyExporter.Callback setTestCaseForThread(@Nullable Callback callback) {
    Callback previousCallback = callbackForThread.get();
    if (callback == null) {
      callbackForThread.remove();
    } else {
      callbackForThread.set(callback);
    }
    return previousCallback;
  }

  static Callback getCallbackForThread() {
    return callbackForThread.get();
  }

  private static class NoOpCallback implements Callback {
    private static final Callback INSTANCE = new NoOpCallback();

    @Override
    public void exportProperty(String name, String value) {
    }

    @Override
    public String exportRepeatedProperty(String name, String value) {
      return name;
    }
  }

}
