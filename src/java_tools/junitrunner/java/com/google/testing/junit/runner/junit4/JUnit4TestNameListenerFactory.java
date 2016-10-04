// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

import com.google.testing.junit.runner.internal.junit4.JUnit4TestNameListener;
import com.google.testing.junit.runner.internal.junit4.SettableCurrentRunningTest;
import com.google.testing.junit.runner.util.Factory;
import com.google.testing.junit.runner.util.Supplier;

/**
 * A factory that supplies {@link JUnit4TestNameListener}.
 */
public final class JUnit4TestNameListenerFactory implements Factory<JUnit4TestNameListener> {
  private final Supplier<SettableCurrentRunningTest> currentRunningTestSupplier;

  public JUnit4TestNameListenerFactory(
      Supplier<SettableCurrentRunningTest> currentRunningTestSupplier) {
    assert currentRunningTestSupplier != null;
    this.currentRunningTestSupplier = currentRunningTestSupplier;
  }

  @Override
  public JUnit4TestNameListener get() {
    return new JUnit4TestNameListener(currentRunningTestSupplier.get());
  }

  public static Factory<JUnit4TestNameListener> create(
      Supplier<SettableCurrentRunningTest> currentRunningTestSupplier) {
    return new JUnit4TestNameListenerFactory(currentRunningTestSupplier);
  }
}
