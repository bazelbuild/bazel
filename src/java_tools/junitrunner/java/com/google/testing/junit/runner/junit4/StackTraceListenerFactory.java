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

import com.google.testing.junit.runner.internal.junit4.JUnit4TestStackTraceListener;
import com.google.testing.junit.runner.util.Factory;
import com.google.testing.junit.runner.util.Supplier;
import org.junit.runner.notification.RunListener;

/**
 * A factory that supplies {@link RunListener} from a {@link JUnit4TestStackTraceListener}.
 */
public final class StackTraceListenerFactory implements Factory<RunListener> {
  private final Supplier<JUnit4TestStackTraceListener> implSupplier;

  public StackTraceListenerFactory(
      Supplier<JUnit4TestStackTraceListener> implSupplier) {
    assert implSupplier != null;
    this.implSupplier = implSupplier;
  }

  @Override
  public RunListener get() {
    RunListener runListener = JUnit4RunnerModule.stackTraceListener(implSupplier.get());
    assert runListener != null;
    return runListener;
  }

  public static Factory<RunListener> create(Supplier<JUnit4TestStackTraceListener> implSupplier) {
    return new StackTraceListenerFactory(implSupplier);
  }
}
