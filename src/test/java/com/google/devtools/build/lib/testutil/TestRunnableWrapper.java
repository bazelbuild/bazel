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
package com.google.devtools.build.lib.testutil;

import com.google.common.base.Throwables;
import com.google.devtools.build.lib.concurrent.ThrowableRecordingRunnableWrapper;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * {@link ThrowableRecordingRunnableWrapper} that can throw if one task has thrown an exception but
 * others are still processing.
 */
public class TestRunnableWrapper extends ThrowableRecordingRunnableWrapper {
  // Because IncrementableCountDownLatch isn't public, we have to use a hacky AtomicInteger.
  private final AtomicInteger runningTasks = new AtomicInteger(0);

  public TestRunnableWrapper(String name) {
    super(name);
  }

  public void waitForTasksAndMaybeThrow() throws Exception {
    Throwable firstThrownError;
    do {
      firstThrownError = getFirstThrownError();
      if (firstThrownError != null) {
        Throwables.propagateIfPossible(firstThrownError);
        throw new RuntimeException(firstThrownError);
      }
      Thread.sleep(100);
    } while (runningTasks.get() > 0);
  }

  @Override
  public Runnable wrap(final Runnable runnable) {
    final Runnable wrapped = super.wrap(runnable);
    return new Runnable() {
      @Override
      public void run() {
        runningTasks.incrementAndGet();
        try {
          wrapped.run();
        } finally {
          runningTasks.decrementAndGet();
        }
      }
    };
  }
}
