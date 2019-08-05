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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Sleeper;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/** Fake sleeper for testing. */
public final class ManualSleeper implements Sleeper {

  private final ManualClock clock;
  private final List<Pair<Long, Runnable>> scheduledRunnables = new ArrayList<>(0);

  public ManualSleeper(ManualClock clock) {
    this.clock = checkNotNull(clock);
  }

  @Override
  public void sleepMillis(long milliseconds) throws InterruptedException {
    checkArgument(milliseconds >= 0, "sleeper can't time travel");
    final long resultedCurrentTimeMillis = clock.advanceMillis(milliseconds);

    Iterator<Pair<Long, Runnable>> iterator = scheduledRunnables.iterator();

    // Run those scheduled Runnables who's time has come.
    while (iterator.hasNext()) {
      Pair<Long, Runnable> scheduledRunnable = iterator.next();

      if (resultedCurrentTimeMillis >= scheduledRunnable.first) {
        iterator.remove();
        scheduledRunnable.second.run();
      }
    }
  }

  /**
   * Schedules a given {@link Runnable} to run when this Sleeper's clock has been adjusted with
   * {@link #sleepMillis(long)} by {@code delayMilliseconds} or greater.
   *
   * @param runnable runnable to run, must not throw exceptions.
   * @param delayMilliseconds delay in milliseconds from current value of {@link ManualClock} used
   *     by this {@link ManualSleeper}.
   */
  public void scheduleRunnable(Runnable runnable, long delayMilliseconds) {
    checkArgument(delayMilliseconds >= 0, "sleeper can't time travel");
    scheduledRunnables.add(new Pair<>(clock.currentTimeMillis() + delayMilliseconds, runnable));
  }
}
