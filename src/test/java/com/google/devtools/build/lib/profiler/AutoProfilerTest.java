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
package com.google.devtools.build.lib.profiler;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.profiler.AutoProfiler.ElapsedTimeReceiver;
import com.google.devtools.build.lib.testutil.ManualClock;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.concurrent.atomic.AtomicLong;

/** Tests for {@link AutoProfiler}. */
@RunWith(JUnit4.class)
public class AutoProfilerTest {

  private ManualClock clock;

  @Before
  public final void init() {
    clock = new ManualClock();
    AutoProfiler.setClock(clock);
  }

  @Test
  public void simple() {
    final AtomicLong elapsedTime = new AtomicLong();
    ElapsedTimeReceiver receiver = new ElapsedTimeReceiver() {
      @Override
      public void accept(long elapsedTimeNanos) {
        elapsedTime.set(elapsedTimeNanos);
      }
    };
    try (AutoProfiler profiler = AutoProfiler.create(receiver)) {
      clock.advanceMillis(42);
    }
    assertThat(elapsedTime.get()).isEqualTo(42 * 1000 * 1000);
  }
}
