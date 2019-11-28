// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.platform;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.ProcessUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SuspendCounter}. */
@RunWith(JUnit4.class)
public final class SuspendCounterTest {

  @Test
  public void testSuspendCounter() throws Exception {
    if (OS.getCurrent() == OS.DARWIN) {
      int startSuspendCount = SuspendCounter.suspendCount();

      // Send a SIGCONT to ourselves.
      ProcessBuilder builder =
          new ProcessBuilder("kill", "-s", "CONT", String.valueOf(ProcessUtils.getpid()));
      Process process = builder.start();
      process.waitFor();

      // Allow 10 seconds for signal to propagate.
      for (int i = 0; i < 1000 && SuspendCounter.suspendCount() <= startSuspendCount; ++i) {
        Thread.sleep(10 /* milliseconds */);
      }
      assertThat(SuspendCounter.suspendCount()).isGreaterThan(startSuspendCount);
    } else {
      assertThat(SuspendCounter.suspendCount()).isEqualTo(0);
    }
  }
}
