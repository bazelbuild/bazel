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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.devtools.build.lib.skyframe.ActionExecutionInactivityWatchdog.InactivityMonitor;
import com.google.devtools.build.lib.skyframe.ActionExecutionInactivityWatchdog.InactivityReporter;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for ActionExecutionInactivityWatchdog. */
@RunWith(JUnit4.class)
public class ActionExecutionInactivityWatchdogTest {

  private static void assertInactivityWatchdogReports(final boolean shouldReport) throws Exception {
    // The monitor implementation below is a state machine. This variable indicates which state
    // it is in.
    final int[] monitorState = new int[] {0};

    // Object that the test thread will wait on.
    final Object monitorFinishedIndicator = new Object();

    // Reported number of action completions in each call to waitForNextCompletion.
    final int[] actionCompletions = new int[] {1, 0, 3, 0, 0, 0, 0, 2};

    // Simulated delay of action completions in each call to waitForNextCompletion.
    final int[] waits = new int[] {5, 10, 3, 10, 30, 60, 60, 1};

    // Log of all Sleep.sleep and InactivityMonitor.waitForNextCompletion calls.
    final List<String> sleepsAndWaits = new ArrayList<>();

    // Mock monitor for this test.
    InactivityMonitor monitor =
        new InactivityMonitor() {
          @Override
          public int waitForNextCompletion(int timeoutSeconds) throws InterruptedException {
            // Simulate the following sequence of events (see actionCompletions):
            // 1. return in 5s (within timeout), 1 action completed; caller will sleep
            // 2. return in 10s (after timeout), 0 action completed; caller will wait
            // 3. return in 3s (within timeout), 3 actions completed (this is possible, since the
            //    waiting (thread doesn't necessarily wake up immediately); caller will sleep
            // 4. return in 10s (after timeout), 0 action completed; caller will wait 30s
            // 5. return in 30s (after timeout), 0 action completed still; caller will wait 60s
            // 6. return in 60s (after timeout), 0 action completed still; caller will wait 60s
            // 7. return in 60s (after timeout), 0 action completed still; caller will wait 60s
            // 8. return in 1s (within timeout), 2 actions completed; caller will sleep, but we
            //    won't record that, because monitorState reached its maximum
            synchronized (monitorFinishedIndicator) {
              if (monitorState[0] >= actionCompletions.length) {
                // Notify the test thread that the test is over.
                monitorFinishedIndicator.notify();
                return 1;
              } else {
                int index = monitorState[0];
                sleepsAndWaits.add("wait:" + waits[index]);
                ++monitorState[0];
                return actionCompletions[index];
              }
            }
          }

          @Override
          public boolean hasStarted() {
            return true;
          }

          @Override
          public int getPending() {
            int index = monitorState[0];
            if (index >= actionCompletions.length) {
              return 0;
            }
            int result = actionCompletions[index];
            while (result == 0) {
              ++index;
              result = actionCompletions[index];
            }
            return result;
          }
        };

    final boolean[] didReportInactivity = new boolean[] {false};
    InactivityReporter reporter =
        new InactivityReporter() {
          @Override
          public void maybeReportInactivity() {
            if (shouldReport) {
              didReportInactivity[0] = true;
            }
          }
        };

    // Mock sleep object; just logs how much the caller's thread would've slept.
    ActionExecutionInactivityWatchdog.Sleep sleep =
        new ActionExecutionInactivityWatchdog.Sleep() {
          @Override
          public void sleep(int durationMilliseconds) throws InterruptedException {
            if (monitorState[0] < actionCompletions.length) {
              sleepsAndWaits.add("sleep:" + durationMilliseconds);
            }
          }
        };

    ActionExecutionInactivityWatchdog watchdog =
        new ActionExecutionInactivityWatchdog(monitor, reporter, 0, sleep);
    try {
      synchronized (monitorFinishedIndicator) {
        watchdog.start();

        long startTime = System.currentTimeMillis();
        boolean done = false;
        while (!done) {
          try {
            monitorFinishedIndicator.wait(5000);
            done = true;
            assertWithMessage("test didn't finish under 5 seconds")
                .that(System.currentTimeMillis() - startTime)
                .isLessThan(5000L);
          } catch (InterruptedException ie) {
            // so-called Spurious Wakeup; ignore
          }
        }
      }
    } finally {
      watchdog.stop();
    }

    assertThat(didReportInactivity[0]).isEqualTo(shouldReport);
    assertThat(sleepsAndWaits)
        .containsExactly(
            "wait:5",
            "sleep:1000",
            "wait:10",
            "wait:3",
            "sleep:1000",
            "wait:10",
            "wait:30",
            "wait:60",
            "wait:60",
            "wait:1")
        .inOrder();
  }

  @Test
  public void testInactivityWatchdogReportsWhenItShould() throws Exception {
    assertInactivityWatchdogReports(true);
  }

  @Test
  public void testInactivityWatchdogDoesNotReportWhenItShouldNot() throws Exception {
    assertInactivityWatchdogReports(false);
  }
}
