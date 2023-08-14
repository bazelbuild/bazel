// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.bugreport.CrashContext;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ThreadUtils}. */
@RunWith(JUnit4.class)
public class ThreadUtilsTest {
  // TODO(b/150299871): inspecting the output of GoogleLogger or mocking it seems too hard for now.
  @Test
  public void smoke() throws Exception {
    SettableFuture<Integer> future = SettableFuture.create();
    int numParkThreads = 11;
    CountDownLatch waitForThreads = new CountDownLatch(numParkThreads + 2);
    List<Thread> parkThreads = new ArrayList<>(numParkThreads);
    for (int i = 0; i < numParkThreads; i++) {
      parkThreads.add(
          new Thread(() -> recursiveMethodPark(0, future, waitForThreads), "parkthread" + i));
    }
    Runnable noParkRunnable = () -> recursiveMethodNoPark(0, waitForThreads);
    Thread noParkThread = new Thread(noParkRunnable, "noparkthread1");
    Thread noParkThread2 = new Thread(noParkRunnable, "noparkthread2");
    AtomicReference<Throwable> reportedException = new AtomicReference<>();
    BugReporter bugReporter =
        new BugReporter() {
          @Override
          public void sendBugReport(Throwable exception, List<String> args, String... values) {
            assertThat(reportedException.get()).isNull();
            reportedException.set(exception);
          }

          @Override
          public void sendNonFatalBugReport(Throwable exception) {
            throw new UnsupportedOperationException();
          }

          @Override
          public void handleCrash(Crash crash, CrashContext ctx) {
            BugReporter.defaultInstance().handleCrash(crash, ctx);
          }
        };
    parkThreads.forEach(Thread::start);
    noParkThread.start();
    noParkThread2.start();
    waitForThreads.await();
    ThreadUtils.warnAboutSlowInterrupt("interrupt message", bugReporter);
    assertThat(reportedException.get())
        .hasCauseThat()
        .hasMessageThat()
        .isEqualTo("(Wrapper exception for longest stack trace) interrupt message");
    // The topmost method is either "sleep" or "sleepNanos0". For example, in JDK 21, "Thread.sleep"
    // calls "sleepNanos" which then calls a "sleepNanos0" native method.
    StackTraceElement[] stackTrace = reportedException.get().getCause().getStackTrace();
    if (stackTrace[0].getMethodName().equals("sleepNanos0")) {
      assertThat(stackTrace[1].getMethodName()).isEqualTo("sleepNanos");
      assertThat(stackTrace[2].getMethodName()).isEqualTo("sleep");
      assertThat(stackTrace[3].getMethodName()).isEqualTo("recursiveMethodNoPark");
    } else {
      assertThat(stackTrace[0].getMethodName()).isEqualTo("sleep");
      assertThat(stackTrace[1].getMethodName()).isEqualTo("recursiveMethodNoPark");
    }

    future.set(1);
    for (Thread thread : parkThreads) {
      thread.join();
    }
    noParkThread.interrupt();
    noParkThread.join();
    noParkThread2.interrupt();
    noParkThread2.join();
  }

  private static void recursiveMethodPark(
      int depth, SettableFuture<Integer> future, CountDownLatch waitForThreads) {
    if (depth < 100) {
      recursiveMethodPark(depth + 1, future, waitForThreads);
      return;
    }
    waitForThreads.countDown();
    try {
      future.get();
    } catch (InterruptedException | ExecutionException e) {
      throw new IllegalStateException(e);
    }
  }

  private static void recursiveMethodNoPark(int depth, CountDownLatch waitForThreads) {
    if (depth < 50) {
      recursiveMethodNoPark(depth + 1, waitForThreads);
      return;
    }
    waitForThreads.countDown();
    try {
      Thread.sleep(Long.MAX_VALUE);
    } catch (InterruptedException e) {
      // Ignored.
    }
  }
}
