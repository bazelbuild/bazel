// Copyright 2022 The Bazel Authors. All rights reserved.
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

import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.CoreMatchers.instanceOf;
import static org.junit.Assert.assertThrows;
import static org.junit.internal.matchers.ThrowableMessageMatcher.hasMessage;

import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.bugreport.CrashContext;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TestInterruptingBugReporter}. */
@RunWith(JUnit4.class)
@SuppressWarnings("ExpectedExceptionChecker") // Need to use an outer rule to test our inner rule.
public final class TestInterruptingBugReporterTest {

  @Rule(order = 0)
  @SuppressWarnings("deprecation") // See above.
  public final ExpectedException thrown = ExpectedException.none();

  @Rule(order = 1)
  public final TestInterruptingBugReporter bugReporter = new TestInterruptingBugReporter();

  private final ExecutorService executor = Executors.newSingleThreadExecutor();

  @After
  public void shutdownExecutor() {
    executor.shutdownNow();
  }

  @Test
  public void passing() {}

  @Test
  public void failing() {
    thrown.expect(IllegalStateException.class);
    thrown.expectMessage("Intentional");
    throw new IllegalStateException("Intentional");
  }

  @Test
  public void interrupted() throws Exception {
    thrown.expect(InterruptedException.class);
    thrown.expectMessage("Manual interrupt");
    throw new InterruptedException("Manual interrupt");
  }

  @Test
  public void mainThread_sendBugReport() {
    thrown.expect(IllegalStateException.class);
    thrown.expectCause(instanceOf(IOException.class));
    thrown.expectCause(hasMessage(equalTo("IO error")));
    throw assertThrows(
        IllegalStateException.class, () -> bugReporter.sendBugReport(new IOException("IO error")));
  }

  @Test
  public void mainThread_handleCrash() {
    thrown.expect(IllegalStateException.class);
    thrown.expectCause(instanceOf(IOException.class));
    thrown.expectCause(hasMessage(equalTo("IO error")));
    throw assertThrows(
        IllegalStateException.class,
        () ->
            bugReporter.handleCrash(Crash.from(new IOException("IO error")), CrashContext.halt()));
  }

  @Test
  public void asyncThread_noException() throws Exception {
    Future<Void> future = doSomethingAsync(() -> {});
    future.get();
  }

  @Test
  public void asyncThread_uncaughtException() throws Exception {
    thrown.expect(IllegalStateException.class);
    thrown.expectMessage("Intentional");
    Future<Void> future =
        doSomethingAsync(
            () -> {
              throw new IllegalStateException("Intentional");
            });
    throw assertThrows(InterruptedException.class, future::get);
  }

  @Test
  public void asyncThread_sendBugReport() throws Exception {
    thrown.expect(IllegalStateException.class);
    thrown.expectMessage("Intentional");
    Future<Void> future =
        doSomethingAsync(() -> bugReporter.sendBugReport(new IllegalStateException("Intentional")));
    throw assertThrows(InterruptedException.class, future::get);
  }

  @Test
  public void asyncThread_handleCrash() throws Exception {
    thrown.expect(IllegalStateException.class);
    thrown.expectMessage("Intentional");
    Future<Void> future =
        doSomethingAsync(
            () ->
                bugReporter.handleCrash(
                    Crash.from(new IllegalStateException("Intentional")), CrashContext.halt()));
    throw assertThrows(InterruptedException.class, future::get);
  }

  private Future<Void> doSomethingAsync(Runnable something) {
    SettableFuture<Void> future = SettableFuture.create();
    executor.execute(
        () -> {
          something.run();
          future.set(null);
        });
    return future;
  }
}
