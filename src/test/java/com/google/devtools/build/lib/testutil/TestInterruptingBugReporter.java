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

import static com.google.common.base.MoreObjects.firstNonNull;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.bugreport.CrashContext;
import java.lang.Thread.UncaughtExceptionHandler;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.rules.TestRule;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;

/**
 * {@link TestRule} that interrupts the main test thread when an unexpected exception in an async
 * thread is encountered.
 *
 * <p>Designed for use in tests that would otherwise hang indefinitely in the event of a bug. In
 * blaze, unexpected exceptions are typically handled by calling {@link Runtime#halt} to terminate
 * the JVM. In Java tests, however, this does not work because:
 *
 * <ul>
 *   <li>If {@link com.google.devtools.build.lib.runtime.BlazeRuntime} is not in scope for the test,
 *       there may not be a relevant {@link UncaughtExceptionHandler} installed.
 *   <li>Calling {@link Runtime#halt} in a Java test is not allowed and leads to a {@link
 *       SecurityException}.
 * </ul>
 *
 * <p>Consider a class with the following method:
 *
 * <pre>{@code
 * public Future<Void> doSomethingAsync() {
 *   SettableFuture<Void> future = SettableFuture.create();
 *   executor.execute(() -> {
 *     ...
 *     Preconditions.checkState(someCondition);
 *     future.set(null);
 *   });
 *   return future;
 * }
 * }</pre>
 *
 * and a corresponding unit test:
 *
 * <pre>{@code
 * @Test
 * public void testSomethingAsync() throws Exception {
 *   Future<Void> future = underTest.doSomethingAsync();
 *   future.get();
 * }
 * }</pre>
 *
 * If the call to {@code Preconditions.checkState} fails, the test hangs indefinitely. Diagnosing
 * the issue would require waiting for the test to time out and then analyzing the test log to find
 * the uncaught exception. Instead, using {@code TestInterruptingBugReporter} will immediately
 * interrupt the main test thread and display the uncaught exception as the test's failure cause.
 *
 * <p>{@code TestInteruptingBugReporter} can also be used as a {@link BugReporter} if the system
 * under test is designed to accept one:
 *
 * <pre>{@code
 * public Future<Void> doSomethingAsync(BugReporter bugReporter) {
 *   SettableFuture<Void> future = SettableFuture.create();
 *   executor.execute(() -> {
 *     ...
 *     if (!someCondition) {
 *       bugReporter.sendBugReport(new IllegalStateException("someCondition was false");
 *     }
 *     future.set(null);
 *   });
 *   return future;
 * }
 * }</pre>
 *
 * Example usage:
 *
 * <pre>{@code
 * @Rule public final TestInterruptingBugReporter bugReporter = new TestInterruptingBugReporter();
 *
 * @Test
 * public void testSomethingAsync() throws Exception {
 *   Future<Void> future = underTest.doSomethingAsync(bugReporter);
 *   future.get();
 * }
 * }</pre>
 */
public final class TestInterruptingBugReporter
    implements BugReporter, UncaughtExceptionHandler, TestRule {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  // This is the main test thread so long as this class is being used as documented (instantiated
  // as a field in the test class).
  private final Thread testThread = Thread.currentThread();

  private final AtomicReference<Throwable> bug = new AtomicReference<>();

  @Override
  public Statement apply(Statement base, Description description) {
    return new Statement() {
      @Override
      public void evaluate() throws Throwable {
        UncaughtExceptionHandler originalHandler = Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler(TestInterruptingBugReporter.this);
        try {
          base.evaluate();
        } catch (InterruptedException e) {
          throw firstNonNull(bug.get(), e);
        } finally {
          Thread.setDefaultUncaughtExceptionHandler(originalHandler);
        }
      }
    };
  }

  @Override
  public void sendBugReport(Throwable exception, List<String> args, String... values) {
    handle(exception, "call to sendBugReport", Thread.currentThread());
  }

  @Override
  public void sendNonFatalBugReport(Throwable exception) {
    handle(exception, "call to sendNonFatalBugReport", Thread.currentThread());
  }

  @Override
  public void handleCrash(Crash crash, CrashContext ctx) {
    handle(crash.getThrowable(), "call to handleCrash", Thread.currentThread());
  }

  @Override
  public void uncaughtException(Thread thread, Throwable exception) {
    handle(exception, "uncaught exception", thread);
  }

  private synchronized void handle(Throwable exception, String context, Thread thread) {
    if (thread.equals(testThread)) {
      throw new IllegalStateException(exception);
    }
    if (bug.compareAndSet(null, exception)) {
      logger.atSevere().withCause(exception).log(
          "Handling %s in thread %s by interrupting the main test thread",
          context, thread.getName());
      testThread.interrupt();
    } else {
      logger.atSevere().withCause(exception).log(
          "Ignoring %s in thread %s since a previous exception was seen",
          context, thread.getName());
      bug.get().addSuppressed(exception);
    }
  }
}
