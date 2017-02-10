// Copyright 2012 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.internal.junit4;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Test;
import org.junit.internal.AssumptionViolatedException;
import org.junit.runner.Description;
import org.junit.runner.JUnitCore;
import org.junit.runner.Request;
import org.junit.runner.Result;
import org.junit.runner.RunWith;
import org.junit.runner.Runner;
import org.junit.runner.notification.Failure;
import org.junit.runner.notification.RunNotifier;
import org.junit.runner.notification.StoppedByUserException;
import org.junit.runners.JUnit4;
import org.junit.runners.Suite;
import org.junit.runners.model.InitializationError;

/**
 * Tests for {@link CancellableRequestFactory}.
 */
@RunWith(JUnit4.class)
public class CancellableRequestFactoryTest {

  private final CancellableRequestFactory cancellableRequestFactory =
      new CancellableRequestFactory();

  @Test
  public void testCancelRunAfterStarting() throws Exception {
    final CountDownLatch testStartLatch = new CountDownLatch(1);
    final CountDownLatch testContinueLatch = new CountDownLatch(1);
    final AtomicBoolean secondTestRan = new AtomicBoolean(false);

    // Simulates a test that hangs
    FakeRunner blockingRunner = new FakeRunner("blocks", new Runnable() {
      @Override
      public void run() {
        testStartLatch.countDown();
        try {
          testContinueLatch.await(1, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException("Timed out waiting for signal to continue test", e);
        }
      }
    });

    // A runner that should never run its test
    FakeRunner secondRunner = new FakeRunner("shouldNotRun", new Runnable() {
      @Override
      public void run() {
        secondTestRan.set(true);
      }
    });

    RunnerSuite fakeSuite = new RunnerSuite(blockingRunner, secondRunner);
    final Request request = cancellableRequestFactory.createRequest(Request.runner(fakeSuite));

    ExecutorService executor = Executors.newSingleThreadExecutor();
    Future<Result> future = executor.submit(new Callable<Result>() {
      @Override
      public Result call() throws Exception {
        JUnitCore core = new JUnitCore();
        return core.run(request);
      }
    });

    // Simulate cancel being called in the middle of the test
    testStartLatch.await(1, TimeUnit.SECONDS);
    cancellableRequestFactory.cancelRun();
    testContinueLatch.countDown();

    try {
      future.get(10, TimeUnit.SECONDS);
      fail("exception expected");
    } catch (ExecutionException e) {
      Throwable runnerException = e.getCause();
      assertTrue(runnerException instanceof RuntimeException);
      assertEquals("Test run interrupted", runnerException.getMessage());
      assertTrue(runnerException.getCause() instanceof StoppedByUserException);
    }

    executor.shutdownNow();
  }

  @Test
  public void testCancelRunBeforeStarting() throws Exception {
    final AtomicBoolean testRan = new AtomicBoolean(false);

    // A runner that should never run its test
    FakeRunner runner = new FakeRunner("shouldNotRun", new Runnable() {
      @Override
      public void run() {
        testRan.set(true);
      }
    });

    Request request = cancellableRequestFactory.createRequest(Request.runner(runner));
    cancellableRequestFactory.cancelRun();
    JUnitCore core = new JUnitCore();

    try {
      core.run(request);
      fail("exception expected");
    } catch (RuntimeException e) {
      assertEquals("Test run interrupted", e.getMessage());
      assertTrue(e.getCause() instanceof StoppedByUserException);
    }

    assertFalse(testRan.get());
  }

  @Test
  public void testNormalRun() {
    final AtomicBoolean testRan = new AtomicBoolean(false);

    // A runner that should run its test
    FakeRunner runner = new FakeRunner("shouldRun", new Runnable() {
      @Override
      public void run() {
        testRan.set(true);
      }
    });

    Request request = cancellableRequestFactory.createRequest(Request.runner(runner));
    JUnitCore core = new JUnitCore();
    Result result = core.run(request);

    assertTrue(testRan.get());
    assertEquals(1, result.getRunCount());
    assertEquals(0, result.getFailureCount());
  }

  @Test
  public void testFailingRun() {
    final AtomicBoolean testRan = new AtomicBoolean(false);
    final RuntimeException expectedFailure = new RuntimeException();

    // A runner that should run its test
    FakeRunner runner = new FakeRunner("shouldRun", new Runnable() {
      @Override
      public void run() {
        testRan.set(true);
        throw expectedFailure;
      }
    });

    Request request = cancellableRequestFactory.createRequest(Request.runner(runner));
    JUnitCore core = new JUnitCore();
    Result result = core.run(request);

    assertTrue(testRan.get());
    assertEquals(1, result.getRunCount());
    assertEquals(1, result.getFailureCount());
    assertSame(expectedFailure, result.getFailures().get(0).getException());
  }


  private static class FakeRunner extends Runner {
    private final Description testDescription;
    private final Runnable test;

    public FakeRunner(String testName, Runnable test) {
      this.test = test;
      testDescription = Description.createTestDescription(FakeRunner.class, testName);
    }

    @Override
    public Description getDescription() {
      return testDescription;
    }

    @Override
    public void run(RunNotifier notifier) {
      notifier.fireTestStarted(testDescription);

      try {
        test.run();
      } catch (AssumptionViolatedException e) {
        notifier.fireTestAssumptionFailed(new Failure(testDescription, e));
      } catch (Throwable e) {
        notifier.fireTestFailure(new Failure(testDescription, e));
      } finally {
        notifier.fireTestFinished(testDescription);
      }
    }
  }

  public static class FakeSuite {
  }

  public static class RunnerSuite extends Suite {

    public RunnerSuite(Runner... runners) throws InitializationError {
      super(FakeSuite.class, Arrays.asList(runners));
    }
  }
}
