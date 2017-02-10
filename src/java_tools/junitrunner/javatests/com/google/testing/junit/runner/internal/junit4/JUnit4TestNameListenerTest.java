// Copyright 2011 The Bazel Authors. All Rights Reserved.
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

import static com.google.common.base.Throwables.throwIfUnchecked;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import com.google.testing.junit.runner.util.TestNameProvider;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link JUnit4TestNameListener}.
 */
@RunWith(JUnit4.class)
public class JUnit4TestNameListenerTest {
  private JUnit4TestNameListener testNameListener;
  private TestNameProvider testNameProviderForTesting;
  @Rule public TestName name = new TestName();

  @Before
  public void setCurrentRunningTest() {
    SettableCurrentRunningTest currentRunningTest = new SettableCurrentRunningTest() {
      @Override
      public void setGlobalTestNameProvider(TestNameProvider provider) {
        testNameProviderForTesting = provider;
      }
    };

    testNameListener = new JUnit4TestNameListener(currentRunningTest);
  }

  @Test
  public void testJUnit4Listener_normalUsage() throws Exception {
    assertNull(testNameProviderForTesting);

    Description description = Description.createSuiteDescription(FakeTest.class);
    testNameListener.testRunStarted(description);
    assertNull(testNameProviderForTesting.get());

    description = Description.createTestDescription(FakeTest.class, "methodName");
    testNameListener.testStarted(description);
    assertEquals(description, testNameProviderForTesting.get());
    testNameListener.testFinished(description);
    assertNull(testNameProviderForTesting.get());

    description = Description.createTestDescription(FakeTest.class, "anotherMethodName");
    testNameListener.testStarted(description);
    assertEquals(description, testNameProviderForTesting.get());
    testNameListener.testFinished(description);
    assertNull(testNameProviderForTesting.get());

    testNameListener.testRunFinished(null);
    assertNull(testNameProviderForTesting.get());
  }

  @Test
  public void testJUnit4Listener_hasExpectedDisplayName() throws Exception {
    Description description = Description.createSuiteDescription(FakeTest.class);
    testNameListener.testRunStarted(description);

    description = Description.createTestDescription(this.getClass(), name.getMethodName());
    testNameListener.testStarted(description);
    assertEquals("testJUnit4Listener_hasExpectedDisplayName("
        + JUnit4TestNameListenerTest.class.getCanonicalName() + ")",
        testNameProviderForTesting.get().getDisplayName());
  }

  @Test
  public void testJUnit4Listener_multipleThreads() throws Exception {
    ExecutorService executorService = Executors.newSingleThreadExecutor();
    Description description1 =
        Description.createTestDescription(FakeTest.class, "methodName");
    final Description description2 =
        Description.createTestDescription(FakeTest.class, "anotherMethodName");

    testNameListener.testRunStarted(Description.createSuiteDescription(FakeTest.class));
    assertNull(testNameProviderForTesting.get());
    testNameListener.testStarted(description1);
    assertEquals(description1, testNameProviderForTesting.get());

    Future<?> startSecondTestFuture =
        executorService.submit(
            new Runnable() {
              @Override
              public void run() {
                assertNull(testNameProviderForTesting.get());
                try {
                  testNameListener.testStarted(description2);
                } catch (Exception e) {
                  throwIfUnchecked(e);
                  throw new RuntimeException(e);
                }
                assertEquals(description2, testNameProviderForTesting.get());
              }
            });
    startSecondTestFuture.get();

    assertEquals(description1, testNameProviderForTesting.get());
    testNameListener.testFinished(description1);
    assertNull(testNameProviderForTesting.get());

    Future<?> endSecondTestFuture =
        executorService.submit(
            new Runnable() {
              @Override
              public void run() {
                assertEquals(description2, testNameProviderForTesting.get());
                try {
                  testNameListener.testFinished(description2);
                } catch (Exception e) {
                  throwIfUnchecked(e);
                  throw new RuntimeException(e);
                }
                assertNull(testNameProviderForTesting.get());
              }
            });
    endSecondTestFuture.get();

    assertNull(testNameProviderForTesting.get());
  }

  /**
   * Typically, {@link junit.framework.TestListener#startTest(junit.framework.Test)}
   * and {@link junit.framework.TestListener#endTest(junit.framework.Test)}
   * should be called in pairs, but if they're not for some reason, the
   * listener will try to handle it as best as possible.
   */
  @Test
  public void testJUnit4Listener_invalidStatesAreHandled() throws Exception {
    testNameListener.testRunStarted(Description.createSuiteDescription(FakeTest.class));

    Description description1 =
        Description.createTestDescription(FakeTest.class, "methodName");
    Description description2 =
        Description.createTestDescription(FakeTest.class, "anotherMethodName");

    testNameListener.testStarted(description1);
    testNameListener.testStarted(description1);
    assertEquals(description1, testNameProviderForTesting.get());

    testNameListener.testStarted(description2);
    assertEquals(description2, testNameProviderForTesting.get());

    testNameListener.testFinished(description1);
    assertNull(testNameProviderForTesting.get());

    testNameListener.testFinished(description2);
    assertNull(testNameProviderForTesting.get());
  }


  private static class FakeTest {
  }
}
