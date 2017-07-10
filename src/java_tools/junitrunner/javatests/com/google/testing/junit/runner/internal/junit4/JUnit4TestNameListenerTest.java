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
import static com.google.common.truth.Truth.assertThat;

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
    assertThat(testNameProviderForTesting).isNull();

    Description description = Description.createSuiteDescription(FakeTest.class);
    testNameListener.testRunStarted(description);
    assertThat(testNameProviderForTesting.get()).isNull();

    description = Description.createTestDescription(FakeTest.class, "methodName");
    testNameListener.testStarted(description);
    assertThat(testNameProviderForTesting.get()).isEqualTo(description);
    testNameListener.testFinished(description);
    assertThat(testNameProviderForTesting.get()).isNull();

    description = Description.createTestDescription(FakeTest.class, "anotherMethodName");
    testNameListener.testStarted(description);
    assertThat(testNameProviderForTesting.get()).isEqualTo(description);
    testNameListener.testFinished(description);
    assertThat(testNameProviderForTesting.get()).isNull();

    testNameListener.testRunFinished(null);
    assertThat(testNameProviderForTesting.get()).isNull();
  }

  @Test
  public void testJUnit4Listener_hasExpectedDisplayName() throws Exception {
    Description description = Description.createSuiteDescription(FakeTest.class);
    testNameListener.testRunStarted(description);

    description = Description.createTestDescription(this.getClass(), name.getMethodName());
    testNameListener.testStarted(description);
    assertThat(testNameProviderForTesting.get().getDisplayName())
        .isEqualTo(
            "testJUnit4Listener_hasExpectedDisplayName("
                + JUnit4TestNameListenerTest.class.getCanonicalName()
                + ")");
  }

  @Test
  public void testJUnit4Listener_multipleThreads() throws Exception {
    ExecutorService executorService = Executors.newSingleThreadExecutor();
    Description description1 =
        Description.createTestDescription(FakeTest.class, "methodName");
    final Description description2 =
        Description.createTestDescription(FakeTest.class, "anotherMethodName");

    testNameListener.testRunStarted(Description.createSuiteDescription(FakeTest.class));
    assertThat(testNameProviderForTesting.get()).isNull();
    testNameListener.testStarted(description1);
    assertThat(testNameProviderForTesting.get()).isEqualTo(description1);

    Future<?> startSecondTestFuture =
        executorService.submit(
            new Runnable() {
              @Override
              public void run() {
                assertThat(testNameProviderForTesting.get()).isNull();
                try {
                  testNameListener.testStarted(description2);
                } catch (Exception e) {
                  throwIfUnchecked(e);
                  throw new RuntimeException(e);
                }
                assertThat(testNameProviderForTesting.get()).isEqualTo(description2);
              }
            });
    startSecondTestFuture.get();

    assertThat(testNameProviderForTesting.get()).isEqualTo(description1);
    testNameListener.testFinished(description1);
    assertThat(testNameProviderForTesting.get()).isNull();

    Future<?> endSecondTestFuture =
        executorService.submit(
            new Runnable() {
              @Override
              public void run() {
                assertThat(testNameProviderForTesting.get()).isEqualTo(description2);
                try {
                  testNameListener.testFinished(description2);
                } catch (Exception e) {
                  throwIfUnchecked(e);
                  throw new RuntimeException(e);
                }
                assertThat(testNameProviderForTesting.get()).isNull();
              }
            });
    endSecondTestFuture.get();

    assertThat(testNameProviderForTesting.get()).isNull();
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
    assertThat(testNameProviderForTesting.get()).isEqualTo(description1);

    testNameListener.testStarted(description2);
    assertThat(testNameProviderForTesting.get()).isEqualTo(description2);

    testNameListener.testFinished(description1);
    assertThat(testNameProviderForTesting.get()).isNull();

    testNameListener.testFinished(description2);
    assertThat(testNameProviderForTesting.get()).isNull();
  }


  private static class FakeTest {
  }
}
