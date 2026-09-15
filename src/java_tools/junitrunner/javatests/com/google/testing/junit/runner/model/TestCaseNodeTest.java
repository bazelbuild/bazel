// Copyright 2015 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.model;

import static com.google.common.truth.Truth.assertThat;
import static com.google.testing.junit.runner.model.TestInstantUtil.advance;
import static com.google.testing.junit.runner.model.TestInstantUtil.testInstant;

import com.google.testing.junit.runner.util.TestClock.TestInstant;
import java.time.Duration;
import java.time.Instant;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit test for {@link TestCaseNode}.
 */
@RunWith(JUnit4.class)
public class TestCaseNodeTest {

  private static final TestInstant NOW = testInstant(Instant.ofEpochMilli(1));
  private static Description suite;
  private static Description testCase;

  @BeforeClass
  public static void createDescriptions() {
    suite = Description.createSuiteDescription(TestSuiteNode.class);
    testCase = Description.createTestDescription(TestSuite.class, "testCase");
    suite.addChild(testCase);
  }

  @Test
  public void assertIsTestCase() {
    assertThat(new TestCaseNode(testCase, new TestSuiteNode(suite)).isTestCase()).isTrue();
  }

  @Test
  public void assertIsFilteredIfNeverPending() {
    TestCaseNode testCaseNode = new TestCaseNode(testCase, new TestSuiteNode(suite));
    assertStatusWithoutTiming(testCaseNode, TestResult.Status.FILTERED);
  }

  @Test
  public void assertIsCancelledIfNotStarted() {
    TestCaseNode testCaseNode = new TestCaseNode(testCase, new TestSuiteNode(suite));
    testCaseNode.pending();
    assertStatusWithoutTiming(testCaseNode, TestResult.Status.CANCELLED);
  }

  @Test
  public void assertIsCancelledIfInterruptedBeforeStart() {
    TestCaseNode testCaseNode = new TestCaseNode(testCase, new TestSuiteNode(suite));
    testCaseNode.pending();
    testCaseNode.testInterrupted(NOW);
    assertStatusAndTiming(testCaseNode, TestResult.Status.CANCELLED, NOW, 0);
  }

  @Test
  public void assertIsCompletedIfFailedBeforeStart() {
    TestCaseNode testCaseNode = new TestCaseNode(testCase, new TestSuiteNode(suite));
    testCaseNode.pending();
    testCaseNode.testFailure(new Exception(), NOW);
    assertStatusAndTiming(testCaseNode, TestResult.Status.COMPLETED, NOW, 0);
  }

  @Test
  public void assertInterruptedIfStartedAndNotFinished() {
    TestCaseNode testCaseNode = new TestCaseNode(testCase, new TestSuiteNode(suite));
    testCaseNode.pending();
    testCaseNode.started(NOW);
    assertStatusAndTiming(testCaseNode, TestResult.Status.INTERRUPTED, NOW, 0);
    // Notice: This is an unexpected ending state, as even interrupted test executions should go
    // through the testCaseNode.interrupted() code path.
  }

  @Test
  public void assertInterruptedIfStartedAndInterrupted() {
    TestCaseNode testCaseNode = new TestCaseNode(testCase, new TestSuiteNode(suite));
    testCaseNode.pending();
    testCaseNode.started(NOW);
    testCaseNode.testInterrupted(advance(NOW, Duration.ofMillis(1)));
    assertStatusAndTiming(testCaseNode, TestResult.Status.INTERRUPTED, NOW, 1);
  }

  @Test
  public void assertSkippedIfStartedAndSkipped() {
    TestCaseNode testCaseNode = new TestCaseNode(testCase, new TestSuiteNode(suite));
    testCaseNode.pending();
    testCaseNode.started(NOW);
    testCaseNode.testSkipped(advance(NOW, Duration.ofMillis(1)));
    assertStatusAndTiming(testCaseNode, TestResult.Status.SKIPPED, NOW, 1);
  }

  @Test
  public void assertCompletedIfStartedAndFinished() {
    TestCaseNode testCaseNode = new TestCaseNode(testCase, new TestSuiteNode(suite));
    testCaseNode.pending();
    testCaseNode.started(NOW);
    testCaseNode.finished(advance(NOW, Duration.ofMillis(1)));
    assertStatusAndTiming(testCaseNode, TestResult.Status.COMPLETED, NOW, 1);
  }

  @Test
  public void assertCompletedIfStartedAndFailedAndFinished() {
    TestCaseNode testCaseNode = new TestCaseNode(testCase, new TestSuiteNode(suite));
    testCaseNode.pending();
    testCaseNode.started(NOW);
    testCaseNode.testFailure(new Exception(), advance(NOW, Duration.ofMillis(1)));
    testCaseNode.finished(advance(NOW, Duration.ofMillis(2)));
    assertStatusAndTiming(testCaseNode, TestResult.Status.COMPLETED, NOW, 2);
  }

  @Test
  public void assertInterruptedIfStartedAndFailedAndInterrupted() {
    TestCaseNode testCaseNode = new TestCaseNode(testCase, new TestSuiteNode(suite));
    testCaseNode.pending();
    testCaseNode.started(NOW);
    testCaseNode.testFailure(new Exception(), advance(NOW, Duration.ofMillis(1)));
    testCaseNode.testInterrupted(advance(NOW, Duration.ofMillis(2)));
    assertStatusAndTiming(testCaseNode, TestResult.Status.INTERRUPTED, NOW, 2);
  }

  @Test
  public void assertTestSuppressedIfNotStartedAndSuppressed() {
    TestCaseNode testCaseNode = new TestCaseNode(testCase, new TestSuiteNode(suite));
    testCaseNode.pending();
    testCaseNode.testSuppressed(NOW);
    assertStatusAndTiming(testCaseNode, TestResult.Status.SUPPRESSED, NOW, 0);
  }

  private void assertStatusAndTiming(
      TestCaseNode testCase, TestResult.Status status, TestInstant start, long duration) {
    TestResult result = testCase.getResult();
    assertThat(result.getStatus()).isEqualTo(status);
    assertThat(result.getRunTimeInterval()).isNotNull();
    assertThat(result.getRunTimeInterval().getStartMillis())
        .isEqualTo(start.wallTime().toEpochMilli());
    assertThat(result.getRunTimeInterval().toDurationMillis()).isEqualTo(duration);
  }

  private void assertStatusWithoutTiming(TestCaseNode testCase, TestResult.Status status) {
    TestResult result = testCase.getResult();
    assertThat(result.getStatus()).isEqualTo(status);
    assertThat(result.getRunTimeInterval()).isNull();
  }

  static class TestSuite {
    @Test public void testCase() {}
  }
}
