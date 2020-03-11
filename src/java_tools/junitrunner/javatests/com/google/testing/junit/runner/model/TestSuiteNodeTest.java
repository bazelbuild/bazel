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
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyZeroInteractions;

import com.google.common.collect.ImmutableMap;
import com.google.testing.junit.runner.util.TestClock.TestInstant;
import java.time.Duration;
import java.time.Instant;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.runners.MockitoJUnitRunner;

/** Unit test for {@link TestSuiteNode}. */
@RunWith(MockitoJUnitRunner.class)
public class TestSuiteNodeTest {

  private static final TestInstant NOW = new TestInstant(Instant.EPOCH, Duration.ZERO);

  @Mock private TestCaseNode testCaseNode;
  private TestSuiteNode testSuiteNode;

  @Before
  public void createTestSuiteNode() {
    testSuiteNode = new TestSuiteNode(Description.createSuiteDescription("suite"));
    testSuiteNode.addTestCase(testCaseNode);
  }

  @Test
  public void testIsTestCase() {
    assertThat(testSuiteNode.isTestCase()).isFalse();
    verifyZeroInteractions(testCaseNode);
  }

  @Test
  public void testInterrupted() {
    testSuiteNode.testInterrupted(NOW);
    verify(testCaseNode, times(1)).testInterrupted(NOW);
  }

  @Test
  public void testTestSkipped() {
    testSuiteNode.testSkipped(NOW);
    verify(testCaseNode, times(1)).testSkipped(NOW);
  }

  @Test
  public void testTestIgnored() {
    testSuiteNode.testSuppressed(NOW);
    verify(testCaseNode, times(1)).testSuppressed(NOW);
  }

  @Test
  public void testTestFailure() {
    Exception failure = new Exception();
    testSuiteNode.testFailure(failure, NOW);
    verify(testCaseNode, times(1)).testFailure(failure, NOW);
  }

  @Test
  public void testDynamicFailure() {
    Description dynamicTestCaseDescription = mock(Description.class);
    Exception failure = new Exception();
    testSuiteNode.dynamicTestFailure(dynamicTestCaseDescription, failure, NOW);
    verify(testCaseNode, times(1)).dynamicTestFailure(dynamicTestCaseDescription, failure, NOW);
    verifyZeroInteractions(dynamicTestCaseDescription);
  }

  @Test
  public void testProperties() {
    ImmutableMap<String, String> properties = ImmutableMap.of("key", "value");
    testSuiteNode = new TestSuiteNode(Description.createSuiteDescription("suite"), properties);

    TestResult result = testSuiteNode.getResult();
    assertThat(result.getProperties()).containsExactlyEntriesIn(properties);
  }
}
