// Copyright 2018 The Bazel Authors. All Rights Reserved.
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

import java.io.IOException;
import java.io.StringWriter;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class AntXmlResultWriterTest {
  private static final long NOW = 1;
  private static TestSuiteNode root;
  private static XmlWriter writer;
  private static AntXmlResultWriter resultWriter;
  private static StringWriter stringWriter;

  @Before
  public void before() {
    stringWriter = new StringWriter();
    writer = XmlWriter.createForTesting(stringWriter);
    resultWriter = new AntXmlResultWriter();
    root = new TestSuiteNode(Description.createSuiteDescription("root"));
  }

  @Test
  public void allPassingTestCasesWritten() throws IOException {
    TestSuiteNode parent = createTestSuite();
    TestCaseNode test1 = createTestCase(parent);
    TestCaseNode test2 = createTestCase(parent);
    runToCompletion(test1);
    runToCompletion(test2);

    resultWriter.writeTestSuites(writer, root.getResult());
    String resultXml = stringWriter.toString();
    assertThat(resultXml).contains("<testcase name='testCase1'");
    assertThat(resultXml).contains("<testcase name='testCase2'");
  }

  @Test
  public void testFilteredCasesNotWritten() throws IOException {
    TestSuiteNode parent = createTestSuite();
    TestCaseNode test1 = createTestCase(parent);
    runToCompletion(test1);

    createTestCase(parent); // creates a test case that is FILTERED by default

    resultWriter.writeTestSuites(writer, root.getResult());

    String resultXml = stringWriter.toString();
    assertThat(resultXml).contains("<testcase name='testCase1'");
    assertThat(resultXml).doesNotContain("<testcase name='testCase2'");
  }

  private void runToCompletion(TestCaseNode test) {
    test.started(NOW);
    test.finished(NOW + 1);
  }

  private TestCaseNode createTestCase(TestSuiteNode parent) {
    int idx = parent.getChildren().size() + 1;
    TestCaseNode testCase =
        new TestCaseNode(Description.createSuiteDescription("testCase" + idx), parent);
    parent.addTestCase(testCase);
    return testCase;
  }

  private TestSuiteNode createTestSuite() {
    Description suite = Description.createSuiteDescription(TestCaseNodeTest.TestSuite.class);
    TestSuiteNode parent = new TestSuiteNode(suite);
    root.addTestSuite(parent);
    return parent;
  }
}
