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
import static com.google.testing.junit.runner.model.TestInstantUtil.testInstant;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.testing.junit.runner.util.FakeTestClock;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.StringWriter;
import java.time.Duration;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

@RunWith(JUnit4.class)
public class AntXmlResultWriterTest {

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

  @Test
  public void testWallTimeAndMonotonicTimestamp() throws Exception {
    FakeTestClock clock = new FakeTestClock();
    Instant startTime = Instant.ofEpochMilli(1560786184600L);
    clock.setWallTimeOffset(startTime);
    TestSuiteNode parent = createTestSuite();
    TestCaseNode test = createTestCase(parent);

    test.started(clock.now());
    // wall time may appear to go back in time in exceptional cases (e.g. daylight saving time)
    clock.advance(Duration.ofMillis(1L));
    clock.setWallTimeOffset(startTime.minus(Duration.ofHours(1)));
    test.finished(clock.now());

    resultWriter.writeTestSuites(writer, root.getResult());

    String resultXml = stringWriter.toString();
    assertThat(resultXml).contains("time=");
    assertThat(resultXml).contains("timestamp=");

    Document document = parseXml(resultXml);
    Element testSuites = document.getDocumentElement();
    Element testSuite = (Element) testSuites.getElementsByTagName("testsuite").item(0);
    assertThat(testSuite.getTagName()).isEqualTo("testsuite");
    assertThat(testSuite.getAttribute("name"))
        .isEqualTo("com.google.testing.junit.runner.model.TestCaseNodeTest$TestSuite");
    assertThat(testSuite.getAttribute("time")).isEqualTo("0.001");
    assertThat(
            Instant.from(
                DateTimeFormatter.ISO_DATE_TIME.parse(testSuite.getAttribute("timestamp"))))
        .isEqualTo(startTime);
  }

  @Test
  public void testSkippedOrSuppressedReportedAsSkipped() throws Exception {
    TestSuiteNode parent = createTestSuite();
    TestCaseNode skipped = createTestCase(parent);
    skipped.started(testInstant(Instant.ofEpochMilli(1)));
    skipped.testSkipped(testInstant(Instant.ofEpochMilli(2)));
    TestCaseNode suppressed = createTestCase(parent);
    suppressed.testSuppressed(testInstant(Instant.ofEpochMilli(4)));

    resultWriter.writeTestSuites(writer, root.getResult());

    Document document = parseXml(stringWriter.toString());
    NodeList caseElems = document.getElementsByTagName("testcase");
    assertThat(caseElems.getLength()).isEqualTo(2);
    for (int i = 0; i < 2; i++) {
      Element caseElem = (Element) caseElems.item(i);
      NodeList skippedElems = caseElem.getElementsByTagName("skipped");
      assertThat(skippedElems.getLength()).isEqualTo(1);
    }
  }

  private void runToCompletion(TestCaseNode test) {
    test.started(testInstant(Instant.ofEpochMilli(1)));
    test.finished(testInstant(Instant.ofEpochMilli(2)));
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

  private static Document parseXml(String testXml)
      throws SAXException, ParserConfigurationException, IOException {
    DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
    factory.setIgnoringElementContentWhitespace(true);
    DocumentBuilder documentBuilder = factory.newDocumentBuilder();
    return documentBuilder.parse(new ByteArrayInputStream(testXml.getBytes(UTF_8)));
  }
}
