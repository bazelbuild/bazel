// Copyright 2009 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import java.io.ByteArrayInputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link TestXmlOutputParser}. */
@RunWith(JUnit4.class)
public class TestXmlOutputParserTest {
  @Test
  public void testCanParsePassedCases() throws Exception {
    TestCase result = parseXml(generateTestSuiteWithTestCase());

    TestCase testSuite = result.getChild(0);
    TestCase testCase = testSuite.getChild(0);

    assertThat(testCase.getStatus()).isEqualTo(TestCase.Status.PASSED);
  }

  @Test
  public void testCanParseFailure() throws Exception {
    TestCase result =
        parseXml(
            generateTestSuiteWithTestCase(
                "<failure message='Bad day' type='horrible'>When you do your best",
                "and it is not enough.</failure>"));

    TestCase testSuite = result.getChild(0);
    TestCase testCase = testSuite.getChild(0);

    assertThat(testCase.getStatus()).isEqualTo(TestCase.Status.FAILED);
  }

  @Test
  public void testCanParseError() throws Exception {
    TestCase result =
        parseXml(
            generateTestSuiteWithTestCase(
                "<error message='Too early.' type='hangover'>Platinum class:",
                "You wish you died ASAP.</error>"));

    TestCase testSuite = result.getChild(0);
    TestCase testCase = testSuite.getChild(0);

    assertThat(testCase.getStatus()).isEqualTo(TestCase.Status.ERROR);
  }

  @Test
  public void testCanParseFallbackXmlWithPropertiesAndTimeout() throws Exception {
    TestCase result =
        parseXml(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
            "<testsuites>",
            "  <testsuite name=\"example/suite\" tests=\"1\" failures=\"0\" errors=\"1\">",
            "    <properties>",
            "      <property name=\"bazel.xml.generator\" value=\"fallback\"/>",
            "      <property name=\"bazel.test.status\" value=\"TIMEOUT\"/>",
            "    </properties>",
            "    <testcase name=\"example/suite\" classname=\"\" status=\"run\" duration=\"60\""
                + " time=\"60\"><error message=\"exited with error code 142\""
                + " type=\"timeout\"></error></testcase>",
            "    <system-out><![CDATA[timed out]]></system-out>",
            "  </testsuite>",
            "</testsuites>");

    TestCase testSuite = result.getChild(0);
    TestCase testCase = testSuite.getChild(0);

    assertThat(testSuite.getStatus()).isEqualTo(TestCase.Status.ERROR);
    assertThat(testCase.getName()).isEqualTo("example/suite");
    assertThat(testCase.getStatus()).isEqualTo(TestCase.Status.ERROR);
  }

  @Test
  public void testCanParseTestSuites() throws Exception {
    TestCase result =
        parseXml(
            "<testsuites name='yo' tests='3' failures='0' errors='1' time='6.9'>", "</testsuites>");

    assertThat(result.getType()).isEqualTo(TestCase.Type.TEST_SUITE);
  }

  @Test
  public void testCanParseTestSuiteInRoot() throws Exception {
    TestCase result =
        parseXml(
            "<testsuite name='' tests='0' failures='0' errors='0' time='0'>",
            "  <testcase name='Goofy' status='run' classname='elite' time='2.0' random='42'>",
            "     <failure message='bam'>boom</failure>",
            "  </testcase>",
            "</testsuite>");

    assertThat(result.getType()).isEqualTo(TestCase.Type.TEST_SUITE);
    assertThat(result.getChildCount()).isEqualTo(1);
  }

  @Test
  public void testReturnsThrowsOnError() throws Exception {
    assertThrows(
        TestXmlOutputParserException.class,
        () ->
            parseXml(
                "Whether 'tis nobler in the mind to suffer",
                "The slings and arrows of outrageous fortune,",
                "Or to take arms against a sea of troubles,",
                "And by opposing end them?"));
  }

  @Test
  public void testCanParseTestDecorator() throws Exception {
    TestCase result =
        parseXml(
            "<testsuites name='' tests='0' failures='0' errors='0' time='0' color='red'>",
            "  <testdecorator classname='Foo'",
            "      tests='3' failures='2' errors='1' time='1.2' memory-delta='3'>",
            "  </testdecorator>",
            "</testsuites>");

    assertThat(result.getChildCount()).isEqualTo(1);
    assertThat(result.getChild(0).getType()).isEqualTo(TestCase.Type.TEST_DECORATOR);
  }

  @Test
  public void testCanParseTestCase() throws Exception {
    TestCase result =
        parseXml(
            "<testsuites name='' tests='0' failures='0' errors='0' time='0' color='red'>",
            "  <testcase classname='Foo' name='foo'",
            "      tests='3' failures='2' errors='1' time='1.2' memory-delta='3'>",
            "  </testcase>",
            "</testsuites>");

    assertThat(result.getChildCount()).isEqualTo(1);
    assertThat(result.getChild(0).getType()).isEqualTo(TestCase.Type.TEST_CASE);
  }

  @Test
  public void testCanParseUnnamedSuites() throws Exception {
    TestCase result =
        parseXml(
            "<testsuites>",
            "  <testcase classname='Foo' name='foo'",
            "      tests='3' failures='2' errors='1' time='1.2' memory-delta='3'>",
            "  </testcase>",
            "</testsuites>");

    assertThat(result.getChildCount()).isEqualTo(1);
    assertThat(result.getChild(0).getType()).isEqualTo(TestCase.Type.TEST_CASE);
  }

  @Test
  public void testCanParseUnnamedCases() throws Exception {
    TestCase result =
        parseXml(
            "<testsuites name='foo'>", "  <testcase time='1.2'>", "  </testcase>", "</testsuites>");

    assertThat(result.getChildCount()).isEqualTo(1);
    assertThat(result.getChild(0).getType()).isEqualTo(TestCase.Type.TEST_CASE);
  }

  @Test
  public void testCanParseIntegerDuration() throws Exception {
    TestCase result =
        parseXml(
            "<testsuites name='yo' tests='3' failures='0' errors='1' time='1500'>",
            "  <testcase name='Goofy' status='run' classname='elite' time='1492'>",
            "     <failure message='bam'>boom</failure>",
            "  </testcase>",
            "</testsuites>");

    assertThat(result.getChildCount()).isEqualTo(1);
    // Other fields are not important here
    assertThat(result.getChild(0).getRunDurationMillis()).isEqualTo(1492);
  }

  @Test
  public void testFailuresPropagateToParents() throws Exception {
    TestCase result = parseXml(generateTestSuiteWithTestCase("<failure>Test failed.</failure>"));

    assertThat(result.getStatus()).isEqualTo(TestCase.Status.FAILED);
  }

  @Test
  public void testErrorsPropagateToParents() throws Exception {
    TestCase result = parseXml(generateTestSuiteWithTestCase("<error>Test errored.</error>"));

    assertThat(result.getStatus()).isEqualTo(TestCase.Status.ERROR);
  }

  @Test
  public void testUnknownEncoding() throws Exception {
    assertThrows(
        TestXmlOutputParserException.class,
        () ->
            parseXml(
                "<?xml version=\"1.0\" encoding=\"ISO8859_1\" ?>",
                "<testsuites name='o' tests='0' failures='0' errors='0' time='0'/>"));
  }

  @Test
  public void testTime() throws Exception {
    String xmlStringTemplate =
        "<testsuites name='qqq' tests='1' failures='0' errors='0' time='%s'></testsuites>";
    assertThat(parseXml(String.format(xmlStringTemplate, "12.34")).getRunDurationMillis())
        .isEqualTo(12340L);
    assertThat(parseXml(String.format(xmlStringTemplate, "1e0")).getRunDurationMillis())
        .isEqualTo(1000L);
    assertThat(parseXml(String.format(xmlStringTemplate, "1.2E3")).getRunDurationMillis())
        .isEqualTo(1200_000L);
    assertThat(parseXml(String.format(xmlStringTemplate, "12")).getRunDurationMillis())
        .isEqualTo(12L);
  }

  private String concatStringArray(String... args) {
    return Joiner.on("\n").join(args);
  }

  private TestCase parseXml(String... args) throws Exception {
    TestXmlOutputParser parser = new TestXmlOutputParser();
    ByteArrayInputStream bytes = new ByteArrayInputStream(concatStringArray(args).getBytes(UTF_8));

    return parser.parseXmlIntoTestResult(bytes);
  }

  private String generateTestSuiteWithTestCase(String... args) {
    String header =
        concatStringArray(
            "<testsuites name='1' tests='1' failures='0' errors='0' time='0.02'>",
            "  <testsuite name='apple' tests='1' failures='0' errors='0' time='0.02'>",
            "    <testcase name='apple' status='run' classname='Apple' time='0.01'>");

    String footer = concatStringArray("    </testcase>", "  </testsuite>", "</testsuites>");

    return header + concatStringArray(args) + footer;
  }
}
