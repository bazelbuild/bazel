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

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import javax.inject.Inject;

/**
 * Writes the JUnit test nodes and their results into Ant-JUnit XML. Ant-JUnit XML is not a
 * standardized format. For this implementation the
 * <a href="http://windyroad.com.au/dl/Open%20Source/JUnit.xsd">XML schema</a> that is generally
 * referred to as the best available source was used as a reference.
 */
public final class AntXmlResultWriter implements XmlResultWriter {
  private static final String JUNIT_ELEMENT_TESTSUITES = "testsuites";
  private static final String JUNIT_ELEMENT_TESTSUITE = "testsuite";
  private static final String JUNIT_ELEMENT_TESTSUITE_PROPERTIES = "properties";
  private static final String JUNIT_ELEMENT_TESTSUITE_SYSTEM_OUT = "system-out";
  private static final String JUNIT_ELEMENT_TESTSUITE_SYSTEM_ERR = "system-err";
  private static final String JUNIT_ELEMENT_PROPERTY = "property";
  private static final String JUNIT_ELEMENT_TESTCASE = "testcase";
  private static final String JUNIT_ELEMENT_FAILURE = "failure";
  private static final String JUNIT_ELEMENT_SKIPPED = "skipped";

  private static final String JUNIT_ATTR_TESTSUITE_ERRORS = "errors";
  private static final String JUNIT_ATTR_TESTSUITE_FAILURES = "failures";
  private static final String JUNIT_ATTR_TESTSUITE_HOSTNAME = "hostname";
  private static final String JUNIT_ATTR_TESTSUITE_NAME = "name";
  private static final String JUNIT_ATTR_TESTSUITE_TESTS = "tests";
  private static final String JUNIT_ATTR_TESTSUITE_TIME = "time";
  private static final String JUNIT_ATTR_TESTSUITE_TIMESTAMP = "timestamp";
  private static final String JUNIT_ATTR_TESTSUITE_ID = "id";
  private static final String JUNIT_ATTR_TESTSUITE_PACKAGE = "package";
  private static final String JUNIT_ATTR_PROPERTY_NAME = "name";
  private static final String JUNIT_ATTR_PROPERTY_VALUE = "value";
  private static final String JUNIT_ATTR_FAILURE_MESSAGE = "message";
  private static final String JUNIT_ATTR_FAILURE_TYPE = "type";
  private static final String JUNIT_ATTR_TESTCASE_NAME = "name";
  private static final String JUNIT_ATTR_TESTCASE_CLASSNAME = "classname";
  private static final String JUNIT_ATTR_TESTCASE_TIME = "time";

  private int testSuiteId;

  @Inject
  public AntXmlResultWriter() {}

  @Override
  public void writeTestSuites(XmlWriter writer, TestResult result) throws IOException {
    testSuiteId = 0;
    writer.startDocument();
    writer.startElement(JUNIT_ELEMENT_TESTSUITES);
    for (TestResult child : result.getChildResults()) {
      writeTestSuite(writer, child, result.getFailures());
    }
    writer.endElement();
    writer.close();
  }

  private void writeTestSuite(XmlWriter writer, TestResult result,
      Iterable<Throwable> parentFailures)
      throws IOException {
    List<Throwable> allFailures = new ArrayList<>();
    for (Throwable failure : parentFailures) {
      allFailures.add(failure);
    }
    allFailures.addAll(result.getFailures());
    parentFailures = allFailures;

    writer.startElement(JUNIT_ELEMENT_TESTSUITE);

    writeTestSuiteAttributes(writer, result);
    writeTestSuiteProperties(writer, result);
    writeTestCases(writer, result, parentFailures);
    writeTestSuiteOutput(writer);

    writer.endElement();

    for (TestResult child : result.getChildResults()) {
      if (!child.getChildResults().isEmpty()) {
        writeTestSuite(writer, child, parentFailures);
      }
    }
  }

  private void writeTestSuiteProperties(XmlWriter writer, TestResult result) throws IOException {
    writer.startElement(JUNIT_ELEMENT_TESTSUITE_PROPERTIES);
    for (Map.Entry<String, String> entry : result.getProperties().entrySet()) {
      writer.startElement(JUNIT_ELEMENT_PROPERTY);
      writer.writeAttribute(JUNIT_ATTR_PROPERTY_NAME, entry.getKey());
      writer.writeAttribute(JUNIT_ATTR_PROPERTY_VALUE, entry.getValue());
      writer.endElement();
    }
    writer.endElement();
  }

  private void writeTestCases(XmlWriter writer, TestResult result,
      Iterable<Throwable> parentFailures) throws IOException {
    for (TestResult child : result.getChildResults()) {
      if (child.getStatus().equals(TestResult.Status.FILTERED)) {
        continue;
      }
      if (child.getChildResults().isEmpty()) {
        writeTestCase(writer, child, parentFailures);
      }
    }
  }

  private void writeTestSuiteOutput(XmlWriter writer) throws IOException {
    writer.startElement(JUNIT_ELEMENT_TESTSUITE_SYSTEM_OUT);
    // TODO(bazel-team) - where to get this from?
    writer.endElement();
    writer.startElement(JUNIT_ELEMENT_TESTSUITE_SYSTEM_ERR);
    // TODO(bazel-team) - where to get this from?
    writer.endElement();
  }

  private void writeTestSuiteAttributes(XmlWriter writer, TestResult result) throws IOException {
    writer.writeAttribute(JUNIT_ATTR_TESTSUITE_NAME, result.getName());
    writer.writeAttribute(JUNIT_ATTR_TESTSUITE_TIMESTAMP, getFormattedTimestamp(
        result.getRunTimeInterval()));
    writer.writeAttribute(JUNIT_ATTR_TESTSUITE_HOSTNAME, "localhost");
    writer.writeAttribute(JUNIT_ATTR_TESTSUITE_TESTS, result.getNumTests());
    writer.writeAttribute(JUNIT_ATTR_TESTSUITE_FAILURES, result.getNumFailures());
    // JUnit 4.x no longer distinguishes between errors and failures, so it should be safe to just
    // report errors as 0 and put everything into failures.
    writer.writeAttribute(JUNIT_ATTR_TESTSUITE_ERRORS, 0);
    writer.writeAttribute(JUNIT_ATTR_TESTSUITE_TIME, getFormattedRunTime(
        result.getRunTimeInterval()));
    // TODO(bazel-team) - do we want to report the package name here? Could we simply get it from
    // result.getClassName() by stripping the last element of the class name?
    writer.writeAttribute(JUNIT_ATTR_TESTSUITE_PACKAGE, "");
    writer.writeAttribute(JUNIT_ATTR_TESTSUITE_ID, this.testSuiteId++);
  }

  private static String getFormattedRunTime(@Nullable TestInterval runTimeInterval) {
    return runTimeInterval == null ? "0.0"
        : String.valueOf(runTimeInterval.toDurationMillis() / 1000.0D);
  }

  private static String getFormattedTimestamp(@Nullable TestInterval runTimeInterval) {
    return runTimeInterval == null ? "" : runTimeInterval.startInstantToString();
  }

  private void writeTestCase(XmlWriter writer, TestResult result,
      Iterable<Throwable> parentFailures)
      throws IOException {
    writer.startElement(JUNIT_ELEMENT_TESTCASE);
    writer.writeAttribute(JUNIT_ATTR_TESTCASE_NAME, result.getName());
    writer.writeAttribute(JUNIT_ATTR_TESTCASE_CLASSNAME, result.getClassName());
    writer.writeAttribute(JUNIT_ATTR_TESTCASE_TIME, getFormattedRunTime(
            result.getRunTimeInterval()));

    for (Throwable failure : parentFailures) {
      writeThrowableToXmlWriter(writer, failure);
    }

    for (Throwable failure : result.getFailures()) {
      writeThrowableToXmlWriter(writer, failure);
    }

    if (result.getStatus().equals(TestResult.Status.SKIPPED)
        || result.getStatus().equals(TestResult.Status.SUPPRESSED)) {
      writer.startElement(JUNIT_ELEMENT_SKIPPED);
      writer.endElement();
    }

    writer.endElement();
  }

  private static void writeThrowableToXmlWriter(XmlWriter writer, Throwable failure)
      throws IOException {
    writer.startElement(JUNIT_ELEMENT_FAILURE);
    writer.writeAttribute(
        JUNIT_ATTR_FAILURE_MESSAGE, (failure.getMessage() == null) ? "" : failure.getMessage());
    writer.writeAttribute(JUNIT_ATTR_FAILURE_TYPE, failure.getClass().getName());
    writer.writeCharacters(formatStackTrace(failure));
    writer.endElement();
  }

  private static String formatStackTrace(Throwable throwable) {
    StringWriter stringWriter = new StringWriter();
    PrintWriter writer = new PrintWriter(stringWriter);
    throwable.printStackTrace(writer);
    return stringWriter.getBuffer().toString();
  }
}
