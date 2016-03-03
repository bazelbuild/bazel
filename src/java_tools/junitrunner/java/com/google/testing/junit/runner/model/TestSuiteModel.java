// Copyright 2010 The Bazel Authors. All Rights Reserved.
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

import static com.google.common.base.Predicates.notNull;
import static com.google.common.collect.Maps.filterValues;
import static com.google.common.collect.Maps.transformValues;
import static java.util.concurrent.TimeUnit.NANOSECONDS;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Ticker;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.testing.junit.junit4.runner.DynamicTestException;
import com.google.testing.junit.runner.sharding.ShardingEnvironment;
import com.google.testing.junit.runner.sharding.ShardingFilters;
import com.google.testing.junit.runner.util.TestPropertyRunnerIntegration;

import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

import java.io.IOException;
import java.io.OutputStream;
import java.io.StringWriter;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.annotation.Nullable;
import javax.inject.Inject;

/**
 * Model of the tests that will be run. The model is agnostic of the particular
 * type of test run (JUnit3 or JUnit4). The test runner uses this class to build
 * the model, and then updates the model during the test run.
 *
 * <p>The leaf nodes in the model are test cases; the other nodes are test suites.
 */
public class TestSuiteModel {
  private final TestSuiteNode rootNode;
  private final ImmutableMap<Description, TestCaseNode> testCaseMap;
  private final ImmutableMap<Description, TestNode> testsMap;
  private final Ticker ticker;
  private final AtomicBoolean wroteXml = new AtomicBoolean(false);
  private final XmlResultWriter xmlResultWriter;
  @Nullable private final Filter shardingFilter;

  private TestSuiteModel(Builder builder) {
    rootNode = builder.rootNode;
    testsMap = ImmutableMap.copyOf(builder.testsMap);
    testCaseMap = ImmutableMap.copyOf(filterTestCases(builder.testsMap));
    ticker = builder.ticker;
    shardingFilter = builder.shardingFilter;
    xmlResultWriter = builder.xmlResultWriter;
  }

  @VisibleForTesting
  public List<TestNode> getTopLevelTestSuites() {
    return rootNode.getChildren();
  }

  /**
   * Gets the sharding filter to use; {@link Filter#ALL} if not sharding.
   */
  public Filter getShardingFilter() {
    return shardingFilter;
  }

  /**
   * Returns the test case node with the given test description.<p>
   *
   * Note that in theory this should never return {@code null}, but
   * if it did we would not want to throw a {@code NullPointerException}
   * because JUnit4 would catch the exception and remove our test
   * listener!
   */
  private TestCaseNode getTestCase(Description description) {
    return testCaseMap.get(description);
  }

  private TestNode getTest(Description description) {
    return testsMap.get(description);
  }

  @VisibleForTesting
  public int getNumTestCases() {
    return testCaseMap.size();
  }

  /**
   * Indicate that the test case with the given key has started.
   *
   * @param description key for a test case
   */
  public void testStarted(Description description) {
    TestCaseNode testCase = getTestCase(description);
    if (testCase != null) {
      testCase.started(currentMillis());
      TestPropertyRunnerIntegration.setTestCaseForThread(testCase);
    }
  }

  /**
   * Indicate that the entire test run was interrupted.
   */
  public void testRunInterrupted() {
    rootNode.testInterrupted(currentMillis());
  }

  /**
   * Indicate that the test case with the given key has requested that
   * a property be written in the XML.<p>
   *
   * @param description key for a test case
   * @param name The property name.
   * @param value The property value.
   */
  public void testEmittedProperty(Description description, String name, String value) {
    TestCaseNode testCase = getTestCase(description);
    if (testCase != null) {
      testCase.exportProperty(name, value);
    }
  }

  /**
   * Adds a failure to the test with the given key. If the specified test is suite, the failure
   * will be added to all its children.
   *
   * @param description key for a test case
   */
  public void testFailure(Description description, Throwable throwable) {
    TestNode test = getTest(description);
    if (test != null) {
      if (throwable instanceof DynamicTestException) {
        DynamicTestException dynamicFailure = (DynamicTestException) throwable;
        test.dynamicTestFailure(
            dynamicFailure.getTest(), dynamicFailure.getCause(), currentMillis());
      } else {
        test.testFailure(throwable, currentMillis());
      }
    }
  }

  /**
   * Indicates that the test case with the given key was skipped
   *
   * @param description key for a test case
   */
  public void testSkipped(Description description) {
    TestNode test = getTest(description);
    if (test != null) {
      test.testSkipped(currentMillis());
    }
  }
  

  /**
   * Indicates that the test case with the given key was ignored or suppressed
   *
   * @param description key for a test case
   */
  public void testSuppressed(Description description) {
    TestNode test = getTest(description);
    if (test != null) {
      test.testSuppressed(currentMillis());
    }
  }

  /**
   * Indicate that the test case with the given description has finished.
   */
  public void testFinished(Description description) {
    TestCaseNode testCase = getTestCase(description);
    if (testCase != null) {
      testCase.finished(currentMillis());
    }

    /*
     * Note: we don't call TestPropertyExporter, so if any properties are
     * exported before the next test runs, they will be associated with the
     * current test.
     */
  }

  private long currentMillis() {
    return NANOSECONDS.toMillis(ticker.read());
  }

  /**
   * Writes the model to XML
   *
   * @param outputStream stream to output to
   * @throws IOException if the underlying writer throws an exception
   */
  public void writeAsXml(OutputStream outputStream) throws IOException {
    write(new XmlWriter(outputStream));
  }

  @VisibleForTesting
  void write(XmlWriter writer) throws IOException {
    if (wroteXml.compareAndSet(false, true)) {
      xmlResultWriter.writeTestSuites(writer, rootNode.getResult());
    }
  }

  @Override
  public int hashCode() {
    return toString().hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof TestSuiteModel)) {
      return false;
    }
    TestSuiteModel that = (TestSuiteModel) obj;

    // We only use this for testing, so using toString() is good enough
    return this.toString().equals(that.toString());
  }

  @Override
  public String toString() {
    try {
      StringWriter stringWriter = new StringWriter();
      write(XmlWriter.createForTesting(stringWriter));
      return stringWriter.toString();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * A builder for creating a model of a test suite.
   */
  public static class Builder {
    private final Ticker ticker;
    private final Map<Description, TestNode> testsMap = new ConcurrentHashMap<>();
    private final ShardingEnvironment shardingEnvironment;
    private final ShardingFilters shardingFilters;
    private final XmlResultWriter xmlResultWriter;
    private TestSuiteNode rootNode;
    private Filter shardingFilter = Filter.ALL;
    private boolean buildWasCalled = false;

    @Inject
    public Builder(Ticker ticker, ShardingFilters shardingFilters,
        ShardingEnvironment shardingEnvironment, XmlResultWriter xmlResultWriter) {
      this.ticker = ticker;
      this.shardingFilters = shardingFilters;
      this.shardingEnvironment = shardingEnvironment;
      this.xmlResultWriter = xmlResultWriter;
    }

    public TestSuiteModel build(String suiteName, Description... topLevelSuites) {
      Preconditions.checkState(!buildWasCalled, "Builder.build() was already called");
      buildWasCalled = true;
      if (shardingEnvironment.isShardingEnabled()) {
        shardingFilter = getShardingFilter(topLevelSuites);
      }
      rootNode = new TestSuiteNode(Description.createSuiteDescription(suiteName));
      for (Description topLevelSuite : topLevelSuites) {
        addTestSuite(rootNode, topLevelSuite);
      }
      return new TestSuiteModel(this);
    }

    private Filter getShardingFilter(Description... topLevelSuites) {
      Collection<Description> tests = Lists.newLinkedList();
      for (Description suite : topLevelSuites) {
        collectTests(suite, tests);
      }
      shardingEnvironment.touchShardFile();
      return shardingFilters.createShardingFilter(tests);
    }

    private static void collectTests(Description desc, Collection<Description> tests) {
      if (desc.isTest()) {
        tests.add(desc);
      } else {
        for (Description child : desc.getChildren()) {
          collectTests(child, tests);
        }
      }
    }

    private void addTestSuite(TestSuiteNode parentSuite, Description suiteDescription) {
      TestSuiteNode suite = new TestSuiteNode(suiteDescription);
      for (Description childDesc : suiteDescription.getChildren()) {
        if (childDesc.isTest()) {
          addTestCase(suite, childDesc);
        } else {
          addTestSuite(suite, childDesc);
        }
      }
      // Empty suites are pruned when sharding.
      if (shardingFilter == Filter.ALL || !suite.getChildren().isEmpty()) {
        parentSuite.addTestSuite(suite);
        testsMap.put(suiteDescription, suite);
      }
    }

    private void addTestCase(TestSuiteNode parentSuite, Description testCaseDesc) {
      Preconditions.checkArgument(testCaseDesc.isTest());
      if (!shardingFilter.shouldRun(testCaseDesc)) {
        return;
      }
      TestCaseNode testCase = new TestCaseNode(testCaseDesc, parentSuite);
      testsMap.put(testCaseDesc, testCase);
      parentSuite.addTestCase(testCase);
    }
  }

  private static Map<Description, TestCaseNode> filterTestCases(Map<Description, TestNode> tests) {
    return filterValues(transformValues(tests, toTestCaseNode()), notNull());
  }

  private static Function<TestNode, TestCaseNode> toTestCaseNode() {
    return new Function<TestNode, TestCaseNode>() {
      @Override
      public TestCaseNode apply(TestNode test) {
        return test instanceof TestCaseNode ? (TestCaseNode) test : null;
      }
    };
  }
}
