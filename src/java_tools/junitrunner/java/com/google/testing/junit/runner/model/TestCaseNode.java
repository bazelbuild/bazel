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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.testing.junit.runner.util.TestPropertyExporter.INITIAL_INDEX_FOR_REPEATED_PROPERTY;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimaps;
import com.google.common.collect.Multiset;
import com.google.testing.junit.runner.model.TestResult.Status;
import com.google.testing.junit.runner.util.TestPropertyExporter;

import org.joda.time.Interval;
import org.junit.runner.Description;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import javax.annotation.Nullable;

/**
 * A leaf in the test suite model.
 */
class TestCaseNode extends TestNode implements TestPropertyExporter.Callback {
  private final TestSuiteNode parent;
  private final Map<String, String> properties = Maps.newConcurrentMap();
  private final Multiset<String> repeatedPropertyNames = ConcurrentHashMultiset.create();
  private final Queue<Throwable> globalFailures = new ConcurrentLinkedQueue<>();
  private final ListMultimap<Description, Throwable> dynamicTestToFailures =
      Multimaps.synchronizedListMultimap(LinkedListMultimap.<Description, Throwable>create());

  @Nullable private volatile Interval runTimeInterval = null;
  private volatile State state = State.INITIAL;

  TestCaseNode(Description description, TestSuiteNode parent) {
    super(description);
    this.parent = parent;
  }

  @VisibleForTesting
  @Override
  public List<TestNode> getChildren() {
    return Collections.emptyList();
  }


  /**
   * Indicates that the test represented by this node has started.
   *
   * @param now Time that the test started
   */
  public void started(long now) {
    compareAndSetState(State.INITIAL, State.STARTED, now);
  }
  
  @Override
  public void testInterrupted(long now) {
    if (compareAndSetState(State.STARTED, State.INTERRUPTED, now)) {
      return;
    }
    compareAndSetState(State.INITIAL, State.CANCELLED, now);
  }

  @Override
  public void exportProperty(String name, String value) {
    properties.put(name, value);
  }

  @Override
  public String exportRepeatedProperty(String name, String value) {
    String propertyName = getRepeatedPropertyName(name);
    properties.put(propertyName, value);
    return propertyName;
  }

  @Override
  public void testSkipped(long now) {
    compareAndSetState(State.STARTED, State.SKIPPED, now);
  }
  

  @Override
  public void testSuppressed(long now) {
    compareAndSetState(State.INITIAL, State.SUPPRESSED, now);
  }

  /**
   * Indicates that the test represented by this node has finished.
   *
   * @param now Time that the test finished
   */
  public void finished(long now) {
    compareAndSetState(State.STARTED, State.FINISHED, now);
  }
  
  @Override
  public void testFailure(Throwable throwable, long now) {
    compareAndSetState(State.INITIAL, State.FINISHED, now);
    globalFailures.add(throwable);
  }
  
  @Override
  public void dynamicTestFailure(Description test, Throwable throwable, long now) {
    compareAndSetState(State.INITIAL, State.FINISHED, now);
    dynamicTestToFailures.put(test, throwable);
  }

  private String getRepeatedPropertyName(String name) {
    int index = repeatedPropertyNames.add(name, 1) + INITIAL_INDEX_FOR_REPEATED_PROPERTY;
    return name + index;
  }

  @Override
  public boolean isTestCase() {
    return true;
  }

  private synchronized boolean compareAndSetState(State fromState, State toState, long now) {
    if (fromState == state && toState != checkNotNull(state)) {
      state = toState;
      runTimeInterval = runTimeInterval == null 
          ? new Interval(now, now) : runTimeInterval.withEndMillis(now);
      return true;
    }
    return false;
  }

  public Optional<Interval> getRuntime() {
    return Optional.fromNullable(runTimeInterval);
  }

  /**
   * @return The equivalent {@link TestResult.Status} if the test execution ends with the FSM
   *      at this state.
   */
  public TestResult.Status getTestResultStatus() {
    return state.getTestResultStatus();
  }

  @Override
  protected TestResult buildResult() {
    // Some test descriptions, like those provided by JavaScript tests, are
    // constructed by Description.createSuiteDescription, not
    // createTestDescription, because they don't have a "class" per se.
    // In this case, getMethodName returns null and we fill in the className
    // attribute with the name of the parent test suite.
    String name = getDescription().getMethodName();
    String className = getDescription().getClassName();
    if (name == null) {
      name = className;
      className = parent.getDescription().getDisplayName();
    }

    // For now, we give each dynamic test an empty properties map and the same
    // run time and status as its parent test case, but this may change.
    List<TestResult> childResults = Lists.newLinkedList();
    for (Description dynamicTest : getDescription().getChildren()) {
      childResults.add(buildDynamicResult(dynamicTest, getRuntime(), getTestResultStatus()));
    }

    int numTests = getDescription().isTest() ? 1 : getDescription().getChildren().size();
    int numFailures = globalFailures.isEmpty() ? dynamicTestToFailures.keySet().size() : numTests;
    return new TestResult.Builder()
        .name(name)
        .className(className)
        .properties(properties)
        .failures(ImmutableList.copyOf(globalFailures))
        .runTimeInterval(getRuntime())
        .status(getTestResultStatus())
        .numTests(numTests)
        .numFailures(numFailures)
        .childResults(childResults)
        .build();
  }

  private TestResult buildDynamicResult(Description test, Optional<Interval> runTime,
      TestResult.Status status) {
    // The dynamic test fails if the testcase itself fails or there is
    // a dynamic failure specifically for the dynamic test.
    List<Throwable> dynamicFailures = dynamicTestToFailures.get(test);
    boolean failed = !globalFailures.isEmpty() || !dynamicFailures.isEmpty();
    return new TestResult.Builder()
        .name(test.getDisplayName())
        .className(getDescription().getDisplayName())
        .properties(ImmutableMap.<String, String>of())
        .failures(dynamicFailures)
        .runTimeInterval(runTime)
        .status(status)
        .numTests(1)
        .numFailures(failed ? 1 : 0)
        .childResults(ImmutableList.<TestResult>of())
        .build();
  }

  /**
   * States of a TestCaseNode (see (link) for all the transitions and states descriptions).
   */
  private static enum State {
    INITIAL(TestResult.Status.SKIPPED), 
    STARTED(TestResult.Status.INTERRUPTED),
    SKIPPED(TestResult.Status.SKIPPED),
    SUPPRESSED(TestResult.Status.SUPPRESSED),
    CANCELLED(TestResult.Status.CANCELLED),
    INTERRUPTED(TestResult.Status.INTERRUPTED),
    FINISHED(TestResult.Status.COMPLETED);

    private final Status status;

    State(TestResult.Status status) {
      this.status = status;
    }

    /**
     * @return The equivalent {@link TestResult.Status} if the test execution ends with the FSM
     *      at this state. 
     */
    public TestResult.Status getTestResultStatus() {
      return status;
    }
  }
}
