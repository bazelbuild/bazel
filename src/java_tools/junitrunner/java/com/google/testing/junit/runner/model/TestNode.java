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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;

import org.junit.runner.Description;

import java.util.List;

import javax.annotation.Nullable;

/**
 * A node in a test suite.
 */
public abstract class TestNode {
  private final Description description;
  @Nullable private TestResult result = null;

  TestNode(Description description) {
    this.description = Preconditions.checkNotNull(description);
  }

  /**
   * {@link Description} of this test node.
   */
  public final Description getDescription() {
    return description;
  }

  /**
   * Returns this node's children (test suites or tests cases).
   */
  @VisibleForTesting
  public abstract List<TestNode> getChildren();

  /**
   * Returns true if this node is a test case (e.g. junit4 test), false otherwise (e.g. junit4 test
   * suite). The {@link TestSuiteModel} distinguishes between test cases and suites based on the
   * value returned by {@link Description#isTest()}.
   */
  public abstract boolean isTestCase();

  /**
   * Indicates that the test represented by this node was skipped.
   */
  public abstract void testSkipped(long now);

  /**
   * Indicates that the test represented by this node was ignored or suppressed due to being 
   * annotated with {@code @Ignore} or {@code @Suppress}.
   */
  public abstract void testSuppressed(long now);

  /**
   * Indicates that the test represented by this node was interrupted.
   */
  public abstract void testInterrupted(long now);

  /**
   * Adds a failure to the test represented by this node.
   */
  public abstract void testFailure(Throwable throwable, long now);

  /**
   * Indicates that a dynamically generated test case or suite failed.
   */
  public abstract void dynamicTestFailure(Description test, Throwable throwable, long now);

  /**
   * Template-method that creates a {@link TestResult} object that represents the test outcome of 
   * this node.
   */
  protected abstract TestResult buildResult();

  final TestResult getResult() {
    if (result == null) {
      result = buildResult();
    }
    return result;
  }
}
