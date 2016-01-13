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

package com.google.testing.junit.junit4.runner;

import com.google.common.base.Preconditions;

import org.junit.runner.Description;

/**
 * The test runner may throw a {@code DynamicTestFailureException} to indicate a
 * test case failed due to a failure in a dynamically-discovered test within
 * a JUnit test case.
 */
public class DynamicTestException extends Exception {
  private final Description test;

  /**
   * Constructs a {@code DynamicTestFailureException} that indicates a
   * dynamically-discovered test, specified as a (@link Description}, failed
   * due to the specified {@code cause}.
   */
  public DynamicTestException(Description test, Throwable cause) {
    super(cause);
    Preconditions.checkArgument(test.isTest());
    this.test = test;
  }

  /**
   * Returns the description of the dynamically-added test case.
   */
  public final Description getTest() {
    return test;
  }
}
