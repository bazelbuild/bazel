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

package com.google.testing.junit.runner.junit4;

import com.google.testing.junit.runner.model.TestSuiteModel;
import java.util.function.Supplier;
import org.junit.runner.Description;
import org.junit.runner.Request;

/**
 * Builds a {@link TestSuiteModel} for JUnit4 tests.
 */
class JUnit4TestModelBuilder implements Supplier<TestSuiteModel> {
  private final Request request;
  private final String suiteName;
  private final TestSuiteModel.Builder builder;

  public JUnit4TestModelBuilder(Request request, String suiteName, TestSuiteModel.Builder builder) {
    this.request = request;
    this.suiteName = suiteName;
    this.builder = builder;
  }

  /**
   * Creates a model for a JUnit4 suite. This can be expensive; callers should
   * consider memoizing the result.
   *
   * @return model.
   */
  @Override
  public TestSuiteModel get() {
    Description root = request.getRunner().getDescription();
    // A test class annotated with @Ignore effectively has no test methods,
    // which is what isSuite() tests for.
    if (!root.isSuite()) {
      return builder.build(suiteName);
    } else {
      return builder.build(suiteName, root);
    }
  }
}
