// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.rules.test;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.view.test.TestStatus.FailedTestCaseDetails;
import com.google.devtools.build.lib.view.test.TestStatus.TestCaseDetail;

import java.util.ArrayList;
import java.util.List;

/**
 * Class for storing test results from the parsed xml before putting then into an
 * actual TestResultData proto.
 */

@Immutable
public class HierarchicalTestResult {
  private final List<HierarchicalTestResult> children;
  private final String name;
  private final String className;        // Not used for suites or decorators.
  private final long runDurationMillis;
  private final int failures;
  private final int errors;

  /**
   * Builder for HierarchicalTestResult. It is the only way to instantiate this class.
   */
  public static class Builder {
    private List<HierarchicalTestResult> children = new ArrayList<>();
    private String name;
    private String className;
    private long duration;
    private int failures;
    private int errors;

    public HierarchicalTestResult build() {
      HierarchicalTestResult result = new HierarchicalTestResult(this.name,
          this.className,
          this.duration,
          this.failures,
          this.errors,
          ImmutableList.copyOf(this.children));
      return result;
    }

    public Builder setName(String name) {
      this.name = name;
      return this;
    }

    public Builder setClassName(String className) {
      this.className = className;
      return this;
    }

    public Builder setRunDurationMillis(long duration) {
      this.duration = duration;
      return this;
    }

    public void addChild(HierarchicalTestResult result) {
      this.children.add(result);
    }

    public Builder incrementFailures() {
      this.failures += 1;
      return this;
    }

    public Builder incrementErrors() {
      this.errors += 1;
      return this;
    }
  }

  private HierarchicalTestResult(String name,
      String className,
      long duration,
      int failures,
      int errors,
      List<HierarchicalTestResult> children) {
    this.name = name;
    this.className = className;
    this.runDurationMillis = duration;
    this.failures = failures;
    this.errors = errors;
    this.children = ImmutableList.copyOf(children);
  }

  public static Builder newBuilder() {
    return new Builder();
  }

  public FailedTestCaseDetails collectFailedTestCases() {
    FailedTestCaseDetails.Builder builder = FailedTestCaseDetails.newBuilder();
    // will be overwritten if something goes wrong.
    builder.setStatus(FailedTestCaseDetails.Status.FULL);
    collectFailedTestCases(builder);
    return builder.build();
  }

  private void collectFailedTestCases(FailedTestCaseDetails.Builder testCaseDetails) {
    if (!this.children.isEmpty()) {
      // This is a non-leaf result. Traverse its children, but do not add its
      // name to the output list. It should not contain any 'failure' or
      // 'error' tags, but we want to be lax here, because the syntax of the
      // test.xml file is also lax.
      for (HierarchicalTestResult child : this.children) {
        child.collectFailedTestCases(testCaseDetails);
      }
    } else {
      // This is a leaf result. If there was a failure or an error, return it.
      boolean passed = this.failures == 0
          && this.errors == 0;
      if (passed) {
        return;
      }

      String name = this.name;
      String className = this.className;
      if (name == null || className == null) {
        // A test case detail is not really interesting if we cannot tell which
        // one it is.
        testCaseDetails.setStatus(FailedTestCaseDetails.Status.PARTIAL);
        return;
      }

      // TODO(bazel-team): The dot separator is only correct for Java.
      String testCaseName = className + "." + name;

      testCaseDetails.addDetail(TestCaseDetail.newBuilder()
          .setName(testCaseName)
          .setStatus(this.errors > 0
              ? TestCaseDetail.Status.ERROR : TestCaseDetail.Status.FAILED)
          .setRunDurationMillis(this.runDurationMillis)
          .build());
    }
  }
}