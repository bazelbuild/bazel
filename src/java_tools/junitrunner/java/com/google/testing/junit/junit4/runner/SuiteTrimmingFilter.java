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

import static com.google.common.base.Preconditions.checkNotNull;

import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

/**
 * A filter that decorates another filter, filtering out any suites
 * that contain no tests.
 */
public final class SuiteTrimmingFilter extends Filter {
  private final Filter delegate;

  public SuiteTrimmingFilter(Filter delegate) {
    this.delegate = checkNotNull(delegate);
  }

  @Override
  public String describe() {
    return delegate.describe();
  }

  @Override
  public final boolean shouldRun(Description description) {
    if (!delegate.shouldRun(description)) {
      return false;
    }

    if (description.isTest()) {
      return true;
    }

    // explicitly check if any children want to run
    for (Description each : description.getChildren()) {
      if (shouldRun(each)) {
        return true;
      }
    }
    return false;           
  }
}
