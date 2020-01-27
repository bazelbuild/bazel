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

import org.junit.runner.Request;
import org.junit.runner.Runner;
import org.junit.runner.manipulation.Filter;
import org.junit.runner.manipulation.NoTestsRemainException;

/**
 * Common filters.
 */
@Deprecated
public final class Filters {

  private Filters() {}

  /**
   * Returns a filter that evaluates to {@code true} if both of its
   * components evaluates to {@code true}. The filters are evaluated in
   * order, and evaluation will be "short-circuited" if the first filter
   * returns {@code false}.
   */
  public static Filter and(Filter delegate1, Filter delegate2) {
    return delegate1 == Filter.ALL ? delegate2
        : (delegate2 == Filter.ALL ? delegate1
            : new AndFilter(delegate1, delegate2));
  }

  /**
   * Returns a Request that only contains those tests that should run when
   * a filter is applied, filtering out all empty suites.<p>
   *
   * Note that if the request passed into this method caches its runner,
   * that runner will be modified to use the given filter. To be safe,
   * do not use the passed-in request after calling this method.
   *
   * @param request Request to filter
   * @param filter Filter to apply
   * @return request
   * @throws NoTestsRemainException if the applying the filter removes all tests
   */
  public static Request apply(Request request, Filter filter) throws NoTestsRemainException {
    filter = new SuiteTrimmingFilter(filter);
    Runner runner = request.getRunner();
    filter.apply(runner);

    return Request.runner(runner);
  }
}
