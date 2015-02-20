// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.query2.engine;

import com.google.common.base.Preconditions;

import java.util.Set;

/**
 * The result of a query evaluation, containing a set of elements.
 *
 * @param <T> the node type of the elements.
 */
public class QueryEvalResult<T> {

  protected final boolean success;
  protected final Set<T> resultSet;

  public QueryEvalResult(
      boolean success, Set<T> resultSet) {
    this.success = success;
    this.resultSet = Preconditions.checkNotNull(resultSet);
  }

  /**
   * Whether the query was successful. This can only be false if the query was run with
   * <code>keep_going</code>, otherwise evaluation will throw a {@link QueryException}.
   */
  public boolean getSuccess() {
    return success;
  }

  /**
   * Returns the result as a set of targets.
   */
  public Set<T> getResultSet() {
    return resultSet;
  }

  @Override
  public String toString() {
    return (getSuccess() ? "Successful" : "Unsuccessful") + ", result size = "
        + getResultSet().size() + ", " + getResultSet();
  }
}
