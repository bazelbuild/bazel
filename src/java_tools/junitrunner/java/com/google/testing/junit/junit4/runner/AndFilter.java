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
 * A filter that returns {@code true} if both of its components return {@code
 * true}.
 */
@Deprecated
class AndFilter extends Filter {
  private final Filter filter1;
  private final Filter filter2;

  public AndFilter(Filter filter1, Filter filter2) {
    this.filter1 = checkNotNull(filter1);
    this.filter2 = checkNotNull(filter2);
  }

  @Override
  public boolean shouldRun(Description description) {
    return filter1.shouldRun(description) && filter2.shouldRun(description);
  }
  
  @Override
  public String describe() {
    return String.format("%s && %s", filter1.describe(), filter2.describe());
  }
}
