// Copyright 2012 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.sharding.testing;

import com.google.testing.junit.runner.sharding.ShardingFilters;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

/**
 * Filter factory that includes only descriptions in the set of descriptions
 * explicitly specified in the constructor.
 */
public class FakeShardingFilters extends ShardingFilters {
  private final Set<Description> descriptionsToRun;

  public FakeShardingFilters(Description... descriptionsToRun) {
    super(null, null);
    this.descriptionsToRun = copyOf(descriptionsToRun);
  }

  @Override
  public Filter createShardingFilter(Collection<Description> allDescriptions) {
    return new ExplicitDescriptionFilter(allDescriptions, descriptionsToRun);
  }


  private static class ExplicitDescriptionFilter extends Filter {
    private final Set<Description> allDescriptions;
    private final Set<Description> descriptionsToRun;

    private ExplicitDescriptionFilter(
        Collection<Description> allDescriptions, Set<Description> descriptionsToRun) {
      this.allDescriptions = copyOf(allDescriptions);
      this.descriptionsToRun = descriptionsToRun;
    }

    @Override
    public boolean shouldRun(Description description) {
      if (description.isSuite()) {
        return true;
      }
      if (!allDescriptions.contains(description)) {
        throw new IllegalArgumentException("Not in the suite: " + description);
      }
      return descriptionsToRun.contains(description);
    }

    @Override
    public String describe() {
      return "explicit description filter";
    }
  }

  private static <T> Set<T> copyOf(T... items) {
    return copyOf(Arrays.asList(items));
  }

  private static <T> Set<T> copyOf(Collection<T> items) {
    return Collections.unmodifiableSet(new HashSet<T>(items));
  }
}
