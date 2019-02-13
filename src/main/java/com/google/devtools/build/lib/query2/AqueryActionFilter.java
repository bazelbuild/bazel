// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Multimap;
import java.util.regex.Pattern;

/** Encapsulate the action filters parsed from aquery command. */
public class AqueryActionFilter {
  // TODO(leba): Use Enum for list of filters.
  private final ImmutableMultimap<String, Pattern> filterMap;

  private AqueryActionFilter(Builder builder) {
    filterMap = ImmutableMultimap.copyOf(builder.filterMap);
  }

  public static AqueryActionFilter emptyInstance() {
    return builder().build();
  }

  public static Builder builder() {
    return new Builder();
  }

  public boolean hasFilterForFunction(String function) {
    return filterMap.containsKey(function);
  }

  /**
   * Returns whether the input string matches ALL the filter patterns of a specific type parsed from
   * aquery command.
   *
   * @param function the name of the aquery function (inputs, outputs, mnemonic)
   * @param input the string to be matched against
   */
  public boolean matchesAllPatternsForFunction(String function, String input) {
    if (!hasFilterForFunction(function)) {
      return false;
    }

    return filterMap.get(function).stream()
        .map(pattern -> pattern.matcher(input).matches())
        .reduce(true, Boolean::logicalAnd);
  }

  /** Builder class for {@code AqueryActionFilter} */
  public static class Builder {
    private final Multimap<String, Pattern> filterMap;

    public Builder() {
      filterMap = HashMultimap.create();
    }

    public Builder put(String key, Pattern value) {
      filterMap.put(key, value);
      return this;
    }

    public AqueryActionFilter build() {
      return new AqueryActionFilter(this);
    }
  }
}
