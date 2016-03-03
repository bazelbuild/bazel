// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * A value for ensuring that an error for a cycle/chain is reported exactly once. This is achieved
 * by forcing the same value key for two logically equivalent errors and letting Skyframe do its
 * magic.
 */
class ChainUniquenessUtils {

  private ChainUniquenessUtils() {}

  /**
   * Create a SkyKey for {@code functionName} with a canonicalized representation of the cycle
   * specified by {@code chain} as the argument. {@code chain} must be non-empty.
   */
  static SkyKey key(SkyFunctionName functionName, ImmutableList<? extends Object> chain) {
    Preconditions.checkState(!chain.isEmpty());
    return SkyKey.create(functionName, canonicalize(chain));
  }

  private static ImmutableList<Object> canonicalize(ImmutableList<? extends Object> cycle) {
    int minPos = 0;
    String minString = cycle.get(0).toString();
    for (int i = 1; i < cycle.size(); i++) {
      // TOOD(bazel-team): Is the toString representation stable enough?
      String candidateString = cycle.get(i).toString();
      if (candidateString.compareTo(minString) < 0) {
        minPos = i;
        minString = candidateString;
      }
    }
    ImmutableList.Builder<Object> builder = ImmutableList.builder();
    for (int i = 0; i < cycle.size(); i++) {
      int pos = (minPos + i) % cycle.size();
      builder.add(cycle.get(pos));
    }
    return builder.build();
  }
}

