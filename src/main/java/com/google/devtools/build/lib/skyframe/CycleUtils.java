// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.Pair;

/** Simple utility for separating path and cycle from a combined iterable. */
class CycleUtils {
  private CycleUtils() {}

  static <S> Pair<ImmutableList<S>, ImmutableList<S>> splitIntoPathAndChain(
      Predicate<S> startOfCycle, Iterable<S> pathAndCycle) {
    boolean inPathToCycle = true;
    ImmutableList.Builder<S> pathToCycleBuilder = ImmutableList.builder();
    ImmutableList.Builder<S> cycleBuilder = ImmutableList.builder();
    for (S elt : pathAndCycle) {
      if (startOfCycle.apply(elt)) {
        inPathToCycle = false;
      }
      if (inPathToCycle) {
        pathToCycleBuilder.add(elt);
      } else {
        cycleBuilder.add(elt);
      }
    }
    return Pair.of(pathToCycleBuilder.build(), cycleBuilder.build());
  }
}
