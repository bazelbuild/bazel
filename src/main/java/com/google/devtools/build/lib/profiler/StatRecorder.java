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
package com.google.devtools.build.lib.profiler;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableMap;

import java.util.Map;

/** An object that can record time statistics about an object. */
public interface StatRecorder {

  /** Add a new time statistic for the object {@code obj}. */
  void addStat(int duration, Object obj);

  /** True if it has not recorded any statistic */
  boolean isEmpty();

  /** A collection of heuristics for VFS kind of stats in order to detect the filesystem type. */
  final class VfsHeuristics {

    private VfsHeuristics() {}

    static Map<String, ? extends Predicate<? super String>> vfsTypeHeuristics =
        ImmutableMap.of(
            "blaze-out", Predicates.containsPattern("/blaze-out/"),
            "source", Predicates.<CharSequence>alwaysTrue());


    public static void setVfsTypeHeuristics(Map<String, ? extends Predicate<? super String>> map) {
      vfsTypeHeuristics = map;
    }
  }
}
