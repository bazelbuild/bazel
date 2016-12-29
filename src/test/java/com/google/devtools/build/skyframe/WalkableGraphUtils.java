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
package com.google.devtools.build.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;

/** Utility methods for querying {@link WalkableGraph}s more concisely. */
public class WalkableGraphUtils {

  public static Iterable<SkyKey> getDirectDeps(WalkableGraph graph, SkyKey key)
      throws InterruptedException {
    return Iterables.getOnlyElement(graph.getDirectDeps(ImmutableList.of(key)).values());
  }

  public static Iterable<SkyKey> getReverseDeps(WalkableGraph graph, SkyKey key)
      throws InterruptedException {
    return Iterables.getOnlyElement(graph.getReverseDeps(ImmutableList.of(key)).values());
  }

  public static boolean exists(SkyKey key, WalkableGraph graph) throws InterruptedException {
    return graph.getValue(key) != null || graph.getException(key) != null || graph.isCycle(key);
  }
}
