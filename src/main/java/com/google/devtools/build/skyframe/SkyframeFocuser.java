// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import java.util.HashSet;
import java.util.Set;

/**
 * SkyframeFocuser is a minimizing optimizer (i.e. garbage collector) for the Skyframe graph, based
 * on a set of known inputs known as a working set, while ensuring correct incremental builds.
 */
public final class SkyframeFocuser {

  private SkyframeFocuser() {}
  ;

  /**
   * The set of SkyKeys kept after focusing. The actual change is done in place with the in-memory
   * graph.
   */
  public static class FocusResult {

    private final ImmutableSet<SkyKey> rdeps;
    private final ImmutableSet<SkyKey> deps;

    private FocusResult(ImmutableSet<SkyKey> rdeps, ImmutableSet<SkyKey> deps) {
      this.rdeps = rdeps;
      this.deps = deps;
    }

    /**
     * Returns the set of SkyKeys that are in the direct dependencies of all roots, and rdeps from
     * the leaves.
     */
    public ImmutableSet<SkyKey> getDeps() {
      return deps;
    }

    /** Returns the set of SkyKeys that are in the reverse dependencies of the leaves. */
    public ImmutableSet<SkyKey> getRdeps() {
      return rdeps;
    }
  }

  /**
   * Minimize the Skyframe graph by traverse it to prune nodes and edges that are not necessary for
   * the build correctness of a working set of files.
   *
   * @param graph the in-memory skyframe graph.
   * @return the set of kept skykeys in the in-memory graph, categorized by deps and rdeps.
   */
  public static FocusResult focus(InMemoryGraph graph, EventHandler eventHandler) {
    Set<SkyKey> keptDeps = new HashSet<>();
    Set<SkyKey> keptRdeps = new HashSet<>();

    eventHandler.handle(
        Event.info(String.format("Skyframe graph has %d nodes", graph.getAllNodeEntries().size())));

    /*
      TODO: b/312819241 - implement graph focusing. The pseudocode is as follows.

      1. Add all leaves into keptRdeps.
      2. Add all roots into keptDeps.
      3. (bottom-up mark phase) For each keptRdep in keptRdeps:
        a) add all direct rdeps of keptRdep into keptRdeps.
        b) add all direct deps of keptRdep into keptDeps.
        c) go back to 3 until keptRdeps is empty.
      4. (node sweep) drop all nodes in the InMemoryGraph nodeMap except for
         union(keptDeps, keptRdeps).
      5. remove all keptRdeps from keptDeps.
      6. (edge sweep) for each keptDep in keptDeps:
        a) remove all direct deps of keptDep.
        b) remove all direct rdeps of keptDep that are not in keptRdeps.
    */

    return new FocusResult(ImmutableSet.copyOf(keptRdeps), ImmutableSet.copyOf(keptDeps));
  }
}
