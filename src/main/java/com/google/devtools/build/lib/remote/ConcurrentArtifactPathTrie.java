// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.concurrent.ConcurrentSkipListMap;

/**
 * A specialized concurrent trie that stores paths of artifacts and allows checking whether a given
 * path is contained in (in the case of a tree artifact) or exactly matches (in any other case) an
 * artifact in the trie.
 *
 * <p>Paths are reference counted: a path is only absent from the trie once it has been {@link
 * #remove}d as many times as it has been {@link #add}ed. Distinct actions can contribute the same
 * path, e.g. the expansion actions of an action template, which all report their common parent tree
 * artifact.
 */
final class ConcurrentArtifactPathTrie {
  // Invariant: no path in this map is a prefix of another path. Values are positive reference
  // counts.
  private final ConcurrentSkipListMap<PathFragment, Integer> paths =
      new ConcurrentSkipListMap<>(PathFragment.HIERARCHICAL_COMPARATOR);

  /**
   * Adds the given {@link ActionInput} to the trie.
   *
   * <p>The caller must ensure that no object's path passed to this method is a prefix of any
   * previously added object's path. Bazel enforces this for non-aggregate artifacts. Callers must
   * not pass in {@link Artifact.TreeFileArtifact}s (which have exec paths that have their parent
   * tree artifact's exec path as a prefix) or non-Artifact {@link ActionInput}s that violate this
   * invariant.
   */
  void add(ActionInput input) {
    Preconditions.checkArgument(
        !(input instanceof Artifact.TreeFileArtifact),
        "TreeFileArtifacts should not be added to the trie: %s",
        input);
    paths.merge(input.getExecPath(), 1, Integer::sum);
  }

  /**
   * Drops one reference to the given {@link ActionInput}, removing it from the trie if this was the
   * last one.
   */
  void remove(ActionInput input) {
    paths.computeIfPresent(input.getExecPath(), (path, count) -> count == 1 ? null : count - 1);
  }

  /** Checks whether the given {@link PathFragment} is contained in an artifact in the trie. */
  boolean contains(PathFragment execPath) {
    if (paths.isEmpty()) {
      return false;
    }
    // By the invariant of this map, there is at most one prefix of execPath in it. Since the
    // comparator sorts all children of a path right after the path itself, if such a prefix
    // exists, it must thus sort right before execPath (or be equal to it).
    var floorPath = paths.floorKey(execPath);
    return floorPath != null && execPath.startsWith(floorPath);
  }
}
