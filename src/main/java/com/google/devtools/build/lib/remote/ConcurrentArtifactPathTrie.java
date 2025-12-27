package com.google.devtools.build.lib.remote;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.concurrent.ConcurrentSkipListSet;

/**
 * A specialized concurrent trie that stores paths of artifacts and allows checking whether a given
 * path is contained in (in the case of a tree artifact) or exactly matches (in any other case) an
 * artifact in the trie.
 */
final class ConcurrentArtifactPathTrie {
  // Invariant: no path in this set is a prefix of another path.
  private final ConcurrentSkipListSet<PathFragment> paths =
      new ConcurrentSkipListSet<>(PathFragment.HIERARCHICAL_COMPARATOR);

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
    paths.add(input.getExecPath());
  }

  /** Checks whether the given {@link PathFragment} is contained in an artifact in the trie. */
  boolean contains(PathFragment execPath) {
    // By the invariant of this set, there is at most one prefix of execPath in the set. Since the
    // comparator sorts all children of a path right after the path itself, if such a prefix
    // exists, it must thus sort right before execPath (or be equal to it).
    var floorPath = paths.floor(execPath);
    return floorPath != null && execPath.startsWith(floorPath);
  }
}
