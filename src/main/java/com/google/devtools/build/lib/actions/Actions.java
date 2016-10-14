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

package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.escape.Escaper;
import com.google.common.escape.Escapers;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

import javax.annotation.Nullable;

/**
 * Helper class for actions.
 */
@ThreadSafe
public final class Actions {
  private static final Escaper PATH_ESCAPER = Escapers.builder()
      .addEscape('_', "_U")
      .addEscape('/', "_S")
      .addEscape('\\', "_B")
      .addEscape(':', "_C")
      .build();

  /**
   * Checks if the two actions are equivalent. This method exists to support sharing actions between
   * configured targets for cases where there is no canonical target that could own the action. In
   * the action graph construction this case shows up as two actions generating the same output
   * file.
   *
   * <p>This method implements an equivalence relationship across actions, based on the action
   * class, the key, and the list of inputs and outputs.
   */
  public static boolean canBeShared(ActionAnalysisMetadata a, ActionAnalysisMetadata b) {
    if (!a.getMnemonic().equals(b.getMnemonic())) {
      return false;
    }

    // Non-Actions cannot be shared.
    if (!(a instanceof Action) || !(b instanceof Action)) {
      return false;
    }

    Action actionA = (Action) a;
    Action actionB = (Action) b;
    if (!actionA.getKey().equals(actionB.getKey())) {
      return false;
    }
    // Don't bother to check input and output counts first; the expected result for these tests is
    // to always be true (i.e., that this method returns true).
    if (!Iterables.elementsEqual(actionA.getMandatoryInputs(), actionB.getMandatoryInputs())) {
      return false;
    }
    if (!Iterables.elementsEqual(actionA.getOutputs(), actionB.getOutputs())) {
      return false;
    }
    return true;
  }

  /**
   * Finds action conflicts. An action conflict happens if two actions generate the same output
   * artifact. Shared actions are tolerated. See {@link #canBeShared} for details.
   *
   * @param actions a list of actions to check for action conflicts
   * @return a map between generated artifacts and their associated generating actions. If there is
   *     more than one action generating the same output artifact, only one action is chosen.
   * @throws ActionConflictException iff there are two actions generate the same output
   */
  public static ImmutableMap<Artifact, ActionAnalysisMetadata> findAndThrowActionConflict(
      Iterable<ActionAnalysisMetadata> actions) throws ActionConflictException {
    return Actions.maybeFilterSharedActionsAndThrowIfConflict(
        actions, /*allowSharedAction=*/ false);
  }

  /**
   * Finds action conflicts. An action conflict happens if two actions generate the same output
   * artifact. Shared actions are tolerated. See {@link #canBeShared} for details.
   *
   * @param actions a list of actions to check for action conflicts
   * @return a map between generated artifacts and their associated generating actions. If there is
   *     more than one action generating the same output artifact, only one action is chosen.
   * @throws ActionConflictException iff there are two unshareable actions generating the same
   *     output
   */
  public static ImmutableMap<Artifact, ActionAnalysisMetadata>
      filterSharedActionsAndThrowActionConflict(
      Iterable<? extends ActionAnalysisMetadata> actions) throws ActionConflictException {
    return Actions.maybeFilterSharedActionsAndThrowIfConflict(
        actions, /*allowSharedAction=*/ true);
  }

  private static ImmutableMap<Artifact, ActionAnalysisMetadata>
      maybeFilterSharedActionsAndThrowIfConflict(
      Iterable<? extends ActionAnalysisMetadata> actions, boolean allowSharedAction)
      throws ActionConflictException {
    Map<Artifact, ActionAnalysisMetadata> generatingActions = new HashMap<>();
    for (ActionAnalysisMetadata action : actions) {
      for (Artifact artifact : action.getOutputs()) {
        ActionAnalysisMetadata previousAction = generatingActions.put(artifact, action);
        if (previousAction != null && previousAction != action) {
          if (!allowSharedAction || !Actions.canBeShared(previousAction, action)) {
            throw new ActionConflictException(artifact, previousAction, action);
          }
        }
      }
    }
    return ImmutableMap.copyOf(generatingActions);
  }

  /**
   * Finds Artifact prefix conflicts between generated artifacts. An artifact prefix conflict
   * happens if one action generates an artifact whose path is a prefix of another artifact's path.
   * Those two artifacts cannot exist simultaneously in the output tree.
   *
   * @param generatingActions a map between generated artifacts and their associated generating
   *     actions.
   * @return a map between actions that generated the conflicting artifacts and their associated
   *     {@link ArtifactPrefixConflictException}.
   */
  public static Map<ActionAnalysisMetadata, ArtifactPrefixConflictException>
      findArtifactPrefixConflicts(Map<Artifact, ActionAnalysisMetadata> generatingActions) {
    TreeMap<PathFragment, Artifact> artifactPathMap = new TreeMap();
    for (Artifact artifact : generatingActions.keySet()) {
      artifactPathMap.put(artifact.getExecPath(), artifact);
    }

    return findArtifactPrefixConflicts(
        new MapBasedImmutableActionGraph(generatingActions), artifactPathMap);
  }

  /**
   * Finds Artifact prefix conflicts between generated artifacts. An artifact prefix conflict
   * happens if one action generates an artifact whose path is a prefix of another artifact's path.
   * Those two artifacts cannot exist simultaneously in the output tree.
   *
   * @param actionGraph the {@link ActionGraph} to query for artifact conflicts
   * @param artifactPathMap a map mapping generated artifacts to their exec paths
   * @return A map between actions that generated the conflicting artifacts and their associated
   *     {@link ArtifactPrefixConflictException}.
   */
  public static Map<ActionAnalysisMetadata, ArtifactPrefixConflictException>
      findArtifactPrefixConflicts(ActionGraph actionGraph,
      SortedMap<PathFragment, Artifact> artifactPathMap) {
    // No actions in graph -- currently happens only in tests. Special-cased because .next() call
    // below is unconditional.
    if (artifactPathMap.isEmpty()) {
      return ImmutableMap.<ActionAnalysisMetadata, ArtifactPrefixConflictException>of();
    }

    // Keep deterministic ordering of bad actions.
    Map<ActionAnalysisMetadata, ArtifactPrefixConflictException> badActions = new LinkedHashMap();
    Iterator<PathFragment> iter = artifactPathMap.keySet().iterator();

    // Report an error for every derived artifact which is a prefix of another.
    // If x << y << z (where x << y means "y starts with x"), then we only report (x,y), (x,z), but
    // not (y,z).
    for (PathFragment pathJ = iter.next(); iter.hasNext(); ) {
      // For each comparison, we have a prefix candidate (pathI) and a suffix candidate (pathJ).
      // At the beginning of the loop, we set pathI to the last suffix candidate, since it has not
      // yet been tested as a prefix candidate, and then set pathJ to the paths coming after pathI,
      // until we come to one that does not contain pathI as a prefix. pathI is then verified not to
      // be the prefix of any path, so we start the next run of the loop.
      PathFragment pathI = pathJ;
      // Compare pathI to the paths coming after it.
      while (iter.hasNext()) {
        pathJ = iter.next();
        if (pathJ.startsWith(pathI)) { // prefix conflict.
          Artifact artifactI = Preconditions.checkNotNull(artifactPathMap.get(pathI), pathI);
          Artifact artifactJ = Preconditions.checkNotNull(artifactPathMap.get(pathJ), pathJ);

          // We ignore the artifact prefix conflict between a TreeFileArtifact and its parent
          // TreeArtifact.
          // We can only have such a conflict here if:
          // 1. The TreeArtifact is generated by an ActionTemplate. And the TreeFileArtifact is
          //    generated by an expanded action created at execution time from the ActionTemplate.
          // 2. This is an incremental build with invalidated configured targets. In this case,
          //    the action graph contains expanded actions from previous builds and they will be
          //    checked for artifact conflicts.
          if (artifactJ.hasParent() && artifactJ.getParent().equals(artifactI)) {
            continue;
          }

          ActionAnalysisMetadata actionI =
              Preconditions.checkNotNull(actionGraph.getGeneratingAction(artifactI), artifactI);
          ActionAnalysisMetadata actionJ =
              Preconditions.checkNotNull(actionGraph.getGeneratingAction(artifactJ), artifactJ);
          if (actionI.shouldReportPathPrefixConflict(actionJ)) {
            ArtifactPrefixConflictException exception = new ArtifactPrefixConflictException(pathI,
                pathJ, actionI.getOwner().getLabel(), actionJ.getOwner().getLabel());
            badActions.put(actionI, exception);
            badActions.put(actionJ, exception);
          }
        } else { // pathJ didn't have prefix pathI, so no conflict possible for pathI.
          break;
        }
      }
    }
    return ImmutableMap.copyOf(badActions);
  }

  /**
   * Returns the escaped name for a given relative path as a string. This takes
   * a short relative path and turns it into a string suitable for use as a
   * filename. Invalid filename characters are escaped with an '_' + a single
   * character token.
   */
  public static String escapedPath(String path) {
    return PATH_ESCAPER.escape(path);
  }

  /**
   * Returns a string that is usable as a unique path component for a label. It is guaranteed
   * that no other label maps to this string.
   */
  public static String escapeLabel(Label label) {
    return PATH_ESCAPER.escape(label.getPackageName() + ":" + label.getName());
  }

  private static class MapBasedImmutableActionGraph implements ActionGraph {
    private final Map<Artifact, ActionAnalysisMetadata> generatingActions;

    MapBasedImmutableActionGraph(
        Map<Artifact, ActionAnalysisMetadata> generatingActions) {
      this.generatingActions = ImmutableMap.copyOf(generatingActions);
    }

    @Nullable
    public ActionAnalysisMetadata getGeneratingAction(Artifact artifact) {
      return generatingActions.get(artifact);
    }
  }
}
