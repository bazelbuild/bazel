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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.escape.Escaper;
import com.google.common.escape.Escapers;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.vfs.OsPathPolicy;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * Helper class for actions.
 *
 * <p>We expect the class to be created exactly once per build doing shared actions check.
 */
@ThreadSafe
public final class Actions {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final Escaper PATH_ESCAPER = Escapers.builder()
      .addEscape('_', "_U")
      .addEscape('/', "_S")
      .addEscape('\\', "_B")
      .addEscape(':', "_C")
      .build();

  /** Flag used to limit the number of warnings about shared actions to one per build. */
  private static final AtomicBoolean issuedWarningForSharedActionsWithTreeArtifactInput =
      new AtomicBoolean();

  // TODO(b/160181927): Remove the warning once we move shared actions detection to execution phase.
  public static void clearSharedActionsWarningFlag() {
    issuedWarningForSharedActionsWithTreeArtifactInput.set(false);
  }

  /**
   * Checks if the two actions are equivalent. This method exists to support sharing actions between
   * configured targets for cases where there is no canonical target that could own the action. In
   * the action graph construction this case shows up as two actions generating the same output
   * file.
   *
   * <p>This method implements an equivalence relationship across actions, based on the action
   * class, the key, and the list of inputs and outputs.
   */
  public static boolean canBeShared(
      ActionKeyContext actionKeyContext, ActionAnalysisMetadata a, ActionAnalysisMetadata b) {
    if (!a.isShareable() || !b.isShareable()) {
      return false;
    }

    if (!a.getMnemonic().equals(b.getMnemonic())) {
      return false;
    }

    // Non-Actions cannot be shared.
    if (!(a instanceof Action) || !(b instanceof Action)) {
      return false;
    }

    Action actionA = (Action) a;
    Action actionB = (Action) b;
    if (!actionA
        .getKey(actionKeyContext, /*artifactExpander=*/ null)
        .equals(actionB.getKey(actionKeyContext, /*artifactExpander=*/ null))) {
      return false;
    }
    // Don't bother to check input and output counts first; the expected result for these tests is
    // to always be true (i.e., that this method returns true).
    if (!artifactsEqualWithoutOwner(
        actionA.getMandatoryInputs().toList(), actionB.getMandatoryInputs().toList())) {
      return false;
    }
    if (!artifactsEqualWithoutOwner(actionA.getOutputs(), actionB.getOutputs())) {
      return false;
    }
    return true;
  }

  /**
   * Checks whether provided actions are equivalent and issues a warning in case we may be overly
   * permissive in the result. Returned result is the same as for {@link
   * #canBeShared(ActionKeyContext, ActionAnalysisMetadata, ActionAnalysisMetadata)}.
   *
   * <p>TODO(b/160181927): Remove the warning once we move shared actions detection to execution
   * phase.
   */
  static boolean canBeSharedWarnForPotentialFalsePositives(
      EventHandler eventHandler,
      ActionKeyContext actionKeyContext,
      ActionAnalysisMetadata actionA,
      ActionAnalysisMetadata actionB) {
    boolean canBeShared = canBeShared(actionKeyContext, actionA, actionB);
    if (canBeShared) {
      Optional<Artifact> treeArtifactInput =
          actionA.getMandatoryInputs().toList().stream()
              .filter(Artifact::isTreeArtifact)
              .findFirst();
      treeArtifactInput.ifPresent(
          treeArtifact -> {
            if (issuedWarningForSharedActionsWithTreeArtifactInput.compareAndSet(false, true)) {
              eventHandler.handle(
                  Event.warn(
                      "Detected shared actions with tree artifact input -- detection is overly"
                          + " permissive in this case and may allow sharing of different"
                          + " actions."));
            }

            logger.atInfo().log(
                "Shared action: %s has a tree artifact input: %s -- shared actions"
                    + " detection is overly permissive in this case and may allow"
                    + " sharing of different actions",
                actionA, treeArtifact);
          });
    }
    return canBeShared;
  }

  private static boolean artifactsEqualWithoutOwner(
      Iterable<Artifact> iterable1, Iterable<Artifact> iterable2) {
    if (iterable1 instanceof Collection && iterable2 instanceof Collection) {
      Collection<?> collection1 = (Collection<?>) iterable1;
      Collection<?> collection2 = (Collection<?>) iterable2;
      if (collection1.size() != collection2.size()) {
        return false;
      }
    }
    Iterator<Artifact> iterator1 = iterable1.iterator();
    Iterator<Artifact> iterator2 = iterable2.iterator();
    while (iterator1.hasNext()) {
      if (!iterator2.hasNext()) {
        return false;
      }
      Artifact artifact1 = iterator1.next();
      Artifact artifact2 = iterator2.next();
      if (!artifact1.equalsWithoutOwner(artifact2)) {
        return false;
      }
    }
    return !iterator2.hasNext();
  }

  /**
   * Assigns generating action keys to artifacts, and finds action conflicts. An action conflict
   * happens if two actions generate the same output artifact. Shared actions are not allowed. See
   * {@link #canBeShared} for details. Should only be called for special action lookup values: does
   * not handle normal configured targets. In particular, {@link
   * GeneratingActions#getArtifactsByOutputLabel} will be empty.
   *
   * @param actions a list of actions to check for action conflict.
   * @return a structure giving the actions, with a level of indirection.
   * @throws ActionConflictException iff there are two actions generate the same output
   */
  public static GeneratingActions assignOwnersAndFindAndThrowActionConflict(
      EventHandler eventHandler,
      ActionKeyContext actionKeyContext,
      ImmutableList<ActionAnalysisMetadata> actions,
      ActionLookupValue.ActionLookupKey actionLookupKey)
      throws ActionConflictException {
    return Actions.assignOwnersAndMaybeFilterSharedActionsAndThrowIfConflict(
        eventHandler,
        actionKeyContext,
        actions,
        actionLookupKey,
        /*allowSharedAction=*/ false,
        /*outputFiles=*/ null);
  }

  /**
   * Assigns generating action keys to artifacts and finds action conflicts. An action conflict
   * happens if two actions generate the same output artifact. Shared actions are tolerated. See
   * {@link #canBeShared} for details. Should be called by a configured target/aspect on the actions
   * it owns. Should not be used for "global" checks of multiple configured targets: use {@link
   * #findArtifactPrefixConflicts} for that.
   *
   * @param actions a list of actions to check for action conflicts, all generated by the same
   *     configured target/aspect.
   * @return a structure giving the actions, with a level of indirection.
   * @throws ActionConflictException iff there are two unshareable actions generating the same
   *     output
   */
  public static GeneratingActions assignOwnersAndFilterSharedActionsAndThrowActionConflict(
      EventHandler eventHandler,
      ActionKeyContext actionKeyContext,
      ImmutableList<ActionAnalysisMetadata> actions,
      ActionLookupKey actionLookupKey,
      @Nullable Collection<OutputFile> outputFiles)
      throws ActionConflictException {
    return Actions.assignOwnersAndMaybeFilterSharedActionsAndThrowIfConflict(
        eventHandler,
        actionKeyContext,
        actions,
        actionLookupKey,
        /*allowSharedAction=*/ true,
        outputFiles);
  }

  private static void verifyGeneratingActionKeys(
      EventHandler eventHandler,
      Artifact.DerivedArtifact output,
      ActionLookupData otherKey,
      boolean allowSharedAction,
      ActionKeyContext actionKeyContext,
      ImmutableList<ActionAnalysisMetadata> actions)
      throws ActionConflictException {
    ActionLookupData firstKey = output.getGeneratingActionKey();
    Preconditions.checkState(
        firstKey.getActionLookupKey().equals(otherKey.getActionLookupKey()),
        "Mismatched lookup keys? %s %s %s",
        output,
        firstKey,
        otherKey);
    int actionIndex = firstKey.getActionIndex();
    int otherIndex = otherKey.getActionIndex();
    if (actionIndex != otherIndex
        && (!allowSharedAction
            || !Actions.canBeSharedWarnForPotentialFalsePositives(
                eventHandler,
                actionKeyContext,
                actions.get(actionIndex),
                actions.get(otherIndex)))) {
      throw new ActionConflictException(
          actionKeyContext, output, actions.get(actionIndex), actions.get(otherIndex));
    }
  }

  /**
   * Checks {@code actions} for conflicts and sets each artifact's generating action key.
   *
   * <p>Conflicts can happen in one of two ways: the same artifact can be the output of multiple
   * unshareable actions (or shareable actions if {@code allowSharedAction} is false), or two
   * artifacts with the same execPath can be the outputs of different unshareable actions.
   *
   * <p>If {@code outputFiles} is non-null, also builds a map of output-file labels to artifacts,
   * for use by output file configured targets when they are retrieving their artifacts from this
   * associated rule configured target.
   */
  private static GeneratingActions assignOwnersAndMaybeFilterSharedActionsAndThrowIfConflict(
      EventHandler eventHandler,
      ActionKeyContext actionKeyContext,
      ImmutableList<ActionAnalysisMetadata> actions,
      ActionLookupKey actionLookupKey,
      boolean allowSharedAction,
      @Nullable Collection<OutputFile> outputFiles)
      throws ActionConflictException {
    Map<PathFragment, Artifact.DerivedArtifact> seenArtifacts = new HashMap<>();
    @Nullable ImmutableMap<String, Label> outputFileNames = null;
    if (outputFiles != null && !outputFiles.isEmpty()) {
      ImmutableMap.Builder<String, Label> outputFileNamesBuilder =
          ImmutableMap.builderWithExpectedSize(outputFiles.size());
      outputFiles.forEach(o -> outputFileNamesBuilder.put(o.getLabel().getName(), o.getLabel()));
      outputFileNames = outputFileNamesBuilder.build();
    }
    @Nullable
    ImmutableMap.Builder<Label, Artifact> artifactsByOutputLabel =
        outputFileNames != null ? ImmutableMap.builderWithExpectedSize(outputFiles.size()) : null;
    @Nullable Label label = actionLookupKey.getLabel();
    @Nullable
    PathFragment packageDirectory =
        outputFileNames != null
            ? Preconditions.checkNotNull(label, actionLookupKey)
                .getPackageIdentifier()
                .getSourceRoot()
            : null;
    // Loop over the actions, looking at all outputs for conflicts.
    int actionIndex = 0;
    for (ActionAnalysisMetadata action : actions) {
      ActionLookupData generatingActionKey = ActionLookupData.create(actionLookupKey, actionIndex);
      for (Artifact artifact : action.getOutputs()) {
        Preconditions.checkState(
            !artifact.isSourceArtifact(),
            "Source in outputs: %s %s %s",
            artifact,
            generatingActionKey,
            action);
        Artifact.DerivedArtifact output = (Artifact.DerivedArtifact) artifact;
        // Has an artifact with this execPath been seen before?
        Artifact.DerivedArtifact equalOutput =
            seenArtifacts.putIfAbsent(output.getExecPath(), output);
        if (equalOutput != null) {
          // Yes: assert that its generating action and this artifact's are compatible.
          verifyGeneratingActionKeys(
              eventHandler,
              equalOutput,
              generatingActionKey,
              allowSharedAction,
              actionKeyContext,
              actions);
        } else {
          // No: populate the output label map with this artifact if applicable: if this
          // artifact corresponds to a target that is an OutputFile with associated rule this label.
          PathFragment rootRelativePath = output.getRootRelativePath();
          if (packageDirectory != null && rootRelativePath.startsWith(packageDirectory)) {
            PathFragment packageRelativePath = rootRelativePath.relativeTo(packageDirectory);
            Label outputLabel = outputFileNames.get(packageRelativePath.getPathString());
            if (outputLabel != null) {
              artifactsByOutputLabel.put(outputLabel, artifact);
            }
          }
        }
        // Was this output already seen, so it has a generating action key set?
        if (!output.hasGeneratingActionKey()) {
          // Common case: artifact hasn't been seen before.
          output.setGeneratingActionKey(generatingActionKey);
        } else {
          // Key is already set: verify that the generating action and this action are compatible.
          verifyGeneratingActionKeys(
              eventHandler,
              output,
              generatingActionKey,
              allowSharedAction,
              actionKeyContext,
              actions);
        }
      }
      actionIndex++;
    }
    return new GeneratingActions(
        actions,
        artifactsByOutputLabel != null ? artifactsByOutputLabel.build() : ImmutableMap.of());
  }

  /**
   * Returns a comparator for use with {@link #findArtifactPrefixConflicts(ActionGraph, SortedMap,
   * boolean)}.
   */
  public static Comparator<PathFragment> comparatorForPrefixConflicts() {
    return PathFragmentPrefixComparator.INSTANCE;
  }

  private static class PathFragmentPrefixComparator implements Comparator<PathFragment> {
    private static final PathFragmentPrefixComparator INSTANCE = new PathFragmentPrefixComparator();

    @Override
    public int compare(PathFragment lhs, PathFragment rhs) {
      // We need to use the OS path policy in case the OS is case insensitive.
      OsPathPolicy os = OsPathPolicy.getFilePathOs();
      String str1 = lhs.getPathString();
      String str2 = rhs.getPathString();
      int len1 = str1.length();
      int len2 = str2.length();
      int n = Math.min(len1, len2);
      for (int i = 0; i < n; ++i) {
        char c1 = str1.charAt(i);
        char c2 = str2.charAt(i);
        int res = os.compare(c1, c2);
        if (res != 0) {
          if (c1 == PathFragment.SEPARATOR_CHAR) {
            return -1;
          } else if (c2 == PathFragment.SEPARATOR_CHAR) {
            return 1;
          }
          return res;
        }
      }
      return len1 - len2;
    }
  }

  /**
   * Finds Artifact prefix conflicts between generated artifacts. An artifact prefix conflict
   * happens if one action generates an artifact whose path is a prefix of another artifact's path.
   * Those two artifacts cannot exist simultaneously in the output tree.
   *
   * @param actionGraph the {@link ActionGraph} to query for artifact conflicts
   * @param artifactPathMap a map mapping generated artifacts to their exec paths. The map must be
   *     sorted using the comparator from {@link #comparatorForPrefixConflicts()}.
   * @param strictConflictChecks report path prefix conflicts, regardless of
   *     shouldReportPathPrefixConflict().
   * @return A map between actions that generated the conflicting artifacts and their associated
   *     {@link ArtifactPrefixConflictException}.
   */
  public static Map<ActionAnalysisMetadata, ArtifactPrefixConflictException>
      findArtifactPrefixConflicts(
          ActionGraph actionGraph,
          SortedMap<PathFragment, Artifact> artifactPathMap,
          boolean strictConflictChecks) {
    // You must construct the sorted map using this comparator for the algorithm to work.
    // The algorithm requires subdirectories to immediately follow parent directories,
    // before any files in that directory.
    // Example: "foo", "foo.obj", foo/bar" must be sorted
    // "foo", "foo/bar", foo.obj"
    Preconditions.checkArgument(
        artifactPathMap.comparator() instanceof PathFragmentPrefixComparator,
        "artifactPathMap must be sorted with PathFragmentPrefixComparator");
    // No actions in graph -- currently happens only in tests. Special-cased because .next() call
    // below is unconditional.
    if (artifactPathMap.isEmpty()) {
      return ImmutableMap.<ActionAnalysisMetadata, ArtifactPrefixConflictException>of();
    }

    // Keep deterministic ordering of bad actions.
    Map<ActionAnalysisMetadata, ArtifactPrefixConflictException> badActions = new LinkedHashMap<>();
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
          if (strictConflictChecks || actionI.shouldReportPathPrefixConflict(actionJ)) {
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

  /**
   * Container class for actions, ensuring they have already been checked for conflicts and their
   * generated artifacts have had owners assigned.
   */
  public static class GeneratingActions {
    public static final GeneratingActions EMPTY =
        new GeneratingActions(ImmutableList.of(), ImmutableMap.of());

    private final ImmutableList<ActionAnalysisMetadata> actions;
    private final ImmutableMap<Label, Artifact> artifactsByOutputLabel;

    private GeneratingActions(
        ImmutableList<ActionAnalysisMetadata> actions,
        ImmutableMap<Label, Artifact> artifactsByOutputLabel) {
      this.actions = actions;
      this.artifactsByOutputLabel = artifactsByOutputLabel;
    }

    /** Used only for the workspace status action. Does not handle duplicate artifacts. */
    public static GeneratingActions fromSingleAction(
        ActionAnalysisMetadata action, ActionLookupValue.ActionLookupKey actionLookupKey) {
      Preconditions.checkState(actionLookupKey.getLabel() == null, actionLookupKey);
      ActionLookupData generatingActionKey = ActionLookupData.create(actionLookupKey, 0);
      for (Artifact output : action.getOutputs()) {
        ((Artifact.DerivedArtifact) output).setGeneratingActionKey(generatingActionKey);
      }
      return new GeneratingActions(ImmutableList.of(action), ImmutableMap.of());
    }

    public ImmutableList<ActionAnalysisMetadata> getActions() {
      return actions;
    }

    public ImmutableMap<Label, Artifact> getArtifactsByOutputLabel() {
      return artifactsByOutputLabel;
    }
  }
}
