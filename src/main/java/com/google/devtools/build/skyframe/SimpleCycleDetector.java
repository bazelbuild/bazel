// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.skyframe.AbstractParallelEvaluator.isDoneForBuild;
import static com.google.devtools.build.skyframe.AbstractParallelEvaluator.maybeMarkRebuilding;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.skyframe.NodeEntry.LifecycleState;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyFunctionEnvironment.UndonePreviouslyRequestedDeps;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Depth-first implementation of cycle detection after a {@link ParallelEvaluator} evaluation has
 * completed with at least one root unfinished.
 */
public class SimpleCycleDetector implements CycleDetector {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** The max number of cycles we will report to the user for a given root, to avoid OOMing. */
  private static final int MAX_CYCLES_TO_STORE = 20;

  private final boolean storeExactCycles;

  public SimpleCycleDetector(boolean storeExactCycles) {
    this.storeExactCycles = storeExactCycles;
  }

  @Override
  public void checkForCycles(
      Iterable<SkyKey> badRoots,
      EvaluationResult.Builder<?> result,
      ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    for (SkyKey root : badRoots) {
      ErrorInfo errorInfo = checkForCycles(root, evaluatorContext);
      if (errorInfo == null) {
        // This node just wasn't finished when evaluation aborted -- there were no cycles below
        // it.
        checkState(
            !evaluatorContext.keepGoing(root),
            "Missing error info with keep going (root=%s, badRoots=%s)",
            root,
            badRoots);
        continue;
      }
      checkState(
          !errorInfo.getCycleInfo().isEmpty(),
          "%s was not evaluated, but was not part of a cycle",
          root);
      result.addError(root, errorInfo);
      if (!evaluatorContext.keepGoing(root)) {
        return;
      }
    }
  }

  /**
   * The algorithm for this cycle detector is as follows. We visit the graph depth-first, keeping
   * track of the path we are currently on. We skip any DONE nodes (they are transitively
   * error-free). If we come to a node already on the path, we immediately construct a cycle. If we
   * are in the noKeepGoing case, we return ErrorInfo with that cycle to the caller. Otherwise, we
   * continue. Once all of a node's children are done, we construct an error value for it, based on
   * those children. Finally, when the original root's node is constructed, we return its ErrorInfo.
   */
  @Nullable
  private ErrorInfo checkForCycles(SkyKey root, ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    // The number of cycles found. Do not keep on searching for more cycles after this many were
    // found.
    int cyclesFound = 0;
    // The path through the graph currently being visited.
    List<SkyKey> graphPath = new ArrayList<>();
    // Set of nodes on the path, to avoid expensive searches through the path for cycles.
    Set<SkyKey> pathSet = new HashSet<>();

    // Maintain a stack explicitly instead of recursion to avoid stack overflows
    // on extreme graphs (with long dependency chains).
    Deque<SkyKey> toVisit = new ArrayDeque<>();

    toVisit.push(root);

    // The procedure for this check is as follows: we visit a node, push it onto the graph path,
    // push a marker value onto the toVisit stack, and then push all of its children onto the
    // toVisit stack. Thus, when the marker node comes to the top of the toVisit stack, we have
    // visited the downward transitive closure of the value. At that point, all of its children must
    // be finished, and so we can build the definitive error info for the node, popping it off the
    // graph path.
    while (!toVisit.isEmpty()) {
      SkyKey key = toVisit.pop();

      NodeEntry entry;
      if (key == CHILDREN_FINISHED) {
        // We have reached the marker node - that means all children of a node have been visited.
        // Since all nodes have errors, we must have found errors in the children at this point.
        key = graphPath.remove(graphPath.size() - 1);
        entry =
            checkNotNull(evaluatorContext.getGraph().get(null, Reason.CYCLE_CHECKING, key), key);
        pathSet.remove(key);
        // Skip this node if it was first/last node of a cycle, and so has already been processed.
        if (entry.isDone()) {
          continue;
        }
        if (!evaluatorContext.keepGoing(key)) {
          // in the --nokeep_going mode, we would have already returned if we'd found a cycle below
          // this node. We haven't, so there are no cycles below this node; skip further evaluation
          continue;
        }
        Set<SkyKey> removedDeps = ImmutableSet.of();
        if (cyclesFound < MAX_CYCLES_TO_STORE || !storeExactCycles) {
          // Value must be ready, because all of its children have finished, so we can build its
          // error.
          checkState(
              !entry.hasUnsignaledDeps(), "%s has unsignaled deps. ValueEntry: %s", key, entry);
        } else if (entry.hasUnsignaledDeps()) {
          removedDeps =
              removeIncompleteChildrenForCycle(
                  key,
                  entry,
                  entry.getTemporaryDirectDeps().getAllElementsAsIterable(),
                  evaluatorContext);
        }
        if (maybeHandleVerifiedCleanNode(key, entry, evaluatorContext, graphPath)) {
          continue;
        }
        maybeMarkRebuilding(entry);
        GroupedDeps directDeps = entry.getTemporaryDirectDeps();
        // Find out which children have errors. Similar logic to that in Evaluate#run().
        List<ErrorInfo> errorDeps =
            getChildrenErrorsForCycle(
                key, directDeps.getAllElementsAsIterable(), entry, evaluatorContext, removedDeps);
        checkState(
            !errorDeps.isEmpty(),
            "Node %s was not successfully evaluated, but had no child errors. NodeEntry: %s",
            key,
            entry);
        SkyFunctionEnvironment env;
        try {
          env =
              SkyFunctionEnvironment.create(
                  key,
                  directDeps,
                  Sets.difference(entry.getAllRemainingDirtyDirectDeps(), removedDeps),
                  entry.getMaxTransitiveSourceVersion(),
                  evaluatorContext);
          // When the environment sets a cycle node to be in error and commits afterwards, it
          // requires all of its deps to be fetched. See `SkyFunctionEnvironment#setError()`'s
          // JavaDoc for more details.
          env.ensurePreviouslyRequestedDepsFetched();
        } catch (UndonePreviouslyRequestedDeps undoneDeps) {
          // All children were finished according to the CHILDREN_FINISHED sentinel, and cycle
          // detection does not do normal SkyFunction evaluation, so no restarting nor child
          // dirtying was possible.
          throw new IllegalStateException(
              "Previously requested deps not done: " + undoneDeps.getDepKeys(), undoneDeps);
        }
        env.setError(entry, ErrorInfo.fromChildErrors(key, errorDeps));
        Set<SkyKey> reverseDeps = env.commitAndGetParents(entry, /* expectDoneDeps= */ true);
        evaluatorContext.signalParentsOnAbort(key, reverseDeps, entry.getVersion());
      } else {
        entry = evaluatorContext.getGraph().get(null, Reason.CYCLE_CHECKING, key);
      }

      checkNotNull(entry, key);
      // Nothing to be done for this node if it already has an entry.
      if (entry.isDone()) {
        continue;
      }
      if (cyclesFound >= MAX_CYCLES_TO_STORE && storeExactCycles) {
        // Do not keep on searching for cycles indefinitely, to avoid excessive runtime/OOMs.
        continue;
      }

      if (pathSet.contains(key)) {
        int cycleStart = graphPath.indexOf(key);
        // Found a cycle!
        cyclesFound++;
        Iterable<SkyKey> cycle = graphPath.subList(cycleStart, graphPath.size());
        // Log the cycle only if storing cycles, as cycle-storing mode ensures that the number
        // of graph cycles is bounded. Otherwise, a cycle-heavy graph could overflow the
        // INFO log.
        if (storeExactCycles) {
          logger.atInfo().log("Found cycle : %s from %s", cycle, graphPath);
        }
        // Put this node into a consistent state for building if it is dirty.
        if (entry.isDirty()) {
          // If this loop runs more than once, we are in the peculiar position of entry not needing
          // rebuilding even though it was signaled with the graph version. This can happen when the
          // entry was previously evaluated at this version, but then invalidated anyway, even
          // though nothing changed.
          int loopCount = 0;
          Version graphVersion = evaluatorContext.getGraphVersion();
          while (entry.getLifecycleState() == LifecycleState.CHECK_DEPENDENCIES) {
            entry.signalDep(graphVersion, null);
            loopCount++;
          }
          if (loopCount > 1 && !entry.getVersion().equals(graphVersion)) {
            BugReport.sendBugReport(
                new IllegalStateException(
                    "Entry needed multiple signaling but didn't have the graph version: "
                        + key
                        + ", "
                        + entry
                        + ", "
                        + graphVersion
                        + ", "
                        + graphPath));
          }
          if (entry.getLifecycleState() == LifecycleState.NEEDS_REBUILDING) {
            entry.markRebuilding();
          } else if (maybeHandleVerifiedCleanNode(key, entry, evaluatorContext, graphPath)) {
            continue;
          }
        }
        if (evaluatorContext.keepGoing(key)) {
          // Any children of this node that we haven't already visited are not worth visiting,
          // since this node is about to be done. Thus, the only child worth visiting is the one in
          // this cycle, the cycleChild (which may == key if this cycle is a self-edge).
          SkyKey cycleChild = selectCycleChild(key, graphPath, cycleStart);
          Set<SkyKey> removedDeps =
              removeDescendantsOfCycleValue(
                  key, entry, cycleChild, toVisit, graphPath.size() - cycleStart, evaluatorContext);
          ValueWithMetadata dummyValue = ValueWithMetadata.wrapWithMetadata(new SkyValue() {});

          SkyFunctionEnvironment env =
              SkyFunctionEnvironment.createForError(
                  key,
                  entry.getTemporaryDirectDeps(),
                  ImmutableMap.of(cycleChild, dummyValue),
                  Sets.difference(entry.getAllRemainingDirtyDirectDeps(), removedDeps),
                  evaluatorContext);

          // Construct full error info for this node. Get errors from children, which are all done
          // except possibly for the cycleChild.
          List<ErrorInfo> allErrors =
              getChildrenErrorsForCycleChecking(
                  entry.getTemporaryDirectDeps().getAllElementsAsIterable(),
                  /* unfinishedChild= */ cycleChild,
                  evaluatorContext);
          CycleInfo cycleInfo =
              storeExactCycles ? CycleInfo.createCycleInfo(cycle) : CycleInfo.cycleInfoNoDetails();
          // Add in this cycle.
          allErrors.add(ErrorInfo.fromCycle(cycleInfo));
          env.setError(entry, ErrorInfo.fromChildErrors(key, allErrors));
          Set<SkyKey> reverseDeps = env.commitAndGetParents(entry, /* expectDoneDeps= */ true);
          evaluatorContext.signalParentsOnAbort(key, reverseDeps, entry.getVersion());
          continue;
        } else {
          // We need to return right away in the noKeepGoing case, so construct the cycle (with the
          // path) and return.
          checkState(
              graphPath.get(0).equals(root),
              "%s not reached from %s. ValueEntry: %s",
              key,
              root,
              entry);
          return ErrorInfo.fromCycle(
              CycleInfo.createCycleInfo(graphPath.subList(0, cycleStart), cycle));
        }
      }

      // This node is not yet known to be in a cycle. So process its children.
      GroupedDeps temporaryDirectDeps = entry.getTemporaryDirectDeps();
      if (temporaryDirectDeps.isEmpty()) {
        continue;
      }
      // Prefetch all children, in case our graph performs better with a primed cache. No need to
      // recurse into done nodes. The fields of done nodes aren't necessary, since we'll filter them
      // out.
      // TODO(janakr): If graph implementations start using these hints for not-done nodes, we may
      // have to change this.
      Iterable<SkyKey> children = temporaryDirectDeps.getAllElementsAsIterable();
      NodeBatch childNodes =
          evaluatorContext.getGraph().getBatch(key, Reason.EXISTENCE_CHECKING, children);

      // This marker flag will tell us when all this node's children have been processed.
      toVisit.push(CHILDREN_FINISHED);
      // This node is now part of the path through the graph.
      graphPath.add(key);
      pathSet.add(key);
      for (SkyKey childKey : children) {
        NodeEntry childEntry =
            checkNotNull(
                childNodes.get(childKey),
                "Missing already declared dep %s (parent=%s)",
                childKey,
                key);
        if (!childEntry.isDone()) {
          toVisit.push(childKey);
        }
      }
    }
    return evaluatorContext.keepGoing(root)
        ? checkDone(root, evaluatorContext.getGraph().get(null, Reason.CYCLE_CHECKING, root))
            .getErrorInfo()
        : null;
  }

  /**
   * Fully process {@code entry} if it is dirty but verified to be clean. This can only happen in
   * rare circumstances where a node with a cycle is invalidated at the same version. Returns true
   * if the entry was successfully processed, meaning that its value has been set and all reverse
   * deps signaled.
   */
  private static boolean maybeHandleVerifiedCleanNode(
      SkyKey key,
      NodeEntry entry,
      ParallelEvaluatorContext evaluatorContext,
      List<SkyKey> graphPathForDebugging)
      throws InterruptedException {
    if (entry.getLifecycleState() != LifecycleState.VERIFIED_CLEAN) {
      return false;
    }
    Set<SkyKey> rdeps = entry.markClean().getRdepsToSignal();
    evaluatorContext.signalParentsOnAbort(key, rdeps, entry.getVersion());
    ErrorInfo error = entry.getErrorInfo();
    if (error.getCycleInfo().isEmpty()) {
      BugReport.sendBugReport(
          new IllegalStateException(
              "Entry was unchanged from last build, but cycle was found this time and not"
                  + " last time: "
                  + key
                  + ", "
                  + entry
                  + ", "
                  + graphPathForDebugging));
    }
    return true;
  }

  /**
   * Marker value that we push onto a stack before we push a node's children on. When the marker
   * value is popped, we know that all the children are finished. We would use null instead, but
   * ArrayDeque does not permit null elements.
   */
  private static final SkyKey CHILDREN_FINISHED = () -> null;

  /**
   * Returns the child of this node that is in the cycle that was just found. If the cycle is a
   * self-edge, returns the node itself.
   */
  private static SkyKey selectCycleChild(SkyKey key, List<SkyKey> graphPath, int cycleStart) {
    return cycleStart + 1 == graphPath.size() ? key : graphPath.get(cycleStart + 1);
  }

  /**
   * Get all the errors of child nodes. There must be at least one cycle amongst them.
   *
   * @param children child nodes to query for errors.
   * @return List of ErrorInfos from all children that had errors.
   */
  private static List<ErrorInfo> getChildrenErrorsForCycle(
      SkyKey parent,
      Iterable<SkyKey> children,
      NodeEntry entryForDebugging,
      ParallelEvaluatorContext evaluatorContext,
      Set<SkyKey> removedDepsForDebugging)
      throws InterruptedException {
    List<ErrorInfo> allErrors = new ArrayList<>();
    boolean foundCycle = false;
    NodeBatch childNodes =
        evaluatorContext.getGraph().getBatch(parent, Reason.CYCLE_CHECKING, children);
    for (SkyKey childKey : children) {
      NodeEntry childEntry =
          checkNotNull(
              childNodes.get(childKey),
              "Missing already declared dep %s (parent=%s)",
              childKey,
              parent);
      checkDone(childKey, childEntry);
      ErrorInfo errorInfo = childEntry.getErrorInfo();
      if (errorInfo != null) {
        foundCycle |= !errorInfo.getCycleInfo().isEmpty();
        allErrors.add(errorInfo);
      }
    }
    checkState(
        foundCycle,
        "Key %s with entry %s had no cycle beneath it: %s; Removed deps: %s",
        parent,
        entryForDebugging,
        allErrors,
        removedDepsForDebugging);
    return allErrors;
  }

  /**
   * Get all the errors of child nodes.
   *
   * @param children child nodes to query for errors.
   * @param unfinishedChild child which is allowed to not be done.
   * @return List of ErrorInfos from all children that had errors.
   */
  private List<ErrorInfo> getChildrenErrorsForCycleChecking(
      Iterable<SkyKey> children, SkyKey unfinishedChild, ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    List<ErrorInfo> allErrors = new ArrayList<>();
    NodeBatch childEntries =
        evaluatorContext.getGraph().getBatch(null, Reason.CYCLE_CHECKING, children);
    for (SkyKey childKey : children) {
      NodeEntry childNodeEntry = childEntries.get(childKey);
      ErrorInfo errorInfo =
          getErrorMaybe(
              childKey, childNodeEntry, /* allowUnfinished= */ childKey.equals(unfinishedChild));
      if (errorInfo != null) {
        // Drop child cycle error if not storing cycles, as these will be redundant with the cycle
        // error of the parent node.
        boolean dropErrorInfo = !storeExactCycles && !errorInfo.getCycleInfo().isEmpty();
        if (!dropErrorInfo) {
          allErrors.add(errorInfo);
        }
      }
    }
    return allErrors;
  }

  @Nullable
  private static ErrorInfo getErrorMaybe(
      SkyKey key, NodeEntry childNodeEntry, boolean allowUnfinished) throws InterruptedException {
    checkNotNull(childNodeEntry, key);
    if (!allowUnfinished) {
      return checkDone(key, childNodeEntry).getErrorInfo();
    }
    return childNodeEntry.isDone() ? childNodeEntry.getErrorInfo() : null;
  }

  /**
   * Removes direct children of key from toVisit and from the entry itself, and makes the entry
   * ready if necessary. We must do this because it would not make sense to try to build the
   * children after building the entry. It would violate the invariant that a parent can only be
   * built after its children are built; See bug "Precondition error while evaluating a Skyframe
   * graph with a cycle".
   *
   * @param key SkyKey of node in a cycle.
   * @param entry NodeEntry of node in a cycle.
   * @param cycleChild direct child of key in the cycle, or key itself if the cycle is a self-edge.
   * @param toVisit list of remaining nodes to visit by the cycle-checker.
   * @param cycleLength the length of the cycle found.
   */
  private static Set<SkyKey> removeDescendantsOfCycleValue(
      SkyKey key,
      NodeEntry entry,
      @Nullable SkyKey cycleChild,
      Iterable<SkyKey> toVisit,
      int cycleLength,
      ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    GroupedDeps directDeps = entry.getTemporaryDirectDeps();
    Set<SkyKey> unvisitedDeps = Sets.newHashSet(directDeps.getAllElementsAsIterable());
    unvisitedDeps.remove(cycleChild);
    // Remove any children from this node that are not part of the cycle we just found. They are
    // irrelevant to the node as it stands, and if they are deleted from the graph because they are
    // not built by the end of cycle-checking, we would have dangling references.
    Set<SkyKey> removedDeps =
        removeIncompleteChildrenForCycle(key, entry, unvisitedDeps, evaluatorContext);
    if (entry.hasUnsignaledDeps()) {
      // The entry has at most one undone dep now, its cycleChild. Signal to make entry ready. Note
      // that the entry can conceivably be ready if its cycleChild already found a different cycle
      // and was built.
      entry.signalDep(evaluatorContext.getGraphVersion(), cycleChild);
    }
    maybeMarkRebuilding(entry);
    checkState(!entry.hasUnsignaledDeps(), "%s %s %s", key, cycleChild, entry);
    Iterator<SkyKey> it = toVisit.iterator();
    while (it.hasNext()) {
      SkyKey descendant = it.next();
      if (descendant == CHILDREN_FINISHED) {
        // Marker value, delineating the end of a group of children that were enqueued.
        cycleLength--;
        if (cycleLength == 0) {
          // We have seen #cycleLength-1 marker values, and have arrived at the one for this value,
          // so we are done.
          return removedDeps;
        }
        continue; // Don't remove marker values.
      }
      if (cycleLength == 1) {
        // Remove the direct children remaining to visit of the cycle node.
        checkState(
            unvisitedDeps.contains(descendant), "%s %s %s %s", key, descendant, cycleChild, entry);
        it.remove();
      }
    }
    throw new IllegalStateException(
        String.format(
            "There were not %d groups of children in %s when trying to remove children of %s other "
                + "than %s",
            cycleLength, toVisit, key, cycleChild));
  }

  private static Set<SkyKey> removeIncompleteChildrenForCycle(
      SkyKey key,
      NodeEntry entry,
      Iterable<SkyKey> children,
      ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    Set<SkyKey> unfinishedDeps = new HashSet<>();
    for (SkyKey child : children) {
      if (removeIncompleteChildForCycle(key, child, evaluatorContext)) {
        unfinishedDeps.add(child);
      }
    }
    entry.removeUnfinishedDeps(unfinishedDeps);
    return unfinishedDeps;
  }

  private static NodeEntry checkDone(SkyKey key, NodeEntry entry) {
    checkNotNull(entry, key);
    checkState(entry.isDone(), "%s %s", key, entry);
    return entry;
  }

  /**
   * If child is not done, removes {@code inProgressParent} from {@code child}'s reverse deps.
   * Returns whether child should be removed from inProgressParent's entry's direct deps.
   */
  private static boolean removeIncompleteChildForCycle(
      SkyKey inProgressParent, SkyKey child, ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    NodeEntry childEntry =
        evaluatorContext.getGraph().get(inProgressParent, Reason.CYCLE_CHECKING, child);
    if (!isDoneForBuild(childEntry)) {
      childEntry.removeReverseDep(inProgressParent);
      return true;
    }
    return false;
  }
}
