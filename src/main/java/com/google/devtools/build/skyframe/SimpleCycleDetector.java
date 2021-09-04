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

import static com.google.devtools.build.skyframe.AbstractParallelEvaluator.isDoneForBuild;
import static com.google.devtools.build.skyframe.AbstractParallelEvaluator.maybeMarkRebuilding;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyFunctionEnvironment.UndonePreviouslyRequestedDeps;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Depth-first implementation of cycle detection after a {@link ParallelEvaluator} evaluation has
 * completed with at least one root unfinished.
 */
public class SimpleCycleDetector implements CycleDetector {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final Duration MIN_LOGGING = Duration.ofMillis(10);

  @Override
  public void checkForCycles(
      Iterable<SkyKey> badRoots,
      EvaluationResult.Builder<?> result,
      ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    try (AutoProfiler p =
        GoogleAutoProfilerUtils.logged("Checking for Skyframe cycles", MIN_LOGGING)) {
      for (SkyKey root : badRoots) {
        ErrorInfo errorInfo = checkForCycles(root, evaluatorContext);
        if (errorInfo == null) {
          // This node just wasn't finished when evaluation aborted -- there were no cycles below
          // it.
          Preconditions.checkState(!evaluatorContext.keepGoing(), "", root, badRoots);
          continue;
        }
        Preconditions.checkState(
            !Iterables.isEmpty(errorInfo.getCycleInfo()),
            "%s was not evaluated, but was not part of a cycle",
            root);
        result.addError(root, errorInfo);
        if (!evaluatorContext.keepGoing()) {
          return;
        }
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
  private static ErrorInfo checkForCycles(SkyKey root, ParallelEvaluatorContext evaluatorContext)
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
            Preconditions.checkNotNull(
                evaluatorContext.getGraph().get(null, Reason.CYCLE_CHECKING, key), key);
        pathSet.remove(key);
        // Skip this node if it was first/last node of a cycle, and so has already been processed.
        if (entry.isDone()) {
          continue;
        }
        if (!evaluatorContext.keepGoing()) {
          // in the --nokeep_going mode, we would have already returned if we'd found a cycle below
          // this node. We haven't, so there are no cycles below this node; skip further evaluation
          continue;
        }
        Set<SkyKey> removedDeps = ImmutableSet.of();
        if (cyclesFound < MAX_CYCLES) {
          // Value must be ready, because all of its children have finished, so we can build its
          // error.
          Preconditions.checkState(entry.isReady(), "%s not ready. ValueEntry: %s", key, entry);
        } else if (!entry.isReady()) {
          removedDeps =
              removeIncompleteChildrenForCycle(
                  key, entry, Iterables.concat(entry.getTemporaryDirectDeps()), evaluatorContext);
        }
        if (maybeHandleVerifiedCleanNode(key, entry, evaluatorContext, graphPath)) {
          continue;
        }
        maybeMarkRebuilding(entry);
        GroupedList<SkyKey> directDeps = entry.getTemporaryDirectDeps();
        // Find out which children have errors. Similar logic to that in Evaluate#run().
        List<ErrorInfo> errorDeps =
            getChildrenErrorsForCycle(
                key,
                Iterables.concat(directDeps),
                directDeps.numElements(),
                entry,
                evaluatorContext);
        Preconditions.checkState(
            !errorDeps.isEmpty(),
            "Node %s was not successfully evaluated, but had no child errors. NodeEntry: %s",
            key,
            entry);
        SkyFunctionEnvironment env;
        try {
          env =
              new SkyFunctionEnvironment(
                  key,
                  directDeps,
                  Sets.difference(entry.getAllRemainingDirtyDirectDeps(), removedDeps),
                  evaluatorContext);
        } catch (UndonePreviouslyRequestedDeps undoneDeps) {
          // All children were finished according to the CHILDREN_FINISHED sentinel, and cycle
          // detection does not do normal SkyFunction evaluation, so no restarting nor child
          // dirtying was possible.
          throw new IllegalStateException(
              "Previously requested dep not done: " + undoneDeps.getDepKeys(), undoneDeps);
        }
        env.setError(entry, ErrorInfo.fromChildErrors(key, errorDeps));
        Set<SkyKey> reverseDeps = env.commitAndGetParents(entry);
        evaluatorContext.signalParentsOnAbort(key, reverseDeps, entry.getVersion());
      } else {
        entry = evaluatorContext.getGraph().get(null, Reason.CYCLE_CHECKING, key);
      }

      Preconditions.checkNotNull(entry, key);
      // Nothing to be done for this node if it already has an entry.
      if (entry.isDone()) {
        continue;
      }
      if (cyclesFound == MAX_CYCLES) {
        // Do not keep on searching for cycles indefinitely, to avoid excessive runtime/OOMs.
        continue;
      }

      if (pathSet.contains(key)) {
        int cycleStart = graphPath.indexOf(key);
        // Found a cycle!
        cyclesFound++;
        Iterable<SkyKey> cycle = graphPath.subList(cycleStart, graphPath.size());
        logger.atInfo().log("Found cycle : %s from %s", cycle, graphPath);
        // Put this node into a consistent state for building if it is dirty.
        if (entry.isDirty()) {
          // If this loop runs more than once, we are in the peculiar position of entry not needing
          // rebuilding even though it was signaled with the graph version. This can happen when the
          // entry was previously evaluated at this version, but then invalidated anyway, even
          // though nothing changed.
          int loopCount = 0;
          Version graphVersion = evaluatorContext.getGraphVersion();
          while (entry.getDirtyState() == NodeEntry.DirtyState.CHECK_DEPENDENCIES) {
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
          if (entry.getDirtyState() == NodeEntry.DirtyState.NEEDS_REBUILDING) {
            entry.markRebuilding();
          } else if (maybeHandleVerifiedCleanNode(key, entry, evaluatorContext, graphPath)) {
            continue;
          }
        }
        if (evaluatorContext.keepGoing()) {
          // Any children of this node that we haven't already visited are not worth visiting,
          // since this node is about to be done. Thus, the only child worth visiting is the one in
          // this cycle, the cycleChild (which may == key if this cycle is a self-edge).
          SkyKey cycleChild = selectCycleChild(key, graphPath, cycleStart);
          Set<SkyKey> removedDeps =
              removeDescendantsOfCycleValue(
                  key, entry, cycleChild, toVisit, graphPath.size() - cycleStart, evaluatorContext);
          ValueWithMetadata dummyValue = ValueWithMetadata.wrapWithMetadata(new SkyValue() {});

          SkyFunctionEnvironment env =
              new SkyFunctionEnvironment(
                  key,
                  entry.getTemporaryDirectDeps(),
                  ImmutableMap.of(cycleChild, dummyValue),
                  Sets.difference(entry.getAllRemainingDirtyDirectDeps(), removedDeps),
                  evaluatorContext);

          // Construct error info for this node. Get errors from children, which are all done
          // except possibly for the cycleChild.
          List<ErrorInfo> allErrors =
              getChildrenErrorsForCycleChecking(
                  Iterables.concat(entry.getTemporaryDirectDeps()),
                  /*unfinishedChild=*/ cycleChild,
                  evaluatorContext);
          CycleInfo cycleInfo = new CycleInfo(cycle);
          // Add in this cycle.
          allErrors.add(ErrorInfo.fromCycle(cycleInfo));
          env.setError(entry, ErrorInfo.fromChildErrors(key, allErrors));
          Set<SkyKey> reverseDeps = env.commitAndGetParents(entry);
          evaluatorContext.signalParentsOnAbort(key, reverseDeps, entry.getVersion());
          continue;
        } else {
          // We need to return right away in the noKeepGoing case, so construct the cycle (with the
          // path) and return.
          Preconditions.checkState(
              graphPath.get(0).equals(root),
              "%s not reached from %s. ValueEntry: %s",
              key,
              root,
              entry);
          return ErrorInfo.fromCycle(new CycleInfo(graphPath.subList(0, cycleStart), cycle));
        }
      }

      // This node is not yet known to be in a cycle. So process its children.
      GroupedList<SkyKey> temporaryDirectDeps = entry.getTemporaryDirectDeps();
      Iterable<SkyKey> children = temporaryDirectDeps.getAllElementsAsIterable();
      if (temporaryDirectDeps.isEmpty()) {
        continue;
      }
      // Prefetch all children, in case our graph performs better with a primed cache. No need to
      // recurse into done nodes. The fields of done nodes aren't necessary, since we'll filter them
      // out.
      // TODO(janakr): If graph implementations start using these hints for not-done nodes, we may
      // have to change this.
      Map<SkyKey, ? extends NodeEntry> childrenNodes =
          evaluatorContext.getGraph().getBatch(key, Reason.EXISTENCE_CHECKING, children);
      if (childrenNodes.size() != temporaryDirectDeps.numElements()) {
        ImmutableSet<SkyKey> childrenSet = ImmutableSet.copyOf(children);
        Set<SkyKey> missingChildren = Sets.difference(childrenSet, childrenNodes.keySet());
        if (missingChildren.isEmpty()) {
          logger.atWarning().log(
              "Mismatch for children?? %d, %d, %s, %s, %s, %s",
              childrenNodes.size(),
              temporaryDirectDeps.numElements(),
              childrenSet,
              childrenNodes,
              key,
              entry);
        } else {
          evaluatorContext
              .getGraphInconsistencyReceiver()
              .noteInconsistencyAndMaybeThrow(
                  key, missingChildren, Inconsistency.ALREADY_DECLARED_CHILD_MISSING);
          entry.removeUnfinishedDeps(missingChildren);
        }
      }
      children = Maps.filterValues(childrenNodes, nodeEntry -> !nodeEntry.isDone()).keySet();

      // This marker flag will tell us when all this node's children have been processed.
      toVisit.push(CHILDREN_FINISHED);
      // This node is now part of the path through the graph.
      graphPath.add(key);
      pathSet.add(key);
      for (SkyKey nextValue : children) {
        toVisit.push(nextValue);
      }
    }
    return evaluatorContext.keepGoing()
        ? getAndCheckDoneForCycle(root, evaluatorContext).getErrorInfo()
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
    if (entry.getDirtyState() != NodeEntry.DirtyState.VERIFIED_CLEAN) {
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

  /** The max number of cycles we will report to the user for a given root, to avoid OOMing. */
  private static final int MAX_CYCLES = 20;

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
      int childrenSize,
      NodeEntry entryForDebugging,
      ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    List<ErrorInfo> allErrors = new ArrayList<>();
    boolean foundCycle = false;
    Map<SkyKey, ? extends NodeEntry> childMap =
        getAndCheckDoneBatchForCycle(parent, children, evaluatorContext);
    if (childMap.size() < childrenSize) {
      Set<SkyKey> missingChildren =
          Sets.difference(ImmutableSet.copyOf(children), childMap.keySet());
      evaluatorContext
          .getGraphInconsistencyReceiver()
          .noteInconsistencyAndMaybeThrow(
              parent, missingChildren, Inconsistency.ALREADY_DECLARED_CHILD_MISSING);
    }
    for (NodeEntry childNode : childMap.values()) {
      ErrorInfo errorInfo = childNode.getErrorInfo();
      if (errorInfo != null) {
        foundCycle |= !Iterables.isEmpty(errorInfo.getCycleInfo());
        allErrors.add(errorInfo);
      }
    }
    Preconditions.checkState(
        foundCycle,
        "Key %s with entry %s had no cycle beneath it: %s",
        parent,
        entryForDebugging,
        allErrors);
    return allErrors;
  }

  /**
   * Get all the errors of child nodes.
   *
   * @param children child nodes to query for errors.
   * @param unfinishedChild child which is allowed to not be done.
   * @return List of ErrorInfos from all children that had errors.
   */
  private static List<ErrorInfo> getChildrenErrorsForCycleChecking(
      Iterable<SkyKey> children, SkyKey unfinishedChild, ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    List<ErrorInfo> allErrors = new ArrayList<>();
    Set<? extends Map.Entry<SkyKey, ? extends NodeEntry>> childEntries =
        evaluatorContext.getBatchValues(null, Reason.CYCLE_CHECKING, children).entrySet();
    for (Map.Entry<SkyKey, ? extends NodeEntry> childMapEntry : childEntries) {
      SkyKey childKey = childMapEntry.getKey();
      NodeEntry childNodeEntry = childMapEntry.getValue();
      ErrorInfo errorInfo =
          getErrorMaybe(
              childKey, childNodeEntry, /*allowUnfinished=*/ childKey.equals(unfinishedChild));
      if (errorInfo != null) {
        allErrors.add(errorInfo);
      }
    }
    return allErrors;
  }

  @Nullable
  private static ErrorInfo getErrorMaybe(
      SkyKey key, NodeEntry childNodeEntry, boolean allowUnfinished) throws InterruptedException {
    Preconditions.checkNotNull(childNodeEntry, key);
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
    GroupedList<SkyKey> directDeps = entry.getTemporaryDirectDeps();
    Set<SkyKey> unvisitedDeps = Sets.newHashSetWithExpectedSize(directDeps.numElements());
    Iterables.addAll(unvisitedDeps, Iterables.concat(directDeps));
    unvisitedDeps.remove(cycleChild);
    // Remove any children from this node that are not part of the cycle we just found. They are
    // irrelevant to the node as it stands, and if they are deleted from the graph because they are
    // not built by the end of cycle-checking, we would have dangling references.
    Set<SkyKey> removedDeps =
        removeIncompleteChildrenForCycle(key, entry, unvisitedDeps, evaluatorContext);
    if (!entry.isReady()) {
      // The entry has at most one undone dep now, its cycleChild. Signal to make entry ready. Note
      // that the entry can conceivably be ready if its cycleChild already found a different cycle
      // and was built.
      entry.signalDep(evaluatorContext.getGraphVersion(), cycleChild);
    }
    maybeMarkRebuilding(entry);
    Preconditions.checkState(entry.isReady(), "%s %s %s", key, cycleChild, entry);
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
        Preconditions.checkState(
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
    Preconditions.checkNotNull(entry, key);
    Preconditions.checkState(entry.isDone(), "%s %s", key, entry);
    return entry;
  }

  private static NodeEntry getAndCheckDoneForCycle(
      SkyKey key, ParallelEvaluatorContext evaluatorContext) throws InterruptedException {
    return checkDone(key, evaluatorContext.getGraph().get(null, Reason.CYCLE_CHECKING, key));
  }

  private static Map<SkyKey, ? extends NodeEntry> getAndCheckDoneBatchForCycle(
      SkyKey parent, Iterable<SkyKey> keys, ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    Map<SkyKey, ? extends NodeEntry> nodes =
        evaluatorContext.getBatchValues(parent, Reason.CYCLE_CHECKING, keys);
    for (Map.Entry<SkyKey, ? extends NodeEntry> nodeEntryMapEntry : nodes.entrySet()) {
      checkDone(nodeEntryMapEntry.getKey(), nodeEntryMapEntry.getValue());
    }
    return nodes;
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
      childEntry.removeInProgressReverseDep(inProgressParent);
      return true;
    }
    return false;
  }
}
