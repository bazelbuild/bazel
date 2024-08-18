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
package com.google.devtools.build.skyframe;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.skyframe.SkyFunction.Reset;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A node in the graph. All operations on this class are thread-safe.
 *
 * <p>This interface is public only for the benefit of alternative graph implementations outside of
 * the package.
 *
 * <p>Certain graph implementations' node entries can throw {@link InterruptedException} on various
 * accesses. Such exceptions should not be caught locally -- they should be allowed to propagate up.
 */
public interface NodeEntry {

  /**
   * Return code for {@link #addReverseDepAndCheckIfDone} and {@link
   * #checkIfDoneForDirtyReverseDep}.
   */
  enum DependencyState {
    /** The node is done. */
    DONE,

    /**
     * The node has not started evaluating, and needs to be scheduled for its first evaluation pass.
     * The caller getting this return value is responsible for scheduling its evaluation and
     * signaling the reverse dependency node when this node is done.
     */
    NEEDS_SCHEDULING,

    /**
     * The node was already created, but isn't done yet. The evaluator is responsible for signaling
     * the reverse dependency node.
     */
    ALREADY_EVALUATING
  }

  /** Represents the various states in a node's lifecycle. */
  enum LifecycleState {
    /**
     * The entry has never started evaluating. The next call to {@link #addReverseDepAndCheckIfDone}
     * will put the entry into the {@link #NEEDS_REBUILDING} state and return {@link
     * DependencyState#NEEDS_SCHEDULING}.
     */
    NOT_YET_EVALUATING,
    /**
     * The node's dependencies need to be checked to see if it needs to be rebuilt. The dependencies
     * must be obtained through calls to {@link #getNextDirtyDirectDeps} and checked.
     */
    CHECK_DEPENDENCIES,
    /**
     * All of the node's dependencies are unchanged, and the value itself was not marked changed, so
     * its current value is still valid -- it need not be rebuilt.
     */
    VERIFIED_CLEAN,
    /**
     * A rebuilding is required for one of the following reasons:
     *
     * <ol>
     *   <li>One of the node's dependencies changed.
     *   <li>The node is built by a {@link FunctionHermeticity#NONHERMETIC} function and its value
     *       is known to have changed due to state outside of Skyframe.
     *   <li>The node was {@linkplain DirtyType#REWIND rewound}.
     * </ol>
     */
    NEEDS_REBUILDING,
    /** A rebuilding is in progress. */
    REBUILDING,
    /** The node {@link #isDone}. */
    DONE,
  }

  /** Ways that a node may be dirtied. */
  enum DirtyType {

    /**
     * Indicates that the node is being marked dirty because it has a dependency that was marked
     * dirty.
     *
     * <p>A node P dirtied with {@code DIRTY} is re-evaluated during the evaluation phase if it is
     * requested and directly depends on some node C whose value changed since the last evaluation
     * of P. If it is requested and there is no such node C, P is {@linkplain #markClean marked
     * clean}.
     */
    DIRTY,

    /**
     * Indicates that the node is being marked dirty because its value from a previous evaluation is
     * no longer valid, even if none of its dependencies change.
     *
     * <p>This is typically used to indicate that a value produced by a {@link
     * FunctionHermeticity#NONHERMETIC} function is no longer valid because some state outside of
     * Skyframe has changed (e.g. a change to the filesystem).
     *
     * <p>A node dirtied with {@code CHANGE} is re-evaluated during the evaluation phase if it is
     * requested, regardless of the state of its dependencies. If it re-evaluates to the same value,
     * dirty parents are not necessarily re-evaluated.
     */
    CHANGE,

    /**
     * Similar to {@link #CHANGE} except may be used intra-evaluation to indicate that the node's
     * value (which may be from either a previous evaluation or the current evaluation) is no longer
     * valid.
     *
     * <p>A node dirtied with {@code REWIND} is re-evaluated during the evaluation phase if it is
     * requested, regardless of the state of its dependencies. Even if it re-evaluates to the same
     * value, dirty parents are re-evaluated.
     *
     * <p>Rewinding is tolerated but no-op if the node is already dirty or is done with an
     * {@linkplain #getErrorInfo() error} (regardless of the error's {@link
     * com.google.devtools.build.skyframe.SkyFunctionException.Transience}).
     */
    REWIND
  }

  /** Returns whether the entry has been built and is finished evaluating. */
  @ThreadSafe
  boolean isDone();

  /** Inverse of {@link #isDone}. */
  @ThreadSafe
  boolean isDirty();

  /**
   * Returns true if the entry is marked changed, meaning that it must be re-evaluated even if its
   * dependencies' values have not changed.
   */
  @ThreadSafe
  boolean isChanged();

  /**
   * Marks this node dirty as specified by the provided {@link DirtyType}.
   *
   * <p>{@code markDirty(DirtyType.DIRTY)} may only be called on a node P for which {@code
   * P.isDone() || P.isChanged()} (the latter is permitted but has no effect). Similarly, {@code
   * markDirty(DirtyType.CHANGE)} may only be called on a node P for which {@code P.isDone() ||
   * !P.isChanged()}. Otherwise, this will throw {@link IllegalStateException}.
   *
   * <p>{@code markDirty(DirtyType.REWIND)} may be called at any time (even multiple times
   * concurrently), although it only has an effect if the node {@link #isDone} with no error.
   *
   * @return if the node transitioned from done to dirty as a result of this call, a {@link
   *     MarkedDirtyResult} which may include the node's reverse deps; otherwise {@code null}
   */
  @Nullable
  @ThreadSafe
  MarkedDirtyResult markDirty(DirtyType dirtyType) throws InterruptedException;

  /**
   * Returned by {@link #markDirty} if that call changed the node from done to dirty.
   *
   * <p>For nodes marked dirty during invalidation ({@link DirtyType#DIRTY} and {@link
   * DirtyType#CHANGE}), contains a {@link Collection} of the node's reverse deps for efficiency, so
   * that the invalidator can schedule the invalidation of a node's reverse deps immediately
   * afterwards.
   *
   * <p>For nodes marked dirty intra-evaluation ({@link DirtyType#REWIND}), reverse deps are not
   * needed by the caller, so {@link #getReverseDepsUnsafe} must not be called.
   *
   * <p>Warning: {@link #getReverseDepsUnsafe()} may return a live view of the reverse deps
   * collection of the marked-dirty node. The consumer of this data must be careful only to iterate
   * over and consume its values while that collection is guaranteed not to change. This is true
   * during invalidation, because reverse deps don't change during invalidation.
   */
  abstract class MarkedDirtyResult {

    private static final MarkedDirtyResult RESULT_FOR_REWINDING =
        new MarkedDirtyResult() {
          @Override
          public Collection<SkyKey> getReverseDepsUnsafe() {
            throw new IllegalStateException("Should not need reverse deps for rewinding");
          }
        };

    public static MarkedDirtyResult withReverseDeps(Collection<SkyKey> reverseDepsUnsafe) {
      return new ResultWithReverseDeps(reverseDepsUnsafe);
    }

    static MarkedDirtyResult forRewinding() {
      return RESULT_FOR_REWINDING;
    }

    private MarkedDirtyResult() {}

    public abstract Collection<SkyKey> getReverseDepsUnsafe();

    private static final class ResultWithReverseDeps extends MarkedDirtyResult {
      private final Collection<SkyKey> reverseDepsUnsafe;

      private ResultWithReverseDeps(Collection<SkyKey> reverseDepsUnsafe) {
        this.reverseDepsUnsafe = checkNotNull(reverseDepsUnsafe);
      }

      @Override
      public Collection<SkyKey> getReverseDepsUnsafe() {
        return reverseDepsUnsafe;
      }
    }
  }

  /**
   * Returns the value stored in this entry, or {@code null} if it has only an error.
   *
   * <p>This method may only be called when the node {@link #isDone}.
   */
  @ThreadSafe
  @Nullable
  SkyValue getValue() throws InterruptedException;

  /**
   * Returns an immutable iterable of the direct deps of this node. This method may only be called
   * after the evaluation of this node is complete.
   *
   * <p>This method is not very efficient, but is only be called in limited circumstances -- when
   * the node is about to be deleted, or when the node is expected to have no direct deps (in which
   * case the overhead is not so bad). It should not be called repeatedly for the same node, since
   * each call takes time proportional to the number of direct deps of the node.
   */
  @ThreadSafe
  Iterable<SkyKey> getDirectDeps() throws InterruptedException;

  /**
   * Returns {@code true} if this node has at least one direct dep.
   *
   * <p>Prefer calling this over {@link #getDirectDeps} if possible.
   *
   * <p>This method may only be called after the evaluation of this node is complete.
   */
  @ThreadSafe
  boolean hasAtLeastOneDep() throws InterruptedException;

  /** Removes a reverse dependency, which must be present. */
  @ThreadSafe
  void removeReverseDep(SkyKey reverseDep) throws InterruptedException;

  /**
   * Removes any reverse dependencies that are in {@code deletedKeys}. Must only be called from an
   * invalidation that is deleting nodes from the graph. Sacrifices correctness checks (that the
   * deleted rdeps were actually rdeps of this entry) for better performance.
   */
  @ThreadSafe
  void removeReverseDepsFromDoneEntryDueToDeletion(Set<SkyKey> deletedKeys);

  /**
   * Returns a copy of the set of reverse dependencies. Note that this introduces a potential
   * check-then-act race; {@link #removeReverseDep} may fail for a key that is returned here.
   *
   * <p>May only be called on a done node entry.
   */
  @ThreadSafe
  Collection<SkyKey> getReverseDepsForDoneEntry() throws InterruptedException;

  /**
   * Returns raw {@link SkyValue} stored in this entry, which may include metadata associated with
   * it (like events and errors).
   *
   * <p>This method returns {@code null} if the evaluation of this node is not complete, i.e., after
   * node creation or dirtying and before {@link #setValue} has been called. Callers should assert
   * that the returned value is not {@code null} whenever they expect the node should be done.
   *
   * <p>Use the static methods of {@link ValueWithMetadata} to extract metadata if necessary.
   */
  @ThreadSafe
  @Nullable
  SkyValue getValueMaybeWithMetadata() throws InterruptedException;

  /**
   * Returns the last known value of this node, even if it was {@linkplain #markDirty marked dirty}.
   *
   * <p>If this node {@link #isDone}, this is equivalent to {@link #getValue}. Unlike {@link
   * #getValue}, however, this method may be called at any point in the node's lifecycle. Returns
   * {@code null} if this node was never built or has no value because it is in error.
   */
  @ThreadSafe
  @Nullable
  SkyValue toValue() throws InterruptedException;

  /**
   * Returns the error, if any, associated to this node. This method may only be called after the
   * evaluation of this node is complete, i.e., after {@link #setValue} has been called.
   */
  @Nullable
  @ThreadSafe
  ErrorInfo getErrorInfo() throws InterruptedException;

  /**
   * Returns the set of reverse deps that have been declared so far this build. Only for use in
   * debugging and when bubbling errors up in the --nokeep_going case, where we need to know what
   * parents this entry has.
   */
  @ThreadSafe
  Set<SkyKey> getInProgressReverseDeps();

  /**
   * Transitions the node from the EVALUATING to the DONE state and simultaneously sets it to the
   * given value and error state. It then returns the set of reverse dependencies that need to be
   * signaled.
   *
   * <p>This is an atomic operation to avoid a race where two threads work on two nodes, where one
   * node depends on another (b depends on a). When a finishes, it signals <b>exactly</b> the set of
   * reverse dependencies that are registered at the time of the {@code setValue} call. If b comes
   * in before a, it is signaled (and re-scheduled) by a, otherwise it needs to do that itself.
   *
   * <p>Nodes may elect to use either {@code graphVersion} or {@code maxTransitiveSourceVersion} (if
   * not {@code null}) for their {@linkplain #getVersion version}. The choice can be distinguished
   * by calling {@link #getMaxTransitiveSourceVersion} - a return of {@code null} indicates that the
   * node uses the graph version.
   *
   * <p>If the entry determines that the new value is equal to the previous value, the entry may
   * keep its current version. Callers can query that version to see if the node considers its value
   * to have changed.
   *
   * @param value the new value of this node
   * @param graphVersion the version of the graph at which this node is being written
   * @param maxTransitiveSourceVersion the maximal version of this node's dependencies from source,
   *     or {@code null} if source versions are not being tracked
   */
  @ThreadSafe
  Set<SkyKey> setValue(
      SkyValue value, Version graphVersion, @Nullable Version maxTransitiveSourceVersion)
      throws InterruptedException;

  /**
   * Sets the max transitive source version of this node so far while it is being evaluated. May
   * only be called when {@link #isDirty()} is {@code true}.
   *
   * <p>This method helps to track the in-progress max transitive source version across Skyframe
   * restarts. The eventual max transitive source version is set when {@link #setValue} is called.
   *
   * <p>This function is a no-op if source versions are not being tracked.
   */
  default void setTemporaryMaxTransitiveSourceVersion(
      @Nullable Version maxTransitiveSourceVersion) {}

  /**
   * Queries if the node is done and adds the given key as a reverse dependency. The return code
   * indicates whether a) the node is done, b) the reverse dependency is the first one, so the node
   * needs to be scheduled, or c) the reverse dependency was added, and the node does not need to be
   * scheduled.
   *
   * <p>This method <b>must</b> be called before any processing of the entry. This encourages
   * callers to check that the entry is ready to be processed.
   *
   * <p>Adding the dependency and checking if the node needs to be scheduled is an atomic operation
   * to avoid a race where two threads work on two nodes, where one depends on the other (b depends
   * on a). In that case, we need to ensure that b is re-scheduled exactly once when a is done.
   * However, a may complete first, in which case b has to re-schedule itself. Also see {@link
   * #setValue}.
   *
   * <p>If the parameter is {@code null}, then no reverse dependency is added, but we still check if
   * the node needs to be scheduled.
   *
   * <p>If {@code reverseDep} is a rebuilding dirty entry that was already a reverse dep of this
   * entry, then {@link #checkIfDoneForDirtyReverseDep} must be called instead.
   */
  @ThreadSafe
  DependencyState addReverseDepAndCheckIfDone(@Nullable SkyKey reverseDep)
      throws InterruptedException;

  /**
   * Similar to {@link #addReverseDepAndCheckIfDone}, except that {@code reverseDep} must already be
   * a reverse dep of this entry. Should be used when reverseDep has been marked dirty and is
   * checking its dependencies for changes or is rebuilding. The caller must treat the return value
   * just as they would the return value of {@link #addReverseDepAndCheckIfDone} by scheduling this
   * node for evaluation if needed.
   */
  @ThreadSafe
  DependencyState checkIfDoneForDirtyReverseDep(SkyKey reverseDep) throws InterruptedException;

  Collection<SkyKey> getAllReverseDepsForNodeBeingDeleted();

  /**
   * Tell this entry that one of its dependencies is now done. Callers must check the return value,
   * and if true, they must re-schedule this node for evaluation.
   *
   * <p>Even if {@code childVersion} is not at most {@link #getVersion}, this entry may not rebuild,
   * in the case that the entry already rebuilt at {@code childVersion} and discovered that it had
   * the same value as at an earlier version. For instance, after evaluating at version v1, at
   * version v2, child has a new value, but parent re-evaluates and finds it has the same value,
   * child.getVersion() will return v2 and parent.getVersion() will return v1. At v3 parent is
   * dirtied and checks its dep on child. child signals parent with version v2. That should not in
   * and of itself trigger a rebuild, since parent has already rebuilt with child at v2.
   *
   * @param childVersion If this entry {@link #isDirty} and the last version at which this entry was
   *     evaluated did not include the changes at version {@code childVersion} (for instance, if
   *     {@code childVersion} is after the last version at which this entry was evaluated), then
   *     this entry records that one of its children has changed since it was last evaluated. Thus,
   *     the next call to {@link #getLifecycleState} will return {@link
   *     LifecycleState#NEEDS_REBUILDING}.
   * @param childForDebugging for use in debugging (can be used to identify specific children that
   *     invalidate this node)
   */
  @ThreadSafe
  boolean signalDep(Version childVersion, @Nullable SkyKey childForDebugging);

  /**
   * Marks this entry as up-to-date at this version.
   *
   * @return {@link NodeValueAndRdepsToSignal} containing the SkyValue and reverse deps to signal.
   */
  @ThreadSafe
  NodeValueAndRdepsToSignal markClean() throws InterruptedException;

  /**
   * Returned by {@link #markClean} after making a node as clean. This is an aggregate object that
   * contains the NodeEntry's SkyValue and its reverse dependencies that signal this node is done (a
   * subset of all of the node's reverse dependencies).
   */
  final class NodeValueAndRdepsToSignal {
    private final SkyValue value;
    private final Set<SkyKey> rDepsToSignal;

    public NodeValueAndRdepsToSignal(SkyValue value, Set<SkyKey> rDepsToSignal) {
      this.value = value;
      this.rDepsToSignal = rDepsToSignal;
    }

    SkyValue getValue() {
      return this.value;
    }

    Set<SkyKey> getRdepsToSignal() {
      return this.rDepsToSignal;
    }
  }

  /**
   * Called on a dirty node during {@linkplain LifecycleState#CHECK_DEPENDENCIES dependency
   * checking} to force the node to be re-evaluated, even if none of its dependencies are known to
   * have changed.
   *
   * <p>Used when a caller has reason to believe that re-evaluating may yield a new result, such as
   * when the prior evaluation encountered a transient error.
   */
  @ThreadSafe
  void forceRebuild();

  /** Returns the current version of this node. */
  @ThreadSafe
  Version getVersion();

  /**
   * Returns the maximal version of this node's dependencies from source.
   *
   * <p>This version should only be tracked when non-hermetic functions {@linkplain
   * SkyFunction.Environment#injectVersionForNonHermeticFunction inject} source versions. Otherwise,
   * returns {@code null} to signal that source versions are not being tracked.
   */
  @ThreadSafe
  @Nullable
  default Version getMaxTransitiveSourceVersion() {
    return null;
  }

  /**
   * Returns the state of this entry as enumerated by {@link LifecycleState}.
   *
   * <p>This method may be called at any time. Returns {@link LifecycleState#DONE} iff the node
   * {@link #isDone}.
   */
  @ThreadSafe
  LifecycleState getLifecycleState();

  /**
   * Should only be called if the entry is in the {@link LifecycleState#CHECK_DEPENDENCIES} state.
   * During the examination to see if the entry must be re-evaluated, this method returns the next
   * group of children to be checked. Callers should have already called {@link #getLifecycleState}
   * and received a return value of {@link LifecycleState#CHECK_DEPENDENCIES} before calling this
   * method -- any other return value from {@link #getLifecycleState} means that this method must
   * not be called, since whether or not the node needs to be rebuilt is already known.
   *
   * <p>Deps are returned in groups. The deps in each group were requested in parallel by the {@code
   * SkyFunction} last build, meaning independently of the values of any other deps in this group
   * (although possibly depending on deps in earlier groups). Thus the caller may check all the deps
   * in this group in parallel, since the deps in all previous groups are verified unchanged. See
   * {@link SkyFunction.Environment#getValuesAndExceptions} for more on dependency groups.
   *
   * @see DirtyBuildingState#getNextDirtyDirectDeps()
   */
  @ThreadSafe
  List<SkyKey> getNextDirtyDirectDeps() throws InterruptedException;

  /**
   * Returns all deps of a node that has not yet finished evaluating. In other words, if a node has
   * a reverse dep on this node, its key will be in the returned set here.
   *
   * <p>The returned set is the union of:
   *
   * <ul>
   *   <li>This node's {@linkplain #getTemporaryDirectDeps temporary direct deps}.
   *   <li>Deps from a previous evaluation, if this this node was {@linkplain #markDirty marked
   *       dirty} (all the elements that would have been returned by successive calls to {@link
   *       #getNextDirtyDirectDeps} or, equivalently, one call to {@link
   *       #getAllRemainingDirtyDirectDeps}).
   *   <li>This node's {@linkplain #getResetDirectDeps reset direct deps}.
   * </ul>
   *
   * <p>This method should only be called when this node is about to be deleted after an aborted
   * evaluation. After such an evaluation, any nodes that did not finish evaluating are deleted, as
   * are any nodes that depend on them, which are necessarily also not done. If this node is to be
   * deleted because of this, we must delete it as a reverse dep from other nodes. This method
   * returns that list of other nodes. This method may not be called on done nodes, since they do
   * not need to be deleted after aborted evaluations.
   *
   * <p>This method must not be called twice: the next thing done to this node after this method is
   * called should be the removal of the node from the graph.
   */
  ImmutableSet<SkyKey> getAllDirectDepsForIncompleteNode() throws InterruptedException;

  /**
   * If an entry {@link #isDirty}, returns all direct deps that were present last build, but have
   * not yet been verified to be present during the current build. Implementations may lazily remove
   * these deps, since in many cases they will be added back during this build, even though the node
   * may have a changed value. However, any elements of this returned set that have not been added
   * back by the end of evaluation <i>must</i> be removed from any done nodes, in order to preserve
   * graph consistency.
   *
   * <p>Returns the empty set if an entry is not dirty. In either case, the entry must already have
   * started evaluation.
   *
   * <p>This method does not mutate the entry. In particular, multiple calls to this method will
   * always produce the same result until the entry finishes evaluation. Contrast with {@link
   * #getAllDirectDepsForIncompleteNode}.
   */
  ImmutableSet<SkyKey> getAllRemainingDirtyDirectDeps() throws InterruptedException;

  /**
   * Notifies a node that it is about to be rebuilt. This method can only be called if the node
   * {@link LifecycleState#NEEDS_REBUILDING}. After this call, this node is ready to be rebuilt (it
   * will be in {@link LifecycleState#REBUILDING}).
   */
  void markRebuilding();

  /**
   * Returns the {@link GroupedDeps} of direct dependencies. This may only be called while the node
   * is being evaluated (i.e. before {@link #setValue} and after {@link #markDirty}.
   */
  @ThreadSafe
  GroupedDeps getTemporaryDirectDeps();

  @ThreadSafe
  boolean noDepsLastBuild();

  /**
   * Remove dep from direct deps. This should only be called if this entry is about to be committed
   * as a cycle node, but some of its children were not checked for cycles, either because the cycle
   * was discovered before some children were checked; some children didn't have a chance to finish
   * before the evaluator aborted; or too many cycles were found when it came time to check the
   * children.
   */
  @ThreadSafe
  void removeUnfinishedDeps(Set<SkyKey> unfinishedDeps);

  /**
   * Prepares this node to reset its evaluation from scratch in order to recover from an
   * inconsistency.
   *
   * <p>Temporary direct deps should be cleared by this call, as they will be added again when
   * requested during the restarted evaluation of this node. If the graph keeps dependency edges,
   * however, the temporary direct deps must be accounted for in {@link #getResetDirectDeps}.
   *
   * <p>Called on a {@link LifecycleState#REBUILDING} node when one of the following scenarios is
   * observed:
   *
   * <ol>
   *   <li>One or more already requested dependencies are not done. This may happen when a
   *       dependency's node was dropped from the graph to save memory, or if a dependency was
   *       {@linkplain DirtyType#REWIND rewound} by another node.
   *   <li>The corresponding {@link SkyFunction} for this node returned {@link Reset} to indicate
   *       that one or more dependencies were done but are in need of {@linkplain DirtyType#REWIND
   *       rewinding} to regenerate their values.
   * </ol>
   *
   * <p>This method is similar to calling {@link #markDirty} with {@link DirtyType#REWIND} with an
   * important distinction: rewinding is initiated on a <em>done</em> node because of an issue with
   * its <em>value</em>, while this method is called on a <em>building</em> node because of an issue
   * with a <em>dependency</em>. The dependency will be rewound if we are in scenario 2 above.
   *
   * <p>Reverse deps on the other hand should be preserved - parents waiting on this node are
   * unaware that it is being restarted and will not register themselves again, yet they still need
   * to be signaled when this node is done.
   */
  @ThreadSafe
  void resetEvaluationFromScratch();

  /**
   * If the graph keeps dependency edges and {@link #resetEvaluationFromScratch} has been called on
   * this node since it was last done, returns the set of temporary direct deps that were registered
   * prior to the restart. Otherwise, returns an empty set.
   *
   * <p>Called on a {@link LifecycleState#REBUILDING} node when it is about to finish evaluating.
   * Used to determine which of its {@linkplain #getTemporaryDirectDeps temporary direct deps} have
   * already registered a corresponding reverse dep, in order to avoid creating duplicate rdep
   * edges.
   *
   * <p>Like {@link #getAllRemainingDirtyDirectDeps}, keys in the returned set are assumed to have
   * already registered an rdep on this node. Unlike {@link #getAllRemainingDirtyDirectDeps},
   * however, deps in the returned set may have only been registered at the current evaluation
   * version, not a previous one.
   *
   * <p>If this node was reset multiple times since it was last done, must return deps requested
   * prior to <em>any</em> of those restarts, not just the most recent one.
   */
  @ThreadSafe
  ImmutableSet<SkyKey> getResetDirectDeps();

  /**
   * Adds a temporary direct dep in its own group.
   *
   * <p>The given dep must not be present in this node's existing temporary direct deps.
   */
  @ThreadSafe
  void addSingletonTemporaryDirectDep(SkyKey dep);

  /**
   * Adds a temporary direct group.
   *
   * <p>The group must be duplicate-free and not contain any deps in common with this node's
   * existing temporary direct deps.
   */
  @ThreadSafe
  void addTemporaryDirectDepGroup(List<SkyKey> group);

  /**
   * Adds temporary direct deps in groups.
   *
   * <p>The iteration order of the given deps along with the {@code groupSizes} parameter dictate
   * how deps are grouped. For example, if {@code deps = {a,b,c}} and {@code groupSizes = [2, 1]},
   * then there will be two groups: {@code [a,b]} and {@code [c]}. The sum of {@code groupSizes}
   * must equal the size of {@code deps}. Note that it only makes sense to call this method with a
   * set implementation that has a stable iteration order.
   *
   * <p>The given set of deps must not contain any deps in common with this node's existing
   * temporary direct deps.
   */
  @ThreadSafe
  void addTemporaryDirectDepsInGroups(Set<SkyKey> deps, List<Integer> groupSizes);

  void addExternalDep();

  /**
   * Returns true if the node has been signaled exactly as many times as it has temporary
   * dependencies, or if {@code getKey().supportsPartialReevaluation()}. This may only be called
   * while the node is being evaluated (i.e. before {@link #setValue} and after {@link #markDirty}).
   */
  @ThreadSafe
  boolean isReadyToEvaluate();

  /**
   * Returns true if the node has not been signaled exactly as many times as it has temporary
   * dependencies. This may only be called while the node is being evaluated (i.e. before {@link
   * #setValue} and after {@link #markDirty}).
   *
   * <p>The node must not complete or be reset while in this state because it may yet be signaled.
   */
  @ThreadSafe
  boolean hasUnsignaledDeps();
}
