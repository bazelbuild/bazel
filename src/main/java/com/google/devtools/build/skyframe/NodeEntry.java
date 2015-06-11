// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import com.google.devtools.build.lib.util.Pair;

import java.util.Collection;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A node in the graph. All operations on this class are thread-safe.
 *
 * <p>This interface is public only for the benefit of alternative graph implementations outside of
 * the package.
 */
public interface NodeEntry {
  /**
   * Return code for {@link #addReverseDepAndCheckIfDone(SkyKey)}.
   */
  enum DependencyState {
    /** The node is done. */
    DONE,

    /**
     * The node was just created and needs to be scheduled for its first evaluation pass. The
     * evaluator is responsible for signaling the reverse dependency node.
     */
    NEEDS_SCHEDULING,

    /**
     * The node was already created, but isn't done yet. The evaluator is responsible for
     * signaling the reverse dependency node.
     */
    ADDED_DEP;
  }

  /**
   * Return code for {@link #getDirtyState}.
   */
  enum DirtyState {
    /**
     * The node's dependencies need to be checked to see if it needs to be rebuilt. The
     * dependencies must be obtained through calls to {@link #getNextDirtyDirectDeps} and checked.
     */
    CHECK_DEPENDENCIES,
    /**
     * All of the node's dependencies are unchanged, and the value itself was not marked changed,
     * so its current value is still valid -- it need not be rebuilt.
     */
    VERIFIED_CLEAN,
    /**
     * A rebuilding is required or in progress, because either the node itself changed or one of
     * its dependencies did.
     */
    REBUILDING
  }

  boolean keepEdges();

  /** Returns whether the entry has been built and is finished evaluating. */
  @ThreadSafe
  boolean isDone();

  /**
   * Returns the value stored in this entry. This method may only be called after the evaluation of
   * this node is complete, i.e., after {@link #setValue} has been called.
   */
  @ThreadSafe
  SkyValue getValue();


  /**
   * Returns the {@link SkyValue} for this entry and the metadata associated with it (Like events
   * and errors). This method may only be called after the evaluation of this node is complete,
   * i.e., after {@link #setValue} has been called.
   */
  @ThreadSafe
  ValueWithMetadata getValueWithMetadata();

  /**
   * Returns the value, even if dirty or changed. Returns null otherwise.
   */
  @ThreadSafe
  SkyValue toValue();

  /**
   * Returns an immutable iterable of the direct deps of this node. This method may only be called
   * after the evaluation of this node is complete, i.e., after {@link #setValue} has been called.
   *
   * <p>This method is not very efficient, but is only be called in limited circumstances --
   * when the node is about to be deleted, or when the node is expected to have no direct deps (in
   * which case the overhead is not so bad). It should not be called repeatedly for the same node,
   * since each call takes time proportional to the number of direct deps of the node.
   */
  @ThreadSafe
  Iterable<SkyKey> getDirectDeps();

  /**
   * Returns the error, if any, associated to this node. This method may only be called after
   * the evaluation of this node is complete, i.e., after {@link #setValue} has been called.
   */
  @Nullable
  @ThreadSafe
  ErrorInfo getErrorInfo();

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
   * node depends on another (b depends on a). When a finishes, it signals <b>exactly</b> the set
   * of reverse dependencies that are registered at the time of the {@code setValue} call. If b
   * comes in before a, it is signaled (and re-scheduled) by a, otherwise it needs to do that
   * itself.
   *
   * <p>{@code version} indicates the graph version at which this node is being written. If the
   * entry determines that the new value is equal to the previous value, the entry will keep its
   * current version. Callers can query that version to see if the node considers its value to have
   * changed.
   */
  @ThreadSafe
  Set<SkyKey> setValue(SkyValue value, Version version);

  /**
   * Queries if the node is done and adds the given key as a reverse dependency. The return code
   * indicates whether a) the node is done, b) the reverse dependency is the first one, so the
   * node needs to be scheduled, or c) the reverse dependency was added, and the node does not
   * need to be scheduled.
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
   * <p>If the parameter is {@code null}, then no reverse dependency is added, but we still check
   * if the node needs to be scheduled.
   */
  @ThreadSafe
  DependencyState addReverseDepAndCheckIfDone(SkyKey reverseDep);

  /**
   * Removes a reverse dependency.
   */
  @ThreadSafe
  void removeReverseDep(SkyKey reverseDep);

  /**
   * Returns a copy of the set of reverse dependencies. Note that this introduces a potential
   * check-then-act race; {@link #removeReverseDep} may fail for a key that is returned here.
   */
  @ThreadSafe
  Iterable<SkyKey> getReverseDeps();

  /**
   * Tell this node that one of its dependencies is now done. Callers must check the return value,
   * and if true, they must re-schedule this node for evaluation. Equivalent to
   * {@code #signalDep(Long.MAX_VALUE)}. Since this entry's version is less than
   * {@link Long#MAX_VALUE}, informing this entry that a child of it has version
   * {@link Long#MAX_VALUE} will force it to re-evaluate.
   */
  @ThreadSafe
  boolean signalDep();

  /**
   * Tell this entry that one of its dependencies is now done. Callers must check the return value,
   * and if true, they must re-schedule this node for evaluation.
   *
   * @param childVersion If this entry {@link #isDirty()} and {@code childVersion} is not at most
   * {@link #getVersion()}, then this entry records that one of its children has changed since it
   * was last evaluated (namely, it was last evaluated at version {@link #getVersion()} and the
   * child was last evaluated at {@code childVersion}. Thus, the next call to
   * {@link #getDirtyState()} will return {@link DirtyState#REBUILDING}.
   */
  @ThreadSafe
  boolean signalDep(Version childVersion);

  /**
   * Returns true if the entry is marked dirty, meaning that at least one of its transitive
   * dependencies is marked changed.
   */
  @ThreadSafe
  boolean isDirty();

  /**
   * Returns true if the entry is marked changed, meaning that it must be re-evaluated even if its
   * dependencies' values have not changed.
   */
  @ThreadSafe
  boolean isChanged();

  /**
   * Marks this node dirty, or changed if {@code isChanged} is true. The node is put in the
   * just-created state. It will be re-evaluated if necessary during the evaluation phase,
   * but if it has not changed, it will not force a re-evaluation of its parents.
   *
   * <p>{@code markDirty(b)} must not be called on an undone node if {@code isChanged() == b}.
   * It is the caller's responsibility to ensure that this does not happen.  Calling
   * {@code markDirty(false)} when {@code isChanged() == true} has no effect. The idea here is that
   * the caller will only ever want to call {@code markDirty()} a second time if a transition from a
   * dirty-unchanged state to a dirty-changed state is required.
   *
   * @return The direct deps and value of this entry, or null if the entry has already been marked
   * dirty. In the latter case, the caller should abort its handling of this node, since another
   * thread is already dealing with it.
   */
  @Nullable
  @ThreadSafe
  Pair<? extends Iterable<SkyKey>, ? extends SkyValue> markDirty(boolean isChanged);

  /**
   * Marks this entry as up-to-date at this version.
   *
   * @return {@link Set} of reverse dependencies to signal that this node is done.
   */
  @ThreadSafe
  Set<SkyKey> markClean();

  /**
   * Forces this node to be re-evaluated, even if none of its dependencies are known to have
   * changed.
   *
   * <p>Used when an external caller has reason to believe that re-evaluating may yield a new
   * result. This method should not be used if one of the normal deps of this node has changed,
   * the usual change-pruning process should work in that case.
   */
  @ThreadSafe
  void forceRebuild();

  /**
   * Gets the current version of this entry.
   */
  @ThreadSafe
  Version getVersion();

  /**
   * Gets the current state of checking this dirty entry to see if it must be re-evaluated. Must be
   * called each time evaluation of a dirty entry starts to find the proper action to perform next,
   * as enumerated by {@link NodeEntry.DirtyState}.
   */
  @ThreadSafe
  NodeEntry.DirtyState getDirtyState();

  /**
   * Should only be called if the entry is dirty. During the examination to see if the entry must be
   * re-evaluated, this method returns the next group of children to be checked. Callers should
   * have already called {@link #getDirtyState} and received a return value of
   * {@link DirtyState#CHECK_DEPENDENCIES} before calling this method -- any other
   * return value from {@link #getDirtyState} means that this method must not be called, since
   * whether or not the node needs to be rebuilt is already known.
   *
   * <p>Deps are returned in groups. The deps in each group were requested in parallel by the
   * {@code SkyFunction} last build, meaning independently of the values of any other deps in this
   * group (although possibly depending on deps in earlier groups). Thus the caller may check all
   * the deps in this group in parallel, since the deps in all previous groups are verified
   * unchanged. See {@link SkyFunction.Environment#getValues} for more on dependency groups.
   *
   * <p>The caller should register these as deps of this entry using {@link #addTemporaryDirectDeps}
   * before checking them.
   *
   * @see BuildingState#getNextDirtyDirectDeps()
   */
  @ThreadSafe
  Collection<SkyKey> getNextDirtyDirectDeps();

  /**
   * Returns the set of direct dependencies. This may only be called while the node is being
   * evaluated, that is, before {@link #setValue} and after {@link #markDirty}.
   */
  @ThreadSafe
  Set<SkyKey> getTemporaryDirectDeps();

  @ThreadSafe
  boolean noDepsLastBuild();

  /**
   * Remove dep from direct deps. This should only be called if this entry is about to be
   * committed as a cycle node, but some of its children were not checked for cycles, either
   * because the cycle was discovered before some children were checked; some children didn't have a
   * chance to finish before the evaluator aborted; or too many cycles were found when it came time
   * to check the children.
   */
  @ThreadSafe
  void removeUnfinishedDeps(Set<SkyKey> unfinishedDeps);

  @ThreadSafe
  void addTemporaryDirectDeps(GroupedListHelper<SkyKey> helper);

  /**
   * Returns true if the node is ready to be evaluated, i.e., it has been signaled exactly as many
   * times as it has temporary dependencies. This may only be called while the node is being
   * evaluated, that is, before {@link #setValue} and after {@link #markDirty}.
   */
  @ThreadSafe
  boolean isReady();
}
