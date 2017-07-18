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

import static com.google.devtools.build.skyframe.ParallelEvaluator.isDoneForBuild;
import static com.google.devtools.build.skyframe.ParallelEvaluator.maybeGetValueFromError;

import com.google.common.base.Function;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationState;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.ParallelEvaluatorContext.EnqueueParentBehavior;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import javax.annotation.Nullable;

/** A {@link SkyFunction.Environment} implementation for {@link ParallelEvaluator}. */
class SkyFunctionEnvironment extends AbstractSkyFunctionEnvironment {
  private static final SkyValue NULL_MARKER = new SkyValue() {};
  private static final boolean PREFETCH_OLD_DEPS =
      Boolean.parseBoolean(
          System.getProperty("skyframe.ParallelEvaluator.PrefetchOldDeps", "true"));

  private boolean building = true;
  private SkyKey depErrorKey = null;
  private final SkyKey skyKey;
  /**
   * The deps requested during the previous build of this node. Used for two reasons: (1) They are
   * fetched eagerly before the node is built, to potentially prime the graph and speed up requests
   * for them during evaluation. (2) When the node finishes building, any deps from the previous
   * build that are not deps from this build must have this node removed from them as a reverse dep.
   * Thus, it is important that all nodes in this set have the property that they have this node as
   * a reverse dep from the last build, but that this node has not added them as a reverse dep on
   * this build. That set is normally {@link NodeEntry#getAllRemainingDirtyDirectDeps()}, but in
   * certain corner cases, like cycles, further filtering may be needed.
   */
  private final Set<SkyKey> oldDeps;

  private SkyValue value = null;
  private ErrorInfo errorInfo = null;
  private final Map<SkyKey, ValueWithMetadata> bubbleErrorInfo;
  /** The values previously declared as dependencies. */
  private final Map<SkyKey, NodeEntry> directDeps;

  /**
   * The grouped list of values requested during this build as dependencies. On a subsequent build,
   * if this value is dirty, all deps in the same dependency group can be checked in parallel for
   * changes. In other words, if dep1 and dep2 are in the same group, then dep1 will be checked in
   * parallel with dep2. See {@link #getValues} for more.
   */
  private final GroupedListHelper<SkyKey> newlyRequestedDeps = new GroupedListHelper<>();

  /** The set of errors encountered while fetching children. */
  private final Collection<ErrorInfo> childErrorInfos = new LinkedHashSet<>();

  private final StoredEventHandler eventHandler =
      new StoredEventHandler() {
        @Override
        @SuppressWarnings("UnsynchronizedOverridesSynchronized") // only delegates to thread-safe.
        public void handle(Event e) {
          checkActive();
          if (evaluatorContext.getStoredEventFilter().apply(e)) {
            super.handle(e);
          } else {
            evaluatorContext.getReporter().handle(e);
          }
        }

        @Override
        @SuppressWarnings("UnsynchronizedOverridesSynchronized") // only delegates to thread-safe.
        public void post(ExtendedEventHandler.Postable e) {
          if (e instanceof ExtendedEventHandler.ProgressLike) {
            evaluatorContext.getReporter().post(e);
          } else {
            super.post(e);
          }
        }
      };
  private final ParallelEvaluatorContext evaluatorContext;

  SkyFunctionEnvironment(
      SkyKey skyKey,
      GroupedList<SkyKey> directDeps,
      Set<SkyKey> oldDeps,
      ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    this(skyKey, directDeps, null, oldDeps, evaluatorContext);
  }

  SkyFunctionEnvironment(
      SkyKey skyKey,
      GroupedList<SkyKey> directDeps,
      @Nullable Map<SkyKey, ValueWithMetadata> bubbleErrorInfo,
      Set<SkyKey> oldDeps,
      ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    this.skyKey = skyKey;
    this.oldDeps = oldDeps;
    this.evaluatorContext = evaluatorContext;
    this.directDeps =
        Collections.<SkyKey, NodeEntry>unmodifiableMap(
            batchPrefetch(
                skyKey, directDeps, oldDeps, /*assertDone=*/ bubbleErrorInfo == null, skyKey));
    this.bubbleErrorInfo = bubbleErrorInfo;
    Preconditions.checkState(
        !this.directDeps.containsKey(ErrorTransienceValue.KEY),
        "%s cannot have a dep on ErrorTransienceValue during building",
        skyKey);
  }

  private Map<SkyKey, ? extends NodeEntry> batchPrefetch(
      SkyKey requestor,
      GroupedList<SkyKey> depKeys,
      Set<SkyKey> oldDeps,
      boolean assertDone,
      SkyKey keyForDebugging)
      throws InterruptedException {
    Iterable<SkyKey> depKeysAsIterable = Iterables.concat(depKeys);
    Iterable<SkyKey> keysToPrefetch = depKeysAsIterable;
    if (PREFETCH_OLD_DEPS) {
      ImmutableSet.Builder<SkyKey> keysToPrefetchBuilder = ImmutableSet.builder();
      keysToPrefetchBuilder.addAll(depKeysAsIterable).addAll(oldDeps);
      keysToPrefetch = keysToPrefetchBuilder.build();
    }
    Map<SkyKey, ? extends NodeEntry> batchMap =
        evaluatorContext.getBatchValues(requestor, Reason.PREFETCH, keysToPrefetch);
    if (PREFETCH_OLD_DEPS) {
      batchMap =
          ImmutableMap.<SkyKey, NodeEntry>copyOf(
              Maps.filterKeys(batchMap, Predicates.in(ImmutableSet.copyOf(depKeysAsIterable))));
    }
    if (batchMap.size() != depKeys.numElements()) {
      throw new IllegalStateException(
          "Missing keys for "
              + keyForDebugging
              + ": "
              + Sets.difference(depKeys.toSet(), batchMap.keySet()));
    }
    if (assertDone) {
      for (Map.Entry<SkyKey, ? extends NodeEntry> entry : batchMap.entrySet()) {
        Preconditions.checkState(
            entry.getValue().isDone(), "%s had not done %s", keyForDebugging, entry);
      }
    }
    return batchMap;
  }

  private void checkActive() {
    Preconditions.checkState(building, skyKey);
  }

  NestedSet<TaggedEvents> buildEvents(NodeEntry entry, boolean missingChildren)
      throws InterruptedException {
    // Aggregate the nested set of events from the direct deps, also adding the events from
    // building this value.
    NestedSetBuilder<TaggedEvents> eventBuilder = NestedSetBuilder.stableOrder();
    ImmutableList<Event> events = eventHandler.getEvents();
    if (!events.isEmpty()) {
      eventBuilder.add(new TaggedEvents(getTagFromKey(), events));
    }
    if (evaluatorContext.getStoredEventFilter().storeEvents()) {
      // Only do the work of processing children if we're going to store events.
      GroupedList<SkyKey> depKeys = entry.getTemporaryDirectDeps();
      Collection<SkyValue> deps = getDepValuesForDoneNodeMaybeFromError(depKeys);
      if (!missingChildren && depKeys.numElements() != deps.size()) {
        throw new IllegalStateException(
            "Missing keys for "
                + skyKey
                + ". Present values: "
                + deps
                + " requested from: "
                + depKeys
                + ", "
                + entry);
      }
      for (SkyValue value : deps) {
        eventBuilder.addTransitive(ValueWithMetadata.getEvents(value));
      }
    }
    return eventBuilder.build();
  }

  NestedSet<Postable> buildPosts(NodeEntry entry) throws InterruptedException {
    NestedSetBuilder<Postable> postBuilder = NestedSetBuilder.stableOrder();
    postBuilder.addAll(eventHandler.getPosts());

    GroupedList<SkyKey> depKeys = entry.getTemporaryDirectDeps();
    Collection<SkyValue> deps = getDepValuesForDoneNodeMaybeFromError(depKeys);
    for (SkyValue value : deps) {
      postBuilder.addTransitive(ValueWithMetadata.getPosts(value));
    }
    return postBuilder.build();
  }

  void setValue(SkyValue newValue) {
    Preconditions.checkState(
        errorInfo == null && bubbleErrorInfo == null,
        "%s %s %s %s",
        skyKey,
        newValue,
        errorInfo,
        bubbleErrorInfo);
    Preconditions.checkState(value == null, "%s %s %s", skyKey, value, newValue);
    value = newValue;
  }

  /**
   * Set this node to be in error. The node's value must not have already been set. However, all
   * dependencies of this node <i>must</i> already have been registered, since this method may
   * register a dependence on the error transience node, which should always be the last dep.
   */
  void setError(NodeEntry state, ErrorInfo errorInfo)  throws InterruptedException {
    Preconditions.checkState(value == null, "%s %s %s", skyKey, value, errorInfo);
    Preconditions.checkState(this.errorInfo == null, "%s %s %s", skyKey, this.errorInfo, errorInfo);

    if (errorInfo.isDirectlyTransient()) {
      NodeEntry errorTransienceNode =
          Preconditions.checkNotNull(
              evaluatorContext
                  .getGraph()
                  .get(skyKey, Reason.RDEP_ADDITION, ErrorTransienceValue.KEY),
              "Null error value? %s",
              skyKey);
      DependencyState triState;
      if (oldDeps.contains(ErrorTransienceValue.KEY)) {
        triState = errorTransienceNode.checkIfDoneForDirtyReverseDep(skyKey);
      } else {
        triState = errorTransienceNode.addReverseDepAndCheckIfDone(skyKey);
      }
      Preconditions.checkState(
          triState == DependencyState.DONE, "%s %s %s", skyKey, triState, errorInfo);
      state.addTemporaryDirectDeps(GroupedListHelper.create(ErrorTransienceValue.KEY));
      state.signalDep();
    }

    this.errorInfo = Preconditions.checkNotNull(errorInfo, skyKey);
  }

  private Map<SkyKey, SkyValue> getValuesMaybeFromError(Iterable<? extends SkyKey> keys)
      throws InterruptedException {
    // Use a HashMap, not an ImmutableMap.Builder, because we have not yet deduplicated these keys
    // and ImmutableMap.Builder does not tolerate duplicates.  The map will be thrown away
    // shortly in any case.
    Map<SkyKey, SkyValue> result = new HashMap<>();
    ArrayList<SkyKey> missingKeys = new ArrayList<>();
    for (SkyKey key : keys) {
      Preconditions.checkState(
          !key.equals(ErrorTransienceValue.KEY),
          "Error transience key cannot be in requested deps of %s",
          skyKey);
      SkyValue value = maybeGetValueFromErrorOrDeps(key);
      if (value == null) {
        missingKeys.add(key);
      } else {
        result.put(key, value);
      }
    }
    Map<SkyKey, ? extends NodeEntry> missingEntries =
        evaluatorContext.getBatchValues(skyKey, Reason.DEP_REQUESTED, missingKeys);
    for (SkyKey key : missingKeys) {
      result.put(key, getValueOrNullMarker(missingEntries.get(key)));
    }
    return result;
  }

  /**
   * Returns just the values of the deps in {@code depKeys}, looking at {@code bubbleErrorInfo},
   * {@link #directDeps}, and the backing {@link #evaluatorContext#graph} in that order. Any deps
   * that are not yet done will not have their values present in the returned collection.
   */
  private Collection<SkyValue> getDepValuesForDoneNodeMaybeFromError(GroupedList<SkyKey> depKeys)
      throws InterruptedException {
    int keySize = depKeys.numElements();
    List<SkyValue> result = new ArrayList<>(keySize);
    // depKeys consists of all known deps of this entry. That should include all the keys in
    // directDeps, and any keys in bubbleErrorInfo. We expect to have to retrieve the keys that
    // are not in either one.
    int expectedMissingKeySize =
        Math.max(
            keySize - directDeps.size() - (bubbleErrorInfo == null ? 0 : bubbleErrorInfo.size()),
            0);
    ArrayList<SkyKey> missingKeys = new ArrayList<>(expectedMissingKeySize);
    for (SkyKey key : Iterables.concat(depKeys)) {
      SkyValue value = maybeGetValueFromErrorOrDeps(key);
      if (value == null) {
        missingKeys.add(key);
      } else {
        result.add(value);
      }
    }
    for (NodeEntry entry :
        evaluatorContext.getBatchValues(skyKey, Reason.DEP_REQUESTED, missingKeys).values()) {
      result.add(getValueOrNullMarker(entry));
    }
    return result;
  }

  @Nullable
  private SkyValue maybeGetValueFromErrorOrDeps(SkyKey key) throws InterruptedException {
    return maybeGetValueFromError(key, directDeps.get(key), bubbleErrorInfo);
  }

  private static SkyValue getValueOrNullMarker(@Nullable NodeEntry nodeEntry)
      throws InterruptedException {
    return isDoneForBuild(nodeEntry) ? nodeEntry.getValueMaybeWithMetadata() : NULL_MARKER;
  }

  @Override
  protected Map<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) throws InterruptedException {
    checkActive();
    Map<SkyKey, SkyValue> values = getValuesMaybeFromError(depKeys);
    for (Map.Entry<SkyKey, SkyValue> depEntry : values.entrySet()) {
      SkyKey depKey = depEntry.getKey();
      SkyValue depValue = depEntry.getValue();
      if (depValue == NULL_MARKER) {
        if (directDeps.containsKey(depKey)) {
          throw new IllegalStateException(
              "Undone key "
                  + depKey
                  + " was already in deps of "
                  + skyKey
                  + "( dep: "
                  + evaluatorContext.getGraph().get(skyKey, Reason.OTHER, depKey)
                  + ", parent: "
                  + evaluatorContext.getGraph().get(null, Reason.OTHER, skyKey));
        }
        valuesMissing = true;
        addDep(depKey);
        continue;
      }
      ErrorInfo errorInfo = ValueWithMetadata.getMaybeErrorInfo(depEntry.getValue());
      if (errorInfo != null) {
        childErrorInfos.add(errorInfo);
        if (bubbleErrorInfo != null) {
          // Set interrupted status, to try to prevent the calling SkyFunction from doing anything
          // fancy after this. SkyFunctions executed during error bubbling are supposed to
          // (quickly) rethrow errors or return a value/null (but there's currently no way to
          // enforce this).
          Thread.currentThread().interrupt();
        }
        if ((!evaluatorContext.keepGoing() && bubbleErrorInfo == null)
            || errorInfo.getException() == null) {
          valuesMissing = true;
          // We arbitrarily record the first child error if we are about to abort.
          if (!evaluatorContext.keepGoing() && depErrorKey == null) {
            depErrorKey = depKey;
          }
        }
      }

      if (!directDeps.containsKey(depKey)) {
        if (bubbleErrorInfo == null) {
          addDep(depKey);
        }
        evaluatorContext
            .getReplayingNestedSetPostableVisitor()
            .visit(ValueWithMetadata.getPosts(depValue));
        evaluatorContext
            .getReplayingNestedSetEventVisitor()
            .visit(ValueWithMetadata.getEvents(depValue));
      }
    }

    return Maps.transformValues(
        values,
        new Function<SkyValue, ValueOrUntypedException>() {
          @Override
          public ValueOrUntypedException apply(SkyValue maybeWrappedValue) {
            if (maybeWrappedValue == NULL_MARKER) {
              return ValueOrExceptionUtils.ofNull();
            }
            SkyValue justValue = ValueWithMetadata.justValue(maybeWrappedValue);
            ErrorInfo errorInfo = ValueWithMetadata.getMaybeErrorInfo(maybeWrappedValue);

            if (justValue != null && (evaluatorContext.keepGoing() || errorInfo == null)) {
              // If the dep did compute a value, it is given to the caller if we are in
              // keepGoing mode or if we are in noKeepGoingMode and there were no errors computing
              // it.
              return ValueOrExceptionUtils.ofValueUntyped(justValue);
            }

            // There was an error building the value, which we will either report by throwing an
            // exception or insulate the caller from by returning null.
            Preconditions.checkNotNull(errorInfo, "%s %s", skyKey, maybeWrappedValue);
            Exception exception = errorInfo.getException();

            if (!evaluatorContext.keepGoing() && exception != null && bubbleErrorInfo == null) {
              // Child errors should not be propagated in noKeepGoing mode (except during error
              // bubbling). Instead we should fail fast.
              return ValueOrExceptionUtils.ofNull();
            }

            if (exception != null) {
              // Give builder a chance to handle this exception.
              return ValueOrExceptionUtils.ofExn(exception);
            }
            // In a cycle.
            Preconditions.checkState(
                !Iterables.isEmpty(errorInfo.getCycleInfo()),
                "%s %s %s",
                skyKey,
                errorInfo,
                maybeWrappedValue);
            return ValueOrExceptionUtils.ofNull();
          }
        });
  }

  @Override
  public <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      Map<SkyKey, ValueOrException5<E1, E2, E3, E4, E5>> getValuesOrThrow(
          Iterable<? extends SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4,
          Class<E5> exceptionClass5)
              throws InterruptedException {
    newlyRequestedDeps.startGroup();
    Map<SkyKey, ValueOrException5<E1, E2, E3, E4, E5>> result =
        super.getValuesOrThrow(
            depKeys,
            exceptionClass1,
            exceptionClass2,
            exceptionClass3,
            exceptionClass4,
            exceptionClass5);
    newlyRequestedDeps.endGroup();
    return result;
  }

  private void addDep(SkyKey key) {
    newlyRequestedDeps.add(key);
  }

  /**
   * If {@code !keepGoing} and there is at least one dep in error, returns a dep in error. Otherwise
   * returns {@code null}.
   */
  @Nullable
  SkyKey getDepErrorKey() {
    return depErrorKey;
  }

  @Override
  public ExtendedEventHandler getListener() {
    checkActive();
    return eventHandler;
  }

  void doneBuilding() {
    building = false;
  }

  GroupedListHelper<SkyKey> getNewlyRequestedDeps() {
    return newlyRequestedDeps;
  }

  Collection<NodeEntry> getDirectDepsValues() {
    return directDeps.values();
  }

  Collection<ErrorInfo> getChildErrorInfos() {
    return childErrorInfos;
  }

  /**
   * Apply the change to the graph (mostly) atomically and signal all nodes that are waiting for
   * this node to complete. Adding nodes and signaling is not atomic, but may need to be changed for
   * interruptibility.
   *
   * <p>Parents are only enqueued if {@code enqueueParents} holds. Parents should be enqueued unless
   * (1) this node is being built after the main evaluation has aborted, or (2) this node is being
   * built with --nokeep_going, and so we are about to shut down the main evaluation anyway.
   *
   * <p>The node entry is informed if the node's value and error are definitive via the flag {@code
   * completeValue}.
   */
  void commit(NodeEntry primaryEntry, EnqueueParentBehavior enqueueParents)
      throws InterruptedException {
    // Construct the definitive error info, if there is one.
    if (errorInfo == null) {
      errorInfo = evaluatorContext.getErrorInfoManager().getErrorInfoToUse(
          skyKey, value != null, childErrorInfos);
    }

    // We have the following implications:
    // errorInfo == null => value != null => enqueueParents.
    // All these implications are strict:
    // (1) errorInfo != null && value != null happens for values with recoverable errors.
    // (2) value == null && enqueueParents happens for values that are found to have errors
    // during a --keep_going build.

    NestedSet<Postable> posts = buildPosts(primaryEntry);
    NestedSet<TaggedEvents> events = buildEvents(primaryEntry, /*missingChildren=*/ false);

    Version valueVersion;
    SkyValue valueWithMetadata;
    if (value == null) {
      Preconditions.checkNotNull(errorInfo, "%s %s", skyKey, primaryEntry);
      valueWithMetadata = ValueWithMetadata.error(errorInfo, events, posts);
    } else {
      // We must be enqueueing parents if we have a value.
      Preconditions.checkState(
          enqueueParents == EnqueueParentBehavior.ENQUEUE, "%s %s", skyKey, primaryEntry);
      valueWithMetadata = ValueWithMetadata.normal(value, errorInfo, events, posts);
    }
    if (!oldDeps.isEmpty()) {
      // Remove the rdep on this entry for each of its old deps that is no longer a direct dep.
      Set<SkyKey> depsToRemove =
          Sets.difference(oldDeps, primaryEntry.getTemporaryDirectDeps().toSet());
      Collection<? extends NodeEntry> oldDepEntries =
          evaluatorContext.getGraph().getBatch(skyKey, Reason.RDEP_REMOVAL, depsToRemove).values();
      for (NodeEntry oldDepEntry : oldDepEntries) {
        oldDepEntry.removeReverseDep(skyKey);
      }
    }
    // If this entry is dirty, setValue may not actually change it, if it determines that
    // the data being written now is the same as the data already present in the entry.
    // We could consider using max(childVersions) here instead of graphVersion. When full
    // versioning is implemented, this would allow evaluation at a version between
    // max(childVersions) and graphVersion to re-use this result.
    Set<SkyKey> reverseDeps =
        primaryEntry.setValue(valueWithMetadata, evaluatorContext.getGraphVersion());
    // Note that if this update didn't actually change the value entry, this version may not
    // be the graph version.
    valueVersion = primaryEntry.getVersion();
    Preconditions.checkState(
        valueVersion.atMost(evaluatorContext.getGraphVersion()),
        "%s should be at most %s in the version partial ordering",
        valueVersion,
        evaluatorContext.getGraphVersion());

    // Tell the receiver that this value was built. If valueVersion.equals(graphVersion), it was
    // evaluated this run, and so was changed. Otherwise, it is less than graphVersion, by the
    // Preconditions check above, and was not actually changed this run -- when it was written
    // above, its version stayed below this update's version, so its value remains the same.
    // We use a SkyValueSupplier here because it keeps a reference to the entry, allowing for
    // the receiver to be confident that the entry is readily accessible in memory.
    evaluatorContext
        .getProgressReceiver()
        .evaluated(
            skyKey,
            new SkyValueSupplier(primaryEntry),
            valueVersion.equals(evaluatorContext.getGraphVersion())
                ? EvaluationState.BUILT
                : EvaluationState.CLEAN);

    evaluatorContext.signalValuesAndEnqueueIfReady(
        skyKey, reverseDeps, valueVersion, enqueueParents);

    evaluatorContext.getReplayingNestedSetPostableVisitor().visit(posts);
    evaluatorContext.getReplayingNestedSetEventVisitor().visit(events);
  }

  @Nullable
  private String getTagFromKey() {
    return evaluatorContext.getSkyFunctions().get(skyKey.functionName()).extractTag(skyKey);
  }

  /**
   * Gets the latch that is counted down when an exception is thrown in {@code
   * AbstractQueueVisitor}. For use in tests to check if an exception actually was thrown. Calling
   * {@code AbstractQueueVisitor#awaitExceptionForTestingOnly} can throw a spurious {@link
   * InterruptedException} because {@link CountDownLatch#await} checks the interrupted bit before
   * returning, even if the latch is already at 0. See bug "testTwoErrors is flaky".
   */
  CountDownLatch getExceptionLatchForTesting() {
    return evaluatorContext.getVisitor().getExceptionLatchForTestingOnly();
  }

  @Override
  public boolean inErrorBubblingForTesting() {
    return bubbleErrorInfo != null;
  }
}
