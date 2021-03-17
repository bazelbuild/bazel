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


import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.collect.compacthashmap.CompactHashMap;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationState;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.ParallelEvaluatorContext.EnqueueParentBehavior;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import javax.annotation.Nullable;

/** A {@link SkyFunction.Environment} implementation for {@link ParallelEvaluator}. */
class SkyFunctionEnvironment extends AbstractSkyFunctionEnvironment {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final SkyValue NULL_MARKER = new SkyValue() {};
  private static final boolean PREFETCH_OLD_DEPS =
      Boolean.parseBoolean(
          System.getProperty("skyframe.ParallelEvaluator.PrefetchOldDeps", "true"));
  private static final boolean PREFETCH_AND_RETAIN_OLD_DEPS =
      Boolean.parseBoolean(
          System.getProperty("skyframe.SkyFunctionEnvironment.PrefetchAndRetainOldDeps", "false"));

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

  private final FunctionHermeticity hermeticity;
  @Nullable private Version maxChildVersion = null;

  /** If present, takes precedence over {@link #maxChildVersion}. */
  @Nullable private Version injectedVersion = null;

  /**
   * This is not {@code null} only during cycle detection and error bubbling. The nullness of this
   * field is used to detect whether evaluation is in one of those special states.
   *
   * <p>When this is not {@code null}, values in this map should be used (while getting
   * dependencies' values, events, or posts) over values from the graph for keys present in this
   * map.
   */
  @Nullable private final Map<SkyKey, ValueWithMetadata> bubbleErrorInfo;

  /**
   * The current entries of the direct deps this node had at the previous version.
   *
   * <p>Used only when {@link #PREFETCH_AND_RETAIN_OLD_DEPS} is {@code true}, and used only for the
   * values stored in the entries; do not do any NodeEntry operations on these.
   */
  private ImmutableMap<SkyKey, ? extends NodeEntry> oldDepsEntries = ImmutableMap.of();

  /**
   * The values previously declared as dependencies.
   *
   * <p>Values in this map are either {@link #NULL_MARKER} or were retrieved via {@link
   * NodeEntry#getValueMaybeWithMetadata}. In the latter case, they should be processed using the
   * static methods of {@link ValueWithMetadata}.
   */
  private final ImmutableMap<SkyKey, SkyValue> previouslyRequestedDepsValues;

  /**
   * The values newly requested from the graph.
   *
   * <p>Values in this map are either {@link #NULL_MARKER} or were retrieved via {@link
   * NodeEntry#getValueMaybeWithMetadata}. In the latter case, they should be processed using the
   * static methods of {@link ValueWithMetadata}.
   */
  private final Map<SkyKey, SkyValue> newlyRequestedDepsValues = new HashMap<>();

  /**
   * Keys of dependencies registered via {@link #registerDependencies} if not using {@link
   * EvaluationVersionBehavior#MAX_CHILD_VERSIONS}.
   *
   * <p>The {@link #registerDependencies} method is hacky. Deps registered through it may not have
   * entries in {@link #newlyRequestedDepsValues}, but they are expected to be done. This set tracks
   * those keys so that they aren't removed when {@link #removeUndoneNewlyRequestedDeps} is called.
   */
  private final Set<SkyKey> newlyRegisteredDeps = new HashSet<>();

  /**
   * The grouped list of values requested during this build as dependencies. On a subsequent build,
   * if this value is dirty, all deps in the same dependency group can be checked in parallel for
   * changes. In other words, if dep1 and dep2 are in the same group, then dep1 will be checked in
   * parallel with dep2. See {@link #getValues} for more.
   */
  private final GroupedListHelper<SkyKey> newlyRequestedDeps = new GroupedListHelper<>();

  /** The set of errors encountered while fetching children. */
  private final Set<ErrorInfo> childErrorInfos = new LinkedHashSet<>();

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
          checkActive();
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
      throws InterruptedException, UndonePreviouslyRequestedDeps {
    super(directDeps);
    this.skyKey = skyKey;
    this.oldDeps = oldDeps;
    this.evaluatorContext = evaluatorContext;
    this.bubbleErrorInfo = null;
    this.hermeticity = skyKey.functionName().getHermeticity();
    this.previouslyRequestedDepsValues =
        batchPrefetch(skyKey, directDeps, oldDeps, /*assertDone=*/ true);
    Preconditions.checkState(
        !this.previouslyRequestedDepsValues.containsKey(ErrorTransienceValue.KEY),
        "%s cannot have a dep on ErrorTransienceValue during building",
        skyKey);
  }

  SkyFunctionEnvironment(
      SkyKey skyKey,
      GroupedList<SkyKey> directDeps,
      Map<SkyKey, ValueWithMetadata> bubbleErrorInfo,
      Set<SkyKey> oldDeps,
      ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    super(directDeps);
    this.skyKey = skyKey;
    this.oldDeps = oldDeps;
    this.evaluatorContext = evaluatorContext;
    this.bubbleErrorInfo = Preconditions.checkNotNull(bubbleErrorInfo);
    this.hermeticity = skyKey.functionName().getHermeticity();
    try {
      this.previouslyRequestedDepsValues =
          batchPrefetch(skyKey, directDeps, oldDeps, /*assertDone=*/ false);
    } catch (UndonePreviouslyRequestedDeps undonePreviouslyRequestedDeps) {
      throw new IllegalStateException(
          "batchPrefetch can't throw UndonePreviouslyRequestedDeps unless assertDone is true",
          undonePreviouslyRequestedDeps);
    }
    Preconditions.checkState(
        !this.previouslyRequestedDepsValues.containsKey(ErrorTransienceValue.KEY),
        "%s cannot have a dep on ErrorTransienceValue during building",
        skyKey);
  }

  private ImmutableMap<SkyKey, SkyValue> batchPrefetch(
      SkyKey requestor, GroupedList<SkyKey> depKeys, Set<SkyKey> oldDeps, boolean assertDone)
      throws InterruptedException, UndonePreviouslyRequestedDeps {
    QueryableGraph.PrefetchDepsRequest request = null;
    if (PREFETCH_OLD_DEPS) {
      request = new QueryableGraph.PrefetchDepsRequest(requestor, oldDeps, depKeys);
      evaluatorContext.getGraph().prefetchDeps(request);
    } else if (PREFETCH_AND_RETAIN_OLD_DEPS) {
      // TODO(b/175215425): Make PREFETCH_AND_RETAIN_OLD_DEPS the only behavior.
      this.oldDepsEntries =
          ImmutableMap.copyOf(evaluatorContext.getBatchValues(requestor, Reason.PREFETCH, oldDeps));
    }
    Map<SkyKey, ? extends NodeEntry> batchMap =
        evaluatorContext.getBatchValues(
            requestor,
            Reason.PREFETCH,
            (request != null && request.excludedKeys != null)
                ? request.excludedKeys
                : depKeys.getAllElementsAsIterable());
    if (batchMap.size() != depKeys.numElements()) {
      Set<SkyKey> difference = Sets.difference(depKeys.toSet(), batchMap.keySet());
      evaluatorContext
          .getGraphInconsistencyReceiver()
          .noteInconsistencyAndMaybeThrow(
              requestor, difference, Inconsistency.ALREADY_DECLARED_CHILD_MISSING);
      throw new UndonePreviouslyRequestedDeps(ImmutableList.copyOf(difference));
    }
    ImmutableMap.Builder<SkyKey, SkyValue> depValuesBuilder =
        ImmutableMap.builderWithExpectedSize(batchMap.size());
    for (Entry<SkyKey, ? extends NodeEntry> entry : batchMap.entrySet()) {
      SkyValue valueMaybeWithMetadata = entry.getValue().getValueMaybeWithMetadata();
      boolean depDone = valueMaybeWithMetadata != null;
      if (assertDone && !depDone) {
        // A previously requested dep may have transitioned from done to dirty between when the node
        // was read during a previous attempt to build this node and now. Notify the graph
        // inconsistency receiver so that we can crash if that's unexpected.
        evaluatorContext
            .getGraphInconsistencyReceiver()
            .noteInconsistencyAndMaybeThrow(
                skyKey,
                ImmutableList.of(entry.getKey()),
                Inconsistency.BUILDING_PARENT_FOUND_UNDONE_CHILD);
        throw new UndonePreviouslyRequestedDeps(ImmutableList.of(entry.getKey()));
      }
      depValuesBuilder.put(entry.getKey(), !depDone ? NULL_MARKER : valueMaybeWithMetadata);
      if (depDone) {
        maybeUpdateMaxChildVersion(entry.getValue());
      }
    }
    return depValuesBuilder.build();
  }

  private void checkActive() {
    Preconditions.checkState(building, skyKey);
  }

  Pair<NestedSet<TaggedEvents>, NestedSet<Postable>> buildAndReportEventsAndPostables(
      NodeEntry entry, boolean expectDoneDeps) throws InterruptedException {
    EventFilter eventFilter = evaluatorContext.getStoredEventFilter();
    if (!eventFilter.storeEventsAndPosts()) {
      return Pair.of(
          NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.emptySet(Order.STABLE_ORDER));
    }

    NestedSetBuilder<TaggedEvents> eventBuilder = NestedSetBuilder.stableOrder();
    ImmutableList<Event> events = eventHandler.getEvents();
    if (!events.isEmpty()) {
      eventBuilder.add(new TaggedEvents(getTagFromKey(), events));
    }
    NestedSetBuilder<Postable> postBuilder = NestedSetBuilder.stableOrder();
    postBuilder.addAll(eventHandler.getPosts());

    GroupedList<SkyKey> depKeys = entry.getTemporaryDirectDeps();
    Collection<SkyValue> deps =
        getDepValuesForDoneNodeFromErrorOrDepsOrGraph(
            Iterables.filter(
                depKeys.getAllElementsAsIterable(),
                eventFilter.depEdgeFilterForEventsAndPosts(skyKey)),
            expectDoneDeps,
            depKeys.numElements());
    for (SkyValue value : deps) {
      eventBuilder.addTransitive(ValueWithMetadata.getEvents(value));
      postBuilder.addTransitive(ValueWithMetadata.getPosts(value));
    }
    NestedSet<TaggedEvents> taggedEvents = eventBuilder.buildInterruptibly();
    NestedSet<Postable> postables = postBuilder.buildInterruptibly();
    evaluatorContext.getReplayingNestedSetEventVisitor().visit(taggedEvents);
    evaluatorContext.getReplayingNestedSetPostableVisitor().visit(postables);
    return Pair.of(taggedEvents, postables);
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
      state.signalDep(evaluatorContext.getGraphVersion(), ErrorTransienceValue.KEY);
      maxChildVersion = evaluatorContext.getGraphVersion();
    }

    this.errorInfo = Preconditions.checkNotNull(errorInfo, skyKey);
  }

  /**
   * Returns a map of {@code keys} to values or {@link #NULL_MARKER}s, populating the map's contents
   * by looking in order at:
   *
   * <ol>
   *   <li>{@link #bubbleErrorInfo}
   *   <li>{@link #previouslyRequestedDepsValues}
   *   <li>{@link #newlyRequestedDepsValues}
   *   <li>{@link #evaluatorContext}'s graph accessing methods
   * </ol>
   *
   * <p>All {@code keys} not previously requested will be added to a new group in {@link
   * #newlyRequestedDeps}. The new group will mirror the order of {@code keys}, minus duplicates.
   *
   * <p>Any key whose {@link NodeEntry}--or absence thereof--had to be read from the graph will also
   * be entered into {@link #newlyRequestedDepsValues} with its value or a {@link #NULL_MARKER}.
   */
  private Map<SkyKey, SkyValue> getValuesFromErrorOrDepsOrGraph(Iterable<? extends SkyKey> keys)
      throws InterruptedException {
    // Do not use an ImmutableMap.Builder, because we have not yet deduplicated these keys
    // and ImmutableMap.Builder does not tolerate duplicates.
    Map<SkyKey, SkyValue> result =
        keys instanceof Collection
            ? CompactHashMap.createWithExpectedSize(((Collection<?>) keys).size())
            : new HashMap<>();
    Set<SkyKey> missingKeys = new HashSet<>();
    newlyRequestedDeps.startGroup();
    for (SkyKey key : keys) {
      Preconditions.checkState(
          !key.equals(ErrorTransienceValue.KEY),
          "Error transience key cannot be in requested deps of %s",
          skyKey);
      SkyValue value = maybeGetValueFromErrorOrDeps(key);
      boolean duplicate;
      if (value == null) {
        duplicate = !missingKeys.add(key);
      } else {
        duplicate = result.put(key, value) != null;
      }
      if (!duplicate && !previouslyRequestedDepsValues.containsKey(key)) {
        newlyRequestedDeps.add(key);
      }
    }
    newlyRequestedDeps.endGroup();

    if (missingKeys.isEmpty()) {
      return result;
    }
    Map<SkyKey, ? extends NodeEntry> missingEntries =
        evaluatorContext.getBatchValues(skyKey, Reason.DEP_REQUESTED, missingKeys);
    for (SkyKey key : missingKeys) {
      NodeEntry depEntry = missingEntries.get(key);
      SkyValue valueOrNullMarker = getValueOrNullMarker(depEntry);
      result.put(key, valueOrNullMarker);
      newlyRequestedDepsValues.put(key, valueOrNullMarker);
      if (valueOrNullMarker != NULL_MARKER) {
        maybeUpdateMaxChildVersion(depEntry);
      }
    }
    return result;
  }

  /**
   * Similar to {@link #getValuesFromErrorOrDepsOrGraph}, but instead of a Map, return a List of
   * SkyValue ordered by the given order of SkyKeys.
   */
  private List<SkyValue> getOrderedValuesFromErrorOrDepsOrGraph(Iterable<? extends SkyKey> keys)
      throws InterruptedException {
    int capacity = keys instanceof Collection ? ((Collection<?>) keys).size() : 16;
    List<SkyValue> result = new ArrayList<>(capacity);

    // Ignoring duplication check here since it's done in GroupedList.
    List<SkyKey> missingKeys = new ArrayList<>();
    newlyRequestedDeps.startGroup();
    for (SkyKey key : keys) {
      Preconditions.checkState(
          !key.equals(ErrorTransienceValue.KEY),
          "Error transience key cannot be in requested deps of %s",
          skyKey);
      SkyValue value = maybeGetValueFromErrorOrDeps(key);
      if (value == null) {
        missingKeys.add(key);
      }
      // To maintain the ordering.
      result.add(value);
      if (!previouslyRequestedDepsValues.containsKey(key)) {
        newlyRequestedDeps.add(key);
      }
    }
    newlyRequestedDeps.endGroup();

    if (missingKeys.isEmpty()) {
      return result;
    }

    Map<SkyKey, ? extends NodeEntry> missingEntries =
        evaluatorContext.getBatchValues(skyKey, Reason.DEP_REQUESTED, missingKeys);
    int i = -1;
    for (SkyKey key : keys) {
      i++;
      if (result.get(i) != null) {
        continue;
      }
      NodeEntry depEntry = missingEntries.get(key);
      SkyValue valueOrNullMarker = getValueOrNullMarker(depEntry);
      result.set(i, valueOrNullMarker);
      newlyRequestedDepsValues.put(key, valueOrNullMarker);
      if (valueOrNullMarker != NULL_MARKER) {
        maybeUpdateMaxChildVersion(depEntry);
      }
    }
    return result;
  }

  /**
   * Returns the values of done deps in {@code depKeys}, by looking in order at:
   *
   * <ol>
   *   <li>{@link #bubbleErrorInfo}
   *   <li>{@link #previouslyRequestedDepsValues}
   *   <li>{@link #newlyRequestedDepsValues}
   *   <li>{@link #oldDepsEntries}
   *   <li>{@link #evaluatorContext}'s graph accessing methods
   * </ol>
   *
   * <p>Any key whose {@link NodeEntry}--or absence thereof--had to be read from the graph will also
   * be entered into {@link #newlyRequestedDepsValues} with its value or a {@link #NULL_MARKER}.
   *
   * <p>This asserts that only keys in {@link #newlyRegisteredDeps} require reading from the graph,
   * because this node is done, and so all other deps must have been previously or newly requested.
   *
   * <p>If {@code assertDone}, this asserts that all deps in {@code depKeys} are done.
   */
  private Collection<SkyValue> getDepValuesForDoneNodeFromErrorOrDepsOrGraph(
      Iterable<SkyKey> depKeys, boolean assertDone, int keySize) throws InterruptedException {
    List<SkyValue> result = new ArrayList<>(keySize);
    // depKeys may contain keys in newlyRegisteredDeps whose values have not yet been retrieved from
    // the graph during this environment's lifetime.
    int expectedMissingKeys = newlyRegisteredDeps.size();
    ArrayList<SkyKey> missingKeys =
        expectedMissingKeys > 0 ? new ArrayList<>(expectedMissingKeys) : null;
    ArrayList<SkyKey> unexpectedlyMissingKeys = null;

    for (SkyKey key : depKeys) {
      SkyValue value = maybeGetValueFromErrorOrDeps(key);
      if (value == null) {
        if (key == ErrorTransienceValue.KEY) {
          continue;
        }
        if (!newlyRegisteredDeps.contains(key)) {
          if (unexpectedlyMissingKeys == null) {
            unexpectedlyMissingKeys = new ArrayList<>();
          }
          unexpectedlyMissingKeys.add(key);
          if (missingKeys == null) {
            missingKeys = new ArrayList<>();
          }
        }
        missingKeys.add(key);
      } else if (value == NULL_MARKER) {
        Preconditions.checkState(!assertDone, "%s had not done %s", skyKey, key);
      } else {
        result.add(value);
      }
    }
    if (unexpectedlyMissingKeys != null && !unexpectedlyMissingKeys.isEmpty()) {
      // This may still crash below, if the dep is not done in the graph, but at least it gives the
      // dep until now to complete its computation, as opposed to the start of this node's
      // evaluation, which is when most of the structures used by #maybeGetValueFromErrorOrDeps were
      // created.
      evaluatorContext
          .getGraphInconsistencyReceiver()
          .noteInconsistencyAndMaybeThrow(
              skyKey, unexpectedlyMissingKeys, Inconsistency.ALREADY_DECLARED_CHILD_MISSING);
    }
    if (missingKeys == null || missingKeys.isEmpty()) {
      return result;
    }
    Map<SkyKey, ? extends NodeEntry> missingEntries =
        evaluatorContext.getBatchValues(skyKey, Reason.DEP_REQUESTED, missingKeys);
    for (SkyKey key : missingKeys) {
      NodeEntry depEntry = missingEntries.get(key);
      SkyValue valueOrNullMarker = getValueOrNullMarker(depEntry);
      newlyRequestedDepsValues.put(key, valueOrNullMarker);
      if (valueOrNullMarker == NULL_MARKER) {
        // TODO(mschaller): handle registered deps that transitioned from done to dirty during eval
        // But how? Restarting the current node may not help, because this dep was *registered*, not
        // requested. For now, no node that gets registered as a dep is eligible for
        // intra-evaluation dirtying, so let it crash.
        Preconditions.checkState(!assertDone, "%s had not done: %s", skyKey, key);
        continue;
      }
      maybeUpdateMaxChildVersion(depEntry);
      result.add(valueOrNullMarker);
    }
    return result;
  }

  /**
   * Returns a value or a {@link #NULL_MARKER} associated with {@code key} by looking in order at:
   *
   * <ol>
   *   <li>{@code bubbleErrorInfo}
   *   <li>{@link #previouslyRequestedDepsValues}
   *   <li>{@link #newlyRequestedDepsValues}
   *   <li>{@link #oldDepsEntries}
   * </ol>
   *
   * <p>Returns {@code null} if no entries for {@code key} were found in any of those three maps.
   * (Note that none of the maps can have {@code null} as a value.)
   */
  @Nullable
  SkyValue maybeGetValueFromErrorOrDeps(SkyKey key) throws InterruptedException {
    if (bubbleErrorInfo != null) {
      ValueWithMetadata bubbleErrorInfoValue = bubbleErrorInfo.get(key);
      if (bubbleErrorInfoValue != null) {
        return bubbleErrorInfoValue;
      }
    }
    SkyValue directDepsValue = previouslyRequestedDepsValues.get(key);
    if (directDepsValue != null) {
      return directDepsValue;
    }
    SkyValue newlyRequestedDepsValue = newlyRequestedDepsValues.get(key);
    if (newlyRequestedDepsValue != null) {
      return newlyRequestedDepsValue;
    }
    SkyValue oldDepsValueOrNullMarker = getValueOrNullMarker(oldDepsEntries.get(key));
    if (oldDepsValueOrNullMarker != NULL_MARKER) {
      return oldDepsValueOrNullMarker;
    }
    return null;
  }

  private static SkyValue getValueOrNullMarker(@Nullable NodeEntry nodeEntry)
      throws InterruptedException {
    if (nodeEntry == null) {
      return NULL_MARKER;
    }
    SkyValue valueMaybeWithMetadata = nodeEntry.getValueMaybeWithMetadata();
    if (valueMaybeWithMetadata == null) {
      return NULL_MARKER;
    }
    return valueMaybeWithMetadata;
  }

  @Override
  protected Map<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) throws InterruptedException {
    checkActive();
    Map<SkyKey, SkyValue> values = getValuesFromErrorOrDepsOrGraph(depKeys);
    for (Map.Entry<SkyKey, SkyValue> depEntry : values.entrySet()) {
      SkyKey depKey = depEntry.getKey();
      SkyValue depValue = depEntry.getValue();

      if (depValue == NULL_MARKER) {
        valuesMissing = true;
        if (previouslyRequestedDepsValues.containsKey(depKey)) {
          Preconditions.checkState(
              bubbleErrorInfo != null,
              "Undone key %s was already in deps of %s( dep: %s, parent: %s )",
              depKey,
              skyKey,
              evaluatorContext.getGraph().get(skyKey, Reason.OTHER, depKey),
              evaluatorContext.getGraph().get(null, Reason.OTHER, skyKey));
        }
        continue;
      }

      ErrorInfo errorInfo = ValueWithMetadata.getMaybeErrorInfo(depValue);
      if (errorInfo != null) {
        errorMightHaveBeenFound = true;
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
    }

    return Maps.transformValues(values, this::transformToValueOrUntypedException);
  }

  @Override
  protected List<ValueOrUntypedException> getOrderedValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) throws InterruptedException {
    checkActive();
    List<SkyValue> values = getOrderedValuesFromErrorOrDepsOrGraph(depKeys);
    int i = 0;
    for (SkyKey depKey : depKeys) {
      SkyValue depValue = values.get(i++);

      if (depValue == NULL_MARKER) {
        valuesMissing = true;
        if (previouslyRequestedDepsValues.containsKey(depKey)) {
          Preconditions.checkState(
              bubbleErrorInfo != null,
              "Undone key %s was already in deps of %s( dep: %s, parent: %s )",
              depKey,
              skyKey,
              evaluatorContext.getGraph().get(skyKey, Reason.OTHER, depKey),
              evaluatorContext.getGraph().get(null, Reason.OTHER, skyKey));
        }
        continue;
      }

      ErrorInfo errorInfo = ValueWithMetadata.getMaybeErrorInfo(depValue);
      if (errorInfo != null) {
        errorMightHaveBeenFound = true;
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
    }

    return Lists.transform(values, this::transformToValueOrUntypedException);
  }

  private ValueOrUntypedException transformToValueOrUntypedException(SkyValue maybeWrappedValue) {
    if (maybeWrappedValue == NULL_MARKER) {
      return ValueOrUntypedException.ofNull();
    }
    SkyValue justValue = ValueWithMetadata.justValue(maybeWrappedValue);
    ErrorInfo errorInfo = ValueWithMetadata.getMaybeErrorInfo(maybeWrappedValue);

    if (justValue != null && (evaluatorContext.keepGoing() || errorInfo == null)) {
      // If the dep did compute a value, it is given to the caller if we are in
      // keepGoing mode or if we are in noKeepGoingMode and there were no errors computing
      // it.
      return ValueOrUntypedException.ofValueUntyped(justValue);
    }

    // There was an error building the value, which we will either report by throwing an
    // exception or insulate the caller from by returning null.
    Preconditions.checkNotNull(errorInfo, "%s %s", skyKey, maybeWrappedValue);
    Exception exception = errorInfo.getException();

    if (!evaluatorContext.keepGoing() && exception != null && bubbleErrorInfo == null) {
      // Child errors should not be propagated in noKeepGoing mode (except during error
      // bubbling). Instead we should fail fast.
      return ValueOrUntypedException.ofNull();
    }

    if (exception != null) {
      // Give builder a chance to handle this exception.
      return ValueOrUntypedException.ofExn(exception);
    }
    // In a cycle.
    Preconditions.checkState(
        !errorInfo.getCycleInfo().isEmpty(), "%s %s %s", skyKey, errorInfo, maybeWrappedValue);
    return ValueOrUntypedException.ofNull();
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

  void removeUndoneNewlyRequestedDeps() {
    HashSet<SkyKey> undoneDeps = new HashSet<>();
    for (SkyKey newlyRequestedDep : newlyRequestedDeps) {
      if (newlyRegisteredDeps.contains(newlyRequestedDep)) {
        continue;
      }
      SkyValue newlyRequestedDepValue =
          Preconditions.checkNotNull(
              newlyRequestedDepsValues.get(newlyRequestedDep), newlyRequestedDep);
      if (newlyRequestedDepValue == NULL_MARKER) {
        // The dep was normally requested, and was not done.
        undoneDeps.add(newlyRequestedDep);
      }
    }
    newlyRequestedDeps.remove(undoneDeps);
  }

  boolean isAnyDirectDepErrorTransitivelyTransient() {
    Preconditions.checkState(
        bubbleErrorInfo == null,
        "Checking dep error transitive transience during error bubbling for: %s",
        skyKey);
    for (SkyValue skyValue : previouslyRequestedDepsValues.values()) {
      ErrorInfo maybeErrorInfo = ValueWithMetadata.getMaybeErrorInfo(skyValue);
      if (maybeErrorInfo != null && maybeErrorInfo.isTransitivelyTransient()) {
        return true;
      }
    }
    return false;
  }

  boolean isAnyNewlyRequestedDepErrorTransitivelyTransient() {
    Preconditions.checkState(
        bubbleErrorInfo == null,
        "Checking dep error transitive transience during error bubbling for: %s",
        skyKey);
    for (SkyValue skyValue : newlyRequestedDepsValues.values()) {
      ErrorInfo maybeErrorInfo = ValueWithMetadata.getMaybeErrorInfo(skyValue);
      if (maybeErrorInfo != null && maybeErrorInfo.isTransitivelyTransient()) {
        return true;
      }
    }
    return false;
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
   * <p>The reverse deps that would have been enqueued are returned if {@code enqueueParents} is
   * {@link EnqueueParentBehavior#SIGNAL} or {@link EnqueueParentBehavior#NO_ACTION}, so that the
   * caller may simulate actions on the parents if desired. Otherwise this method returns null.
   */
  Set<SkyKey> commit(NodeEntry primaryEntry, EnqueueParentBehavior enqueueParents)
      throws InterruptedException {
    // Construct the definitive error info, if there is one.
    if (errorInfo == null) {
      errorInfo = evaluatorContext.getErrorInfoManager().getErrorInfoToUse(
          skyKey, value != null, childErrorInfos);
      // TODO(b/166268889, b/172223413): remove when fixed.
      if (errorInfo != null && errorInfo.getException() instanceof IOException) {
        logger.atInfo().withCause(errorInfo.getException()).log(
            "Synthetic errorInfo for %s", skyKey);
      }
    }

    // We have the following implications:
    // errorInfo == null => value != null => enqueueParents.
    // All these implications are strict:
    // (1) errorInfo != null && value != null happens for values with recoverable errors.
    // (2) value == null && enqueueParents happens for values that are found to have errors
    // during a --keep_going build.

    Pair<NestedSet<TaggedEvents>, NestedSet<Postable>> eventsAndPostables =
        buildAndReportEventsAndPostables(primaryEntry, /*expectDoneDeps=*/ true);

    SkyValue valueWithMetadata;
    if (value == null) {
      Preconditions.checkNotNull(errorInfo, "%s %s", skyKey, primaryEntry);
      valueWithMetadata =
          ValueWithMetadata.error(errorInfo, eventsAndPostables.first, eventsAndPostables.second);
    } else {
      // We must be enqueueing parents if we have a value.
      Preconditions.checkState(
          enqueueParents == EnqueueParentBehavior.ENQUEUE, "%s %s", skyKey, primaryEntry);
      valueWithMetadata =
          ValueWithMetadata.normal(
              value, errorInfo, eventsAndPostables.first, eventsAndPostables.second);
    }
    GroupedList<SkyKey> temporaryDirectDeps = primaryEntry.getTemporaryDirectDeps();
    if (evaluatorContext.getGraph().storesReverseDeps() && !oldDeps.isEmpty()) {
      // Remove the rdep on this entry for each of its old deps that is no longer a direct dep.
      Set<SkyKey> depsToRemove = Sets.difference(oldDeps, temporaryDirectDeps.toSet());
      Collection<? extends NodeEntry> oldDepEntries =
          evaluatorContext.getGraph().getBatch(skyKey, Reason.RDEP_REMOVAL, depsToRemove).values();
      for (NodeEntry oldDepEntry : oldDepEntries) {
        oldDepEntry.removeReverseDep(skyKey);
      }
    }

    Version evaluationVersion = maxChildVersion;
    if (bubbleErrorInfo != null) {
      // Cycles can lead to a state where the versions of done children don't accurately reflect the
      // state that led to this node's value. Be conservative then.
      evaluationVersion = evaluatorContext.getGraphVersion();
    } else if (injectedVersion != null) {
      evaluationVersion = injectedVersion;
    } else if (evaluatorContext.getEvaluationVersionBehavior()
            == EvaluationVersionBehavior.GRAPH_VERSION
        || hermeticity == FunctionHermeticity.NONHERMETIC) {
      evaluationVersion = evaluatorContext.getGraphVersion();
    } else if (evaluationVersion == null) {
      Preconditions.checkState(
          temporaryDirectDeps.isEmpty(),
          "No max child version found, but have direct deps: %s %s",
          skyKey,
          primaryEntry);
      evaluationVersion = evaluatorContext.getGraphVersion();
    }
    Version previousVersion = primaryEntry.getVersion();
    // If this entry is dirty, setValue may not actually change it, if it determines that
    // the data being written now is the same as the data already present in the entry.
    Set<SkyKey> reverseDeps = primaryEntry.setValue(valueWithMetadata, evaluationVersion);

    // Note that if this update didn't actually change the entry, this version may not be
    // evaluationVersion.
    Version currentVersion = primaryEntry.getVersion();
    // Tell the receiver that this value was built. If currentVersion.equals(evaluationVersion), it
    // was evaluated this run, and so was changed. Otherwise, it is less than evaluationVersion, by
    // the Preconditions check above, and was not actually changed this run -- when it was written
    // above, its version stayed below this update's version, so its value remains the same.
    // We use a SkyValueSupplier here because it keeps a reference to the entry, allowing for
    // the receiver to be confident that the entry is readily accessible in memory.
    EvaluationState evaluationState =
        currentVersion.equals(previousVersion) ? EvaluationState.CLEAN : EvaluationState.BUILT;
    evaluatorContext
        .getProgressReceiver()
        .evaluated(
            skyKey,
            evaluationState == EvaluationState.BUILT ? value : null,
            evaluationState == EvaluationState.BUILT ? errorInfo : null,
            EvaluationSuccessStateSupplier.fromSkyValue(valueWithMetadata),
            evaluationState);

    evaluatorContext.signalValuesAndEnqueueIfReady(
        skyKey, reverseDeps, currentVersion, enqueueParents);

    return enqueueParents == EnqueueParentBehavior.ENQUEUE ? null : reverseDeps;
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

  @Override
  public void registerDependencies(Iterable<SkyKey> keys) throws InterruptedException {
    if (EvaluationVersionBehavior.MAX_CHILD_VERSIONS.equals(
        evaluatorContext.getEvaluationVersionBehavior())) {
      // Need versions when doing MAX_CHILD_VERSIONS, so can't use optimization. To use the
      // optimization, the caller would have to know the versions of the passed-in keys. Extensions
      // of the SkyFunction.Environment interface to make that possible could happen.
      Map<SkyKey, SkyValue> checkSizeMap = getValues(keys);
      ImmutableSet<SkyKey> keysSet = ImmutableSet.copyOf(keys);
      if (checkSizeMap.size() != keysSet.size()) {
        throw new IllegalStateException(
            "Missing keys when checking dependencies for "
                + skyKey
                + ": "
                + Sets.difference(keysSet, checkSizeMap.keySet()));
      }
      return;
    }
    newlyRequestedDeps.startGroup();
    for (SkyKey key : keys) {
      if (!previouslyRequestedDepsValues.containsKey(key)) {
        newlyRequestedDeps.add(key);
        newlyRegisteredDeps.add(key);
      }
    }
    newlyRequestedDeps.endGroup();
  }

  @Override
  public void injectVersionForNonHermeticFunction(Version version) {
    Preconditions.checkState(hermeticity == FunctionHermeticity.NONHERMETIC, skyKey);
    injectedVersion = version;
  }

  private void maybeUpdateMaxChildVersion(NodeEntry depEntry) {
    if (hermeticity != FunctionHermeticity.NONHERMETIC
        && evaluatorContext.getEvaluationVersionBehavior()
            == EvaluationVersionBehavior.MAX_CHILD_VERSIONS) {
      Version depVersion = depEntry.getVersion();
      if (maxChildVersion == null || maxChildVersion.atMost(depVersion)) {
        maxChildVersion = depVersion;
      }
    }
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("skyKey", skyKey)
        .add("oldDeps", oldDeps)
        .add("value", value)
        .add("errorInfo", errorInfo)
        .add("previouslyRequestedDepsValues", previouslyRequestedDepsValues)
        .add("newlyRequestedDepsValues", newlyRequestedDepsValues)
        .add("newlyRegisteredDeps", newlyRegisteredDeps)
        .add("newlyRequestedDeps", newlyRequestedDeps)
        .add("childErrorInfos", childErrorInfos)
        .add("depErrorKey", depErrorKey)
        .add("hermeticity", hermeticity)
        .add("maxChildVersion", maxChildVersion)
        .add("injectedVersion", injectedVersion)
        .add("bubbleErrorInfo", bubbleErrorInfo)
        .add("evaluatorContext", evaluatorContext)
        .toString();
  }

  @Override
  public boolean restartPermitted() {
    return evaluatorContext.restartPermitted();
  }

  /** Thrown during environment construction if previously requested deps are no longer done. */
  static class UndonePreviouslyRequestedDeps extends Exception {
    private final ImmutableList<SkyKey> depKeys;

    UndonePreviouslyRequestedDeps(ImmutableList<SkyKey> depKeys) {
      this.depKeys = depKeys;
    }

    ImmutableList<SkyKey> getDepKeys() {
      return depKeys;
    }
  }
}
