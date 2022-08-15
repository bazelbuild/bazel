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

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.collect.compacthashmap.CompactHashMap;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reportable;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationState;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** A {@link SkyFunction.Environment} implementation for {@link ParallelEvaluator}. */
final class SkyFunctionEnvironment extends AbstractSkyFunctionEnvironment
    implements ExtendedEventHandler {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final SkyValue NULL_MARKER = new SkyValue() {};

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

  @Nullable private Version maxTransitiveSourceVersion;

  /**
   * This is not {@code null} only during cycle detection and error bubbling. The nullness of this
   * field is used to detect whether evaluation is in one of those special states.
   *
   * <p>When this is not {@code null}, values in this map should be used (while getting
   * dependencies' values, events, or posts) over values from the graph for keys present in this
   * map.
   */
  @Nullable private final Map<SkyKey, ValueWithMetadata> bubbleErrorInfo;

  private boolean encounteredErrorDuringBubbling = false;

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
   * Keys of dependencies registered via {@link #registerDependencies}.
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
   * parallel with dep2. See {@link SkyFunction.Environment#getValuesAndExceptions} for more.
   */
  private final GroupedListHelper<SkyKey> newlyRequestedDeps = new GroupedListHelper<>();

  /** The set of errors encountered while fetching children. */
  private final Set<ErrorInfo> childErrorInfos = new LinkedHashSet<>();

  private final ParallelEvaluatorContext evaluatorContext;

  private final List<Reportable> eventsToReport = new ArrayList<>();

  static SkyFunctionEnvironment create(
      SkyKey skyKey,
      GroupedList<SkyKey> directDeps,
      Set<SkyKey> oldDeps,
      ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException, UndonePreviouslyRequestedDep {
    return new SkyFunctionEnvironment(
        skyKey,
        directDeps,
        /*bubbleErrorInfo=*/ null,
        oldDeps,
        evaluatorContext,
        /*throwIfPreviouslyRequestedDepsUndone=*/ true);
  }

  static SkyFunctionEnvironment createForError(
      SkyKey skyKey,
      GroupedList<SkyKey> directDeps,
      Map<SkyKey, ValueWithMetadata> bubbleErrorInfo,
      Set<SkyKey> oldDeps,
      ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    try {
      return new SkyFunctionEnvironment(
          skyKey,
          directDeps,
          checkNotNull(bubbleErrorInfo),
          oldDeps,
          evaluatorContext,
          /*throwIfPreviouslyRequestedDepsUndone=*/ false);
    } catch (UndonePreviouslyRequestedDep undonePreviouslyRequestedDep) {
      throw new IllegalStateException(undonePreviouslyRequestedDep);
    }
  }

  private SkyFunctionEnvironment(
      SkyKey skyKey,
      GroupedList<SkyKey> directDeps,
      @Nullable Map<SkyKey, ValueWithMetadata> bubbleErrorInfo,
      Set<SkyKey> oldDeps,
      ParallelEvaluatorContext evaluatorContext,
      boolean throwIfPreviouslyRequestedDepsUndone)
      throws UndonePreviouslyRequestedDep, InterruptedException {
    super(directDeps);
    this.skyKey = checkNotNull(skyKey);
    this.bubbleErrorInfo = bubbleErrorInfo;
    this.oldDeps = checkNotNull(oldDeps);
    this.evaluatorContext = checkNotNull(evaluatorContext);
    // Cycles can lead to a state where the versions of done children don't accurately reflect the
    // state that led to this node's value. Be conservative then.
    this.maxTransitiveSourceVersion =
        bubbleErrorInfo == null
                && skyKey.functionName().getHermeticity() != FunctionHermeticity.NONHERMETIC
            ? evaluatorContext.getMinimalVersion()
            : null;
    this.previouslyRequestedDepsValues = batchPrefetch(throwIfPreviouslyRequestedDepsUndone);
    checkState(
        !this.previouslyRequestedDepsValues.containsKey(ErrorTransienceValue.KEY),
        "%s cannot have a dep on ErrorTransienceValue during building",
        skyKey);
  }

  private ImmutableMap<SkyKey, SkyValue> batchPrefetch(boolean throwIfPreviouslyRequestedDepsUndone)
      throws InterruptedException, UndonePreviouslyRequestedDep {
    GroupedList<SkyKey> previouslyRequestedDeps = getTemporaryDirectDeps();
    ImmutableSet<SkyKey> excludedKeys =
        evaluatorContext.getGraph().prefetchDeps(skyKey, oldDeps, previouslyRequestedDeps);
    Iterable<SkyKey> keysToPrefetch =
        excludedKeys != null ? excludedKeys : previouslyRequestedDeps.getAllElementsAsIterable();
    Map<SkyKey, ? extends NodeEntry> batchMap =
        evaluatorContext.getGraph().getBatchMap(skyKey, Reason.PREFETCH, keysToPrefetch);
    ImmutableMap.Builder<SkyKey, SkyValue> depValuesBuilder =
        ImmutableMap.builderWithExpectedSize(batchMap.size());
    for (SkyKey depKey : keysToPrefetch) {
      NodeEntry entry =
          checkNotNull(
              batchMap.get(depKey),
              "Missing previously requested dep %s (parent=%s)",
              depKey,
              skyKey);
      SkyValue valueMaybeWithMetadata = entry.getValueMaybeWithMetadata();
      boolean depDone = valueMaybeWithMetadata != null;
      if (throwIfPreviouslyRequestedDepsUndone && !depDone) {
        // A previously requested dep may have transitioned from done to dirty between when the node
        // was read during a previous attempt to build this node and now. Notify the graph
        // inconsistency receiver so that we can crash if that's unexpected.
        evaluatorContext
            .getGraphInconsistencyReceiver()
            .noteInconsistencyAndMaybeThrow(
                skyKey, ImmutableList.of(depKey), Inconsistency.BUILDING_PARENT_FOUND_UNDONE_CHILD);
        throw new UndonePreviouslyRequestedDep(depKey);
      }
      depValuesBuilder.put(depKey, !depDone ? NULL_MARKER : valueMaybeWithMetadata);
      if (depDone) {
        maybeUpdateMaxTransitiveSourceVersion(entry);
      }
    }
    return depValuesBuilder.buildOrThrow();
  }

  private void checkActive() {
    checkState(building, skyKey);
  }

  /**
   * Reports events which were temporarily stored in this environment per the specification of
   * {@link SkyFunction.Environment#getListener}. Returns events that should be stored for potential
   * replay on a future evaluation.
   */
  NestedSet<Reportable> reportEventsAndGetEventsToStore(NodeEntry entry, boolean expectDoneDeps)
      throws InterruptedException {
    EventFilter eventFilter = evaluatorContext.getStoredEventFilter();
    if (!eventFilter.storeEvents()) {
      if (!eventsToReport.isEmpty()) {
        String tag = getTagFromKey();
        for (Reportable event : eventsToReport) {
          event.withTag(tag).reportTo(evaluatorContext.getReporter());
        }
      }
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    GroupedList<SkyKey> depKeys = entry.getTemporaryDirectDeps();
    if (eventsToReport.isEmpty() && depKeys.isEmpty()) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    NestedSetBuilder<Reportable> eventBuilder = NestedSetBuilder.stableOrder();
    if (!eventsToReport.isEmpty()) {
      String tag = getTagFromKey();
      eventBuilder.addAll(Lists.transform(eventsToReport, event -> event.withTag(tag)));
    }

    addTransitiveEventsFromDepValuesForDoneNode(
        eventBuilder,
        // When there's no boundary between analysis & execution, we don't filter any dep.
        evaluatorContext.mergingSkyframeAnalysisExecutionPhases()
            ? depKeys.getAllElementsAsIterable()
            : Iterables.filter(
                depKeys.getAllElementsAsIterable(),
                depKey -> eventFilter.shouldPropagate(depKey, skyKey)),
        expectDoneDeps);

    NestedSet<Reportable> events = eventBuilder.buildInterruptibly();
    evaluatorContext.getReplayingNestedSetEventVisitor().visit(events);
    return events;
  }

  /**
   * Adds transitive events from done deps in {@code depKeys}, by looking in order at:
   *
   * <ol>
   *   <li>{@link #bubbleErrorInfo}
   *   <li>{@link #previouslyRequestedDepsValues}
   *   <li>{@link #newlyRequestedDepsValues}
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
  private void addTransitiveEventsFromDepValuesForDoneNode(
      NestedSetBuilder<Reportable> eventBuilder, Iterable<SkyKey> depKeys, boolean assertDone)
      throws InterruptedException {
    // depKeys may contain keys in newlyRegisteredDeps whose values have not yet been retrieved from
    // the graph during this environment's lifetime.
    List<SkyKey> missingKeys =
        newlyRegisteredDeps.isEmpty() ? null : new ArrayList<>(newlyRegisteredDeps.size());

    for (SkyKey key : depKeys) {
      SkyValue value = maybeGetValueFromErrorOrDeps(key);
      if (value == null) {
        if (key == ErrorTransienceValue.KEY) {
          continue;
        }
        checkState(
            newlyRegisteredDeps.contains(key),
            "Missing already declared dep %s (parent=%s)",
            key,
            skyKey);
        missingKeys.add(key);
      } else if (value == NULL_MARKER) {
        checkState(!assertDone, "%s had not done %s", skyKey, key);
      } else {
        eventBuilder.addTransitive(ValueWithMetadata.getEvents(value));
      }
    }
    if (missingKeys == null || missingKeys.isEmpty()) {
      return;
    }
    Map<SkyKey, ? extends NodeEntry> missingEntries =
        evaluatorContext.getGraph().getBatchMap(skyKey, Reason.DEP_REQUESTED, missingKeys);
    for (SkyKey key : missingKeys) {
      NodeEntry depEntry = missingEntries.get(key);
      SkyValue valueOrNullMarker = getValueOrNullMarker(depEntry);
      newlyRequestedDepsValues.put(key, valueOrNullMarker);
      if (valueOrNullMarker == NULL_MARKER) {
        // TODO(mschaller): handle registered deps that transitioned from done to dirty during eval
        // But how? Restarting the current node may not help, because this dep was *registered*, not
        // requested. For now, no node that gets registered as a dep is eligible for
        // intra-evaluation dirtying, so let it crash.
        checkState(!assertDone, "%s had not done: %s", skyKey, key);
        continue;
      }
      maybeUpdateMaxTransitiveSourceVersion(depEntry);
      eventBuilder.addTransitive(ValueWithMetadata.getEvents(valueOrNullMarker));
    }
  }

  void setValue(SkyValue newValue) {
    checkState(
        errorInfo == null && bubbleErrorInfo == null,
        "%s %s %s %s",
        skyKey,
        newValue,
        errorInfo,
        bubbleErrorInfo);
    checkState(value == null, "%s %s %s", skyKey, value, newValue);
    value = newValue;
  }

  /**
   * Set this node to be in error. The node's value must not have already been set. However, all
   * dependencies of this node <i>must</i> already have been registered, since this method may
   * register a dependence on the error transience node, which should always be the last dep.
   */
  void setError(NodeEntry state, ErrorInfo errorInfo) throws InterruptedException {
    checkState(value == null, "%s %s %s", skyKey, value, errorInfo);
    checkState(this.errorInfo == null, "%s %s %s", skyKey, this.errorInfo, errorInfo);

    if (errorInfo.isDirectlyTransient()) {
      NodeEntry errorTransienceNode =
          checkNotNull(
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
      checkState(triState == DependencyState.DONE, "%s %s %s", skyKey, triState, errorInfo);
      state.addTemporaryDirectDeps(GroupedListHelper.create(ErrorTransienceValue.KEY));
      state.signalDep(evaluatorContext.getGraphVersion(), ErrorTransienceValue.KEY);
      maxTransitiveSourceVersion = null;
    }

    this.errorInfo = checkNotNull(errorInfo, skyKey);
  }

  /**
   * Returns a value or a {@link #NULL_MARKER} associated with {@code key} by looking in order at:
   *
   * <ol>
   *   <li>{@link #bubbleErrorInfo}
   *   <li>{@link #previouslyRequestedDepsValues}
   *   <li>{@link #newlyRequestedDepsValues}
   * </ol>
   *
   * <p>Returns {@code null} if no entries for {@code key} were found in any of those three maps.
   * (Note that none of the maps can have {@code null} as a value.)
   */
  @Nullable
  SkyValue maybeGetValueFromErrorOrDeps(SkyKey key) {
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
    return newlyRequestedDepsValues.get(key);
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
    // Do not use an ImmutableMap.Builder, because we have not yet deduplicated these keys
    // and ImmutableMap.Builder does not tolerate duplicates.
    Map<SkyKey, ValueOrUntypedException> result =
        depKeys instanceof Collection
            ? CompactHashMap.createWithExpectedSize(((Collection<?>) depKeys).size())
            : new HashMap<>();
    Set<SkyKey> missingKeys = new HashSet<>();
    newlyRequestedDeps.startGroup();
    for (SkyKey key : depKeys) {
      checkState(
          !key.equals(ErrorTransienceValue.KEY),
          "Error transience key cannot be in requested deps of %s",
          skyKey);
      SkyValue value = maybeGetValueFromErrorOrDeps(key);
      boolean duplicate;
      if (value == null) {
        duplicate = !missingKeys.add(key);
      } else {
        duplicate = result.put(key, transformToValueOrUntypedException(value)) != null;
        if (!duplicate) {
          processDepValue(key, value);
        }
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
        evaluatorContext.getGraph().getBatchMap(skyKey, Reason.DEP_REQUESTED, missingKeys);
    for (SkyKey key : missingKeys) {
      NodeEntry depEntry = missingEntries.get(key);
      SkyValue valueOrNullMarker = getValueOrNullMarker(depEntry);
      result.put(key, transformToValueOrUntypedException(valueOrNullMarker));
      processDepValue(key, valueOrNullMarker);
      newlyRequestedDepsValues.put(key, valueOrNullMarker);
      if (valueOrNullMarker != NULL_MARKER) {
        maybeUpdateMaxTransitiveSourceVersion(depEntry);
      }
    }
    return result;
  }

  @Override
  protected List<ValueOrUntypedException> getOrderedValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) throws InterruptedException {
    checkActive();
    int capacity = depKeys instanceof Collection ? ((Collection<?>) depKeys).size() : 16;
    List<ValueOrUntypedException> result = new ArrayList<>(capacity);

    // Ignoring duplication check here since it's done in GroupedList.
    List<SkyKey> missingKeys = new ArrayList<>();
    newlyRequestedDeps.startGroup();
    for (SkyKey key : depKeys) {
      checkState(
          !key.equals(ErrorTransienceValue.KEY),
          "Error transience key cannot be in requested deps of %s",
          skyKey);
      SkyValue value = maybeGetValueFromErrorOrDeps(key);
      if (value == null) {
        missingKeys.add(key);
        result.add(null); // Add null as a placeholder to maintain the ordering.
      } else {
        result.add(transformToValueOrUntypedException(value));
        processDepValue(key, value);
      }
      if (!previouslyRequestedDepsValues.containsKey(key)) {
        newlyRequestedDeps.add(key);
      }
    }
    newlyRequestedDeps.endGroup();

    if (missingKeys.isEmpty()) {
      return result;
    }

    Map<SkyKey, ? extends NodeEntry> missingEntries =
        evaluatorContext.getGraph().getBatchMap(skyKey, Reason.DEP_REQUESTED, missingKeys);
    int i = 0;
    for (SkyKey key : missingKeys) {
      while (result.get(i) != null) {
        i++; // Fast-forward to the next null placeholder.
      }
      NodeEntry depEntry = missingEntries.get(key);
      SkyValue valueOrNullMarker = getValueOrNullMarker(depEntry);
      result.set(i, transformToValueOrUntypedException(valueOrNullMarker));
      processDepValue(key, valueOrNullMarker);
      newlyRequestedDepsValues.put(key, valueOrNullMarker);
      if (valueOrNullMarker != NULL_MARKER) {
        maybeUpdateMaxTransitiveSourceVersion(depEntry);
      }
    }
    return result;
  }

  private void processDepValue(SkyKey depKey, SkyValue depValue) throws InterruptedException {
    if (depValue == NULL_MARKER) {
      valuesMissing = true;
      if (bubbleErrorInfo == null && previouslyRequestedDepsValues.containsKey(depKey)) {
        throw new IllegalStateException(
            String.format(
                "Undone key %s was already in deps of %s (dep=%s, parent=%s)",
                depKey,
                skyKey,
                evaluatorContext.getGraph().get(skyKey, Reason.OTHER, depKey),
                evaluatorContext.getGraph().get(null, Reason.OTHER, skyKey)));
      }
      return;
    }

    ErrorInfo errorInfo = ValueWithMetadata.getMaybeErrorInfo(depValue);
    if (errorInfo == null) {
      return;
    }
    errorMightHaveBeenFound = true;
    childErrorInfos.add(errorInfo);
    if (bubbleErrorInfo != null) {
      encounteredErrorDuringBubbling = true;
      // Set interrupted status, to try to prevent the calling SkyFunction from doing anything fancy
      // after this. SkyFunctions executed during error bubbling are supposed to (quickly) rethrow
      // errors or return a value/null (but there's currently no way to enforce this).
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
    checkNotNull(errorInfo, "%s %s", skyKey, maybeWrappedValue);
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
    checkState(
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

  @CanIgnoreReturnValue
  @Override
  public ExtendedEventHandler getListener() {
    checkActive();
    return this;
  }

  @Override
  public void handle(Event event) {
    reportEvent(event);
  }

  @Override
  public void post(Postable obj) {
    reportEvent(obj);
  }

  private void reportEvent(Reportable event) {
    checkActive();
    if (event.storeForReplay()) {
      eventsToReport.add(event);
    } else {
      event.reportTo(evaluatorContext.getReporter());
    }
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
          checkNotNull(newlyRequestedDepsValues.get(newlyRequestedDep), newlyRequestedDep);
      if (newlyRequestedDepValue == NULL_MARKER) {
        // The dep was normally requested, and was not done.
        undoneDeps.add(newlyRequestedDep);
      }
    }
    newlyRequestedDeps.remove(undoneDeps);
  }

  boolean isAnyDirectDepErrorTransitivelyTransient() {
    checkState(
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
    checkState(
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
   * Applies the change to the graph (mostly) atomically and returns parents to potentially signal
   * and enqueue.
   *
   * <p>Parents should be enqueued unless (1) this node is being built after the main evaluation has
   * aborted, or (2) this node is being built with {@code --nokeep_going}, and so we are about to
   * shut down the main evaluation anyway.
   */
  Set<SkyKey> commitAndGetParents(NodeEntry primaryEntry) throws InterruptedException {
    // Construct the definitive error info, if there is one.
    if (errorInfo == null) {
      errorInfo =
          evaluatorContext
              .getErrorInfoManager()
              .getErrorInfoToUse(skyKey, value != null, childErrorInfos);
      // TODO(b/166268889, b/172223413): remove when fixed.
      if (errorInfo != null && errorInfo.getException() instanceof IOException) {
        String skyFunctionName = skyKey.functionName().getName();
        if (!skyFunctionName.startsWith("FILE")
            && !skyFunctionName.startsWith("DIRECTORY_LISTING")) {
          logger.atInfo().withCause(errorInfo.getException()).log(
              "Synthetic errorInfo for %s", skyKey);
        }
      }
    }

    // We have the following implications:
    // errorInfo == null => value != null => enqueueParents.
    // All these implications are strict:
    // (1) errorInfo != null && value != null happens for values with recoverable errors.
    // (2) value == null && enqueueParents happens for values that are found to have errors
    // during a --keep_going build.

    NestedSet<Reportable> events =
        reportEventsAndGetEventsToStore(primaryEntry, /*expectDoneDeps=*/ true);

    SkyValue valueWithMetadata;
    if (value == null) {
      checkNotNull(errorInfo, "%s %s", skyKey, primaryEntry);
      valueWithMetadata = ValueWithMetadata.error(errorInfo, events);
    } else {
      valueWithMetadata = ValueWithMetadata.normal(value, errorInfo, events);
    }
    GroupedList<SkyKey> temporaryDirectDeps = primaryEntry.getTemporaryDirectDeps();
    if (!oldDeps.isEmpty()) {
      // Remove the rdep on this entry for each of its old deps that is no longer a direct dep.
      Set<SkyKey> depsToRemove = Sets.difference(oldDeps, temporaryDirectDeps.toSet());
      Collection<? extends NodeEntry> oldDepEntries =
          evaluatorContext
              .getGraph()
              .getBatchMap(skyKey, Reason.RDEP_REMOVAL, depsToRemove)
              .values();
      for (NodeEntry oldDepEntry : oldDepEntries) {
        oldDepEntry.removeReverseDep(skyKey);
      }
    }

    checkState(
        maxTransitiveSourceVersion == null || newlyRegisteredDeps.isEmpty(),
        "Dependency registration not supported when tracking max transitive source versions");

    // If this entry is dirty, setValue may not actually change it, if it determines that the data
    // being written now is the same as the data already present in the entry. We detect this case
    // by comparing versions before and after setting the value.
    Version previousVersion = primaryEntry.getVersion();
    Set<SkyKey> reverseDeps =
        primaryEntry.setValue(
            valueWithMetadata, evaluatorContext.getGraphVersion(), maxTransitiveSourceVersion);
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

    return reverseDeps;
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
  public boolean inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors() {
    return bubbleErrorInfo != null;
  }

  @Override
  public void registerDependencies(Iterable<SkyKey> keys) {
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
    checkState(skyKey.functionName().getHermeticity() == FunctionHermeticity.NONHERMETIC, skyKey);
    checkState(
        maxTransitiveSourceVersion == null,
        "Multiple injected versions (%s, %s) for %s",
        maxTransitiveSourceVersion,
        version,
        skyKey);
    checkNotNull(version, skyKey);
    checkState(
        !evaluatorContext.getGraphVersion().lowerThan(version),
        "Invalid injected version (%s > %s) for %s",
        version,
        evaluatorContext.getGraphVersion(),
        skyKey);
    maxTransitiveSourceVersion = version;
  }

  private void maybeUpdateMaxTransitiveSourceVersion(NodeEntry depEntry) {
    if (maxTransitiveSourceVersion == null
        || skyKey.functionName().getHermeticity() == FunctionHermeticity.NONHERMETIC) {
      return;
    }
    Version depMtsv = depEntry.getMaxTransitiveSourceVersion();
    if (depMtsv == null || maxTransitiveSourceVersion.atMost(depMtsv)) {
      maxTransitiveSourceVersion = depMtsv;
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
        .add("maxTransitiveSourceVersion", maxTransitiveSourceVersion)
        .add("bubbleErrorInfo", bubbleErrorInfo)
        .add("evaluatorContext", evaluatorContext)
        .toString();
  }

  @Override
  public boolean restartPermitted() {
    return evaluatorContext.restartPermitted();
  }

  @SuppressWarnings("unchecked")
  @Override
  public <T extends SkyKeyComputeState> T getState(Supplier<T> stateSupplier) {
    return (T) evaluatorContext.stateCache().get(skyKey, k -> stateSupplier.get());
  }

  boolean encounteredErrorDuringBubbling() {
    return encounteredErrorDuringBubbling;
  }

  @Override
  @Nullable
  public Version getMaxTransitiveSourceVersionSoFar() {
    return maxTransitiveSourceVersion;
  }

  /** Thrown during environment construction if a previously requested dep is no longer done. */
  static final class UndonePreviouslyRequestedDep extends Exception {
    private final SkyKey depKey;

    private UndonePreviouslyRequestedDep(SkyKey depKey) {
      this.depKey = depKey;
    }

    SkyKey getDepKey() {
      return depKey;
    }
  }
}
