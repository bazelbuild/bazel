// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.util.concurrent.Futures.whenAllSucceed;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.EmptyFileOpNode.EMPTY_FILE_OP_NODE;
import static com.google.devtools.build.lib.skyframe.serialization.proto.DataType.DATA_TYPE_ANALYSIS_NODE;
import static com.google.devtools.build.lib.skyframe.serialization.proto.DataType.DATA_TYPE_EXECUTION_NODE;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupSummaryKey;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.QuiescingFuture;
import com.google.devtools.build.lib.profiler.CounterSeriesCollector;
import com.google.devtools.build.lib.profiler.CounterSeriesTask;
import com.google.devtools.build.lib.profiler.CounterSeriesTask.Color;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNodeOrEmpty;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FutureFileOpNode;
import com.google.devtools.build.lib.skyframe.serialization.AsyncSerializationTask;
import com.google.devtools.build.lib.skyframe.serialization.DeserializedSkyValue;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.KeyValueWriter;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.ProfileCollector;
import com.google.devtools.build.lib.skyframe.serialization.SerializationResult;
import com.google.devtools.build.lib.skyframe.serialization.SharedValueSerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatus;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FutureFileDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FutureListingDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FutureNodeDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.InvalidationDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.NodeInvalidationDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingEventListener.SerializedNodeEvent;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;
import javax.annotation.Nullable;

/**
 * Supports uploading a selected set of {@link SkyKey}s to a {@link FingerprintValueService}.
 *
 * <p>Execution is encapsulated via the {@link #uploadSelection} method.
 *
 * <p>Each persisted entry consists of the following:
 *
 * <ul>
 *   <li><b>Key</b>: formatted as {@code fingerprint(<version stamp>, <serialized SkyKey>)}.
 *   <li><b>Invalidation Data and Value</b>: formatted as {@code <invalidation data>, <serialized
 *       SkyValue>}. The {@code invalidation data} consists of a {@link DataType} number followed by
 *       a corresponding key, if it is not {@link DATA_TYPE_EMPTY}.
 * </ul>
 *
 * <p>Note that both {@code <invalidation data>} and {@code <serialized SkyValue>} are intended to
 * have relatively small immediate representations. When there is a large amount of data, it will be
 * expressed via references (e.g., keys to a other {@link FingerprintValueService} entries).
 */
final class SelectedEntrySerializer {
  // Chosen completely arbitrarily and the first attempt worked out quite well
  private static final int MAX_PENDING_SKYVALUES = 20_000;

  /**
   * Counters for the progress of serialization.
   *
   * <p>Logged in the trace profile.
   */
  private static final class Counters implements CounterSeriesCollector {
    private final AtomicLong entriesWaitingForKeyBytes = new AtomicLong();
    private final AtomicLong entriesWaitingForValueBytes = new AtomicLong();
    private final AtomicLong entriesWaitingForInvalidationInfo = new AtomicLong();
    private final AtomicLong entriesWaitingForInvalidationBytes = new AtomicLong();
    private final AtomicLong entriesWaitingForUpload = new AtomicLong();
    private final AtomicLong entriesUploaded = new AtomicLong();
    private final AtomicLong keyBytesWaitingForUpload = new AtomicLong();
    private final AtomicLong valueBytesWaitingForUpload = new AtomicLong();
    private final AtomicLong keyBytesUploaded = new AtomicLong();
    private final AtomicLong valueBytesUploaded = new AtomicLong();

    private static final CounterSeriesTask ENTRIES_WAITING_FOR_KEY_BYTES =
        new CounterSeriesTask(
            "Skycache: SkyValues: Waiting for key bytes", "SkyValues", Color.RAIL_LOAD);

    private static final CounterSeriesTask ENTRIES_WAITING_FOR_VALUE_BYTES =
        new CounterSeriesTask(
            "Skycache: SkyValues: Waiting for value bytes", "SkyValues", Color.RAIL_LOAD);

    private static final CounterSeriesTask ENTRIES_WAITING_FOR_INVALIDATION_INFO =
        new CounterSeriesTask(
            "Skycache: SkyValues: Waiting for invalidation info", "SkyValues", Color.RAIL_LOAD);

    private static final CounterSeriesTask ENTRIES_WAITING_FOR_INVALIDATION_BYTES =
        new CounterSeriesTask(
            "Skycache: SkyValues: Waiting for invalidation bytes", "SkyValues", Color.RAIL_LOAD);

    private static final CounterSeriesTask ENTRIES_WAITING_FOR_UPLOAD =
        new CounterSeriesTask(
            "Skycache: SkyValues: Waiting for upload", "SkyValues", Color.RAIL_LOAD);
    private static final CounterSeriesTask ENTRIES_UPLOADED =
        new CounterSeriesTask("Skycache: SkyValues: Uploaded", "SkyValues", Color.RAIL_RESPONSE);

    private static final CounterSeriesTask KEY_BYTES_WAITING_FOR_UPLOAD =
        new CounterSeriesTask("Skycache: SkyValue bytes: Pending", "Key", Color.RAIL_LOAD);

    private static final CounterSeriesTask VALUE_BYTES_WAITING_FOR_UPLOAD =
        new CounterSeriesTask("Skycache: SkyValue bytes: Pending", "Value", Color.RAIL_LOAD);

    private static final CounterSeriesTask KEY_BYTES_UPLOADED =
        new CounterSeriesTask("Skycache: SkyValue bytes: Uploaded", "Key", Color.RAIL_RESPONSE);

    private static final CounterSeriesTask VALUE_BYTES_UPLOADED =
        new CounterSeriesTask("Skycache: SkyValue bytes: Uploaded", "Value", Color.RAIL_RESPONSE);

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {
      consumer.accept(ENTRIES_WAITING_FOR_KEY_BYTES, (double) entriesWaitingForKeyBytes.get());
      consumer.accept(ENTRIES_WAITING_FOR_VALUE_BYTES, (double) entriesWaitingForValueBytes.get());
      consumer.accept(
          ENTRIES_WAITING_FOR_INVALIDATION_INFO, (double) entriesWaitingForInvalidationInfo.get());
      consumer.accept(
          ENTRIES_WAITING_FOR_INVALIDATION_BYTES,
          (double) entriesWaitingForInvalidationBytes.get());
      consumer.accept(ENTRIES_WAITING_FOR_UPLOAD, (double) entriesWaitingForUpload.get());
      consumer.accept(ENTRIES_UPLOADED, (double) entriesUploaded.get());
      consumer.accept(KEY_BYTES_WAITING_FOR_UPLOAD, (double) keyBytesWaitingForUpload.get());
      consumer.accept(KEY_BYTES_UPLOADED, (double) keyBytesUploaded.get());
      consumer.accept(VALUE_BYTES_WAITING_FOR_UPLOAD, (double) valueBytesWaitingForUpload.get());
      consumer.accept(VALUE_BYTES_UPLOADED, (double) valueBytesUploaded.get());
    }
  }

  static final class SerializationStats {
    private final AtomicLong analysisNodes = new AtomicLong(0);
    private final AtomicLong executionNodes = new AtomicLong(0);

    SerializationStats() {}

    void registerAnalysisNode() {
      analysisNodes.incrementAndGet();
    }

    void registerExecutionNode() {
      executionNodes.incrementAndGet();
    }

    long analysisNodes() {
      return analysisNodes.get();
    }

    long executionNodes() {
      return executionNodes.get();
    }
  }

  private final InMemoryGraph graph;
  private final ObjectCodecs codecs;
  private final FrontierNodeVersion frontierVersion;

  private final FingerprintValueService fingerprintValueService;

  private final FileOpNodeMemoizingLookup fileOpNodes;
  private final FileDependencySerializer fileDependencySerializer;

  private final SerializationStatus writeStatuses;

  private final boolean shouldDiscardMemory;

  private final EventBus eventBus;
  private final ProfileCollector profileCollector;
  private final SerializationStats serializationStats;
  private final boolean emitUploadedEvents;

  /**
   * Tracks the number of selected configured target keys remaining to be uploaded per package.
   *
   * <p>Used to perform early memory reclamation of {@link PackageValue} nodes from the Skyframe
   * graph as soon as all of their selected configured targets have finished uploading.
   *
   * <p>Null if {@link #shouldDiscardMemory} is false.
   */
  @Nullable private final ImmutableMap<PackageIdentifier, AtomicInteger> packageRefcounts;

  /** Uploads the entries of {@code selection} to {@code fingerprintValueService}. */
  static QuiescingFuture<ImmutableList<Throwable>> uploadSelection(
      InMemoryGraph graph,
      LongVersionGetter versionGetter,
      ObjectCodecs codecs,
      FrontierNodeVersion frontierVersion,
      ImmutableSet<SkyKey> selection,
      FingerprintValueService fingerprintValueService,
      KeyValueWriter fileInvalidationWriter,
      boolean shouldDiscardMemory,
      EventBus eventBus,
      ProfileCollector profileCollector,
      SerializationStats serializationStats,
      boolean emitUploadedEvents,
      FileOpNodeMemoizingLookup fileOpNodes)
      throws InterruptedException {
    ImmutableMap<PackageIdentifier, AtomicInteger> packageRefcounts = null;
    if (shouldDiscardMemory) {
      var tempRefcounts = new ConcurrentHashMap<PackageIdentifier, AtomicInteger>();
      selection.parallelStream()
          .forEach(
              key -> {
                if (!(key instanceof ConfiguredTargetKey ctKey)) {
                  return;
                }
                tempRefcounts
                    .computeIfAbsent(
                        getActualPackageIdentifier(graph, ctKey), unused -> new AtomicInteger(0))
                    .incrementAndGet();
              });
      packageRefcounts = ImmutableMap.copyOf(tempRefcounts);
    }
    fileOpNodes.setMemoryReclamationParameters(
        selection, shouldDiscardMemory, shouldDiscardMemory ? packageRefcounts.keySet() : null);
    var fileDependencySerializer =
        new FileDependencySerializer(
            versionGetter,
            graph,
            fileInvalidationWriter,
            fingerprintValueService.getExecutor(),
            profileCollector);
    var writeStatuses = new SerializationStatus(fileDependencySerializer.getCounters());
    var serializer =
        new SelectedEntrySerializer(
            graph,
            codecs,
            frontierVersion,
            fingerprintValueService,
            fileOpNodes,
            fileDependencySerializer,
            writeStatuses,
            shouldDiscardMemory,
            packageRefcounts,
            eventBus,
            profileCollector,
            serializationStats,
            emitUploadedEvents);

    // A topological sort prevents the antipattern where one serializes a high level node, walks
    // its whole transitive closure, then serializes lower level nodes, thus revisiting the
    // transitive closure again.
    //
    // This doesn't help a lot when one only serializes a frontier or a small active set (since
    // the majority of the Skyframe graph is not serialized), but does help when serializing a
    // lot of nodes.
    ImmutableList<SkyKey> sortedSelection = sortTopologically(selection, graph);

    for (SkyKey selectedKey : sortedSelection) {
      serializer.upload(selectedKey);
    }

    writeStatuses.notifyAllStarted();
    return writeStatuses;
  }

  SelectedEntrySerializer(
      InMemoryGraph graph,
      ObjectCodecs codecs,
      FrontierNodeVersion frontierVersion,
      FingerprintValueService fingerprintValueService,
      FileOpNodeMemoizingLookup fileOpNodes,
      FileDependencySerializer fileDependencySerializer,
      SerializationStatus writeStatuses,
      boolean shouldDiscardMemory,
      @Nullable ImmutableMap<PackageIdentifier, AtomicInteger> packageRefcounts,
      EventBus eventBus,
      ProfileCollector profileCollector,
      SerializationStats serializationStats,
      boolean emitUploadedEvents) {
    this.graph = graph;
    this.codecs = codecs;
    this.frontierVersion = frontierVersion;
    this.fingerprintValueService = fingerprintValueService;
    this.fileOpNodes = fileOpNodes;
    this.fileDependencySerializer = fileDependencySerializer;
    this.writeStatuses = writeStatuses;
    this.shouldDiscardMemory = shouldDiscardMemory;
    this.packageRefcounts = packageRefcounts;
    this.eventBus = eventBus;
    this.profileCollector = profileCollector;
    this.serializationStats = serializationStats;
    this.emitUploadedEvents = emitUploadedEvents;
  }

  public void upload(SkyKey key) throws InterruptedException {
    InMemoryNodeEntry entry = graph.getIfPresent(key);
    if (entry != null && entry.getValue() instanceof DeserializedSkyValue) {
      return;
    }
    // TODO: b/371508153 - only upload nodes that were freshly computed by this invocation and
    // unaffected by local, un-submitted changes.
    writeStatuses.selectedEntryStartingCapped();
    try {
      switch (key) {
        case ActionLookupKey actionLookupKey -> {
          if (entry == null) {
            throw new MissingSkyframeEntryException(actionLookupKey);
          }
          serializationStats.registerAnalysisNode();
          uploadAnalysisEntry(actionLookupKey, entry.getValue(), entry.getDirectDeps());
        }
        case ActionLookupData lookupData -> {
          serializationStats.registerExecutionNode();
          uploadExecutionEntry(lookupData, null);
        }
        case DerivedArtifact artifact -> {
          // This case handles the subclasses of DerivedArtifact. DerivedArtifact itself will show
          // up here as ActionLookupData.
          serializationStats.registerExecutionNode();
          uploadExecutionEntry(artifact, null);
        }
        case ActionLookupSummaryKey summaryKey -> {
          serializationStats.registerExecutionNode();
          uploadExecutionEntry(summaryKey, null);
        }
        default -> throw new AssertionError("Unexpected selected type: " + key.getCanonicalName());
      }
      eventBus.post(new SerializedNodeEvent(key));
    } catch (MissingSkyframeEntryException e) {
      writeStatuses.selectedEntryFailed(e);
    } catch (Throwable t) {
      writeStatuses.selectedEntryFailed(t);
    }
  }

  /**
   * Uploads an analysis phase entry to Skycache.
   *
   * <p>Direct deps must always be given.
   */
  public void uploadAnalysisEntry(
      ActionLookupKey key, SkyValue value, Iterable<SkyKey> directDeps) {
    // For analysis phase entries, we register their own dependencies in the invalidation data
    uploadEntry(key, value, key, directDeps);
  }

  /**
   * Uploads an execution phase entry to Skycache.
   *
   * <p>If {@code value} is not given, it is read from Skyframe.
   */
  public void uploadExecutionEntry(SkyKey key, @Nullable SkyValue value)
      throws MissingSkyframeEntryException {
    // For execution phase entries, we register the dependencies of their owner in the invalidation
    // data. This way, every action owned by the same action has the same invalidation data, which
    // reduces storage use a little bit and keeps things compatible with a Google-specific
    // optimization.
    Preconditions.checkArgument(!(key instanceof ActionLookupKey));

    if (value == null) {
      InMemoryNodeEntry entry = graph.getIfPresent(key);
      if (entry == null) {
        throw new MissingSkyframeEntryException(key);
      }
      value = entry.getValue();
    }
    ActionLookupKey dependencyKey = getDependencyKey(key);

    // We don't pass directDeps in because the code must be tolerant to those not being available:
    // if we delete nodes as we upload them, the NodeEntry to dependencyKey might not be available
    // anymore. In this case, FileOpNodeMemoizingLookup will definitely contain an entry for it,
    // since creating one is a side effect of uploading. If we are not deleting them, it will do
    // a graph lookup anyway.
    uploadEntry(key, value, dependencyKey, null);
  }

  private static ActionLookupKey getDependencyKey(SkyKey key) {
    return switch (key) {
      case ActionLookupData ald -> ald.getActionLookupKey();
      case DerivedArtifact artifact -> artifact.getArtifactOwner();
      case ActionLookupSummaryKey alsk -> alsk.argument();
      default -> throw new IllegalStateException("unexpected key: " + key.getCanonicalName());
    };
  }

  /**
   * Uploads the entry associated with {@code key} persisting alongside it, the file dependencies
   * associated with {@code dependencyKey}.
   *
   * @param key the {@link SkyKey} of the entry being uploaded
   * @param value the value of the Skyframe node
   * @param dependencyKey the {@link SkyKey} whose file system dependencies are to be used
   * @param dependencyDeps the dependencies to traverse. These should be the direct deps of {@code
   *     dependencyDeps}. If null, Skyframe will be asked for the deps of {@code key}
   */
  private void uploadEntry(
      SkyKey key,
      SkyValue value,
      ActionLookupKey dependencyKey,
      @Nullable Iterable<SkyKey> dependencyDeps) {
    new UploadTask(key, value, dependencyKey, dependencyDeps).submit();
  }

  private final class UploadTask implements Runnable, FutureCallback<FileOpNodeOrEmpty> {
    private final SkyKey key;
    private final SkyValue value;
    private final ActionLookupKey dependencyKey;
    @Nullable private final Iterable<SkyKey> dependencyDeps;
    private final boolean isExecutionValue;

    // Keys are always stored as fingerprints so their detailed profiles are omitted.
    private AsyncSerializationTask keyResultTask;
    private AsyncSerializationTask valueResultTask;

    private UploadTask(
        SkyKey key,
        SkyValue value,
        ActionLookupKey dependencyKey,
        @Nullable Iterable<SkyKey> dependencyDeps) {
      this.key = key;
      this.value = value;
      this.dependencyKey = dependencyKey;
      this.dependencyDeps = dependencyDeps;
      this.isExecutionValue = isExecutionValue(key);
    }

    void submit() {
      try {
        fingerprintValueService.getExecutor().execute(this);
      } catch (RejectedExecutionException e) {
        writeStatuses.selectedEntryFailed(e);
        throw e;
      }
    }

    @Override
    public void run() {
      try {
        if (writeStatuses.hasError()) {
          writeStatuses.selectedEntryDone();
          eventBus.post(new SerializedNodeEvent(key));
          return;
        }

        writeStatuses.counters.entriesWaitingForKeyBytes.incrementAndGet();
        writeStatuses.counters.entriesWaitingForValueBytes.incrementAndGet();
        writeStatuses.counters.entriesWaitingForInvalidationInfo.incrementAndGet();

        this.keyResultTask =
            codecs.serializeMemoizedAsync(
                fingerprintValueService, key, /* profileCollector= */ null);
        fingerprintValueService.getExecutor().execute(keyResultTask);
        this.valueResultTask =
            codecs.serializeMemoizedAsync(fingerprintValueService, value, profileCollector);
        fingerprintValueService.getExecutor().execute(valueResultTask);

        keyResultTask.addListener(
            () -> writeStatuses.counters.entriesWaitingForKeyBytes.decrementAndGet(),
            directExecutor());
        valueResultTask.addListener(
            () -> writeStatuses.counters.entriesWaitingForValueBytes.decrementAndGet(),
            directExecutor());

        // We pass a null value for execution entries to maintain the invariant that value is
        // non-null only for analysis entries.
        FileOpNodeOrFuture fileOpNodeOrFuture =
            fileOpNodes.computeNode(dependencyKey, isExecutionValue ? null : value, dependencyDeps);
        switch (fileOpNodeOrFuture) {
          case FileOpNodeOrEmpty nodeOrEmpty -> onSuccess(nodeOrEmpty);
          case FutureFileOpNode future ->
              Futures.addCallback(future, this, fingerprintValueService.getExecutor());
        }
        eventBus.post(new SerializedNodeEvent(key));

      } catch (Throwable t) {
        writeStatuses.selectedEntryFailed(t);
      }
    }

    /**
     * Saves the entry for to the {@link FingerprintValueService}.
     *
     * <p>The entry includes both value and the associated invalidation data. More precisely, it
     * consists of the following components.
     *
     * <ol>
     *   <li>If {@code dataInfo} is null, the invalidation data is the {@code DATA_TYPE_EMPTY} value
     *       only.
     *   <li>Otherwise when {@code dataInfo} is non-null, the invalidation data starts with a type
     *       value, {@code DATA_TYPE_FILE}, {@code DATA_TYPE_LISTING}, {@code
     *       DATA_TYPE_ANALYSIS_NODE} or {@code DATA_TYPE_EXECUTION_NODE}, depending on {@code
     *       dataInfo}'s and {@link UploadTask#value}'s type. The invalidation data cache key
     *       follows the type value.
     *   <li>The {@link #value}'s value bytes.
     * </ol>
     */
    @Override
    public void onSuccess(FileOpNodeOrEmpty nodeOrEmpty) {
      try {
        writeStatuses.counters.entriesWaitingForInvalidationInfo.decrementAndGet();
        writeStatuses.counters.entriesWaitingForInvalidationBytes.incrementAndGet();

        ListenableFuture<InvalidationDataInfo> futureDataInfo =
            switch (nodeOrEmpty) {
              case FileOpNode node ->
                  switch (fileDependencySerializer.registerDependency(node)) {
                    case InvalidationDataInfo dataInfo ->
                        whenAllSucceed(keyResultTask, valueResultTask)
                            .call(() -> dataInfo, directExecutor());
                    case FutureFileDataInfo futureFile ->
                        whenAllSucceed(keyResultTask, valueResultTask, futureFile)
                            .call(() -> Futures.getDone(futureFile), directExecutor());
                    case FutureListingDataInfo futureListing ->
                        whenAllSucceed(keyResultTask, valueResultTask, futureListing)
                            .call(() -> Futures.getDone(futureListing), directExecutor());
                    case FutureNodeDataInfo futureNode ->
                        whenAllSucceed(keyResultTask, valueResultTask, futureNode)
                            .call(() -> Futures.getDone(futureNode), directExecutor());
                  };
              case EMPTY_FILE_OP_NODE ->
                  whenAllSucceed(keyResultTask, valueResultTask).call(() -> null, directExecutor());
            };
        Futures.addCallback(
            futureDataInfo,
            new InvalidationDataInfoHandler(),
            fingerprintValueService.getExecutor());
      } catch (Throwable t) {
        writeStatuses.counters.entriesWaitingForInvalidationBytes.decrementAndGet();
        writeStatuses.selectedEntryFailed(t);
      }
    }

    @Override
    public void onFailure(Throwable t) {
      writeStatuses.counters.entriesWaitingForInvalidationInfo.decrementAndGet();
      writeStatuses.selectedEntryFailed(t);
    }

    private final class InvalidationDataInfoHandler
        implements FutureCallback<InvalidationDataInfo> {

      /**
       * Saves the entry for to the {@link FingerprintValueService}.
       *
       * <p>The entry includes both value and the associated invalidation data. More precisely, it
       * consists of the following components.
       *
       * <ol>
       *   <li>If {@code dataInfo} is null, the invalidation data is the {@link DATA_TYPE_EMPTY}
       *       value only.
       *   <li>Otherwise when {@code dataInfo} is non-null, the invalidation data starts with a type
       *       value, {@link DATA_TYPE_FILE}, {@link DATA_TYPE_LISTING}, {@link
       *       DATA_TYPE_ANALYSIS_NODE} or {@link DATA_TYPE_EXECUTION_NODE}, depending on {@code
       *       dataInfo}'s and {@link UploadTask#key}'s type. The invalidation data cache key
       *       follows the type value.
       *   <li>The {@link UploadTask#value}'s value bytes.
       * </ol>
       */
      @Override
      public void onSuccess(@Nullable InvalidationDataInfo dataInfo) {
        if (shouldDiscardMemory) {
          // Reclaim memory early: once a selected entry is successfully serialized and uploaded,
          // its value is no longer needed in the evaluator. If it's a ConfiguredTargetKey, we
          // also decrement the refcount of its package. Once all selected configured targets in
          // the package are uploaded, the PackageValue is also discarded, releasing substantial
          // memory early.
          if (key instanceof ConfiguredTargetKey ctKey) {
            PackageIdentifier pkgId = getActualPackageIdentifier(graph, ctKey);
            if (packageRefcounts.get(pkgId).decrementAndGet() <= 0) {
              graph.removeIfDone(pkgId);
            }
          }
          graph.removeIfDone(key);
        }
        try {
          ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
          CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);

          SerializationResult<ByteString> keyResult;
          SerializationResult<ByteString> valueResult;
          try {
            keyResult = Futures.getDone(keyResultTask);
            valueResult = Futures.getDone(valueResultTask);
          } catch (ExecutionException e) {
            throw new IllegalStateException(
                "should have succeeded as part of the FutureCombiner", e);
          }
          writeStatuses.addWriteStatus(valueResult.getFutureToBlockWritesOn());
          writeStatuses.addWriteStatus(keyResult.getFutureToBlockWritesOn());

          // Entries representing SkyValues should always have a node as their invalidation info
          // so that that node can be reported as the invalidation fingerprint to Blaze.
          if (!(dataInfo instanceof NodeInvalidationDataInfo node)) {
            throw new IllegalStateException(
                String.format(
                    "Expected NodeInvalidationDataInfo for %s, but got %s",
                    key, dataInfo == null ? "null" : dataInfo.getClass().getSimpleName()));
          }

          codedOut.writeEnumNoTag(
              (isExecutionValue ? DATA_TYPE_EXECUTION_NODE : DATA_TYPE_ANALYSIS_NODE).getNumber());
          node.cacheKey().writeTo(codedOut);
          writeStatuses.addWriteStatus(node.writeStatus());
          codedOut.writeRawBytes(valueResult.getObject());
          codedOut.flush();

          PackedFingerprint versionedKey =
              fingerprintValueService.fingerprint(
                  frontierVersion.concat(keyResult.getObject().toByteArray()));
          byte[] entryBytes = bytesOut.toByteArray();

          // Put this in a separate variable so that we don't close over potentially large byte
          // arrays
          long keyByteCount = versionedKey.toBytes().length;
          long valueByteCount = entryBytes.length;
          writeStatuses.counters.entriesWaitingForInvalidationBytes.decrementAndGet();
          writeStatuses.counters.entriesWaitingForUpload.incrementAndGet();
          writeStatuses.counters.keyBytesWaitingForUpload.addAndGet(keyByteCount);
          writeStatuses.counters.valueBytesWaitingForUpload.addAndGet(valueByteCount);

          // We wait for the invalidation data and the value to be uploaded before we upload the
          // entry. Otherwise readers would get a cache miss on this entry if they happened to
          // try to read the entry before the upload finishes, but potentially not before doing a
          // lot of useless work. We don't wait for the key data to be uploaded because it is
          // never deserialized and it's a wart that we upload anything as part of serializing the
          // key anyway because we only need its fingerprint.
          ArrayList<ListenableFuture<?>> futuresToBlockOn = new ArrayList<>(2);
          if (valueResult.getFutureToBlockWritesOn() != null) {
            futuresToBlockOn.add(valueResult.getFutureToBlockWritesOn());
          }
          futuresToBlockOn.add(node.writeStatus());
          ListenableFuture<Void> blockedOn =
              whenAllSucceed(futuresToBlockOn).call(() -> null, directExecutor());
          Futures.addCallback(
              blockedOn,
              new FutureCallback<>() {
                @Override
                public void onSuccess(Void unused) {
                  uploadEntryBytes(versionedKey, entryBytes, keyByteCount, valueByteCount);
                }

                @Override
                public void onFailure(Throwable t) {
                  onUploadFailure(t, keyByteCount, valueByteCount);
                }
              },
              fingerprintValueService.getExecutor());
        } catch (Throwable t) {
          onFailure(t);
        }
      }

      @Override
      public void onFailure(Throwable t) {
        writeStatuses.counters.entriesWaitingForInvalidationBytes.decrementAndGet();
        writeStatuses.selectedEntryFailed(t);
      }
    }

    private void uploadEntryBytes(
        PackedFingerprint versionedKey, byte[] entryBytes, long keyByteCount, long valueByteCount) {
      try {
        WriteStatus putStatus = fingerprintValueService.put(versionedKey, entryBytes);
        valueResultTask.registerWriteStatus(putStatus);

        putStatus.addListener(
            () -> {
              writeStatuses.counters.entriesWaitingForUpload.decrementAndGet();
              writeStatuses.counters.keyBytesWaitingForUpload.addAndGet(-keyByteCount);
              writeStatuses.counters.valueBytesWaitingForUpload.addAndGet(-valueByteCount);
              boolean shouldUpdateCounts;
              try {
                // Avoids updating counts if the writes are marked as duplicates. Note that
                // duplicate detection is not ordinarily enabled.
                shouldUpdateCounts = Futures.getDone(putStatus);
              } catch (ExecutionException e) {
                // This error is propagated to the main control flow via `writeStatuses`.
                shouldUpdateCounts = false;
              }
              if (shouldUpdateCounts) {
                writeStatuses.counters.entriesUploaded.incrementAndGet();
                writeStatuses.counters.keyBytesUploaded.addAndGet(keyByteCount);
                writeStatuses.counters.valueBytesUploaded.addAndGet(valueByteCount);
                if (emitUploadedEvents) {
                  eventBus.post(new SkyValueUploadedEvent(key, frontierVersion, versionedKey));
                }
              }
            },
            directExecutor());

        writeStatuses.addWriteStatus(putStatus);
        writeStatuses.selectedEntryDone();
      } catch (Throwable t) {
        onUploadFailure(t, keyByteCount, valueByteCount);
      }
    }

    private void onUploadFailure(Throwable t, long keyByteCount, long valueByteCount) {
      writeStatuses.counters.entriesWaitingForUpload.decrementAndGet();
      writeStatuses.counters.keyBytesWaitingForUpload.addAndGet(-keyByteCount);
      writeStatuses.counters.valueBytesWaitingForUpload.addAndGet(-valueByteCount);
      writeStatuses.selectedEntryFailed(t);
    }
  }

  /** Serialization status. */
  public static class SerializationStatus extends QuiescingFuture<ImmutableList<Throwable>>
      implements FutureCallback<Object>, CounterSeriesCollector {
    private final Semaphore inflightSkyframeEntrySemaphore = new Semaphore(MAX_PENDING_SKYVALUES);
    private final ConcurrentLinkedQueue<Throwable> errors = new ConcurrentLinkedQueue<>();
    private final FileDependencySerializer.Counters fileDependencySerializerCounters;
    private final Counters counters = new Counters();

    private static final CounterSeriesTask BYTES_WAITING_FOR_FUTURE_PUTS =
        new CounterSeriesTask(
            "Skycache: Serialization: Bytes: Pending", "Waiting for future puts", Color.RAIL_LOAD);
    private static final CounterSeriesTask BYTES_WAITING_FOR_UPLOAD =
        new CounterSeriesTask(
            "Skycache: Serialization: Bytes: Pending", "Waiting for upload", Color.RAIL_LOAD);
    private static final CounterSeriesTask BYTES_UPLOADED =
        new CounterSeriesTask(
            "Skycache: Serialization: Bytes: Uploaded", "Written", Color.RAIL_RESPONSE);
    private static final CounterSeriesTask OBJECTS_WAITING_FOR_SERIALIZATION =
        new CounterSeriesTask(
            "Skycache: Serialization: Objects: Pending",
            "Waiting for serialization",
            Color.RAIL_LOAD);
    private static final CounterSeriesTask OBJECTS_WAITING_FOR_FUTURE_PUTS =
        new CounterSeriesTask(
            "Skycache: Serialization: Objects: Pending",
            "Waiting for future puts",
            Color.RAIL_LOAD);
    private static final CounterSeriesTask OBJECTS_WAITING_FOR_UPLOAD =
        new CounterSeriesTask(
            "Skycache: Serialization: Objects: Pending", "Waiting for upload", Color.RAIL_LOAD);
    private static final CounterSeriesTask OBJECTS_UPLOADED =
        new CounterSeriesTask(
            "Skycache: Serialization: Objects: Uploaded", "done", Color.RAIL_RESPONSE);

    SerializationStatus(FileDependencySerializer.Counters fileDependencySerializerCounters) {
      super(directExecutor());

      this.fileDependencySerializerCounters = fileDependencySerializerCounters;

      Profiler.instance().registerCounterSeriesCollector(this);
      Profiler.instance().registerCounterSeriesCollector(counters);
      Profiler.instance().registerCounterSeriesCollector(fileDependencySerializerCounters);
    }

    void selectedEntryStartingCapped() throws InterruptedException {
      inflightSkyframeEntrySemaphore.acquire();
      increment();
    }

    void selectedEntryDone() {
      decrement();
      inflightSkyframeEntrySemaphore.release();
    }

    void selectedEntryFailed(Throwable t) {
      notifyWriteFailure(t);
      inflightSkyframeEntrySemaphore.release();
    }

    void notifyAllStarted() {
      decrement();
    }

    private void notifyWriteFailure(Throwable t) {
      errors.add(t);
      decrement();
    }

    private boolean hasError() {
      return !errors.isEmpty();
    }

    @Override
    protected ImmutableList<Throwable> getValue() {
      Profiler.instance().unregisterCounterSeriesCollector(this);
      Profiler.instance().unregisterCounterSeriesCollector(counters);
      Profiler.instance().unregisterCounterSeriesCollector(fileDependencySerializerCounters);
      return ImmutableList.copyOf(errors);
    }

    private void addWriteStatus(@Nullable ListenableFuture<?> writeStatus) {
      if (writeStatus == null) {
        return;
      }
      increment();
      Futures.addCallback(writeStatus, (FutureCallback<Object>) this, directExecutor());
    }

    /**
     * Implementation of {@link FutureCallback<Object>}.
     *
     * @deprecated only for use via {@link #addWriteStatus}
     */
    @Override
    @Deprecated // only called via addWriteStatus
    public void onSuccess(Object unused) {
      decrement();
    }

    /**
     * Implementation of {@link FutureCallback<void>}.
     *
     * @deprecated only for use via {@link #addWriteStatus}
     */
    @Override
    @Deprecated // only called via addWriteStatus
    public void onFailure(Throwable t) {
      notifyWriteFailure(t);
    }

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {
      // This should really be a method on StatsCollector but that means that serialization must
      // depend on profiler, and profiler transitively depends on serialization because it depends
      // on
      // common/options, which has EnvVar, which is marked as @AutoCodec.

      consumer.accept(
          BYTES_WAITING_FOR_FUTURE_PUTS,
          (double) SharedValueSerializationContext.COUNTERS.getBytesWaitingForFuturePuts());
      consumer.accept(
          BYTES_WAITING_FOR_UPLOAD,
          (double) SharedValueSerializationContext.COUNTERS.getBytesWaitingForUpload());
      consumer.accept(
          BYTES_UPLOADED, (double) SharedValueSerializationContext.COUNTERS.getBytesUploaded());
      consumer.accept(
          OBJECTS_WAITING_FOR_SERIALIZATION,
          (double) SharedValueSerializationContext.COUNTERS.getObjectsWaitingForSerialization());
      consumer.accept(
          OBJECTS_WAITING_FOR_FUTURE_PUTS,
          (double) SharedValueSerializationContext.COUNTERS.getObjectsWaitingForFuturePuts());
      consumer.accept(
          OBJECTS_WAITING_FOR_UPLOAD,
          (double) SharedValueSerializationContext.COUNTERS.getObjectsWaitingForUpload());
      consumer.accept(
          OBJECTS_UPLOADED, (double) SharedValueSerializationContext.COUNTERS.getObjectsUploaded());
    }
  }

  private static boolean isExecutionValue(SkyKey key) {
    // TODO: b/439060530: consider whether this is correct for ActionTemplateExpansionValue keys.
    return !(key instanceof ActionLookupKey);
  }

  /** Sorts {@code selection} topologically based on the edges in {@code graph}. */
  private static ImmutableList<SkyKey> sortTopologically(
      ImmutableSet<SkyKey> selection, InMemoryGraph graph) {
    ArrayList<SkyKey> result = new ArrayList<>(selection.size());
    Set<SkyKey> visited = Sets.newHashSetWithExpectedSize(selection.size());
    for (SkyKey key : selection) {
      if (visited.add(key)) {
        dfs(key, selection, graph, visited, result);
      }
    }
    return ImmutableList.copyOf(result);
  }

  private static void dfs(
      SkyKey key,
      ImmutableSet<SkyKey> selection,
      InMemoryGraph graph,
      Set<SkyKey> visited,
      List<SkyKey> result) {
    InMemoryNodeEntry entry = graph.getIfPresent(key);
    if (entry == null) {
      return;
    }

    for (SkyKey dep : entry.getDirectDeps()) {
      // This is suboptimal when "selection" is non-contiguous, but makes it possible to avoid
      // visiting the whole Skyframe graph when serializing just a frontier
      if (selection.contains(dep) && visited.add(dep)) {
        dfs(dep, selection, graph, visited, result);
      }
    }

    result.add(key);
  }

  /**
   * Returns the real package associated with a configured target.
   *
   * <p>The configured target may be an alias where the referent package contains its target data.
   */
  private static PackageIdentifier getActualPackageIdentifier(
      InMemoryGraph graph, ConfiguredTargetKey key) {
    return ((ConfiguredTargetValue) graph.getIfPresent(key).getValue())
        .getConfiguredTarget()
        .getLabel()
        .getPackageIdentifier();
  }
}
