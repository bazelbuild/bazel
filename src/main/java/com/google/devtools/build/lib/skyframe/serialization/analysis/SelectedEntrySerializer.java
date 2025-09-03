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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.util.concurrent.Futures.whenAllSucceed;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.EmptyFileOpNode.EMPTY_FILE_OP_NODE;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.ConstantFileData.CONSTANT_FILE;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.ConstantListingData.CONSTANT_LISTING;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.ConstantNodeData.CONSTANT_NODE;
import static com.google.devtools.build.lib.skyframe.serialization.proto.DataType.DATA_TYPE_ANALYSIS_NODE;
import static com.google.devtools.build.lib.skyframe.serialization.proto.DataType.DATA_TYPE_EMPTY;
import static com.google.devtools.build.lib.skyframe.serialization.proto.DataType.DATA_TYPE_EXECUTION_NODE;
import static com.google.devtools.build.lib.skyframe.serialization.proto.DataType.DATA_TYPE_FILE;
import static com.google.devtools.build.lib.skyframe.serialization.proto.DataType.DATA_TYPE_LISTING;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.concurrent.QuiescingFuture;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNodeOrEmpty;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FutureFileOpNode;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.ProfileCollector;
import com.google.devtools.build.lib.skyframe.serialization.SerializationResult;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FileInvalidationDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FutureFileDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FutureListingDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FutureNodeDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.InvalidationDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.ListingInvalidationDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.NodeInvalidationDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingEventListener.SerializedNodeEvent;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
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
final class SelectedEntrySerializer implements Consumer<SkyKey> {
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

  private final WriteStatusesFuture writeStatuses;

  private final EventBus eventBus;
  private final ProfileCollector profileCollector;
  private final SerializationStats serializationStats;

  /** Uploads the entries of {@code selection} to {@code fingerprintValueService}. */
  static ListenableFuture<ImmutableList<Throwable>> uploadSelection(
      InMemoryGraph graph,
      LongVersionGetter versionGetter,
      ObjectCodecs codecs,
      FrontierNodeVersion frontierVersion,
      ImmutableSet<SkyKey> selection,
      FingerprintValueService fingerprintValueService,
      EventBus eventBus,
      ProfileCollector profileCollector,
      SerializationStats serializationStats) {
    var fileOpNodes = new FileOpNodeMemoizingLookup(graph);
    var fileDependencySerializer =
        new FileDependencySerializer(versionGetter, graph, fingerprintValueService);
    var writeStatuses = new WriteStatusesFuture();
    var serializer =
        new SelectedEntrySerializer(
            graph,
            codecs,
            frontierVersion,
            fingerprintValueService,
            fileOpNodes,
            fileDependencySerializer,
            writeStatuses,
            eventBus,
            profileCollector,
            serializationStats);
    selection.parallelStream().forEach(serializer);
    writeStatuses.notifyAllStarted();
    return writeStatuses;
  }

  private SelectedEntrySerializer(
      InMemoryGraph graph,
      ObjectCodecs codecs,
      FrontierNodeVersion frontierVersion,
      FingerprintValueService fingerprintValueService,
      FileOpNodeMemoizingLookup fileOpNodes,
      FileDependencySerializer fileDependencySerializer,
      WriteStatusesFuture writeStatuses,
      EventBus eventBus,
      ProfileCollector profileCollector,
      SerializationStats serializationStats) {
    this.graph = graph;
    this.codecs = codecs;
    this.frontierVersion = frontierVersion;

    this.fingerprintValueService = fingerprintValueService;
    this.fileOpNodes = fileOpNodes;
    this.fileDependencySerializer = fileDependencySerializer;
    this.writeStatuses = writeStatuses;
    this.eventBus = eventBus;
    this.profileCollector = profileCollector;
    this.serializationStats = serializationStats;
  }

  @Override
  public void accept(SkyKey key) {
    // TODO: b/371508153 - only upload nodes that were freshly computed by this invocation and
    // unaffected by local, un-submitted changes.
    try {
      switch (key) {
        case ActionLookupKey actionLookupKey:
          serializationStats.registerAnalysisNode();
          uploadEntry(actionLookupKey, actionLookupKey);
          break;
        case ActionLookupData lookupData:
          serializationStats.registerExecutionNode();
          uploadEntry(lookupData, checkNotNull(lookupData.getActionLookupKey(), lookupData));
          break;
        case DerivedArtifact artifact:
          // This case handles the subclasses of DerivedArtifact. DerivedArtifact itself will show
          // up here as ActionLookupData.
          serializationStats.registerExecutionNode();
          uploadEntry(artifact, checkNotNull(artifact.getArtifactOwner(), artifact));
          break;
        default:
          throw new AssertionError("Unexpected selected type: " + key.getCanonicalName());
      }
      eventBus.post(new SerializedNodeEvent(key));
    } catch (MissingSkyframeEntryException e) {
      writeStatuses.notifyWriteFailure(e);
    }
  }

  /**
   * Uploads the entry associated with {@code key} persisting alongside it, the file dependencies
   * associated with {@code dependencyKey}.
   */
  private void uploadEntry(SkyKey key, ActionLookupKey dependencyKey)
      throws MissingSkyframeEntryException {
    if (writeStatuses.hasError()) {
      return;
    }

    writeStatuses.selectedEntryStarting();

    InMemoryNodeEntry nodeEntry = graph.getIfPresent(key);
    if (nodeEntry == null) {
      // TODO: b/400460727 - add some coverage for this code path
      throw new MissingSkyframeEntryException(key);
    }

    ListenableFuture<SerializationResult<ByteString>> futureKeyBytes =
        Futures.submitAsync(
            () -> codecs.serializeMemoizedAsync(fingerprintValueService, key, profileCollector),
            fingerprintValueService.getExecutor());

    ListenableFuture<SerializationResult<ByteString>> futureValueBytes =
        Futures.submitAsync(
            () ->
                codecs.serializeMemoizedAsync(
                    fingerprintValueService, nodeEntry.getValue(), profileCollector),
            fingerprintValueService.getExecutor());

    new FileOpNodeProcessor(futureKeyBytes, futureValueBytes, isExecutionValue(key), dependencyKey)
        .run();
  }

  private final class FileOpNodeProcessor implements FutureCallback<FileOpNodeOrEmpty>, Runnable {
    private final ListenableFuture<SerializationResult<ByteString>> futureKeyBytes;
    private final ListenableFuture<SerializationResult<ByteString>> futureValueBytes;
    private final boolean isExecutionValue;
    private final ActionLookupKey dependencyKey;

    private FileOpNodeProcessor(
        ListenableFuture<SerializationResult<ByteString>> futureKeyBytes,
        ListenableFuture<SerializationResult<ByteString>> futureValueBytes,
        boolean isExecutionValue,
        ActionLookupKey dependencyKey) {
      this.futureKeyBytes = futureKeyBytes;
      this.futureValueBytes = futureValueBytes;
      this.isExecutionValue = isExecutionValue;
      this.dependencyKey = dependencyKey;
    }

    @Override
    public void run() {
      switch (fileOpNodes.computeNode(dependencyKey)) {
        case FileOpNodeOrEmpty nodeOrEmpty:
          onSuccess(nodeOrEmpty);
          break;
        case FutureFileOpNode future:
          Futures.addCallback(future, this, fingerprintValueService.getExecutor());
          break;
      }
    }

    @Override
    public final void onSuccess(FileOpNodeOrEmpty nodeOrEmpty) {
      ListenableFuture<InvalidationDataInfo> futureDataInfo =
          switch (nodeOrEmpty) {
            case FileOpNode node ->
                switch (fileDependencySerializer.registerDependency(node)) {
                  case InvalidationDataInfo dataInfo ->
                      whenAllSucceed(futureKeyBytes, futureValueBytes)
                          .call(() -> dataInfo, directExecutor());
                  case FutureFileDataInfo futureFile ->
                      whenAllSucceed(futureKeyBytes, futureValueBytes, futureFile)
                          .call(() -> Futures.getDone(futureFile), directExecutor());
                  case FutureListingDataInfo futureListing ->
                      whenAllSucceed(futureKeyBytes, futureValueBytes, futureListing)
                          .call(() -> Futures.getDone(futureListing), directExecutor());
                  case FutureNodeDataInfo futureNode ->
                      whenAllSucceed(futureKeyBytes, futureValueBytes, futureNode)
                          .call(() -> Futures.getDone(futureNode), directExecutor());
                };
            case EMPTY_FILE_OP_NODE ->
                whenAllSucceed(futureKeyBytes, futureValueBytes).call(() -> null, directExecutor());
          };
      Futures.addCallback(
          futureDataInfo, new InvalidationDataInfoHandler(), fingerprintValueService.getExecutor());
    }

    @Override
    public final void onFailure(Throwable t) {
      writeStatuses.notifyWriteFailure(t);
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
       *       dataInfo}'s and {@link FileOpNodeProcessor#entry}'s type. The invalidation data cache
       *       key follows the type value.
       *   <li>The {@link #entry}'s value bytes.
       * </ol>
       */
      @Override
      public void onSuccess(@Nullable InvalidationDataInfo dataInfo) {
        ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
        CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);

        SerializationResult<ByteString> keyBytes;
        SerializationResult<ByteString> valueBytes;
        try {
          keyBytes = Futures.getDone(futureKeyBytes);
          valueBytes = Futures.getDone(futureValueBytes);
        } catch (ExecutionException e) {
          throw new IllegalStateException("should have succeeded as part of the FutureCombiner", e);
        }
        writeStatuses.addWriteStatus(valueBytes.getFutureToBlockWritesOn());
        writeStatuses.addWriteStatus(keyBytes.getFutureToBlockWritesOn());

        try {
          switch (dataInfo) {
            case CONSTANT_FILE:
            case CONSTANT_LISTING:
            case CONSTANT_NODE:
            // fall through
            case null:
              codedOut.writeEnumNoTag(DATA_TYPE_EMPTY.getNumber());
              break;
            case FileInvalidationDataInfo file:
              codedOut.writeEnumNoTag(DATA_TYPE_FILE.getNumber());
              codedOut.writeStringNoTag(file.cacheKey());
              writeStatuses.addWriteStatus(file.writeStatus());
              break;
            case ListingInvalidationDataInfo listing:
              codedOut.writeEnumNoTag(DATA_TYPE_LISTING.getNumber());
              codedOut.writeStringNoTag(listing.cacheKey());
              writeStatuses.addWriteStatus(listing.writeStatus());
              break;
            case NodeInvalidationDataInfo node:
              codedOut.writeEnumNoTag(
                  (isExecutionValue ? DATA_TYPE_EXECUTION_NODE : DATA_TYPE_ANALYSIS_NODE)
                      .getNumber());
              node.cacheKey().writeTo(codedOut);
              writeStatuses.addWriteStatus(node.writeStatus());
              break;
          }
          codedOut.writeRawBytes(valueBytes.getObject());
          codedOut.flush();
        } catch (IOException e) {
          // A ByteArrayOutputStream backed CodedOutputStream doesn't throw IOExceptions.
          throw new AssertionError(e);
        }
        PackedFingerprint versionedKey =
            fingerprintValueService.fingerprint(
                frontierVersion.concat(keyBytes.getObject().toByteArray()));
        byte[] entryBytes = bytesOut.toByteArray();
        writeStatuses.addWriteStatus(fingerprintValueService.put(versionedKey, entryBytes));

        // IMPORTANT: when this completes, no more write statuses can be added.
        writeStatuses.selectedEntryDone();
      }

      @Override
      public final void onFailure(Throwable t) {
        writeStatuses.notifyWriteFailure(t);
      }
    }
  }

  private static class WriteStatusesFuture extends QuiescingFuture<ImmutableList<Throwable>>
      implements FutureCallback<Void> {
    private final ConcurrentLinkedQueue<Throwable> errors = new ConcurrentLinkedQueue<>();

    private WriteStatusesFuture() {
      super(directExecutor());
    }

    private void selectedEntryStarting() {
      increment();
    }

    private void selectedEntryDone() {
      decrement();
    }

    private void notifyAllStarted() {
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
      return ImmutableList.copyOf(errors);
    }

    private void addWriteStatus(@Nullable ListenableFuture<Void> writeStatus) {
      if (writeStatus == null) {
        return;
      }
      increment();
      Futures.addCallback(writeStatus, (FutureCallback<Void>) this, directExecutor());
    }

    /**
     * Implementation of {@link FutureCallback<void>}.
     *
     * @deprecated only for use via {@link #addWriteStatus}
     */
    @Override
    @Deprecated // only called via addWriteStatus
    public void onSuccess(Void unused) {
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
  }

  private static boolean isExecutionValue(SkyKey key) {
    // TODO: b/439060530: consider whether this is correct for ActionTemplateExpansionValue keys.
    return !(key instanceof ActionLookupKey);
  }
}
