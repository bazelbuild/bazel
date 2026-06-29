// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static java.util.concurrent.ForkJoinPool.commonPool;
import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueCache;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.Fingerprinter;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.KeyValueWriter;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievalResult;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * Implementation of remote analysis cache functionality that is needed for reading and writing.
 *
 * <p>The parts that are needed for maintenance of Skyframe, etc. are in {@link
 * RemoteAnalysisCacheManager}.
 */
public class RemoteAnalysisCacheDeps
    implements SerializationDependenciesProvider, RemoteAnalysisCacheReaderDepsProvider {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final long CLIENT_LOOKUP_TIMEOUT_SEC = 20L;

  private final RemoteAnalysisCacheMode mode;
  private final boolean bailOutOnMissingFingerprint;
  private final boolean minimizeMemory;
  private final String serializedFrontierProfile;
  private final Optional<Predicate<PackageIdentifier>> activeDirectoriesMatcher;
  private final RemoteAnalysisCachingEventListener listener;
  private final FrontierNodeVersion frontierNodeVersion;
  private final boolean skycacheAnalysisOnly;
  private final boolean emitUploadedEvents;

  private final ListenableFuture<ObjectCodecs> objectCodecs;
  @Nullable private final ListenableFuture<FingerprintValueService> fingerprintValueServiceFuture;
  @Nullable private final ListenableFuture<? extends RemoteAnalysisCacheClient> analysisCacheClient;
  @Nullable private final ListenableFuture<? extends RemoteAnalysisMetadataWriter> metadataWriter;

  // Volatile because double-checked locking is used in the getter
  @Nullable private volatile SkyValueRetriever skyValueRetriever;
  @Nullable private volatile SkycacheUploadClient skycacheUploadClient;
  @Nullable private volatile FileOpNodeMemoizingLookup fileOpNodes;

  private final InMemoryGraph graph;
  private final EventBus eventBus;
  private final LongVersionGetter versionGetter;

  private final AtomicBoolean bailedOut = new AtomicBoolean();
  private final ExtendedEventHandler eventHandler;

  public static RemoteAnalysisCacheDeps createDisabled() {
    return new RemoteAnalysisCacheDeps();
  }

  RemoteAnalysisCacheDeps(
      ExtendedEventHandler eventHandler,
      RemoteAnalysisCacheMode mode,
      boolean bailOutOnMissingFingerprint,
      boolean minimizeMemory,
      RemoteAnalysisCachingServicesSupplier servicesSupplier,
      RemoteAnalysisCachingEventListener listener,
      ListenableFuture<ObjectCodecs> objectCodecs,
      FrontierNodeVersion frontierNodeVersion,
      Optional<Predicate<PackageIdentifier>> activeDirectoriesMatcher,
      String serializedFrontierProfile,
      boolean skycacheAnalysisOnly,
      boolean emitUploadedEvents,
      Fingerprinter fingerprinterForAnalysisCaching,
      InMemoryGraph graph,
      EventBus eventBus,
      LongVersionGetter versionGetter) {
    this.mode = mode;
    this.bailOutOnMissingFingerprint = bailOutOnMissingFingerprint;
    this.skycacheAnalysisOnly = skycacheAnalysisOnly;
    this.emitUploadedEvents = emitUploadedEvents;
    this.minimizeMemory = minimizeMemory;
    this.serializedFrontierProfile = serializedFrontierProfile;
    this.activeDirectoriesMatcher = activeDirectoriesMatcher;
    this.eventHandler = eventHandler;

    this.objectCodecs = objectCodecs;
    this.listener = listener;

    this.frontierNodeVersion = frontierNodeVersion;

    this.fingerprintValueServiceFuture =
        servicesSupplier.getFingerprintValueStore() == null
            ? null
            : Futures.transform(
                servicesSupplier.getFingerprintValueStore(),
                store ->
                    new FingerprintValueService(
                        // This pool is surfaced via FingerprintValueService.getExecutor and is only
                        // used for pure deserialization CPU work.
                        //
                        // TODO: b/390533627 - consider if a different executor should be used for
                        // better isolation.
                        commonPool(),
                        store,
                        new FingerprintValueCache(FingerprintValueCache.SyncMode.NOT_LINKED),
                        fingerprinterForAnalysisCaching),
                directExecutor());
    this.metadataWriter = servicesSupplier.getMetadataWriter();
    this.analysisCacheClient = servicesSupplier.getAnalysisCacheClient();

    this.graph = graph;
    this.eventBus = eventBus;
    this.versionGetter = versionGetter;
  }

  private RemoteAnalysisCacheDeps() {
    this.mode = RemoteAnalysisCacheMode.OFF;
    this.bailOutOnMissingFingerprint = false;
    this.minimizeMemory = false;
    this.skycacheAnalysisOnly = false;
    this.emitUploadedEvents = false;
    this.serializedFrontierProfile = "";
    this.activeDirectoriesMatcher = Optional.empty();
    this.eventHandler = null;
    this.objectCodecs = null;
    this.listener = null;
    this.frontierNodeVersion = null;
    this.fingerprintValueServiceFuture = null;
    this.metadataWriter = null;
    this.analysisCacheClient = null;

    this.graph = null;
    this.eventBus = null;
    this.versionGetter = null;
  }

  static <T> T resolveWithTimeout(Future<? extends T> future, String what)
      throws InterruptedException {
    if (future == null) {
      return null;
    }
    try (SilentCloseable unused = Profiler.instance().profile("resolveWithTimeout: " + what)) {
      return future.get(CLIENT_LOOKUP_TIMEOUT_SEC, SECONDS);
    } catch (ExecutionException | TimeoutException e) {
      logger.atWarning().withCause(e).log("Unable to initialize %s", what);
      return null;
    }
  }

  private void checkEnabled() {
    Preconditions.checkState(
        mode != RemoteAnalysisCacheMode.OFF, "Remote analysis cache is disabled");
  }

  @Override
  public RemoteAnalysisCacheMode mode() {
    return mode;
  }

  @Override
  public boolean shouldMinimizeMemory() {
    checkEnabled();
    return minimizeMemory;
  }

  @Override
  public String getSerializedFrontierProfile() {
    checkEnabled();
    return serializedFrontierProfile;
  }

  @Override
  public Optional<Predicate<PackageIdentifier>> getActiveDirectoriesMatcher() {
    checkEnabled();
    return activeDirectoriesMatcher;
  }

  @Override
  public FrontierNodeVersion getSkyValueVersion() {
    checkEnabled();
    return frontierNodeVersion;
  }

  @Override
  public ObjectCodecs getObjectCodecs() throws InterruptedException {
    checkEnabled();
    try {
      return objectCodecs.get();
    } catch (ExecutionException e) {
      throw new IllegalStateException("Failed to initialize ObjectCodecs", e);
    }
  }

  @Nullable
  ListenableFuture<FingerprintValueService> getFingerprintValueServiceFuture() {
    return fingerprintValueServiceFuture;
  }

  @Override
  public FingerprintValueService getFingerprintValueService() throws InterruptedException {
    checkEnabled();
    return resolveWithTimeout(fingerprintValueServiceFuture, "fingerprint value service");
  }

  @Override
  public KeyValueWriter getFileInvalidationWriter() throws InterruptedException {
    checkEnabled();
    return getFingerprintValueService();
  }

  @Override
  @Nullable
  public RemoteAnalysisCacheClient getAnalysisCacheClient() throws InterruptedException {
    checkEnabled();
    return resolveWithTimeout(analysisCacheClient, "analysis cache client");
  }

  private FileOpNodeMemoizingLookup getFileOpNodes() throws InterruptedException {
    if (fileOpNodes != null) {
      return fileOpNodes;
    }
    synchronized (this) {
      if (fileOpNodes == null) {
        fileOpNodes =
            new FileOpNodeMemoizingLookup(
                getFingerprintValueService().getExecutor(),
                graph,
                ImmutableSet.of(),
                /* shouldDiscardMemory= */ false,
                /* referencedPackages= */ null);
      }
      return fileOpNodes;
    }
  }

  @Override
  public SkyValueRetriever getSkyValueRetriever() throws InterruptedException {
    checkEnabled();
    if (skyValueRetriever != null) {
      return skyValueRetriever;
    }

    synchronized (this) {
      if (skyValueRetriever == null) {
        skyValueRetriever =
            new SkyValueRetriever(
                getFingerprintValueService(),
                getObjectCodecs(),
                frontierNodeVersion,
                getFileOpNodes());
      }
      return skyValueRetriever;
    }
  }

  @Override
  public SkycacheUploadClient getSkycacheUploadClient() throws InterruptedException {
    checkEnabled();
    if (skycacheUploadClient != null) {
      return skycacheUploadClient;
    }

    synchronized (this) {
      if (skycacheUploadClient == null) {
        ObjectCodecs codecs = getObjectCodecs();
        FingerprintValueService fvs = getFingerprintValueService();
        skycacheUploadClient =
            new SkycacheUploadClient(
                fvs,
                codecs,
                frontierNodeVersion,
                graph,
                eventBus,
                versionGetter,
                emitUploadedEvents,
                getFileOpNodes());
      }
      return skycacheUploadClient;
    }
  }

  @Override
  public void waitForUploadCompletion() throws InterruptedException, ExecutionException {
    var client = skycacheUploadClient;
    if (client != null) {
      client.waitForCompletion();
    }
  }

  @Override
  @Nullable
  public RemoteAnalysisMetadataWriter getMetadataWriter() throws InterruptedException {
    checkEnabled();
    return resolveWithTimeout(metadataWriter, "metadata writer");
  }

  @Override
  public void recordRetrievalResult(RetrievalResult retrievalResult, SkyKey key) {
    checkEnabled();
    listener.recordRetrievalResult(retrievalResult, key);
  }

  @Override
  public void recordSerializationException(SerializationException e, SkyKey key) {
    checkEnabled();
    listener.recordSerializationException(e, key);
  }

  @Override
  public boolean shouldBailOutOnMissingFingerprint() {
    checkEnabled();
    if (!bailOutOnMissingFingerprint) {
      return false;
    }
    if (bailedOut.get()) {
      return true;
    }

    try {
      FingerprintValueService service = getFingerprintValueService();
      boolean retVal = service != null && service.getStats().entriesNotFound() > 0;
      if (retVal) {
        bailedOut.set(true);
        eventHandler.handle(
            Event.warn(
                "Skycache: falling back to local evaluation due to unexpected missing cache"
                    + " entries"));
        analysisCacheClient.get().bailOutDueToMissingFingerprint();
      }
      return retVal;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return false;
    } catch (ExecutionException e) {
      throw new IllegalStateException(
          "At this point the Skycache client should have been initialized", e);
    }
  }

  @Override
  public boolean getSkycacheAnalysisOnly() {
    checkEnabled();
    return skycacheAnalysisOnly;
  }

  @Override
  public boolean getEmitUploadedEvents() {
    checkEnabled();
    return emitUploadedEvents;
  }
}
