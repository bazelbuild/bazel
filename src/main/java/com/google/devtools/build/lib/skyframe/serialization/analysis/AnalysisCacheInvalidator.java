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
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static java.lang.Math.min;
import static java.util.concurrent.TimeUnit.MICROSECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.common.math.IntMath;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.RemoteAnalysisCacheStatistics.InvalidationLookupMetrics;
import com.google.devtools.build.lib.concurrent.safeexecutor.RejectionHandlingRunnable;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.serialization.AsyncSerializationTask;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.skyframe.SkyKey;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Helper class for checking which keys should be invalidated using a remote analysis cache service.
 */
public final class AnalysisCacheInvalidator {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  // Same method used by java.util.stream.AbstractTask.suggestTargetSize.
  private static final int TARGET_WORK_UNITS = Runtime.getRuntime().availableProcessors() * 4;
  private static final long INVALIDATION_TIMEOUT_SECONDS = 10;

  private final RemoteAnalysisCacheClient analysisCacheClient;
  private final ObjectCodecs codecs;
  private final FingerprintValueService fingerprintService;
  private final ExtendedEventHandler eventHandler;
  private final RemoteAnalysisCachingEventListener eventListener;
  private final FrontierNodeVersion currentVersion;
  private final ClientId currentClientId;
  private final SafeExecutor executor;

  public AnalysisCacheInvalidator(
      RemoteAnalysisCacheClient analysisCacheClient,
      ObjectCodecs objectCodecs,
      FingerprintValueService fingerprintValueService,
      FrontierNodeVersion currentVersion,
      ClientId currentClientId,
      ExtendedEventHandler eventHandler,
      RemoteAnalysisCachingEventListener eventListener,
      SafeExecutor executor) {
    this.analysisCacheClient = checkNotNull(analysisCacheClient, "analysisCacheClient");
    this.codecs = checkNotNull(objectCodecs, "objectCodecs");
    this.fingerprintService = checkNotNull(fingerprintValueService, "fingerprintValueService");
    this.currentVersion = checkNotNull(currentVersion, "currentVersion");
    this.currentClientId = checkNotNull(currentClientId, "currentClientId");
    this.eventHandler = checkNotNull(eventHandler, "eventHandler");
    this.eventListener = checkNotNull(eventListener, "eventListener");
    this.executor = checkNotNull(executor, "executor");
  }

  /**
   * Looks up the given keys in the analysis cache service to determine which ones should be
   * invalidated.
   *
   * @param keysToLookupSupplier The supplier of set of SkyKeys to check.
   * @return The subset of keysToLookup that got a cache miss should be invalidated locally.
   */
  public ImmutableSet<SkyKey> lookupKeysToInvalidate(
      Supplier<ImmutableSet<SkyKey>> keysToLookupSupplier,
      RemoteAnalysisCachingServerState serverState)
      throws InterruptedException {
    var previousVersion = serverState.version();
    if (previousVersion == null) {
      // TODO: b/439857268 - it looks like this can happen if the previous build was interrupted,
      // but the exact way that leads to the previous version being unset is not entirely clear.
      logger.atWarning().log(
          "Skycache: no previous version was found during invalidation check. Invalidating"
              + " everything");
      return keysToLookupSupplier.get(); // invalidate everything
    }

    if (!previousVersion.equals(currentVersion)) {
      logger.atInfo().log(
          "Skycache: Version changed during invalidation check. Previous version: %s, current"
              + " version: %s.",
          previousVersion, currentVersion);
      return keysToLookupSupplier.get(); // everything must be invalidated
    }

    if (Objects.equals(currentClientId, serverState.clientId())) {
      // The current client state is the same as the previous client state, so
      // no invalidation is needed because all deserialized keys are still valid.
      return ImmutableSet.of();
    }

    ImmutableSet<SkyKey> keysToLookup = keysToLookupSupplier.get();

    if (keysToLookup.isEmpty()) {
      logger.atInfo().log("Skycache: No keys to lookup for invalidation check.");
      return ImmutableSet.of();
    }

    Stopwatch stopwatch = Stopwatch.createStarted();
    int numInvalidatedKeys = 0;
    InvalidationLookupMetrics.Status status = null;
    ImmutableSet<SkyKey> keysToInvalidate;
    try {
      List<ListenableFuture<Optional<SkyKey>>> futures =
          submitInvalidationLookupsBatched(keysToLookup);
      if (futures == null) {
        numInvalidatedKeys = keysToLookup.size();
        status = InvalidationLookupMetrics.Status.TIMED_OUT;
        return keysToLookup;
      }

      try (SilentCloseable unused = Profiler.instance().profile("waitInvalidationLookups")) {
        try {
          keysToInvalidate =
              Futures.allAsList(futures).get(INVALIDATION_TIMEOUT_SECONDS, SECONDS).stream()
                  // Flatten Optionals, keeping only non-empty ones (keys to invalidate)
                  .flatMap(Optional::stream)
                  .collect(toImmutableSet());
          status = InvalidationLookupMetrics.Status.OK;
          numInvalidatedKeys = keysToInvalidate.size();
        } catch (ExecutionException e) {
          status = InvalidationLookupMetrics.Status.ERROR;
          numInvalidatedKeys = keysToLookup.size();
          logger.atWarning().withCause(e).log(
              "Skycache: Error waiting for analysis cache responses during invalidation check."
                  + " Invalidating everything.");
          return keysToLookup;
        } catch (TimeoutException e) {
          status = InvalidationLookupMetrics.Status.TIMED_OUT;
          numInvalidatedKeys = keysToLookup.size();
          logger.atWarning().log(
              "Skycache: Timeout waiting for analysis cache responses during invalidation check."
                  + " Invalidating everything.");
          return keysToLookup;
        }
      }
    } finally {
      stopwatch.stop();
      if (status != null) {
        eventListener.setInvalidationLookupMetrics(
            InvalidationLookupMetrics.newBuilder()
                .setLatencyMicros(stopwatch.elapsed(MICROSECONDS))
                .setStatus(status)
                .setNumKeys(keysToLookup.size())
                .setNumInvalidatedKeys(numInvalidatedKeys)
                .build());
      }
    }
    eventHandler.handle(
        Event.info(
            String.format(
                "Skycache: Invalidation lookup took %s. %s/%s keys will be invalidated.",
                stopwatch, keysToInvalidate.size(), keysToLookup.size())));
    return keysToInvalidate;
  }

  /**
   * Checks if the given node should be invalidated by submitting the node's fingerprint to the
   * analysis cache.
   *
   * <p>Returns the node's SkyKey if the node should be invalidated (i.e. cache miss), otherwise
   * returns an empty Optional.
   *
   * <p>Note: only lookup SkyKeys that were deserialized! Sending a key that was never serialized
   * will result in a cache miss for every build.
   */
  private ListenableFuture<Optional<SkyKey>> submitInvalidationLookup(SkyKey key) {
    // 1. Serialize the key
    AsyncSerializationTask serializeKeyTask =
        codecs.serializeMemoizedAsync(fingerprintService, key, null);
    serializeKeyTask.run();

    // 2. Compute the fingerprint from the serialized blob
    ListenableFuture<PackedFingerprint> fingerprint =
        Futures.transform(
            serializeKeyTask,
            k -> fingerprintService.fingerprint(currentVersion.concat(k.getObject().toByteArray())),
            directExecutor());

    // 3. Submit the fingerprint to the analysis cache service
    ListenableFuture<LookupResult> responseFuture =
        Futures.transformAsync(
            fingerprint, f -> analysisCacheClient.lookup(f.toBytes()), directExecutor());

    // 4. Transform result to return keys that should be invalidated (i.e.
    // empty response, cache miss)
    return Futures.transform(
        responseFuture,
        response -> (response.value().length == 0) ? Optional.of(key) : Optional.empty(),
        directExecutor());
  }

  /**
   * Dispatches {@code keysToLookup} in batched work units on {@link #executor}.
   *
   * <p>This method approximates {@code parallelStream} which cannot be used here because that uses
   * {@code ForkJoinPool.commonPool}. Using a custom executor rather than {@code commonPool}
   * provides a way to scope the lifetime of threads, i.e., so they don't cross over into the next
   * build.
   */
  @Nullable // returns null if there was a timeout
  private List<ListenableFuture<Optional<SkyKey>>> submitInvalidationLookupsBatched(
      ImmutableSet<SkyKey> keysToLookup) throws InterruptedException {
    int totalKeys = keysToLookup.size();
    @SuppressWarnings("unchecked") // exactly how other generic containers are implemented
    var futures =
        (List<ListenableFuture<Optional<SkyKey>>>) (List<?>) Arrays.asList(new Object[totalKeys]);
    try (SilentCloseable unused = Profiler.instance().profile("submitInvalidationLookups")) {
      ImmutableList<SkyKey> keysToLookupList = keysToLookup.asList();
      int batchSize = totalKeys / TARGET_WORK_UNITS;
      if (batchSize < 1) {
        batchSize = 1;
      }
      int batches = IntMath.divide(totalKeys, batchSize, RoundingMode.CEILING);
      var allSet = new CountDownLatch(batches);
      for (int start = 0; start < totalKeys; start += batchSize) {
        final int begin = start;
        final int limit = min(start + batchSize, totalKeys);

        executor.execute(
            new RejectionHandlingRunnable() {
              @Override
              public void run() {
                try {
                  for (int i = begin; i < limit; i++) {
                    futures.set(i, submitInvalidationLookup(keysToLookupList.get(i)));
                  }
                } finally {
                  allSet.countDown();
                }
              }

              @Override
              public void handleRejection(Throwable t) {
                allSet.countDown();
              }
            });
      }
      if (!allSet.await(INVALIDATION_TIMEOUT_SECONDS, SECONDS)) {
        logger.atWarning().log(
            "Skycache: Timeout waiting to submit (%d) analysis cache invalidation requests."
                + " Invalidating everything.",
            totalKeys);
        return null;
      }
      return futures;
    }
  }
}
