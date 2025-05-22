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
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SkyKeySerializationHelper;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.FrontierNodeVersion;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.protobuf.ByteString;
import java.util.Optional;
import java.util.concurrent.ExecutionException;

/**
 * Helper class for checking which keys should be invalidated using a remote analysis cache service.
 */
public final class AnalysisCacheInvalidator {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final RemoteAnalysisCacheClient analysisCacheClient;
  private final ObjectCodecs codecs;
  private final FingerprintValueService fingerprintService;
  private final ExtendedEventHandler eventHandler;
  private final FrontierNodeVersion currentVersion;

  public AnalysisCacheInvalidator(
      RemoteAnalysisCacheClient analysisCacheClient,
      ObjectCodecs objectCodecs,
      FingerprintValueService fingerprintValueService,
      FrontierNodeVersion currentVersion,
      ExtendedEventHandler eventHandler) {
    this.analysisCacheClient = checkNotNull(analysisCacheClient, "analysisCacheClient");
    this.codecs = checkNotNull(objectCodecs, "objectCodecs");
    this.fingerprintService = checkNotNull(fingerprintValueService, "fingerprintValueService");
    this.currentVersion = checkNotNull(currentVersion, "currentVersion");
    this.eventHandler = checkNotNull(eventHandler, "eventHandler");
  }

  /**
   * Looks up the given keys in the analysis cache service to determine which ones should be
   * invalidated.
   *
   * @param keysToLookup The set of SkyKeys to check.
   * @return The subset of keysToLookup that got a cache miss should be invalidated locally.
   */
  public ImmutableSet<SkyKey> lookupKeysToInvalidate(
      RemoteAnalysisCachingState remoteAnalysisCachingState) {
    if (remoteAnalysisCachingState.deserializedKeys().isEmpty()) {
      logger.atInfo().log("Skycache: No keys to lookup for invalidation check.");
      return ImmutableSet.of();
    }

    var previousVersion = remoteAnalysisCachingState.version();
    checkState(previousVersion != null, "Version is null, but there are keys to lookup.");

    if (!previousVersion.equals(currentVersion)) {
      logger.atInfo().log(
          "Skycache: Version changed during invalidation check. Previous version: %s, current"
              + " version: %s.",
          previousVersion, currentVersion);
      return remoteAnalysisCachingState.deserializedKeys(); // everything must be invalidated
    }

    Stopwatch stopwatch = Stopwatch.createStarted();

    ImmutableList<ListenableFuture<Optional<SkyKey>>> futures;
    try (SilentCloseable unused = Profiler.instance().profile("submitInvalidationLookups")) {
      futures =
          remoteAnalysisCachingState.deserializedKeys().parallelStream()
              .map(this::submitInvalidationLookup)
              .collect(toImmutableList());
    }

    try (SilentCloseable unused = Profiler.instance().profile("waitInvalidationLookups")) {
      ImmutableSet<SkyKey> keysToInvalidate;
      try {
        keysToInvalidate =
            Futures.allAsList(futures).get().stream()
                // Flatten Optionals, keeping only non-empty ones (keys to invalidate)
                .flatMap(Optional::stream)
                .collect(toImmutableSet());
      } catch (ExecutionException e) {
        throw new IllegalStateException(
            "Skycache: Error waiting for analysis cache responses during invalidation check.", e);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        logger.atWarning().log("Skycache: Interrupted while waiting for analysis cache responses.");
        throw new IllegalStateException(
            "Skycache: Interrupted while waiting for analysis cache responses", e);
      }
      stopwatch.stop();
      eventHandler.handle(
          Event.info(
              String.format(
                  "Remote analysis caching service lookup took %s. %s/%s keys will be"
                      + " invalidated.",
                  stopwatch, keysToInvalidate.size(), futures.size())));
      return keysToInvalidate;
    }
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
    try {
      // 1. Compute the fingerprint for the versioned key
      PackedFingerprint cacheKey =
          SkyKeySerializationHelper.computeFingerprint(
              codecs, fingerprintService, key, currentVersion);

      // 2. Submit the fingerprint to the analysis cache service
      ListenableFuture<ByteString> responseFuture =
          analysisCacheClient.lookup(ByteString.copyFrom(cacheKey.toBytes()));

      // 3. Transform result to return keys that should be invalidated (i.e.
      // empty response, cache miss)
      return Futures.transform(
          responseFuture,
          response -> response.isEmpty() ? Optional.of(key) : Optional.empty(),
          directExecutor());
    } catch (SerializationException e) {
      // Wrap serialization errors in a failed future
      logger.atWarning().withCause(e).log("Skycache: Failed to serialize key: %s", key);
      return immediateFailedFuture(
          new IllegalStateException("Skycache: Failed to serialize key: " + key, e));
    }
  }
}
