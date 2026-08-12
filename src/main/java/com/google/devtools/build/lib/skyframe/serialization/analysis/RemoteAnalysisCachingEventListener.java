// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multiset;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.RemoteAnalysisCacheStatistics.InvalidationLookupMetrics;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.NoCachedData;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.Restart;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievalPhase;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievalResult;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievedValue;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingServicesSupplier.Peer;
import com.google.devtools.build.lib.skyframe.serialization.analysis.proto.MissReason;
import com.google.devtools.build.lib.util.DecimalBucketer;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/** An {@link com.google.common.eventbus.EventBus} listener for remote analysis caching events. */
@ThreadSafety.ThreadSafe
public class RemoteAnalysisCachingEventListener {

  /**
   * An event for when a Skyframe node has been serialized, but its associated write futures (i.e.
   * RPC latency) may not be done yet.
   */
  public record SerializedNodeEvent(SkyKey key) {
    public SerializedNodeEvent {
      checkNotNull(key);
    }
  }

  /** A SkyFunction/retrieval phase pair for logging. */
  public record FunctionAndPhase(SkyFunctionName functionName, RetrievalPhase phase) {
    public FunctionAndPhase {
      checkNotNull(functionName);
      checkNotNull(phase);
    }
  }

  private final Set<SkyKey> serializedKeys = ConcurrentHashMap.newKeySet();
  private final Set<SkyKey> cacheHits = ConcurrentHashMap.newKeySet();
  private final Set<SkyKey> cacheMisses = ConcurrentHashMap.newKeySet();
  private final ConcurrentHashMap<Peer, AtomicLong> peers = new ConcurrentHashMap<>();
  private final Set<SerializationException> serializationExceptions = ConcurrentHashMap.newKeySet();
  private final ConcurrentHashMap<SkyFunctionName, AtomicLong> hitsBySkyFunctionName =
      new ConcurrentHashMap<>();
  private final ConcurrentHashMap<SkyFunctionName, AtomicLong> missesBySkyFunctionName =
      new ConcurrentHashMap<>();
  private final ConcurrentHashMap<FunctionAndPhase, DecimalBucketer> hitLatenciesBySkyFunctionName =
      new ConcurrentHashMap<>();
  private final ConcurrentHashMap<FunctionAndPhase, DecimalBucketer>
      missLatenciesBySkyFunctionName = new ConcurrentHashMap<>();

  private final ConcurrentHashMap<MissReason, AtomicLong> missesByReason =
      new ConcurrentHashMap<>();
  private final AtomicReference<InvalidationLookupMetrics> invalidationLookupMetrics =
      new AtomicReference<>();

  private final AtomicReference<FrontierNodeVersion> skyValueVersion = new AtomicReference<>();

  private FingerprintValueStore.Stats fingerprintValueStoreStats =
      FingerprintValueStore.EMPTY_STATS;
  private RemoteAnalysisCacheClient.Stats remoteAnalysisCacheStats =
      RemoteAnalysisCacheClient.EMPTY_STATS;

  @Nullable private ClientId clientId;

  @Subscribe
  @AllowConcurrentEvents
  @SuppressWarnings("unused")
  public void onSerializationComplete(SerializedNodeEvent event) {
    serializedKeys.add(event.key());
  }

  /** Returns the counts of {@link SkyFunctionName} from serialized nodes of this invocation. */
  public Multiset<SkyFunctionName> getSkyfunctionCounts() {
    Multiset<SkyFunctionName> counts = HashMultiset.create();
    serializedKeys.forEach(key -> counts.add(key.functionName()));
    return counts;
  }

  /** Returns the count of serialized nodes of this invocation. */
  public int getSerializedKeysCount() {
    return serializedKeys.size();
  }

  public Set<SkyKey> getSerializedKeys() {
    return ImmutableSet.copyOf(serializedKeys);
  }

  public Set<SkyKey> getCacheHits() {
    return ImmutableSet.copyOf(cacheHits);
  }

  public Set<SkyKey> getCacheMisses() {
    return ImmutableSet.copyOf(cacheMisses);
  }

  public void recordPeers(Map<Peer, AtomicLong> newPeers) {
    if (newPeers != null) {
      newPeers.forEach(
          (peer, count) ->
              this.peers.computeIfAbsent(peer, k -> new AtomicLong()).addAndGet(count.get()));
    }
  }

  public Map<Peer, AtomicLong> getPeers() {
    return ImmutableMap.copyOf(peers);
  }

  public void recordServiceStats(
      FingerprintValueStore.Stats fvsStats, RemoteAnalysisCacheClient.Stats raccStats) {
    fingerprintValueStoreStats = checkNotNull(fvsStats);
    remoteAnalysisCacheStats = checkNotNull(raccStats);
  }

  public FingerprintValueStore.Stats getFingerprintValueStoreStats() {
    return fingerprintValueStoreStats;
  }

  public RemoteAnalysisCacheClient.Stats getRemoteAnalysisCacheStats() {
    return remoteAnalysisCacheStats;
  }

  @ThreadSafe
  public void recordRetrievalResult(
      RetrievalResult result, SkyKey key, ImmutableMap<RetrievalPhase, Long> phaseDurationMicros) {
    switch (result) {
      case RetrievedValue unusedValue -> {
        if (!cacheHits.add(key)) {
          return;
        }
        hitsBySkyFunctionName
            .computeIfAbsent(key.functionName(), k -> new AtomicLong())
            .incrementAndGet();
        recordLatencies(hitLatenciesBySkyFunctionName, key.functionName(), phaseDurationMicros);
      }
      case NoCachedData(MissReason reason) -> recordCacheMiss(key, reason, phaseDurationMicros);
      case Restart.RESTART -> {}
    }
  }

  @VisibleForTesting
  public void recordRetrievalResult(RetrievalResult result, SkyKey key, long elapsedTimeMicros) {
    recordRetrievalResult(result, key, ImmutableMap.of(RetrievalPhase.TOTAL, elapsedTimeMicros));
  }

  private static void recordLatencies(
      ConcurrentHashMap<FunctionAndPhase, DecimalBucketer> latenciesMap,
      SkyFunctionName functionName,
      ImmutableMap<RetrievalPhase, Long> phaseDurationMicros) {
    phaseDurationMicros.forEach(
        (phase, micros) ->
            latenciesMap
                .computeIfAbsent(
                    new FunctionAndPhase(functionName, phase), k -> new DecimalBucketer())
                .add(micros));
  }

  /** Returns the number of cache hits grouped by SkyFunction name. */
  public ImmutableMap<SkyFunctionName, AtomicLong> getHitsBySkyFunctionName() {
    return ImmutableMap.copyOf(hitsBySkyFunctionName);
  }

  /** Returns the number of cache misses grouped by SkyFunction name. */
  public ImmutableMap<SkyFunctionName, AtomicLong> getMissesBySkyFunctionName() {
    return ImmutableMap.copyOf(missesBySkyFunctionName);
  }

  /** Returns the latency distribution of cache hits grouped by SkyFunction name and Phase. */
  public ImmutableMap<FunctionAndPhase, DecimalBucketer> getHitLatenciesBySkyFunctionName() {
    return ImmutableMap.copyOf(hitLatenciesBySkyFunctionName);
  }

  /** Returns the latency distribution of cache misses grouped by SkyFunction name and Phase. */
  public ImmutableMap<FunctionAndPhase, DecimalBucketer> getMissLatenciesBySkyFunctionName() {
    return ImmutableMap.copyOf(missLatenciesBySkyFunctionName);
  }

  public ImmutableMap<MissReason, AtomicLong> getMissesByReason() {
    return ImmutableMap.copyOf(missesByReason);
  }

  /** Records a {@link SerializationException} encountered during SkyValue retrievals. */
  public void recordSerializationException(
      SerializationException e,
      SkyKey key,
      ImmutableMap<RetrievalPhase, Long> phaseDurationMicros) {
    serializationExceptions.add(e);
    recordCacheMiss(key, e.getReason(), phaseDurationMicros);
  }

  @VisibleForTesting
  /**
   * Returns the number of {@link SerializationException}s that were thrown during this invocation.
   */
  public int getSerializationExceptionCounts() {
    return serializationExceptions.size();
  }

  public void recordSkyValueVersion(FrontierNodeVersion version) {
    this.skyValueVersion.set(version);
  }

  public FrontierNodeVersion getSkyValueVersion() {
    return skyValueVersion.get();
  }

  public void setClientId(ClientId clientId) {
    this.clientId = clientId;
  }

  public ClientId getClientId() {
    return clientId;
  }

  public void setInvalidationLookupMetrics(InvalidationLookupMetrics invalidationLookupMetrics) {
    this.invalidationLookupMetrics.set(invalidationLookupMetrics);
  }

  public InvalidationLookupMetrics getInvalidationLookupMetrics() {
    return invalidationLookupMetrics.get();
  }

  private void recordCacheMiss(
      SkyKey key, MissReason reason, ImmutableMap<RetrievalPhase, Long> phaseDurationMicros) {
    if (reason == MissReason.MISS_REASON_NOT_ATTEMPTED) {
      // Not actually a cache miss
      return;
    }

    if (!cacheMisses.add(key)) {
      return;
    }
    missesBySkyFunctionName
        .computeIfAbsent(key.functionName(), k -> new AtomicLong())
        .incrementAndGet();
    recordLatencies(missLatenciesBySkyFunctionName, key.functionName(), phaseDurationMicros);

    missesByReason.computeIfAbsent(reason, r -> new AtomicLong()).incrementAndGet();
  }
}
