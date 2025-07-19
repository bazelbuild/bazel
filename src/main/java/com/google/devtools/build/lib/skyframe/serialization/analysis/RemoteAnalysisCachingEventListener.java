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

import com.google.common.collect.HashMultiset;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multiset;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.NoCachedData;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.Restart;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievalResult;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievedValue;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
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

  private final Set<SkyKey> serializedKeys = ConcurrentHashMap.newKeySet();
  private final Set<SkyKey> cacheHits = ConcurrentHashMap.newKeySet();
  private final Set<SkyKey> cacheMisses = ConcurrentHashMap.newKeySet();
  private final Set<SerializationException> serializationExceptions = ConcurrentHashMap.newKeySet();
  private final ConcurrentHashMap<SkyFunctionName, AtomicInteger> hitsBySkyFunctionName =
      new ConcurrentHashMap<>();
  private final ConcurrentHashMap<SkyFunctionName, AtomicInteger> missesBySkyFunctionName =
      new ConcurrentHashMap<>();

  private final AtomicReference<FrontierNodeVersion> skyValueVersion = new AtomicReference<>();

  @Nullable private FingerprintValueStore.Stats fingerprintValueStoreStats;
  @Nullable private RemoteAnalysisCacheClient.Stats remoteAnalysisCacheStats;

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

  public void recordServiceStats(
      @Nullable FingerprintValueStore.Stats fvsStats,
      @Nullable RemoteAnalysisCacheClient.Stats raccStats) {
    fingerprintValueStoreStats = fvsStats;
    remoteAnalysisCacheStats = raccStats;
  }

  public FingerprintValueStore.Stats getFingerprintValueStoreStats() {
    return fingerprintValueStoreStats;
  }

  public RemoteAnalysisCacheClient.Stats getRemoteAnalysisCacheStats() {
    return remoteAnalysisCacheStats;
  }

  @ThreadSafe
  public void recordRetrievalResult(RetrievalResult result, SkyKey key) {
    if (result instanceof Restart) {
      return;
    }

    switch (result) {
      case RetrievedValue unusedValue -> {
        if (!cacheHits.add(key)) {
          return;
        }
        hitsBySkyFunctionName
            .computeIfAbsent(key.functionName(), k -> new AtomicInteger())
            .incrementAndGet();
      }
      case NoCachedData unusedNoCachedData -> {
        if (!cacheMisses.add(key)) {
          return;
        }
        missesBySkyFunctionName
            .computeIfAbsent(key.functionName(), k -> new AtomicInteger())
            .incrementAndGet();
      }
      case Restart unusedRestart ->
          throw new IllegalStateException(
              "should have returned earlier"); // restart counts are not useful (yet).
    }
  }

  /** Returns the number of cache hits grouped by SkyFunction name. */
  public ImmutableMap<SkyFunctionName, AtomicInteger> getHitsBySkyFunctionName() {
    return ImmutableMap.copyOf(hitsBySkyFunctionName);
  }

  /** Returns the number of cache misses grouped by SkyFunction name. */
  public ImmutableMap<SkyFunctionName, AtomicInteger> getMissesBySkyFunctionName() {
    return ImmutableMap.copyOf(missesBySkyFunctionName);
  }

  /** Records a {@link SerializationException} encountered during SkyValue retrievals. */
  public void recordSerializationException(SerializationException e) {
    serializationExceptions.add(e);
  }

  /**
   * Returns the number of {@link SerializationException}s that were thrown during this invocation.
   */
  public int getSerializationExceptionCounts() {
    return serializationExceptions.size();
  }

  public void recordSkyValueVersion(FrontierNodeVersion version) {
    this.skyValueVersion.set(version);
  }

  /**
   * Returns the {@link RemoteAnalysisCachingState} for this build with information about all the
   * cache hits.
   */
  public RemoteAnalysisCachingState getLatestBuildState() {
    return new RemoteAnalysisCachingState(skyValueVersion.get(), ImmutableSet.copyOf(cacheHits));
  }
}
