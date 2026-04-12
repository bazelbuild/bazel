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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.serialization.SkycacheMetadataParams;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCacheClient.LookupTopLevelTargetsResult;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingOptions.RemoteAnalysisCacheMode;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Collection;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeoutException;
import java.util.function.Predicate;
import java.util.function.Supplier;

/** A collection of dependencies and minor bits of functionality for remote analysis caching. */
// Non-final for mockability
public class RemoteAnalysisCacheManager implements RemoteAnalysisCachingDependenciesProvider {
  private final RemoteAnalysisCacheMode mode;

  private final Future<? extends RemoteAnalysisCacheClient> analysisCacheClient;
  private final Future<? extends AnalysisCacheInvalidator> analysisCacheInvalidator;

  private final Collection<Label> topLevelTargets;
  private final Optional<Predicate<PackageIdentifier>> activeDirectoriesMatcher;

  private final ExtendedEventHandler eventHandler;

  private final boolean areMetadataQueriesEnabled;
  private final SkycacheMetadataParams skycacheMetadataParams;

  private boolean bailedOut;

  private final boolean minimizeMemory;

  /**
   * A collection of various parts of this class that various parts of Bazel (cache reading, cache
   * writing, in-memory bookkeeping) need.
   */
  public record AnalysisDeps(
      RemoteAnalysisCachingDependenciesProvider deps,
      RemoteAnalysisCacheReaderDepsProvider readerDeps,
      SerializationDependenciesProvider serializationDeps) {}

  public static RemoteAnalysisCacheManager createDisabled() {
    return new RemoteAnalysisCacheManager();
  }

  private RemoteAnalysisCacheManager() {
    this.mode = RemoteAnalysisCacheMode.OFF;
    this.analysisCacheClient = null;
    this.analysisCacheInvalidator = null;
    this.topLevelTargets = ImmutableList.of();
    this.activeDirectoriesMatcher = Optional.empty();
    this.minimizeMemory = false;
    this.eventHandler = null;
    this.skycacheMetadataParams = null;
    this.areMetadataQueriesEnabled = false;
  }

  RemoteAnalysisCacheManager(
      RemoteAnalysisCacheMode mode,
      boolean areMetadataQueriesEnabled,
      ExtendedEventHandler eventHandler,
      SkycacheMetadataParams skycacheMetadataParams,
      Future<? extends RemoteAnalysisCacheClient> analysisCacheClient,
      Future<? extends AnalysisCacheInvalidator> analysisCacheInvalidator,
      Collection<Label> topLevelTargets,
      Optional<Predicate<PackageIdentifier>> activeDirectoriesMatcher,
      boolean minimizeMemory) {
    this.mode = mode;
    this.analysisCacheClient = analysisCacheClient;
    this.analysisCacheInvalidator = analysisCacheInvalidator;
    this.topLevelTargets = topLevelTargets;
    this.activeDirectoriesMatcher = activeDirectoriesMatcher;
    this.minimizeMemory = minimizeMemory;
    this.eventHandler = eventHandler;
    this.skycacheMetadataParams = skycacheMetadataParams;
    this.areMetadataQueriesEnabled = areMetadataQueriesEnabled;
  }

  @Override
  public RemoteAnalysisCacheMode mode() {
    return mode;
  }

  @Override
  public void queryMetadataAndMaybeBailout() throws InterruptedException {
    Preconditions.checkState(mode == RemoteAnalysisCacheMode.DOWNLOAD);
    if (!areMetadataQueriesEnabled) {
      return;
    }
    if (skycacheMetadataParams.getTargets().isEmpty()) {
      eventHandler.handle(
          Event.warn("Skycache: Not querying Skycache metadata because invocation has no targets"));
    } else {
      try {
        LookupTopLevelTargetsResult result =
            RemoteAnalysisCacheDeps.resolveWithTimeout(analysisCacheClient, "analysis cache client")
                .lookupTopLevelTargets(
                    skycacheMetadataParams.getEvaluatingVersion(),
                    skycacheMetadataParams.getConfigurationHash(),
                    skycacheMetadataParams.getUseFakeStampData(),
                    skycacheMetadataParams.getBazelVersion());

        Event event =
            switch (result.status()) {
              case MATCH_STATUS_MATCH -> Event.info("Skycache: " + result.statusMessage());
              case MATCH_STATUS_FAILURE -> Event.warn("Skycache: " + result.statusMessage());
              default -> {
                bailedOut = true;
                yield Event.warn("Skycache: " + result.statusMessage());
              }
            };
        eventHandler.handle(event);
      } catch (ExecutionException | TimeoutException e) {
        eventHandler.handle(Event.warn("Skycache: Error with metadata store: " + e.getMessage()));
      }
    }
  }

  private void checkEnabled() {
    Preconditions.checkState(
        mode != RemoteAnalysisCacheMode.OFF, "Remote analysis cache is disabled");
  }

  @Override
  public Set<SkyKey> lookupKeysToInvalidate(
      Supplier<ImmutableSet<SkyKey>> keysToLookupSupplier,
      RemoteAnalysisCachingServerState remoteAnalysisCachingState)
      throws InterruptedException {
    checkEnabled();
    AnalysisCacheInvalidator invalidator =
        RemoteAnalysisCacheDeps.resolveWithTimeout(
            analysisCacheInvalidator, "analysis cache invalidator");
    if (invalidator == null) {
      // We need to know which keys to invalidate but we don't have an invalidator, presumably
      // because the backend services couldn't be contacted. Play if safe and invalidate every
      // value retrieved from the remote cache.
      return keysToLookupSupplier.get();
    }
    return invalidator.lookupKeysToInvalidate(keysToLookupSupplier, remoteAnalysisCachingState);
  }

  @Override
  public boolean bailedOut() {
    checkEnabled();
    return bailedOut;
  }

  @Override
  public void computeSelectionAndMinimizeMemory(InMemoryGraph graph) {
    checkEnabled();
    FrontierSerializer.computeSelectionAndMinimizeMemory(
        graph, topLevelTargets, activeDirectoriesMatcher);
  }

  @Override
  public boolean shouldMinimizeMemory() {
    checkEnabled();
    return minimizeMemory;
  }

}
