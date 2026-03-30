// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel;

import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.bazel.repository.cache.DownloadCacheHitEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.repository.RepositoryFailedEvent;
import com.google.devtools.build.lib.repository.RepositoryFetchProgress;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.Pair;
import java.net.URI;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** Module reporting about cache hits in external repositories in case of failures */
public final class CacheHitReportingModule extends BlazeModule {
  private Reporter reporter;
  private ConcurrentHashMap<String, Set<Pair<String, URI>>> cacheHitsByContext;

  @Override
  public void beforeCommand(CommandEnvironment env) {
    env.getEventBus().register(this);
    this.reporter = env.getReporter();
    this.cacheHitsByContext = new ConcurrentHashMap<>();
  }

  @Override
  public void afterCommand() {
    this.reporter = null;
    this.cacheHitsByContext = null;
  }

  @Subscribe
  @AllowConcurrentEvents
  public void cacheHit(DownloadCacheHitEvent event) {
    cacheHitsByContext
        .computeIfAbsent(event.context(), k -> ConcurrentHashMap.newKeySet())
        .add(Pair.of(event.fileHash(), event.uri()));
  }

  @Subscribe
  public void failed(RepositoryFailedEvent event) {
    // TODO(wyv): add an event for the failure of a module extension too
    String context = RepositoryFetchProgress.repositoryFetchContextString(event.getRepo());
    Set<Pair<String, URI>> cacheHits = cacheHitsByContext.get(context);
    if (cacheHits != null && !cacheHits.isEmpty()) {
      StringBuilder info = new StringBuilder();

      info.append(context)
          .append(
              "' used the following cache hits instead of downloading the corresponding file.\n");
      for (Pair<String, URI> hit : cacheHits) {
        info.append(" * Hash '")
            .append(hit.getFirst())
            .append("' for ")
            .append(hit.getSecond().toString())
            .append("\n");
      }
      info.append("If the definition of '")
          .append(context)
          .append("' was updated, verify that the hashes were also updated.");
      reporter.handle(Event.info(info.toString()));
    }
  }
}
