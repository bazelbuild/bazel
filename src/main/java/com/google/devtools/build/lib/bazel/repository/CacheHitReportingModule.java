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
package com.google.devtools.build.lib.bazel.repository;

import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCacheHitEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.repository.RepositoryFailedEvent;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.Pair;
import java.net.URL;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** Module reporting about cache hits in external repositories in case of failures */
public final class CacheHitReportingModule extends BlazeModule {
  private Reporter reporter;
  private Map<String, Set<Pair<String, URL>>> cacheHitsByRepo;

  @Override
  public void beforeCommand(CommandEnvironment env) {
    env.getEventBus().register(this);
    this.reporter = env.getReporter();
    this.cacheHitsByRepo = new HashMap<String, Set<Pair<String, URL>>>();
  }

  @Override
  public void afterCommand() {
    this.reporter = null;
    this.cacheHitsByRepo = null;
  }

  @Subscribe
  public synchronized void cacheHit(RepositoryCacheHitEvent event) {
    String repo = event.getRepo();
    if (cacheHitsByRepo.get(repo) == null) {
      cacheHitsByRepo.put(repo, new HashSet<Pair<String, URL>>());
    }
    cacheHitsByRepo.get(repo).add(Pair.of(event.getFileHash(), event.getUrl()));
  }

  @Subscribe
  public void failed(RepositoryFailedEvent event) {
    String repo = event.getRepo();
    Set<Pair<String, URL>> cacheHits = cacheHitsByRepo.get(repo);
    if (cacheHits != null && !cacheHits.isEmpty()) {
      StringBuilder info = new StringBuilder();

      info.append("Repository '")
          .append(repo)
          .append(
              "' used the following cache hits instead of downloading the corresponding file.\n");
      for (Pair<String, URL> hit : cacheHits) {
        info.append(" * Hash '")
            .append(hit.getFirst())
            .append("' for ")
            .append(hit.getSecond().toString())
            .append("\n");
      }
      info.append("If the definition of '")
          .append(repo)
          .append("' was updated, verify that the hashes were also updated.");
      reporter.handle(Event.info(info.toString()));
    }
  }
}
