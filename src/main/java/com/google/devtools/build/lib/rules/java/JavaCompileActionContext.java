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

package com.google.devtools.build.lib.rules.java;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.view.proto.Deps;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** Context for compiling Java files. */
public class JavaCompileActionContext implements ActionContext {

  // TODO(djasper): Investigate caching across builds.
  private final ConcurrentHashMap<Artifact, Deps.Dependencies> cache = new ConcurrentHashMap<>();

  @SuppressWarnings("AllowVirtualThreads")
  private final ExecutorService executor =
      Executors.newThreadPerTaskExecutor(Thread.ofVirtual().name("jdeps-reader-", 0).factory());

  void addDependencies(
      ImmutableList<Artifact> jdepsFiles,
      ActionExecutionContext actionExecutionContext,
      Set<String> deps)
      throws IOException, InterruptedException {
    List<Future<Deps.Dependencies>> uncached = new ArrayList<>();
    for (Artifact jdepsFile : jdepsFiles) {
      // Reading a jdeps file is potentially expensive, e.g. when we have to download an input with
      // actionFS, so we use an ExecutorService instead of computeIfAbsent here. The downside is
      // that potentially we parse the same jdeps file twice, but at least we are not blocking all
      // other threads on the lock for the cache.
      Deps.Dependencies cachedDeps = cache.get(jdepsFile);
      if (cachedDeps == null) {
        uncached.add(
            executor.submit(() -> readAndCacheJdepsFile(jdepsFile, actionExecutionContext)));
      } else {
        for (Deps.Dependency dep : cachedDeps.getDependencyList()) {
          deps.add(dep.getPath());
        }
      }
    }

    for (Future<Deps.Dependencies> future : uncached) {
      Deps.Dependencies result;
      try {
        result = future.get();
      } catch (ExecutionException e) {
        Throwables.throwIfInstanceOf(e.getCause(), IOException.class);
        throw new IllegalStateException(e);
      }

      for (Deps.Dependency dep : result.getDependencyList()) {
        deps.add(dep.getPath());
      }
    }
  }

  private Deps.Dependencies readAndCacheJdepsFile(
      Artifact jdepsFile, ActionExecutionContext actionExecutionContext) throws IOException {
    Deps.Dependencies deps;
    try (InputStream input = actionExecutionContext.getInputPath(jdepsFile).getInputStream()) {
      deps = Deps.Dependencies.parseFrom(input);
    }
    cache.putIfAbsent(jdepsFile, deps);
    return deps;
  }

  void insertDependencies(Artifact jdepsFile, Deps.Dependencies dependencies) {
    cache.put(jdepsFile, dependencies);
  }
}
