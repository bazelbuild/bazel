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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.proto.Deps;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;

/** Context for compiling Java files. */
public class JavaCompileActionContext implements ActionContext {

  // TODO(djasper): Investigate caching across builds.
  private final ConcurrentHashMap<Artifact, ImmutableList<PathFragment>> cache =
      new ConcurrentHashMap<>();

  private final ListeningExecutorService executor =
      MoreExecutors.listeningDecorator(
          Executors.newThreadPerTaskExecutor(
              Thread.ofVirtual().name("jdeps-reader-", 0).factory()));

  void addDependencies(
      ImmutableList<Artifact> jdepsFiles,
      ActionExecutionContext actionExecutionContext,
      Set<PathFragment> deps)
      throws IOException, InterruptedException {
    List<ListenableFuture<ImmutableList<PathFragment>>> uncached = new ArrayList<>();
    for (Artifact jdepsFile : jdepsFiles) {
      // Reading a jdeps file is potentially expensive, e.g. when we have to download an input with
      // actionFS, so we use an ExecutorService instead of computeIfAbsent here. The downside is
      // that potentially we parse the same jdeps file twice, but at least we are not blocking all
      // other threads on the lock for the cache.
      ImmutableList<PathFragment> cachedDeps = cache.get(jdepsFile);
      if (cachedDeps == null) {
        uncached.add(
            executor.submit(() -> readAndCacheJdepsFile(jdepsFile, actionExecutionContext)));
      } else {
        deps.addAll(cachedDeps);
      }
    }

    // Wait for every read to finish, even after one fails, so that the full set of lost jdeps files
    // can be discovered by RemoteActionFileSystem (if used).
    try {
      Futures.whenAllComplete(uncached).run(() -> {}, directExecutor()).get();
    } catch (ExecutionException e) {
      // whenAllComplete never completes exceptionally.
      throw new IllegalStateException(e);
    }

    IOException ioException = null;
    for (ListenableFuture<ImmutableList<PathFragment>> future : uncached) {
      ImmutableList<PathFragment> result;
      try {
        result = Futures.getDone(future);
      } catch (ExecutionException e) {
        if (e.getCause() instanceof IOException cause) {
          if (ioException == null) {
            ioException = cause;
          } else {
            ioException.addSuppressed(cause);
          }
          continue;
        }
        throw new IllegalStateException(e);
      }

      deps.addAll(result);
    }
    if (ioException != null) {
      throw ioException;
    }
  }

  private ImmutableList<PathFragment> readAndCacheJdepsFile(
      Artifact jdepsFile, ActionExecutionContext actionExecutionContext) throws IOException {
    Deps.Dependencies deps;
    try (InputStream input = actionExecutionContext.getInputPath(jdepsFile).getInputStream()) {
      deps = Deps.Dependencies.parseFrom(input);
    }
    ImmutableList<PathFragment> paths = toPaths(deps);
    cache.putIfAbsent(jdepsFile, paths);
    return paths;
  }

  void insertDependencies(Artifact jdepsFile, Deps.Dependencies dependencies) {
    cache.put(jdepsFile, toPaths(dependencies));
  }

  private static ImmutableList<PathFragment> toPaths(Deps.Dependencies dependencies) {
    return dependencies.getDependencyList().stream()
        .map(dep -> PathFragment.create(dep.getPath()))
        .collect(toImmutableList());
  }
}
