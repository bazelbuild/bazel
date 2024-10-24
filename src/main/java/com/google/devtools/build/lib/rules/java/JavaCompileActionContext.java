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

import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.view.proto.Deps;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.concurrent.ConcurrentHashMap;

/** Context for compiling Java files. */
public class JavaCompileActionContext implements ActionContext {
  private final ConcurrentHashMap<Artifact, Deps.Dependencies> cache = new ConcurrentHashMap<>();

  Deps.Dependencies getDependencies(
      Artifact jdepsFile, ActionExecutionContext actionExecutionContext) throws IOException {
    // TODO(djasper): Investigate caching across builds.
    try {
      // The cache value computation is potentially expensive, e.g. when we have
      // to download an input with actionFS, so we are explicitly not using
      // computeIfAbsent here.
      // The downside is that potentially we parse the same jdepsFile twice, but
      // at least we are not blocking all other threads on the lock for the
      // cache.
      Deps.Dependencies deps = cache.get(jdepsFile);
      if (deps != null) {
        return deps;
      }
      try (InputStream input = actionExecutionContext.getInputPath(jdepsFile).getInputStream()) {
        deps = Deps.Dependencies.parseFrom(input);
        cache.putIfAbsent(jdepsFile, deps);
        return deps;
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    } catch (UncheckedIOException e) {
      throw e.getCause();
    }
  }

  void insertDependencies(Artifact jdepsFile, Deps.Dependencies dependencies) {
    cache.put(jdepsFile, dependencies);
  }
}
