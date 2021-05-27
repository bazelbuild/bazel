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

import com.google.common.annotations.VisibleForTesting;
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
      return cache.computeIfAbsent(
          jdepsFile,
          file -> {
            try (InputStream input = actionExecutionContext.getInputPath(file).getInputStream()) {
              return Deps.Dependencies.parseFrom(input);
            } catch (IOException e) {
              throw new UncheckedIOException(e);
            }
          });
    } catch (UncheckedIOException e) {
      throw e.getCause();
    }
  }

  @VisibleForTesting
  void insertDependencies(Artifact jdepsFile, Deps.Dependencies dependencies) {
    cache.put(jdepsFile, dependencies);
  }
}
