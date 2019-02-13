// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

class CachedSkylarkImportLookupValueAndDeps {
  private final SkylarkImportLookupValue value;
  // We store the deps separately so that we can let go of the value when the cache drops it.
  private final CachedSkylarkImportLookupFunctionDeps deps;

  private CachedSkylarkImportLookupValueAndDeps(
      SkylarkImportLookupValue value, CachedSkylarkImportLookupFunctionDeps deps) {
    this.value = value;
    this.deps = deps;
  }

  void traverse(
      DepGroupConsumer depGroupConsumer,
      Set<CachedSkylarkImportLookupFunctionDeps> visitedGlobalDeps)
      throws InterruptedException {
    deps.traverse(depGroupConsumer, visitedGlobalDeps);
  }

  SkylarkImportLookupValue getValue() {
    return value;
  }
  static CachedSkylarkImportLookupValueAndDeps.Builder newBuilder() {
    return new CachedSkylarkImportLookupValueAndDeps.Builder();
  }

  static class Builder {
    private final CachedSkylarkImportLookupFunctionDeps.Builder depsBuilder =
        CachedSkylarkImportLookupFunctionDeps.newBuilder();
    private SkylarkImportLookupValue value;

    @CanIgnoreReturnValue
    Builder addDep(SkyKey key) {
      depsBuilder.addDep(key);
      return this;
    }

    @CanIgnoreReturnValue
    Builder addDeps(Iterable<SkyKey> keys) {
      depsBuilder.addDeps(keys);
      return this;
    }

    @CanIgnoreReturnValue
    Builder noteException(Exception e) {
      depsBuilder.noteException(e);
      return this;
    }

    @CanIgnoreReturnValue
    Builder addTransitiveDeps(CachedSkylarkImportLookupValueAndDeps transitiveDeps) {
      depsBuilder.addTransitiveDeps(transitiveDeps.deps);
      return this;
    }

    @CanIgnoreReturnValue
    Builder setValue(SkylarkImportLookupValue value) {
      this.value = value;
      return this;
    }

    CachedSkylarkImportLookupValueAndDeps build() {
      Preconditions.checkNotNull(
          value, "Expected value to be set. directDeps: %s", depsBuilder.directDeps);
      return new CachedSkylarkImportLookupValueAndDeps(value, depsBuilder.build());
    }
  }

  static class CachedSkylarkImportLookupFunctionDeps {
    private final ImmutableList<Iterable<SkyKey>> directDeps;
    private final ImmutableList<CachedSkylarkImportLookupFunctionDeps> transitiveDeps;

    private CachedSkylarkImportLookupFunctionDeps(
        ImmutableList<Iterable<SkyKey>> directDeps,
        ImmutableList<CachedSkylarkImportLookupFunctionDeps> transitiveDeps) {
      this.directDeps = directDeps;
      this.transitiveDeps = transitiveDeps;
    }

    private static CachedSkylarkImportLookupFunctionDeps.Builder newBuilder() {
      return new CachedSkylarkImportLookupFunctionDeps.Builder();
    }

    private static class Builder {
      private final List<Iterable<SkyKey>> directDeps = new ArrayList<>();
      private final List<CachedSkylarkImportLookupFunctionDeps> transitiveDeps = new ArrayList<>();
      private final AtomicReference<Exception> exceptionSeen = new AtomicReference<>(null);

      // We only add the ASTFileLookupFunction through this so we don't need to worry about memory
      // optimizations by adding it raw.
      private void addDep(SkyKey key) {
        Preconditions.checkState(transitiveDeps.isEmpty(),
            "Expected transitive deps to be loaded last. directDeps: %s", directDeps);
        directDeps.add(ImmutableList.of(key));
      }

      private void addDeps(Iterable<SkyKey> keys) {
        Preconditions.checkState(transitiveDeps.isEmpty(),
            "Expected transitive deps to be loaded last. directDeps: %s", directDeps);
        directDeps.add(keys);
      }

      private void noteException(Exception e) {
        exceptionSeen.set(e);
      }

      private void addTransitiveDeps(CachedSkylarkImportLookupFunctionDeps transitiveDeps) {
        this.transitiveDeps.add(transitiveDeps);
      }

      CachedSkylarkImportLookupFunctionDeps build() {
        // We expect that we don't handle any exceptions in SkylarkLookupImportFunction directly.
        Preconditions.checkState(
            exceptionSeen.get() == null,
            "Caching a value in error?: %s %s",
            directDeps,
            exceptionSeen.get());
        return new CachedSkylarkImportLookupFunctionDeps(
            ImmutableList.copyOf(directDeps), ImmutableList.copyOf(transitiveDeps));
      }
    }

    // Reference equality is fine here as most often, the deps objects for any given import
    // label will be the same. It is technically possible for the same key to have two copies
    // of associated cached deps but this just means we may visit the same deps more than once. We
    // don't run the risk of chasing a cycle because these are detected prior to having any
    // cached values.
    private void traverse(
        DepGroupConsumer depGroupConsumer, Set<CachedSkylarkImportLookupFunctionDeps> visitedDeps)
        throws InterruptedException {
      for (Iterable<SkyKey> directDepGroup : directDeps) {
        depGroupConsumer.accept(directDepGroup);
      }
      for (CachedSkylarkImportLookupFunctionDeps indirectDeps : transitiveDeps) {
        if (visitedDeps.add(indirectDeps)) {
          indirectDeps.traverse(depGroupConsumer, visitedDeps);
        }
      }
    }
  }

  /** A consumer of dependency groups that can be interrupted. */
  interface DepGroupConsumer {
    void accept(Iterable<SkyKey> keys) throws InterruptedException;
  }
}
