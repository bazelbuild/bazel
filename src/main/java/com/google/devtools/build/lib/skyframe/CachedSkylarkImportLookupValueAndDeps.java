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
import com.google.common.collect.Iterables;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

class CachedSkylarkImportLookupValueAndDeps {
  private final SkylarkImportLookupValue value;
  // We store the deps separately so that we can let go of the value when the cache drops it.
  final CachedSkylarkImportLookupFunctionDeps deps;

  private CachedSkylarkImportLookupValueAndDeps(
      SkylarkImportLookupValue value, CachedSkylarkImportLookupFunctionDeps deps) {
    this.value = value;
    this.deps = deps;
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
      return new CachedSkylarkImportLookupValueAndDeps(value, depsBuilder.build());
    }
  }

  static class CachedSkylarkImportLookupFunctionDeps implements Iterable<Iterable<SkyKey>> {
    private final ImmutableList<Iterable<SkyKey>> deps;

    private CachedSkylarkImportLookupFunctionDeps(ImmutableList<Iterable<SkyKey>> deps) {
      this.deps = deps;
    }

    static CachedSkylarkImportLookupFunctionDeps.Builder newBuilder() {
      return new CachedSkylarkImportLookupFunctionDeps.Builder();
    }

    @Override
    public Iterator<Iterable<SkyKey>> iterator() {
      return deps.iterator();
    }

    static class Builder {
      private final List<Iterable<SkyKey>> deps = new ArrayList<>();
      private final AtomicReference<Exception> exceptionSeen = new AtomicReference<>(null);

      // We only add the ASTFileLookupFunction through this so we don't need to worry about memory
      // optimizations by adding it raw.
      @CanIgnoreReturnValue
      Builder addDep(SkyKey key) {
        deps.add(ImmutableList.of(key));
        return this;
      }

      @CanIgnoreReturnValue
      Builder addDeps(Iterable<SkyKey> keys) {
        deps.add(keys);
        return this;
      }

      @CanIgnoreReturnValue
      Builder noteException(Exception e) {
        exceptionSeen.set(e);
        return this;
      }

      @CanIgnoreReturnValue
      Builder addTransitiveDeps(CachedSkylarkImportLookupFunctionDeps transitiveDeps) {
        Iterables.addAll(deps, transitiveDeps);
        return this;
      }

      CachedSkylarkImportLookupFunctionDeps build() {
        // We expect that we don't handle any exceptions in SkylarkLookupImportFunction directly.
        Preconditions.checkState(
            exceptionSeen.get() == null,
            "Caching a value in error?: %s %s",
            deps,
            exceptionSeen.get());
        return new CachedSkylarkImportLookupFunctionDeps(ImmutableList.copyOf(deps));
      }
    }
  }
}
