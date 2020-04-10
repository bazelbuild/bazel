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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.skyframe.StarlarkImportLookupValue.SkylarkImportLookupKey;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

class CachedSkylarkImportLookupValueAndDeps {
  private final SkylarkImportLookupKey key;
  private final StarlarkImportLookupValue value;
  private final ImmutableList<Iterable<SkyKey>> directDeps;
  private final ImmutableList<CachedSkylarkImportLookupValueAndDeps> transitiveDeps;

  private CachedSkylarkImportLookupValueAndDeps(
      SkylarkImportLookupKey key,
      StarlarkImportLookupValue value,
      ImmutableList<Iterable<SkyKey>> directDeps,
      ImmutableList<CachedSkylarkImportLookupValueAndDeps> transitiveDeps) {
    this.key = key;
    this.value = value;
    this.directDeps = directDeps;
    this.transitiveDeps = transitiveDeps;
  }

  void traverse(
      DepGroupConsumer depGroupConsumer,
      Map<SkylarkImportLookupKey, CachedSkylarkImportLookupValueAndDeps> visitedDeps)
      throws InterruptedException {
    for (Iterable<SkyKey> directDepGroup : directDeps) {
      depGroupConsumer.accept(directDepGroup);
    }
    for (CachedSkylarkImportLookupValueAndDeps indirectDeps : transitiveDeps) {
      if (!visitedDeps.containsKey(indirectDeps.key)) {
        visitedDeps.put(indirectDeps.key, indirectDeps);
        indirectDeps.traverse(depGroupConsumer, visitedDeps);
      }
    }
  }

  StarlarkImportLookupValue getValue() {
    return value;
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof CachedSkylarkImportLookupValueAndDeps) {
      // With the interner, force there to be exactly one cached value per key at any given point
      // in time.
      return this.key.equals(((CachedSkylarkImportLookupValueAndDeps) obj).key);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return key.hashCode();
  }

  /** A consumer of dependency groups that can be interrupted. */
  interface DepGroupConsumer {
    void accept(Iterable<SkyKey> keys) throws InterruptedException;
  }

  static class Builder {
    Builder(Interner<CachedSkylarkImportLookupValueAndDeps> interner) {
      this.interner = interner;
    }

    private final Interner<CachedSkylarkImportLookupValueAndDeps> interner;
    private final List<Iterable<SkyKey>> directDeps = new ArrayList<>();
    private final List<CachedSkylarkImportLookupValueAndDeps> transitiveDeps = new ArrayList<>();
    private final AtomicReference<Exception> exceptionSeen = new AtomicReference<>(null);
    private StarlarkImportLookupValue value;
    private SkylarkImportLookupKey key;

    @CanIgnoreReturnValue
    Builder addDep(SkyKey key) {
      Preconditions.checkState(
          transitiveDeps.isEmpty(), "Expected transitive deps to be loaded last: %s", this);
      directDeps.add(ImmutableList.of(key));
      return this;
    }

    @CanIgnoreReturnValue
    Builder addDeps(Iterable<SkyKey> keys) {
      Preconditions.checkState(
          transitiveDeps.isEmpty(), "Expected transitive deps to be loaded last: %s", this);
      directDeps.add(keys);
      return this;
    }

    @CanIgnoreReturnValue
    Builder noteException(Exception e) {
      exceptionSeen.set(e);
      return this;
    }

    @CanIgnoreReturnValue
    Builder addTransitiveDeps(CachedSkylarkImportLookupValueAndDeps transitiveDeps) {
      this.transitiveDeps.add(transitiveDeps);
      return this;
    }

    @CanIgnoreReturnValue
    Builder setValue(StarlarkImportLookupValue value) {
      this.value = value;
      return this;
    }

    @CanIgnoreReturnValue
    Builder setKey(SkylarkImportLookupKey key) {
      this.key = key;
      return this;
    }

    CachedSkylarkImportLookupValueAndDeps build() {
      // We expect that we don't handle any exceptions in SkylarkLookupImportFunction directly.
      Preconditions.checkState(exceptionSeen.get() == null, "Caching a value in error?: %s", this);
      Preconditions.checkNotNull(value, "Expected value to be set: %s", this);
      Preconditions.checkNotNull(key, "Expected key to be set: %s", this);
      return interner.intern(
          new CachedSkylarkImportLookupValueAndDeps(
              key, value, ImmutableList.copyOf(directDeps), ImmutableList.copyOf(transitiveDeps)));
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(CachedSkylarkImportLookupValueAndDeps.Builder.class)
          .add("key", key)
          .add("value", value)
          .add("directDeps", directDeps)
          .add("transitiveDeps", transitiveDeps)
          .add("exceptionSeen", exceptionSeen)
          .toString();
    }

  }
}
