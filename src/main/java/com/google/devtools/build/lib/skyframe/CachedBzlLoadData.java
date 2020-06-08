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
import com.google.devtools.build.skyframe.SkyKey;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A saved {@link BzlLoadFunction} computation, used when inlining that Skyfunction.
 *
 * <p>This holds a requested key, its computed value, and the direct and transitive Skyframe
 * dependencies that are needed to compute it from scratch (i.e. if it weren't cached). Here
 * "transitive" means "underneath another {@code BzlLoadFunction} computation"; we split them into
 * other {@code CachedBzlLoadData} objects so they can be shared by other requesting bzls.
 */
class CachedBzlLoadData {
  private final BzlLoadValue.Key key;
  private final BzlLoadValue value;
  private final ImmutableList<Iterable<SkyKey>> directDeps;
  private final ImmutableList<CachedBzlLoadData> transitiveDeps;

  private CachedBzlLoadData(
      BzlLoadValue.Key key,
      BzlLoadValue value,
      ImmutableList<Iterable<SkyKey>> directDeps,
      ImmutableList<CachedBzlLoadData> transitiveDeps) {
    this.key = key;
    this.value = value;
    this.directDeps = directDeps;
    this.transitiveDeps = transitiveDeps;
  }

  /**
   * Adds all deps (direct and transitive) of this value to the {@code visitedDeps} set and passes
   * them to the consumer (with unspecified order and grouping). The traversal does not include
   * nodes already contained in {@code visitedDeps}.
   */
  void traverse(
      DepGroupConsumer depGroupConsumer, Map<BzlLoadValue.Key, CachedBzlLoadData> visitedDeps)
      throws InterruptedException {
    if (visitedDeps.putIfAbsent(key, this) != null) {
      return;
    }

    for (Iterable<SkyKey> directDepGroup : directDeps) {
      depGroupConsumer.accept(directDepGroup);
    }
    for (CachedBzlLoadData indirectDeps : transitiveDeps) {
      indirectDeps.traverse(depGroupConsumer, visitedDeps);
    }
  }

  BzlLoadValue getValue() {
    return value;
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof CachedBzlLoadData) {
      // With the interner, force there to be exactly one cached value per key at any given point
      // in time.
      return this.key.equals(((CachedBzlLoadData) obj).key);
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
    Builder(Interner<CachedBzlLoadData> interner) {
      this.interner = interner;
    }

    private final Interner<CachedBzlLoadData> interner;
    private final List<Iterable<SkyKey>> directDeps = new ArrayList<>();
    private final List<CachedBzlLoadData> transitiveDeps = new ArrayList<>();
    private final AtomicReference<Exception> exceptionSeen = new AtomicReference<>(null);
    private BzlLoadValue value;
    private BzlLoadValue.Key key;

    @CanIgnoreReturnValue
    Builder addDep(SkyKey key) {
      directDeps.add(ImmutableList.of(key));
      return this;
    }

    @CanIgnoreReturnValue
    Builder addDeps(Iterable<SkyKey> keys) {
      directDeps.add(keys);
      return this;
    }

    @CanIgnoreReturnValue
    Builder noteException(Exception e) {
      exceptionSeen.set(e);
      return this;
    }

    @CanIgnoreReturnValue
    Builder addTransitiveDeps(CachedBzlLoadData transitiveDeps) {
      this.transitiveDeps.add(transitiveDeps);
      return this;
    }

    @CanIgnoreReturnValue
    Builder setValue(BzlLoadValue value) {
      this.value = value;
      return this;
    }

    @CanIgnoreReturnValue
    Builder setKey(BzlLoadValue.Key key) {
      this.key = key;
      return this;
    }

    CachedBzlLoadData build() {
      // We expect that we don't handle any exceptions in BzlLoadFunction directly.
      Preconditions.checkState(exceptionSeen.get() == null, "Caching a value in error?: %s", this);
      Preconditions.checkNotNull(value, "Expected value to be set: %s", this);
      Preconditions.checkNotNull(key, "Expected key to be set: %s", this);
      return interner.intern(
          new CachedBzlLoadData(
              key, value, ImmutableList.copyOf(directDeps), ImmutableList.copyOf(transitiveDeps)));
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(CachedBzlLoadData.Builder.class)
          .add("key", key)
          .add("value", value)
          .add("directDeps", directDeps)
          .add("transitiveDeps", transitiveDeps)
          .add("exceptionSeen", exceptionSeen)
          .toString();
    }

  }
}
