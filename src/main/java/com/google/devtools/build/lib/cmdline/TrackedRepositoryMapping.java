// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.cmdline;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * A {@link RepositoryMapping} whose lookups are recorded by the {@link RepoMappingLookups.Scope} of
 * the current thread, if any.
 *
 * <p>Lookups of covered names aren't recorded since the owner of this mapping already depends on
 * them.
 */
public final class TrackedRepositoryMapping extends RepositoryMapping {
  private static final ConcurrentHashMap<RepositoryName, TrackedRepositoryMapping> sharedInstances =
      new ConcurrentHashMap<>();

  private final RepositoryMapping snapshot;
  private final Object mappingSource;
  private volatile ImmutableSet<String> covered = ImmutableSet.of();
  @Nullable private volatile ImmutableMap<String, RepositoryName> overrides;

  /**
   * Creates a mapping that answers lookups not recorded by a scope from the latest mapping of its
   * repo, or from {@code snapshot} if there is none.
   *
   * @param mappingSource identifies the entries of the mapping, which may be shared with other
   *     repos
   */
  public TrackedRepositoryMapping(RepositoryMapping snapshot, Object mappingSource) {
    super(snapshot);
    this.snapshot = snapshot;
    this.mappingSource = mappingSource;
  }

  /**
   * Returns a mapping as created by the constructor that may be shared with other callers, so it
   * must not be {@linkplain #cover covered}.
   */
  public static TrackedRepositoryMapping shared(RepositoryMapping snapshot, Object mappingSource) {
    return sharedInstances.compute(
        snapshot.contextRepo(),
        (repo, existing) ->
            existing != null
                    && existing.snapshot == snapshot
                    && existing.mappingSource.equals(mappingSource)
                ? existing
                : new TrackedRepositoryMapping(snapshot, mappingSource));
  }

  public Object mappingSource() {
    return mappingSource;
  }

  /** Returns the mapping that lookups which aren't recorded are answered from. */
  public RepositoryMapping untracked() {
    RepositoryMapping latest = RepoMappingLookups.getLatest(contextRepo());
    return latest != null ? latest : snapshot;
  }

  /**
   * Stops recording lookups of the given names. {@code overrides} replaces the answers for names on
   * which {@link #untracked()} is out of date.
   */
  public void cover(
      ImmutableSet<String> names, @Nullable ImmutableMap<String, RepositoryName> overrides) {
    this.overrides = overrides;
    this.covered = names;
  }

  @Override
  public ImmutableMap<String, RepositoryName> entries() {
    return untracked().entries();
  }

  @Override
  public RepositoryName get(String preMappingName) {
    var overrides = this.overrides;
    if (overrides != null) {
      RepositoryName override = overrides.get(preMappingName);
      if (override != null) {
        return override;
      }
    }
    if (!covered.contains(preMappingName)) {
      var scope = RepoMappingLookups.currentScope();
      if (scope != null) {
        return scope.get(this, preMappingName);
      }
    }
    return untracked().get(preMappingName);
  }

  @Override
  public Optional<String> getInverse(RepositoryName postMappingName) {
    var scope = RepoMappingLookups.currentScope();
    if (scope != null) {
      return scope.getInverse(this, postMappingName);
    }
    return untracked().getInverse(postMappingName);
  }
}
