// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * A class to distinguish repository mappings for repos from WORKSPACE and Bzlmod.
 *
 * <p>For repositories from the WORKSPACE file, if the requested repo doesn't exist in the mapping,
 * we fallback to the requested name. For repositories from Bzlmod, we return null to let the caller
 * decide what to do. This class won't be needed if one day we don't define external repositories in
 * the WORKSPACE file since {@code fallback} would always be false.
 */
@AutoValue
public abstract class RepositoryMapping {

  // Always fallback to the requested name
  public static final RepositoryMapping ALWAYS_FALLBACK = createAllowingFallback(ImmutableMap.of());

  /** Returns all the entries in this repo mapping. */
  public abstract ImmutableMap<String, RepositoryName> entries();

  /**
   * The owner repo of this repository mapping. It is for providing useful debug information when
   * repository mapping fails due to enforcing strict dependency, therefore it's only recorded when
   * we don't fallback to the requested repo name.
   */
  @Nullable
  abstract RepositoryName ownerRepo();

  public static RepositoryMapping create(
      Map<String, RepositoryName> entries, RepositoryName ownerRepo) {
    return createInternal(
        Preconditions.checkNotNull(entries), Preconditions.checkNotNull(ownerRepo));
  }

  public static RepositoryMapping createAllowingFallback(Map<String, RepositoryName> entries) {
    return createInternal(Preconditions.checkNotNull(entries), null);
  }

  private static RepositoryMapping createInternal(
      Map<String, RepositoryName> entries, RepositoryName ownerRepo) {
    return new AutoValue_RepositoryMapping(ImmutableMap.copyOf(entries), ownerRepo);
  }

  /**
   * Create a new {@link RepositoryMapping} instance based on existing repo mappings and given
   * additional mappings. If there are conflicts, existing mappings will take precedence.
   */
  public RepositoryMapping withAdditionalMappings(Map<String, RepositoryName> additionalMappings) {
    HashMap<String, RepositoryName> allMappings = new HashMap<>(additionalMappings);
    allMappings.putAll(entries());
    return createInternal(allMappings, ownerRepo());
  }

  /**
   * Create a new {@link RepositoryMapping} instance based on existing repo mappings and given
   * additional mappings. If there are conflicts, existing mappings will take precedence. The owner
   * repo of the given additional mappings is ignored.
   */
  public RepositoryMapping withAdditionalMappings(RepositoryMapping additionalMappings) {
    return withAdditionalMappings(additionalMappings.entries());
  }

  /**
   * Returns the canonical repository name associated with the given apparent repo name. The
   * provided apparent repo name is assumed to be valid.
   */
  public RepositoryName get(String preMappingName) {
    RepositoryName canonicalRepoName = entries().get(preMappingName);
    if (canonicalRepoName != null) {
      return canonicalRepoName;
    }
    // If the owner repo is not present, that means we should fall back to the requested repo name.
    if (ownerRepo() == null) {
      return RepositoryName.createUnvalidated(preMappingName);
    } else {
      return RepositoryName.createUnvalidated(preMappingName).toNonVisible(ownerRepo());
    }
  }

  /**
   * Whether the repo with this mapping is subject to strict deps; when strict deps is off, unknown
   * apparent names are silently treated as canonical names.
   */
  public boolean usesStrictDeps() {
    return ownerRepo() != null;
  }

  /**
   * Returns the first apparent name in this mapping that maps to the given canonical name, if any.
   */
  public Optional<String> getInverse(RepositoryName postMappingName) {
    return entries().entrySet().stream()
        .filter(e -> e.getValue().equals(postMappingName))
        .map(Entry::getKey)
        .findFirst();
  }

  /**
   * Creates a new {@link RepositoryMapping} instance that is the equivalent of composing this
   * {@link RepositoryMapping} with another one. That is, {@code a.composeWith(b).get(name) ===
   * b.get(a.get(name))} (treating {@code b} as allowing fallback).
   *
   * <p>Since we're treating the result of {@code a.get(name)} as an apparent repo name instead of a
   * canonical repo name, this only really makes sense when {@code a} does not use strict deps (i.e.
   * allows fallback).
   */
  public RepositoryMapping composeWith(RepositoryMapping other) {
    Preconditions.checkArgument(
        !usesStrictDeps(), "only an allow-fallback mapping can be composed with other mappings");
    HashMap<String, RepositoryName> entries = new HashMap<>(other.entries());
    for (Map.Entry<String, RepositoryName> entry : entries().entrySet()) {
      RepositoryName mappedName = other.get(entry.getValue().getName());
      entries.put(entry.getKey(), mappedName.isVisible() ? mappedName : entry.getValue());
    }
    return createInternal(entries, null);
  }
}
