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

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import net.starlark.java.spelling.SpellChecker;

/**
 * Stores the mapping from apparent repo name to canonical repo name, from the viewpoint of a
 * context repo.
 *
 * <p>This class must not implement {@link net.starlark.java.eval.StarlarkValue} since instances of
 * this class are used as markers by {@link
 * com.google.devtools.build.lib.analysis.starlark.StarlarkCustomCommandLine}.
 */
public class RepositoryMapping {
  /* An empty repo mapping with the main repo as the context repo. */
  public static final RepositoryMapping EMPTY =
      create(ImmutableMap.of("", RepositoryName.MAIN), RepositoryName.MAIN);

  private final ImmutableMap<String, RepositoryName> entries;
  private final RepositoryName contextRepo;

  private RepositoryMapping(Map<String, RepositoryName> entries, RepositoryName contextRepo) {
    this.entries = ImmutableMap.copyOf(entries);
    this.contextRepo = contextRepo;
  }

  /** Returns all the entries in this repo mapping. */
  public final ImmutableMap<String, RepositoryName> entries() {
    return entries;
  }

  /**
   * The context repo of this repository mapping. It is for providing useful debug information when
   * repository mapping fails due to enforcing strict dependency.
   */
  public final RepositoryName contextRepo() {
    return contextRepo;
  }

  @Override
  public boolean equals(Object o) {
    return this == o
        || (o instanceof RepositoryMapping that
            && Objects.equal(entries, that.entries)
            && Objects.equal(contextRepo, that.contextRepo));
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(entries, contextRepo);
  }

  @Override
  public String toString() {
    return String.format("RepositoryMapping{entries=%s, contextRepo=%s}", entries, contextRepo);
  }

  public static RepositoryMapping create(
      ImmutableMap<String, RepositoryName> entries, RepositoryName contextRepo) {
    return new RepositoryMapping(
        Preconditions.checkNotNull(entries), Preconditions.checkNotNull(contextRepo));
  }

  /**
   * Create a new {@link RepositoryMapping} instance based on existing repo mappings and given
   * additional mappings. If there are conflicts, existing mappings will take precedence.
   */
  public RepositoryMapping withAdditionalMappings(
      ImmutableMap<String, RepositoryName> additionalMappings) {
    return new RepositoryMapping(
        ImmutableMap.<String, RepositoryName>builderWithExpectedSize(
                entries().size() + additionalMappings.size())
            .putAll(additionalMappings)
            .putAll(entries())
            .buildKeepingLast(),
        contextRepo());
  }

  /**
   * Create a new {@link RepositoryMapping} instance based on existing repo mappings and given
   * additional mappings. If there are conflicts, existing mappings will take precedence. The owner
   * repo of the given additional mappings is ignored.
   */
  public RepositoryMapping withAdditionalMappings(RepositoryMapping additionalMappings) {
    return withAdditionalMappings(
        (additionalMappings == null) ? ImmutableMap.of() : additionalMappings.entries());
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
    return RepositoryName.createUnvalidated(preMappingName)
        .toNonVisible(contextRepo(), SpellChecker.didYouMean(preMappingName, entries().keySet()));
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
   * Returns a new {@link RepositoryMapping} instance with identical contents, except that the
   * inverse mapping is cached, causing {@link #getInverse} to be much more efficient. This is
   * particularly important for the main repo mapping, as it's often used to generate display-form
   * labels ({@link Label#getDisplayForm}).
   */
  public RepositoryMapping withCachedInverseMap() {
    var inverse = Maps.<RepositoryName, String>newHashMapWithExpectedSize(entries.size());
    for (Map.Entry<String, RepositoryName> entry : entries.entrySet()) {
      inverse.putIfAbsent(entry.getValue(), entry.getKey());
    }
    var inverseCopy = ImmutableMap.copyOf(inverse);
    return new RepositoryMapping(entries, contextRepo) {
      @Override
      public Optional<String> getInverse(RepositoryName postMappingName) {
        return Optional.ofNullable(inverseCopy.get(postMappingName));
      }
    };
  }
}
