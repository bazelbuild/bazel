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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
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
  // entries has an expensive hashCode() method that isn't cached.
  private final int hashCode;
  private final boolean recordsLookups;

  private RepositoryMapping(
      Map<String, RepositoryName> entries,
      int entriesHashCode,
      RepositoryName contextRepo,
      boolean recordsLookups) {
    this.entries = ImmutableMap.copyOf(entries);
    this.contextRepo = contextRepo;
    this.hashCode = 31 * entriesHashCode + contextRepo.hashCode();
    this.recordsLookups = recordsLookups;
  }

  /**
   * Creates a mapping with the same entries and context repo as {@code other} that reports its
   * lookups to the current {@link RepoMappingRecorder}.
   */
  protected RepositoryMapping(RepositoryMapping other) {
    this(other, /* recordsLookups= */ true);
  }

  private RepositoryMapping(RepositoryMapping other, boolean recordsLookups) {
    this.entries = other.entries;
    this.contextRepo = other.contextRepo;
    this.hashCode = other.hashCode;
    this.recordsLookups = recordsLookups;
  }

  /** Returns all the entries in this repo mapping. */
  public ImmutableMap<String, RepositoryName> entries() {
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
            && Objects.equals(entries, that.entries)
            && Objects.equals(contextRepo, that.contextRepo));
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  @Override
  public String toString() {
    return String.format("RepositoryMapping{entries=%s, contextRepo=%s}", entries, contextRepo);
  }

  public static RepositoryMapping create(
      ImmutableMap<String, RepositoryName> entries, RepositoryName contextRepo) {
    return createWithKnownEntriesHashCode(
        entries, entries.hashCode(), Preconditions.checkNotNull(contextRepo));
  }

  /**
   * Use this over {@link #create} if the {@link ImmutableMap#hashCode()} of the entries map is
   * already known.
   *
   * <p>Avoids repeated recomputation for RepositoryMapping instances with identical entries maps.
   */
  public static RepositoryMapping createWithKnownEntriesHashCode(
      ImmutableMap<String, RepositoryName> entries,
      int entriesHashCode,
      RepositoryName contextRepo) {
    return new RepositoryMapping(
        Preconditions.checkNotNull(entries),
        entriesHashCode,
        Preconditions.checkNotNull(contextRepo),
        /* recordsLookups= */ false);
  }

  /**
   * Returns a mapping with the same entries and context repo as this one that reports its lookups
   * to the current {@link RepoMappingRecorder}. This is meant for the mapping of a repo as computed
   * by Skyframe, so that recorded lookups can be checked against it later.
   */
  public RepositoryMapping withRecordedLookups() {
    return recordsLookups ? this : new RepositoryMapping(this, /* recordsLookups= */ true);
  }

  /**
   * Returns a mapping that resolves the names in {@code overrides} to their values and all other
   * names through this mapping, so that only lookups of the latter are recorded.
   */
  public RepositoryMapping withOverrides(ImmutableMap<String, RepositoryName> overrides) {
    RepositoryMapping base = this;
    ImmutableMap<String, RepositoryName> allEntries =
        ImmutableMap.<String, RepositoryName>builderWithExpectedSize(
                entries.size() + overrides.size())
            .putAll(entries)
            .putAll(overrides)
            .buildKeepingLast();
    return new RepositoryMapping(
        allEntries, allEntries.hashCode(), contextRepo, /* recordsLookups= */ false) {
      @Override
      public RepositoryName get(String preMappingName) {
        RepositoryName override = overrides.get(preMappingName);
        if (override != null) {
          return override;
        }
        RepositoryName canonicalName = base.get(preMappingName);
        // Suggest the overridden names as well.
        return canonicalName.isVisible() ? canonicalName : getUnrecorded(preMappingName);
      }
    };
  }

  /**
   * Create a new {@link RepositoryMapping} instance based on existing repo mappings and given
   * additional mappings. If there are conflicts, existing mappings will take precedence.
   */
  public RepositoryMapping withAdditionalMappings(
      ImmutableMap<String, RepositoryName> additionalMappings) {
    return create(
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
    if (recordsLookups) {
      RepoMappingRecorder recorder = RepoMappingRecorder.current();
      if (recorder != null) {
        return recorder.get(this, preMappingName);
      }
    }
    return getUnrecorded(preMappingName);
  }

  /**
   * Like {@link #get}, but doesn't report the lookup to the current {@link RepoMappingRecorder}.
   */
  public RepositoryName getUnrecorded(String preMappingName) {
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
    if (recordsLookups) {
      RepoMappingRecorder recorder = RepoMappingRecorder.current();
      if (recorder != null) {
        return recorder.getInverse(this, postMappingName);
      }
    }
    return getInverseUnrecorded(postMappingName);
  }

  /**
   * Like {@link #getInverse}, but doesn't report the lookup to the current {@link
   * RepoMappingRecorder}.
   */
  public Optional<String> getInverseUnrecorded(RepositoryName postMappingName) {
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
    return new RepositoryMapping(entries, entries.hashCode(), contextRepo, recordsLookups) {
      @Override
      public Optional<String> getInverseUnrecorded(RepositoryName postMappingName) {
        return Optional.ofNullable(inverseCopy.get(postMappingName));
      }
    };
  }
}
