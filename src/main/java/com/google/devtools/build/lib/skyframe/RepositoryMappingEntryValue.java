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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * A single fact derived from the {@link RepositoryMappingValue} of a repo: the result of one
 * forward lookup, one inverse lookup, or the repo's associated module.
 *
 * <p>Consumers that depend on these values instead of the full {@link RepositoryMappingValue} are
 * only invalidated if a fact they observed changes.
 */
public sealed interface RepositoryMappingEntryValue extends SkyValue {

  /**
   * The fact as observed through {@link com.google.devtools.build.lib.cmdline.RepositoryMapping}.
   */
  Object payload();

  static ForwardKey forwardKey(RepositoryName contextRepo, String apparentName) {
    return ForwardKey.INTERNER.intern(
        new ForwardKey(contextRepo, ForwardKey.NAME_INTERNER.intern(apparentName)));
  }

  static InverseKey inverseKey(RepositoryName contextRepo, RepositoryName canonicalName) {
    return InverseKey.INTERNER.intern(new InverseKey(contextRepo, canonicalName));
  }

  static ModuleInfoKey moduleInfoKey(RepositoryName repo) {
    return ModuleInfoKey.INTERNER.intern(new ModuleInfoKey(repo));
  }

  /** Common supertype of all keys. */
  sealed interface Key extends SkyKey {
    RepositoryName contextRepo();

    @Override
    default SkyFunctionName functionName() {
      return SkyFunctions.REPOSITORY_MAPPING_ENTRY;
    }
  }

  /** Key for the canonical name of {@code apparentName} as seen from {@code contextRepo}. */
  record ForwardKey(RepositoryName contextRepo, String apparentName) implements Key {
    private static final SkyKeyInterner<ForwardKey> INTERNER = SkyKey.newInterner();
    // Apparent names recur across many context repos.
    private static final Interner<String> NAME_INTERNER = BlazeInterners.newWeakInterner();

    public ForwardKey {
      checkNotNull(contextRepo);
      checkNotNull(apparentName);
    }

    @Override
    public SkyKeyInterner<ForwardKey> getSkyKeyInterner() {
      return INTERNER;
    }
  }

  /** Key for the first apparent name of {@code canonicalName} as seen from {@code contextRepo}. */
  record InverseKey(RepositoryName contextRepo, RepositoryName canonicalName) implements Key {
    private static final SkyKeyInterner<InverseKey> INTERNER = SkyKey.newInterner();

    public InverseKey {
      checkNotNull(contextRepo);
      checkNotNull(canonicalName);
    }

    @Override
    public SkyKeyInterner<InverseKey> getSkyKeyInterner() {
      return INTERNER;
    }
  }

  /** Key for the module associated with {@code contextRepo}. */
  record ModuleInfoKey(RepositoryName contextRepo) implements Key {
    private static final SkyKeyInterner<ModuleInfoKey> INTERNER = SkyKey.newInterner();

    public ModuleInfoKey {
      checkNotNull(contextRepo);
    }

    @Override
    public SkyKeyInterner<ModuleInfoKey> getSkyKeyInterner() {
      return INTERNER;
    }
  }

  /** Value for {@link ForwardKey}. Null if the context repo doesn't exist. */
  record Forward(@Nullable RepositoryName canonicalName) implements RepositoryMappingEntryValue {
    private static final Interner<Forward> INTERNER = BlazeInterners.newWeakInterner();
    static final Forward REPO_NOT_FOUND = new Forward(null);

    static Forward of(RepositoryName canonicalName) {
      return INTERNER.intern(new Forward(checkNotNull(canonicalName)));
    }

    @Override
    @Nullable
    public Object payload() {
      return canonicalName;
    }
  }

  /** Value for {@link InverseKey}. */
  record Inverse(Optional<String> apparentName) implements RepositoryMappingEntryValue {
    private static final Interner<Inverse> INTERNER = BlazeInterners.newWeakInterner();
    static final Inverse NONE = new Inverse(Optional.empty());

    static Inverse of(Optional<String> apparentName) {
      return apparentName.isEmpty() ? NONE : INTERNER.intern(new Inverse(apparentName));
    }

    @Override
    public Object payload() {
      return apparentName;
    }
  }

  /** Value for {@link ModuleInfoKey}. */
  record ModuleInfo(
      boolean repoExists,
      Optional<String> associatedModuleName,
      Optional<String> associatedModuleVersion,
      @Nullable String moduleRepoName)
      implements RepositoryMappingEntryValue {
    static final ModuleInfo REPO_NOT_FOUND =
        new ModuleInfo(false, Optional.empty(), Optional.empty(), null);

    @Override
    public Object payload() {
      return this;
    }
  }
}
