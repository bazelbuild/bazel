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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepoMappingLookups;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TrackedRepositoryMapping;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * Records the repo mapping lookups of a Skyframe evaluation as dependencies on {@link
 * RepositoryMappingEntryValue}s.
 *
 * <p>Lookups whose value hasn't been computed yet are answered from the latest known mapping and
 * leave {@link Environment#valuesMissing()} set. Callers either redo the evaluation after a restart
 * or keep its result if {@link #registerDepsAndCheck} confirms all answers.
 */
final class RepoMappingTracker implements RepoMappingLookups.Scope {
  private Environment env;
  // Answers taken from Skyframe, which take precedence in later attempts.
  private final ConcurrentHashMap<RepositoryMappingEntryValue.Key, Object> authoritative =
      new ConcurrentHashMap<>();
  // Answers given in the current attempt.
  private ConcurrentHashMap<RepositoryMappingEntryValue.Key, Object> recorded =
      new ConcurrentHashMap<>();

  RepoMappingTracker(Environment env) {
    this.env = env;
  }

  /** Sets the environment of the current Skyframe restart. */
  void setEnv(Environment env) {
    this.env = env;
  }

  /** Forgets the answers given in a previous attempt. */
  void startAttempt() {
    recorded = new ConcurrentHashMap<>();
  }

  RepoMappingLookups.ScopeHandle enter() {
    return RepoMappingLookups.enter(this);
  }

  /** Adds the values of the given keys that are done as authoritative answers. */
  void addAuthoritative(
      SkyframeLookupResult result, Iterable<? extends RepositoryMappingEntryValue.Key> keys) {
    for (var key : keys) {
      var value = (RepositoryMappingEntryValue) result.get(key);
      if (value != null && value.payload() != null) {
        authoritative.put(key, value.payload());
      }
    }
  }

  @Override
  public RepositoryName get(TrackedRepositoryMapping mapping, String apparentName) {
    var key = RepositoryMappingEntryValue.forwardKey(mapping.contextRepo(), apparentName);
    var result = (RepositoryName) lookUp(key);
    if (result == null) {
      result = mapping.untracked().get(apparentName);
    }
    recorded.putIfAbsent(key, result);
    return result;
  }

  @Override
  @SuppressWarnings("unchecked")
  public Optional<String> getInverse(
      TrackedRepositoryMapping mapping, RepositoryName canonicalName) {
    var key = RepositoryMappingEntryValue.inverseKey(mapping.contextRepo(), canonicalName);
    var result = (Optional<String>) lookUp(key);
    if (result == null) {
      result = mapping.untracked().getInverse(canonicalName);
    }
    recorded.putIfAbsent(key, result);
    return result;
  }

  @Nullable
  private Object lookUp(RepositoryMappingEntryValue.Key key) {
    Object result = authoritative.get(key);
    if (result != null) {
      return result;
    }
    RepositoryMappingEntryValue value;
    try {
      value = (RepositoryMappingEntryValue) env.getValue(key);
    } catch (InterruptedException e) {
      // Starlark evaluation notices the interrupt soon.
      Thread.currentThread().interrupt();
      return null;
    }
    if (value == null || value.payload() == null) {
      return null;
    }
    authoritative.put(key, value.payload());
    return value.payload();
  }

  /** Returns the apparent names looked up in the given repo in the current attempt. */
  ImmutableSet<String> recordedForwardNames(RepositoryName repo) {
    var names = ImmutableSet.<String>builder();
    for (var key : recorded.keySet()) {
      if (key instanceof RepositoryMappingEntryValue.ForwardKey forwardKey
          && forwardKey.contextRepo().equals(repo)) {
        names.add(forwardKey.apparentName());
      }
    }
    return names.build();
  }

  @Nullable
  Object authoritativeAnswer(RepositoryMappingEntryValue.Key key) {
    return authoritative.get(key);
  }

  /**
   * Requests the {@link RepositoryMappingEntryValue}s for all recorded lookups.
   *
   * @return null if some values are missing, otherwise whether all recorded answers agree with
   *     Skyframe
   */
  @Nullable
  Boolean registerDepsAndCheck() throws InterruptedException {
    var keys = ImmutableList.copyOf(recorded.keySet());
    SkyframeLookupResult result = env.getValuesAndExceptions(keys);
    boolean consistent = true;
    for (var key : keys) {
      var value = (RepositoryMappingEntryValue) result.get(key);
      if (value == null || value.payload() == null) {
        // A null payload means that the context repo no longer exists, which fails the lookup of
        // the package or .bzl file instead.
        continue;
      }
      authoritative.put(key, value.payload());
      if (!Objects.equals(value.payload(), recorded.get(key))) {
        consistent = false;
      }
    }
    return env.valuesMissing() ? null : consistent;
  }
}
