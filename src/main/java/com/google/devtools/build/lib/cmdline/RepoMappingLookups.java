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

import java.util.Optional;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * Routes lookups through {@link TrackedRepositoryMapping}s to the evaluation running on the current
 * thread, which records them as dependencies.
 */
public final class RepoMappingLookups {
  /** Whether repo mapping lookups are tracked individually. */
  public static final boolean ENABLED =
      Boolean.parseBoolean(System.getProperty("bazel.fine_grained_repo_mapping", "true"));

  /** Answers and records the lookups of the evaluation running on the current thread. */
  public interface Scope {
    RepositoryName get(TrackedRepositoryMapping mapping, String apparentName);

    Optional<String> getInverse(TrackedRepositoryMapping mapping, RepositoryName canonicalName);
  }

  /** Restores the previous scope when closed. */
  public interface ScopeHandle extends AutoCloseable {
    @Override
    void close();
  }

  private static final ThreadLocal<Scope> currentScope = new ThreadLocal<>();

  // Returns the current mapping of a repo if it has been computed. Lookups that aren't recorded are
  // answered from it.
  private static volatile Function<RepositoryName, RepositoryMapping> latestMappings = repo -> null;

  private RepoMappingLookups() {}

  public static ScopeHandle enter(Scope scope) {
    Scope previous = currentScope.get();
    currentScope.set(scope);
    return () -> currentScope.set(previous);
  }

  @Nullable
  static Scope currentScope() {
    return currentScope.get();
  }

  public static void setLatestMappings(Function<RepositoryName, RepositoryMapping> mappings) {
    latestMappings = mappings;
  }

  @Nullable
  public static RepositoryMapping getLatest(RepositoryName repo) {
    return latestMappings.apply(repo);
  }
}
