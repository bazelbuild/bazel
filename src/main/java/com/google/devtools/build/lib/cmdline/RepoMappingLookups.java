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

import java.util.function.Function;
import javax.annotation.Nullable;

/** Global state for {@link TrackedRepositoryMapping}s. */
public final class RepoMappingLookups {
  /** Whether repo mapping lookups are tracked individually. */
  public static final boolean ENABLED =
      Boolean.parseBoolean(System.getProperty("bazel.fine_grained_repo_mapping", "true"));

  // Returns the current mapping of a repo if it has been computed. Lookups through tracked mappings
  // are answered from it.
  private static volatile Function<RepositoryName, RepositoryMapping> latestMappings = repo -> null;

  private RepoMappingLookups() {}

  public static void setLatestMappings(Function<RepositoryName, RepositoryMapping> mappings) {
    latestMappings = mappings;
  }

  @Nullable
  public static RepositoryMapping getLatest(RepositoryName repo) {
    return latestMappings.apply(repo);
  }
}
