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
import java.util.Map;
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

  abstract ImmutableMap<RepositoryName, RepositoryName> repositoryMapping();

  abstract boolean fallback();

  public static RepositoryMapping create(Map<RepositoryName, RepositoryName> repositoryMapping) {
    return new AutoValue_RepositoryMapping(
        ImmutableMap.copyOf(Preconditions.checkNotNull(repositoryMapping)), /* fallback= */ false);
  }

  public static RepositoryMapping createAllowingFallback(
      Map<RepositoryName, RepositoryName> repositoryMapping) {
    return new AutoValue_RepositoryMapping(
        ImmutableMap.copyOf(Preconditions.checkNotNull(repositoryMapping)), /* fallback= */ true);
  }

  @Nullable
  public RepositoryName get(RepositoryName repositoryName) {
    if (fallback()) {
      return repositoryMapping().getOrDefault(repositoryName, repositoryName);
    } else {
      return repositoryMapping().get(repositoryName);
    }
  }
}
