// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * A value that represents the 'mappings' of an external Bazel workspace, as defined in the main
 * WORKSPACE file. The SkyValue contains the mappings themselves, with the key being the name of the
 * external repository.
 *
 * <p>Given the following rule:
 *
 * <pre>{@code
 * local_repository(
 *   name = "a",
 *   path = "../a",
 *   repo_mapping = {"@x" : "@y"}
 * )
 * }</pre>
 *
 * <p>The SkyKey would be {@code "@a"} and the SkyValue would be the map {@code {"@x" : "@y"}}
 *
 * <p>This is kept as a separate value with trivial change pruning so as to not necessitate a
 * dependency from every {@link PackageValue} to the //external {@link PackageValue}, so that
 * changes to things in the WORKSPACE other than the mappings (and name) won't require reloading all
 * packages. If the mappings are changed then the external packages need to be reloaded.
 */
public class RepositoryMappingValue implements SkyValue {
  public static final Key KEY_FOR_ROOT_MODULE_WITHOUT_WORKSPACE_REPOS =
      Key.create(RepositoryName.MAIN, /*rootModuleShouldSeeWorkspaceRepos=*/ false);

  private final RepositoryMapping repositoryMapping;

  private RepositoryMappingValue(RepositoryMapping repositoryMapping) {
    Preconditions.checkNotNull(repositoryMapping);
    this.repositoryMapping = repositoryMapping;
  }

  /** Returns the workspace mappings. */
  public RepositoryMapping getRepositoryMapping() {
    return repositoryMapping;
  }

  /** Returns the {@link Key} for {@link RepositoryMappingValue}s. */
  public static Key key(RepositoryName repositoryName) {
    return RepositoryMappingValue.Key.create(
        repositoryName, /*rootModuleShouldSeeWorkspaceRepos=*/ true);
  }

  /** Returns a {@link RepositoryMappingValue} for a workspace with the given name. */
  public static RepositoryMappingValue withMapping(RepositoryMapping repositoryMapping) {
    return new RepositoryMappingValue(Preconditions.checkNotNull(repositoryMapping));
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof RepositoryMappingValue)) {
      return false;
    }
    RepositoryMappingValue other = (RepositoryMappingValue) o;
    return Objects.equals(repositoryMapping, other.repositoryMapping);
  }

  @Override
  public int hashCode() {
    return Objects.hash(repositoryMapping);
  }

  @Override
  public String toString() {
    return repositoryMapping.toString();
  }

  /** {@link SkyKey} for {@link RepositoryMappingValue}. */
  @AutoValue
  abstract static class Key implements SkyKey {

    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    /** The name of the repo to grab mappings for. */
    abstract RepositoryName repoName();

    /**
     * Whether the root module should see repos defined in WORKSPACE. This only takes effect when
     * {@link #repoName} is the main repo.
     */
    abstract boolean rootModuleShouldSeeWorkspaceRepos();

    @AutoCodec.Instantiator
    static Key create(RepositoryName repoName, boolean rootModuleShouldSeeWorkspaceRepos) {
      return interner.intern(
          new AutoValue_RepositoryMappingValue_Key(repoName, rootModuleShouldSeeWorkspaceRepos));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.REPOSITORY_MAPPING;
    }
  }
}
