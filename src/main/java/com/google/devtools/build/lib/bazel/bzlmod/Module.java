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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Represents a node in the external dependency graph.
 *
 * <p>In particular, it represents a specific version of a module; there can be multiple {@link
 * Module}s in a dependency graph with the same name but with different versions (when there's a
 * multiple_version_override in play).
 *
 * <p>For the intermediate type used during module resolution, see {@link InterimModule}.
 */
@AutoValue
public abstract class Module extends ModuleBase {

  /**
   * The resolved direct dependencies of this module. The key type is the repo name of the dep, and
   * the value type is the ModuleKey ({@link #getKey()}) of the dep.
   */
  public abstract ImmutableMap<String, ModuleKey> getDeps();

  /**
   * Returns a {@link RepositoryMapping} with only Bazel module repos and no repos from module
   * extensions. For the full mapping, see {@link BazelDepGraphValue#getFullRepoMapping}.
   */
  public final RepositoryMapping getRepoMappingWithBazelDepsOnly(
      ImmutableMap<ModuleKey, RepositoryName> moduleKeyToRepositoryNames) {
    ImmutableMap.Builder<String, RepositoryName> mapping = ImmutableMap.builder();
    // If this is the root module, then the main repository should be visible as `@`.
    if (getKey().equals(ModuleKey.ROOT)) {
      mapping.put("", RepositoryName.MAIN);
    }
    // Every module should be able to reference itself as @<module repo name>.
    // If this is the root module, this perfectly falls into @<module repo name> => @
    RepositoryName owner = moduleKeyToRepositoryNames.get(getKey());
    if (!getRepoName().isEmpty()) {
      mapping.put(getRepoName(), owner);
    }
    for (Map.Entry<String, ModuleKey> dep : getDeps().entrySet()) {
      // Special note: if `dep` is actually the root module, its ModuleKey would be ROOT whose
      // canonicalRepoName is the empty string. This perfectly maps to the main repo ("@").
      mapping.put(dep.getKey(), moduleKeyToRepositoryNames.get(dep.getValue()));
    }
    return RepositoryMapping.create(mapping.buildOrThrow(), owner);
  }

  /**
   * The repo spec for this module (information about the attributes of its repository rule). This
   * is only non-null for modules coming from registries (i.e. without non-registry overrides).
   */
  @Nullable
  public abstract RepoSpec getRepoSpec();

  /** Returns a new, empty {@link Builder}. */
  public static Builder builder() {
    return new AutoValue_Module.Builder();
  }

  /** Builder type for {@link Module}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setName(String value);

    public abstract Builder setVersion(Version value);

    public abstract Builder setKey(ModuleKey value);

    public abstract Builder setRepoName(String value);

    public abstract Builder setExecutionPlatformsToRegister(ImmutableList<String> value);

    public abstract Builder setToolchainsToRegister(ImmutableList<String> value);

    public abstract Builder setDeps(ImmutableMap<String, ModuleKey> value);

    abstract ImmutableMap.Builder<String, ModuleKey> depsBuilder();

    @CanIgnoreReturnValue
    public Builder addDep(String depRepoName, ModuleKey depKey) {
      depsBuilder().put(depRepoName, depKey);
      return this;
    }

    public abstract Builder setRepoSpec(RepoSpec value);

    public abstract Builder setExtensionUsages(ImmutableList<ModuleExtensionUsage> value);

    abstract ImmutableList.Builder<ModuleExtensionUsage> extensionUsagesBuilder();

    @CanIgnoreReturnValue
    public Builder addExtensionUsage(ModuleExtensionUsage value) {
      extensionUsagesBuilder().add(value);
      return this;
    }

    public abstract Module build();
  }
}
