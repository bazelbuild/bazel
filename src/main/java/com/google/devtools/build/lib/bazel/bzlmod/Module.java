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
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Map;
import java.util.Optional;
import java.util.function.UnaryOperator;
import javax.annotation.Nullable;

/**
 * Represents a node in the external dependency graph.
 *
 * <p>In particular, it represents a specific version of a module; there can be multiple {@link
 * Module}s in a dependency graph with the same name but with different versions (such as after
 * discovery but before selection, or when there's a multiple_version_override in play).
 */
@AutoValue
public abstract class Module {

  /**
   * The name of the module, as specified in this module's MODULE.bazel file. Can be empty if this
   * is the root module.
   */
  public abstract String getName();

  /**
   * The version of the module, as specified in this module's MODULE.bazel file. Can be empty if
   * this is the root module, or if this module comes from a {@link NonRegistryOverride}.
   */
  public abstract Version getVersion();

  /**
   * The key of this module in the dependency graph. Note that, although a {@link ModuleKey} is also
   * just a (name, version) pair, its semantics differ from {@link #getName} and {@link
   * #getVersion}, which are always as specified in the MODULE.bazel file. The {@link ModuleKey}
   * returned by this method, however, will have the following special semantics:
   *
   * <ul>
   *   <li>The name of the {@link ModuleKey} is the same as {@link #getName}, unless this is the
   *       root module, in which case the name of the {@link ModuleKey} must be empty.
   *   <li>The version of the {@link ModuleKey} is the same as {@link #getVersion}, unless this is
   *       the root module OR this module has a {@link NonRegistryOverride}, in which case the
   *       version of the {@link ModuleKey} must be empty.
   * </ul>
   */
  public abstract ModuleKey getKey();

  public final RepositoryName getCanonicalRepoName() {
    return getKey().getCanonicalRepoName();
  }

  /**
   * The compatibility level of the module, which essentially signifies the "major version" of the
   * module in terms of SemVer.
   */
  public abstract int getCompatibilityLevel();

  /**
   * The name of the repository representing this module, as seen by the module itself. By default,
   * the name of the repo is the name of the module. This can be specified to ease migration for
   * projects that have been using a repo name for itself that differs from its module name.
   */
  public abstract String getRepoName();

  /** List of bazel compatible versions that would run/fail this module */
  public abstract ImmutableList<String> getBazelCompatibility();

  /**
   * Target patterns identifying execution platforms to register when this module is selected. Note
   * that these are what was written in module files verbatim, and don't contain canonical repo
   * names.
   */
  public abstract ImmutableList<String> getExecutionPlatformsToRegister();

  /**
   * Target patterns identifying toolchains to register when this module is selected. Note that
   * these are what was written in module files verbatim, and don't contain canonical repo names.
   */
  public abstract ImmutableList<String> getToolchainsToRegister();

  /**
   * The resolved direct dependencies of this module, which can be either the original ones,
   * overridden by a {@code single_version_override}, by a {@code multiple_version_override}, or by
   * a {@link NonRegistryOverride} (the version will be ""). The key type is the repo name of the
   * dep, and the value type is the ModuleKey (name+version) of the dep.
   */
  public abstract ImmutableMap<String, ModuleKey> getDeps();

  /**
   * The original direct dependencies of this module as they are declared in their MODULE file. The
   * key type is the repo name of the dep, and the value type is the ModuleKey (name+version) of the
   * dep.
   */
  public abstract ImmutableMap<String, ModuleKey> getOriginalDeps();

  /**
   * Returns a {@link RepositoryMapping} with only Bazel module repos and no repos from module
   * extensions. For the full mapping, see {@link BazelDepGraphValue#getFullRepoMapping}.
   */
  public final RepositoryMapping getRepoMappingWithBazelDepsOnly() {
    ImmutableMap.Builder<String, RepositoryName> mapping = ImmutableMap.builder();
    // If this is the root module, then the main repository should be visible as `@`.
    if (getKey().equals(ModuleKey.ROOT)) {
      mapping.put("", RepositoryName.MAIN);
    }
    // Every module should be able to reference itself as @<module repo name>.
    // If this is the root module, this perfectly falls into @<module repo name> => @
    if (!getRepoName().isEmpty()) {
      mapping.put(getRepoName(), getCanonicalRepoName());
    }
    for (Map.Entry<String, ModuleKey> dep : getDeps().entrySet()) {
      // Special note: if `dep` is actually the root module, its ModuleKey would be ROOT whose
      // canonicalRepoName is the empty string. This perfectly maps to the main repo ("@").
      mapping.put(dep.getKey(), dep.getValue().getCanonicalRepoName());
    }
    return RepositoryMapping.create(mapping.buildOrThrow(), getCanonicalRepoName());
  }

  /**
   * The registry where this module came from. Must be null iff the module has a {@link
   * NonRegistryOverride}.
   */
  @Nullable
  public abstract Registry getRegistry();

  /** The module extensions used in this module. */
  public abstract ImmutableList<ModuleExtensionUsage> getExtensionUsages();

  /** Returns a {@link Builder} that starts out with the same fields as this object. */
  abstract Builder toBuilder();

  /** Returns a new, empty {@link Builder}. */
  public static Builder builder() {
    return new AutoValue_Module.Builder()
        .setName("")
        .setVersion(Version.EMPTY)
        .setKey(ModuleKey.ROOT)
        .setCompatibilityLevel(0);
  }

  /**
   * Returns a new {@link Module} with all values in {@link #getDeps} transformed using the given
   * function.
   */
  public Module withDepKeysTransformed(UnaryOperator<ModuleKey> transform) {
    return toBuilder()
        .setDeps(ImmutableMap.copyOf(Maps.transformValues(getDeps(), transform::apply)))
        .build();
  }

  /** Builder type for {@link Module}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Optional; defaults to the empty string. */
    public abstract Builder setName(String value);

    /** Optional; defaults to {@link Version#EMPTY}. */
    public abstract Builder setVersion(Version value);

    /** Optional; defaults to {@link ModuleKey#ROOT}. */
    public abstract Builder setKey(ModuleKey value);

    /** Optional; defaults to {@code 0}. */
    public abstract Builder setCompatibilityLevel(int value);

    /** Optional; defaults to {@link #setName}. */
    public abstract Builder setRepoName(String value);

    abstract ImmutableList.Builder<String> bazelCompatibilityBuilder();

    @CanIgnoreReturnValue
    public final Builder addBazelCompatibilityValues(Iterable<String> values) {
      bazelCompatibilityBuilder().addAll(values);
      return this;
    }

    abstract ImmutableList.Builder<String> executionPlatformsToRegisterBuilder();

    @CanIgnoreReturnValue
    public final Builder addExecutionPlatformsToRegister(Iterable<String> values) {
      executionPlatformsToRegisterBuilder().addAll(values);
      return this;
    }

    abstract ImmutableList.Builder<String> toolchainsToRegisterBuilder();

    @CanIgnoreReturnValue
    public final Builder addToolchainsToRegister(Iterable<String> values) {
      toolchainsToRegisterBuilder().addAll(values);
      return this;
    }

    public abstract Builder setOriginalDeps(ImmutableMap<String, ModuleKey> value);

    public abstract Builder setDeps(ImmutableMap<String, ModuleKey> value);

    abstract ImmutableMap.Builder<String, ModuleKey> depsBuilder();

    @CanIgnoreReturnValue
    public Builder addDep(String depRepoName, ModuleKey depKey) {
      depsBuilder().put(depRepoName, depKey);
      return this;
    }

    abstract ImmutableMap.Builder<String, ModuleKey> originalDepsBuilder();

    @CanIgnoreReturnValue
    public Builder addOriginalDep(String depRepoName, ModuleKey depKey) {
      originalDepsBuilder().put(depRepoName, depKey);
      return this;
    }

    public abstract Builder setRegistry(Registry value);

    public abstract Builder setExtensionUsages(ImmutableList<ModuleExtensionUsage> value);

    abstract ImmutableList.Builder<ModuleExtensionUsage> extensionUsagesBuilder();

    @CanIgnoreReturnValue
    public Builder addExtensionUsage(ModuleExtensionUsage value) {
      extensionUsagesBuilder().add(value);
      return this;
    }

    abstract String getName();

    abstract Optional<String> getRepoName();

    abstract Module autoBuild();

    final Module build() {
      if (getRepoName().isEmpty()) {
        setRepoName(getName());
      }
      return autoBuild();
    }
  }
}
