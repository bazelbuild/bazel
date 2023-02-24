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
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import javax.annotation.Nullable;

/**
 * Represents a node in the unresolved external dependency graph.
 *
 * <p>In particular, it represents a specific version of a module; there can be multiple {@link
 * UnresolvedModule}s in a dependency graph with the same name but with different versions (such as
 * after discovery but before selection, or when there's a multiple_version_override in play).
 */
@AutoValue
public abstract class UnresolvedModule {

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

  /* TODO: Intermediary, unresolved deps */
  public abstract ImmutableMap<String, UnresolvedModuleKey> getUnresolvedDeps();

  /**
   * The original direct dependencies of this module as they are declared in their MODULE file. The
   * key type is the repo name of the dep, and the value type is the UnresolvedModuleKey
   * (name+version+max_compatibility_level) of the dep.
   */
  public abstract ImmutableMap<String, UnresolvedModuleKey> getOriginalDeps();

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
    return new AutoValue_UnresolvedModule.Builder()
        .setName("")
        .setVersion(Version.EMPTY)
        .setKey(ModuleKey.ROOT)
        .setCompatibilityLevel(0);
  }

  /**
   * Returns a new {@link UnresolvedModule} with all values in {@link #getUnresolvedDeps}
   * transformed using the given function.
   */
  public UnresolvedModule withUnresolvedDepKeysTransformed(
      UnaryOperator<UnresolvedModuleKey> transform) {
    return toBuilder()
        .setUnresolvedDeps(
            ImmutableMap.copyOf(Maps.transformValues(getUnresolvedDeps(), transform::apply)))
        .build();
  }

  /**
   * Returns a new {@link Module} with {@link #getDeps} set to the result of transforming all values
   * in {@link #getUnresolvedDeps} using the given function.
   */
  public Module withResolvedDepKeys(Function<UnresolvedModuleKey, ModuleKey> transform) {
    return Module.builder()
        .setName(getName())
        .setVersion(getVersion())
        .setKey(getKey())
        .setCompatibilityLevel(getCompatibilityLevel())
        .setRepoName(getRepoName())
        .addBazelCompatibilityValues(getBazelCompatibility())
        .addExecutionPlatformsToRegister(getExecutionPlatformsToRegister())
        .addToolchainsToRegister(getToolchainsToRegister())
        .setOriginalDeps(getOriginalDeps())
        .setRegistry(getRegistry())
        .setExtensionUsages(getExtensionUsages())
        .setDeps(ImmutableMap.copyOf(Maps.transformValues(getUnresolvedDeps(), transform::apply)))
        .build();
  }

  /** Builder type for {@link UnresolvedModule}. */
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

    public abstract Builder setUnresolvedDeps(ImmutableMap<String, UnresolvedModuleKey> value);

    abstract ImmutableMap.Builder<String, UnresolvedModuleKey> unresolvedDepsBuilder();

    @CanIgnoreReturnValue
    public Builder addUnresolvedDep(String depRepoName, UnresolvedModuleKey depKey) {
      unresolvedDepsBuilder().put(depRepoName, depKey);
      return this;
    }

    public abstract Builder setOriginalDeps(ImmutableMap<String, UnresolvedModuleKey> value);

    abstract ImmutableMap.Builder<String, UnresolvedModuleKey> originalDepsBuilder();

    @CanIgnoreReturnValue
    public Builder addOriginalDep(String depRepoName, UnresolvedModuleKey depKey) {
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

    abstract UnresolvedModule autoBuild();

    final UnresolvedModule build() {
      if (getRepoName().isEmpty()) {
        setRepoName(getName());
      }
      return autoBuild();
    }
  }
}
