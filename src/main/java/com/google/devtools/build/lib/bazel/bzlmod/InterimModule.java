// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static java.util.Objects.requireNonNull;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Optional;
import java.util.function.UnaryOperator;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.StarlarkList;

/**
 * Represents a node in the external dependency graph during module resolution (discovery &
 * selection).
 *
 * <p>In particular, it represents a specific version of a module; there can be multiple {@link
 * InterimModule}s in a dependency graph with the same name but with different versions (such as
 * after discovery but before selection, or when there's a multiple_version_override in play).
 *
 * <p>Compared to {@link Module}, which is used after module resolution, this class holds some more
 * information that's useful only during resolution, such as the {@code max_compatibility_level} for
 * each dep, the {@code compatibility_level}, the {@code registry} the module comes from, etc.
 */
@AutoValue
public abstract class InterimModule extends ModuleBase {

  /**
   * The compatibility level of the module, which essentially signifies the "major version" of the
   * module in terms of SemVer.
   */
  public abstract int getCompatibilityLevel();

  /** List of bazel compatible versions that would run/fail this module */
  public abstract ImmutableList<String> getBazelCompatibility();

  /** The specification of a dependency. */
  @AutoCodec
  public record DepSpec(String name, Version version, int maxCompatibilityLevel) {
    public DepSpec {
      requireNonNull(name, "name");
      requireNonNull(version, "version");
    }

    public static DepSpec create(String name, Version version, int maxCompatibilityLevel) {
      return new DepSpec(name, version, maxCompatibilityLevel);
    }

    public static DepSpec fromModuleKey(ModuleKey key) {
      return create(key.name(), key.version(), -1);
    }

    public final ModuleKey toModuleKey() {
      return new ModuleKey(name(), version());
    }
  }

  /**
   * The resolved direct dependencies of this module, which can be either the original ones,
   * overridden by a {@code single_version_override}, by a {@code multiple_version_override}, or by
   * a {@link NonRegistryOverride} (the version will be ""). The key type is the repo name of the
   * dep.
   */
  public abstract ImmutableMap<String, DepSpec> getDeps();

  /**
   * The original direct dependencies of this module as they are declared in their MODULE file. The
   * key type is the repo name of the dep.
   */
  public abstract ImmutableMap<String, DepSpec> getOriginalDeps();

  /**
   * The "nodep" dependencies of this module: these don't actually add a dependency on the specified
   * module, but if specified module is somehow in the dependency graph, it'll be at least at this
   * version.
   */
  public abstract ImmutableList<DepSpec> getNodepDeps();

  /**
   * The registry where this module came from. Must be null iff the module has a {@link
   * NonRegistryOverride}.
   */
  @Nullable
  public abstract Registry getRegistry();

  /** Returns a {@link Builder} that starts out with the same fields as this object. */
  abstract Builder toBuilder();

  /** Returns a new, empty {@link Builder}. */
  public static Builder builder() {
    return new AutoValue_InterimModule.Builder()
        .setName("")
        .setVersion(Version.EMPTY)
        .setKey(ModuleKey.ROOT)
        .setCompatibilityLevel(0);
  }

  /**
   * Returns a new {@link InterimModule} with all values in {@link #getDeps} transformed using the
   * given function.
   */
  public InterimModule withDepsTransformed(UnaryOperator<DepSpec> transform) {
    return toBuilder()
        .setDeps(ImmutableMap.copyOf(Maps.transformValues(getDeps(), transform::apply)))
        .build();
  }

  /**
   * Returns a new {@link InterimModule} with all values in {@link #getDeps} and {@link
   * #getNodepDeps} transformed using the given function.
   */
  public InterimModule withDepsAndNodepDepsTransformed(UnaryOperator<DepSpec> transform) {
    return toBuilder()
        .setDeps(ImmutableMap.copyOf(Maps.transformValues(getDeps(), transform::apply)))
        .setNodepDeps(ImmutableList.copyOf(Lists.transform(getNodepDeps(), transform::apply)))
        .build();
  }

  /** Builder type for {@link InterimModule}. */
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

    public abstract Builder setOriginalDeps(ImmutableMap<String, DepSpec> value);

    public abstract Builder setDeps(ImmutableMap<String, DepSpec> value);

    abstract ImmutableList.Builder<DepSpec> nodepDepsBuilder();

    @CanIgnoreReturnValue
    public final Builder addNodepDep(DepSpec value) {
      nodepDepsBuilder().add(value);
      return this;
    }

    public abstract Builder setNodepDeps(ImmutableList<DepSpec> value);

    public abstract Builder setRegistry(Registry value);

    public abstract Builder setExtensionUsages(ImmutableList<ModuleExtensionUsage> value);

    abstract ImmutableList.Builder<ModuleExtensionUsage> extensionUsagesBuilder();

    @CanIgnoreReturnValue
    public Builder addExtensionUsage(ModuleExtensionUsage value) {
      extensionUsagesBuilder().add(value);
      return this;
    }

    abstract ModuleKey getKey();

    abstract String getName();

    abstract Optional<String> getRepoName();

    abstract InterimModule autoBuild();

    final InterimModule build() {
      if (getRepoName().isEmpty()) {
        setRepoName(getName());
      }
      return autoBuild();
    }
  }

  /**
   * Builds a {@link Module} from an {@link InterimModule}, discarding unnecessary fields and adding
   * extra necessary ones (such as the repo spec).
   *
   * @param remoteRepoSpec the {@link RepoSpec} for the module obtained from a registry or null if
   *     the module has a non-registry override
   */
  static Module toModule(
      InterimModule interim, @Nullable ModuleOverride override, @Nullable RepoSpec remoteRepoSpec) {
    return Module.builder()
        .setName(interim.getName())
        .setVersion(interim.getVersion())
        .setKey(interim.getKey())
        .setRepoName(interim.getRepoName())
        .setExecutionPlatformsToRegister(interim.getExecutionPlatformsToRegister())
        .setToolchainsToRegister(interim.getToolchainsToRegister())
        .setDeps(ImmutableMap.copyOf(Maps.transformValues(interim.getDeps(), DepSpec::toModuleKey)))
        .setRepoSpec(maybeAppendAdditionalPatches(remoteRepoSpec, override))
        .setExtensionUsages(interim.getExtensionUsages())
        .build();
  }

  private static RepoSpec maybeAppendAdditionalPatches(
      @Nullable RepoSpec repoSpec, @Nullable ModuleOverride override) {
    if (!(override instanceof SingleVersionOverride singleVersion)) {
      return repoSpec;
    }
    if (singleVersion.patches().isEmpty()) {
      return repoSpec;
    }
    Dict.Builder<String, Object> attrBuilder = Dict.builder();
    attrBuilder.putAll(repoSpec.attributes().attributes());
    attrBuilder.put("patches", StarlarkList.immutableCopyOf(singleVersion.patches()));
    attrBuilder.put("patch_cmds", StarlarkList.immutableCopyOf(singleVersion.patchCmds()));
    attrBuilder.put("patch_args", StarlarkList.immutableOf("-p" + singleVersion.patchStrip()));
    return new RepoSpec(
        repoSpec.repoRuleId(), AttributeValues.create(attrBuilder.buildImmutable()));
  }
}
