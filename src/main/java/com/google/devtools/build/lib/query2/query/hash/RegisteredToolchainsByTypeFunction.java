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
package com.google.devtools.build.lib.query2.query.hash;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.Module;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.TargetPatternUtil;
import com.google.devtools.build.lib.skyframe.TargetPatternUtil.InvalidTargetPatternException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Groups the toolchains registered with {@code register_toolchains} in {@code MODULE.bazel} files
 * by the {@code toolchain_type} they implement, for {@code query --generate_hashes}.
 *
 * <p>Toolchain resolution needs a configuration, but finding the registered {@code toolchain()}
 * targets and their (non-configurable) {@code toolchain_type} only needs loading. {@link
 * TransitiveHashes} uses this to make every {@code toolchain_type} depend on all toolchains
 * registered for it, so that a change to any of them reaches the targets that may use it.
 *
 * <p>Toolchains added with {@code --extra_toolchains} are not included: they are part of the
 * configuration.
 */
public final class RegisteredToolchainsByTypeFunction implements SkyFunction {
  public static final SkyFunctionName NAME =
      SkyFunctionName.createHermetic("QUERY_REGISTERED_TOOLCHAINS_BY_TYPE");

  /** The only key of this function. */
  public static final SkyKey KEY = new Key();

  private static final class Key implements SkyKey {
    @Override
    public SkyFunctionName functionName() {
      return NAME;
    }
  }

  /** The registered toolchains, grouped by toolchain type. */
  public static final class Value implements SkyValue {
    private final ImmutableSetMultimap<Label, Label> toolchainsByType;
    private final ImmutableSortedSet<Label> toolchainsOfUnknownType;

    Value(
        ImmutableSetMultimap<Label, Label> toolchainsByType,
        ImmutableSortedSet<Label> toolchainsOfUnknownType) {
      this.toolchainsByType = toolchainsByType;
      this.toolchainsOfUnknownType = toolchainsOfUnknownType;
    }

    /**
     * Returns the toolchains that may implement {@code toolchainType}: those registered for it,
     * plus those whose type could not be determined because they failed to load.
     */
    public ImmutableSet<Label> getToolchains(Label toolchainType) {
      ImmutableSet<Label> toolchains = toolchainsByType.get(toolchainType);
      if (toolchainsOfUnknownType.isEmpty()) {
        return toolchains;
      }
      return ImmutableSet.<Label>builder()
          .addAll(toolchains)
          .addAll(toolchainsOfUnknownType)
          .build();
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof Value other
          && toolchainsByType.equals(other.toolchainsByType)
          && toolchainsOfUnknownType.equals(other.toolchainsOfUnknownType);
    }

    @Override
    public int hashCode() {
      return 31 * toolchainsByType.hashCode() + toolchainsOfUnknownType.hashCode();
    }
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws RegisteredToolchainsByTypeFunctionException, InterruptedException {
    ImmutableList<TargetPattern> patterns = getBzlmodToolchainPatterns(env);
    if (patterns == null) {
      return null;
    }
    ImmutableSet<Label> toolchainLabels;
    try {
      toolchainLabels =
          TargetPatternUtil.expandTargetPatterns(
              env,
              TargetPatternUtil.toSigned(patterns),
              FilteringPolicies.ruleTypeExplicit("toolchain"));
    } catch (InvalidTargetPatternException e) {
      throw new RegisteredToolchainsByTypeFunctionException(e);
    }
    if (toolchainLabels == null) {
      return null;
    }

    // The packages were loaded to expand the patterns, so these lookups don't load anything new.
    ImmutableSet<PackageIdentifier> packageIds =
        toolchainLabels.stream().map(Label::getPackageIdentifier).collect(toImmutableSet());
    SkyframeLookupResult packages = env.getValuesAndExceptions(packageIds);
    if (env.valuesMissing()) {
      return null;
    }
    ImmutableSetMultimap.Builder<Label, Label> toolchainsByType =
        ImmutableSetMultimap.<Label, Label>builder()
            .orderKeysBy(Ordering.natural())
            .orderValuesBy(Ordering.natural());
    ImmutableSortedSet.Builder<Label> toolchainsOfUnknownType = ImmutableSortedSet.naturalOrder();
    for (Label toolchain : toolchainLabels) {
      Label toolchainType = null;
      try {
        PackageValue packageValue =
            (PackageValue)
                packages.getOrThrow(toolchain.getPackageIdentifier(), NoSuchPackageException.class);
        if (packageValue == null) {
          return null;
        }
        Package pkg = packageValue.getPackage();
        Target target = pkg.getTarget(toolchain.getName());
        if (target instanceof Rule rule) {
          toolchainType =
              NonconfigurableAttributeMapper.of(rule).get("toolchain_type", BuildType.LABEL);
        }
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        // Fall through: a toolchain that can't be read conservatively counts for every type.
      }
      if (toolchainType != null) {
        toolchainsByType.put(toolchainType, toolchain);
      } else {
        toolchainsOfUnknownType.add(toolchain);
      }
    }
    return new Value(toolchainsByType.build(), toolchainsOfUnknownType.build());
  }

  /**
   * Returns the toolchain patterns registered by all modules, like {@code
   * RegisteredToolchainsFunction} does, or null if a Skyframe value is missing.
   */
  @Nullable
  private static ImmutableList<TargetPattern> getBzlmodToolchainPatterns(Environment env)
      throws InterruptedException, RegisteredToolchainsByTypeFunctionException {
    BazelDepGraphValue bazelDepGraphValue =
        (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
    if (bazelDepGraphValue == null) {
      return null;
    }
    ImmutableList.Builder<TargetPattern> patterns = ImmutableList.builder();
    for (Module module : bazelDepGraphValue.getDepGraph().values()) {
      List<String> toolchains = module.getToolchainsToRegister();
      if (toolchains.isEmpty()) {
        continue;
      }
      RepositoryName repoName =
          bazelDepGraphValue.getCanonicalRepoNameLookup().inverse().get(module.getKey());
      RepositoryMappingValue repoMapping =
          (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(repoName));
      if (repoMapping == null) {
        continue;
      }
      TargetPattern.Parser parser =
          new TargetPattern.Parser(
              PathFragment.EMPTY_FRAGMENT, repoName, repoMapping.repositoryMapping());
      for (String pattern : toolchains) {
        try {
          patterns.add(parser.parse(pattern));
        } catch (TargetParsingException e) {
          throw new RegisteredToolchainsByTypeFunctionException(e);
        }
      }
    }
    return env.valuesMissing() ? null : patterns.build();
  }

  private static final class RegisteredToolchainsByTypeFunctionException
      extends SkyFunctionException {
    RegisteredToolchainsByTypeFunctionException(Exception e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
