// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.bazel.BazelVersion;
import com.google.devtools.build.lib.bazel.bzlmod.InterimModule.DepSpec;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.BazelCompatibilityMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Discovers the whole dependency graph and runs selection algorithm on it to produce the pruned
 * dependency graph and runs checks on it.
 */
public class BazelModuleResolutionFunction implements SkyFunction {

  public static final Precomputed<CheckDirectDepsMode> CHECK_DIRECT_DEPENDENCIES =
      new Precomputed<>("check_direct_dependency");
  public static final Precomputed<BazelCompatibilityMode> BAZEL_COMPATIBILITY_MODE =
      new Precomputed<>("bazel_compatibility_mode");
  public static final Precomputed<List<String>> ALLOWED_YANKED_VERSIONS =
      new Precomputed<>("allowed_yanked_versions");

  public static final String BZLMOD_ALLOWED_YANKED_VERSIONS_ENV = "BZLMOD_ALLOW_YANKED_VERSIONS";

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {

    ClientEnvironmentValue allowedYankedVersionsFromEnv =
        (ClientEnvironmentValue)
            env.getValue(ClientEnvironmentFunction.key(BZLMOD_ALLOWED_YANKED_VERSIONS_ENV));
    if (allowedYankedVersionsFromEnv == null) {
      return null;
    }
    RootModuleFileValue root =
        (RootModuleFileValue) env.getValue(ModuleFileValue.KEY_FOR_ROOT_MODULE);
    if (root == null) {
      return null;
    }
    ImmutableMap<ModuleKey, InterimModule> initialDepGraph = Discovery.run(env, root);
    if (initialDepGraph == null) {
      return null;
    }

    Selection.Result selectionResult;
    try {
      selectionResult = Selection.run(initialDepGraph, root.getOverrides());
    } catch (ExternalDepsException e) {
      throw new BazelModuleResolutionFunctionException(e, Transience.PERSISTENT);
    }
    ImmutableMap<ModuleKey, InterimModule> resolvedDepGraph = selectionResult.getResolvedDepGraph();

    verifyRootModuleDirectDepsAreAccurate(
        initialDepGraph.get(ModuleKey.ROOT),
        resolvedDepGraph.get(ModuleKey.ROOT),
        Objects.requireNonNull(CHECK_DIRECT_DEPENDENCIES.get(env)),
        env.getListener());

    checkBazelCompatibility(
        resolvedDepGraph.values(),
        Objects.requireNonNull(BAZEL_COMPATIBILITY_MODE.get(env)),
        env.getListener());

    verifyYankedVersions(
        resolvedDepGraph,
        parseYankedVersions(
            allowedYankedVersionsFromEnv.getValue(),
            Objects.requireNonNull(ALLOWED_YANKED_VERSIONS.get(env))),
        env.getListener());

    ImmutableMap<ModuleKey, Module> finalDepGraph =
        computeFinalDepGraph(resolvedDepGraph, root.getOverrides(), env.getListener());

    return BazelModuleResolutionValue.create(finalDepGraph, selectionResult.getUnprunedDepGraph());
  }

  private static void verifyRootModuleDirectDepsAreAccurate(
      InterimModule discoveredRootModule,
      InterimModule resolvedRootModule,
      CheckDirectDepsMode mode,
      EventHandler eventHandler)
      throws BazelModuleResolutionFunctionException {
    if (mode == CheckDirectDepsMode.OFF) {
      return;
    }

    boolean failure = false;
    for (Map.Entry<String, DepSpec> dep : discoveredRootModule.getDeps().entrySet()) {
      ModuleKey resolved = resolvedRootModule.getDeps().get(dep.getKey()).toModuleKey();
      if (!dep.getValue().toModuleKey().equals(resolved)) {
        String message =
            String.format(
                "For repository '%s', the root module requires module version %s, but got %s in the"
                    + " resolved dependency graph.",
                dep.getKey(), dep.getValue().toModuleKey(), resolved);
        if (mode == CheckDirectDepsMode.WARNING) {
          eventHandler.handle(Event.warn(message));
        } else {
          eventHandler.handle(Event.error(message));
          failure = true;
        }
      }
    }

    if (failure) {
      throw new BazelModuleResolutionFunctionException(
          ExternalDepsException.withMessage(
              Code.VERSION_RESOLUTION_ERROR, "Direct dependency check failed."),
          Transience.PERSISTENT);
    }
  }

  public static void checkBazelCompatibility(
      ImmutableCollection<InterimModule> modules,
      BazelCompatibilityMode mode,
      EventHandler eventHandler)
      throws BazelModuleResolutionFunctionException {
    if (mode == BazelCompatibilityMode.OFF) {
      return;
    }

    String currentBazelVersion = BlazeVersionInfo.instance().getVersion();
    if (Strings.isNullOrEmpty(currentBazelVersion)) {
      return;
    }

    BazelVersion curVersion = BazelVersion.parse(currentBazelVersion);
    for (InterimModule module : modules) {
      for (String compatVersion : module.getBazelCompatibility()) {
        if (!curVersion.satisfiesCompatibility(compatVersion)) {
          String message =
              String.format(
                  "Bazel version %s is not compatible with module \"%s@%s\" (bazel_compatibility:"
                      + " %s)",
                  curVersion.getOriginal(),
                  module.getName(),
                  module.getVersion().getOriginal(),
                  module.getBazelCompatibility());

          if (mode == BazelCompatibilityMode.WARNING) {
            eventHandler.handle(Event.warn(message));
          } else {
            eventHandler.handle(Event.error(message));
            throw new BazelModuleResolutionFunctionException(
                ExternalDepsException.withMessage(
                    Code.VERSION_RESOLUTION_ERROR, "Bazel compatibility check failed"),
                Transience.PERSISTENT);
          }
        }
      }
    }
  }

  /**
   * Parse a set of allowed yanked version from command line flag (--allowed_yanked_versions) and
   * environment variable (ALLOWED_YANKED_VERSIONS). If `all` is specified, return Optional.empty();
   * otherwise returns the set of parsed modulel key.
   */
  private Optional<ImmutableSet<ModuleKey>> parseYankedVersions(
      String allowedYankedVersionsFromEnv, List<String> allowedYankedVersionsFromFlag)
      throws BazelModuleResolutionFunctionException {
    ImmutableSet.Builder<ModuleKey> allowedYankedVersionBuilder = new ImmutableSet.Builder<>();
    if (allowedYankedVersionsFromEnv != null) {
      if (parseModuleKeysFromString(
          allowedYankedVersionsFromEnv,
          allowedYankedVersionBuilder,
          String.format(
              "envirnoment variable %s=%s",
              BZLMOD_ALLOWED_YANKED_VERSIONS_ENV, allowedYankedVersionsFromEnv))) {
        return Optional.empty();
      }
    }
    for (String allowedYankedVersions : allowedYankedVersionsFromFlag) {
      if (parseModuleKeysFromString(
          allowedYankedVersions,
          allowedYankedVersionBuilder,
          String.format("command line flag --allow_yanked_versions=%s", allowedYankedVersions))) {
        return Optional.empty();
      }
    }
    return Optional.of(allowedYankedVersionBuilder.build());
  }

  /**
   * Parse of a comma-separated list of module version(s) of the form '<module name>@<version>' or
   * 'all' from the string. Returns true if 'all' is present, otherwise returns false.
   */
  private boolean parseModuleKeysFromString(
      String input, ImmutableSet.Builder<ModuleKey> allowedYankedVersionBuilder, String context)
      throws BazelModuleResolutionFunctionException {
    ImmutableList<String> moduleStrs = ImmutableList.copyOf(Splitter.on(',').split(input));

    for (String moduleStr : moduleStrs) {
      if (moduleStr.equals("all")) {
        return true;
      }

      if (moduleStr.isEmpty()) {
        continue;
      }

      String[] pieces = moduleStr.split("@", 2);

      if (pieces.length != 2) {
        throw new BazelModuleResolutionFunctionException(
            ExternalDepsException.withMessage(
                Code.VERSION_RESOLUTION_ERROR,
                "Parsing %s failed, module versions must be of the form '<module name>@<version>'",
                context),
            Transience.PERSISTENT);
      }

      if (!RepositoryName.VALID_MODULE_NAME.matcher(pieces[0]).matches()) {
        throw new BazelModuleResolutionFunctionException(
            ExternalDepsException.withMessage(
                Code.VERSION_RESOLUTION_ERROR,
                "Parsing %s failed, invalid module name '%s': valid names must 1) only contain"
                    + " lowercase letters (a-z), digits (0-9), dots (.), hyphens (-), and"
                    + " underscores (_); 2) begin with a lowercase letter; 3) end with a lowercase"
                    + " letter or digit.",
                context,
                pieces[0]),
            Transience.PERSISTENT);
      }

      Version version;
      try {
        version = Version.parse(pieces[1]);
      } catch (ParseException e) {
        throw new BazelModuleResolutionFunctionException(
            ExternalDepsException.withCauseAndMessage(
                Code.VERSION_RESOLUTION_ERROR,
                e,
                "Parsing %s failed, invalid version specified for module: %s",
                context,
                pieces[1]),
            Transience.PERSISTENT);
      }

      allowedYankedVersionBuilder.add(ModuleKey.create(pieces[0], version));
    }
    return false;
  }

  private static void verifyYankedVersions(
      ImmutableMap<ModuleKey, InterimModule> depGraph,
      Optional<ImmutableSet<ModuleKey>> allowedYankedVersions,
      ExtendedEventHandler eventHandler)
      throws BazelModuleResolutionFunctionException, InterruptedException {
    // Check whether all resolved modules are either not yanked or allowed. Modules with a
    // NonRegistryOverride are ignored as their metadata is not available whatsoever.
    for (InterimModule m : depGraph.values()) {
      if (m.getKey().equals(ModuleKey.ROOT) || m.getRegistry() == null) {
        continue;
      }
      Optional<ImmutableMap<Version, String>> yankedVersions;
      try {
        yankedVersions = m.getRegistry().getYankedVersions(m.getKey().getName(), eventHandler);
      } catch (IOException e) {
        eventHandler.handle(
            Event.warn(
                String.format(
                    "Could not read metadata file for module %s: %s", m.getKey(), e.getMessage())));
        continue;
      }
      if (yankedVersions.isEmpty()) {
        continue;
      }
      String yankedInfo = yankedVersions.get().get(m.getVersion());
      if (yankedInfo != null
          && allowedYankedVersions.isPresent()
          && !allowedYankedVersions.get().contains(m.getKey())) {
        throw new BazelModuleResolutionFunctionException(
            ExternalDepsException.withMessage(
                Code.VERSION_RESOLUTION_ERROR,
                "Yanked version detected in your resolved dependency graph: %s, for the reason: "
                    + "%s.\nYanked versions may contain serious vulnerabilities and should not be "
                    + "used. To fix this, use a bazel_dep on a newer version of this module. To "
                    + "continue using this version, allow it using the --allow_yanked_versions "
                    + "flag or the BZLMOD_ALLOW_YANKED_VERSIONS env variable.",
                m.getKey(),
                yankedInfo),
            Transience.PERSISTENT);
      }
    }
  }

  private static RepoSpec maybeAppendAdditionalPatches(RepoSpec repoSpec, ModuleOverride override) {
    if (!(override instanceof SingleVersionOverride)) {
      return repoSpec;
    }
    SingleVersionOverride singleVersion = (SingleVersionOverride) override;
    if (singleVersion.getPatches().isEmpty()) {
      return repoSpec;
    }
    ImmutableMap.Builder<String, Object> attrBuilder = ImmutableMap.builder();
    attrBuilder.putAll(repoSpec.attributes().attributes());
    attrBuilder.put("patches", singleVersion.getPatches());
    attrBuilder.put("patch_cmds", singleVersion.getPatchCmds());
    attrBuilder.put("patch_args", ImmutableList.of("-p" + singleVersion.getPatchStrip()));
    return RepoSpec.builder()
        .setBzlFile(repoSpec.bzlFile())
        .setRuleClassName(repoSpec.ruleClassName())
        .setAttributes(AttributeValues.create(attrBuilder.buildOrThrow()))
        .build();
  }

  @Nullable
  private static RepoSpec computeRepoSpec(
      InterimModule interimModule, ModuleOverride override, ExtendedEventHandler eventHandler)
      throws BazelModuleResolutionFunctionException, InterruptedException {
    if (interimModule.getRegistry() == null) {
      // This module has a non-registry override. We don't need to store the repo spec in this case.
      return null;
    }
    try {
      RepoSpec moduleRepoSpec =
          interimModule
              .getRegistry()
              .getRepoSpec(
                  interimModule.getKey(), interimModule.getCanonicalRepoName(), eventHandler);
      return maybeAppendAdditionalPatches(moduleRepoSpec, override);
    } catch (IOException e) {
      throw new BazelModuleResolutionFunctionException(
          ExternalDepsException.withMessage(
              Code.ERROR_ACCESSING_REGISTRY,
              "Unable to get module repo spec from registry: %s",
              e.getMessage()),
          Transience.PERSISTENT);
    }
  }

  /**
   * Builds a {@link Module} from an {@link InterimModule}, discarding unnecessary fields and adding
   * extra necessary ones (such as the repo spec).
   */
  static Module moduleFromInterimModule(
      InterimModule interim, ModuleOverride override, ExtendedEventHandler eventHandler)
      throws BazelModuleResolutionFunctionException, InterruptedException {
    return Module.builder()
        .setName(interim.getName())
        .setVersion(interim.getVersion())
        .setKey(interim.getKey())
        .setRepoName(interim.getRepoName())
        .setExecutionPlatformsToRegister(interim.getExecutionPlatformsToRegister())
        .setToolchainsToRegister(interim.getToolchainsToRegister())
        .setDeps(ImmutableMap.copyOf(Maps.transformValues(interim.getDeps(), DepSpec::toModuleKey)))
        .setRepoSpec(computeRepoSpec(interim, override, eventHandler))
        .setExtensionUsages(interim.getExtensionUsages())
        .build();
  }

  private static ImmutableMap<ModuleKey, Module> computeFinalDepGraph(
      ImmutableMap<ModuleKey, InterimModule> resolvedDepGraph,
      ImmutableMap<String, ModuleOverride> overrides,
      ExtendedEventHandler eventHandler)
      throws BazelModuleResolutionFunctionException, InterruptedException {
    ImmutableMap.Builder<ModuleKey, Module> finalDepGraph = ImmutableMap.builder();
    for (Map.Entry<ModuleKey, InterimModule> entry : resolvedDepGraph.entrySet()) {
      finalDepGraph.put(
          entry.getKey(),
          moduleFromInterimModule(
              entry.getValue(), overrides.get(entry.getKey().getName()), eventHandler));
    }
    return finalDepGraph.buildOrThrow();
  }

  static class BazelModuleResolutionFunctionException extends SkyFunctionException {
    BazelModuleResolutionFunctionException(ExternalDepsException e, Transience transience) {
      super(e, transience);
    }
  }
}
