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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.bazel.BazelVersion;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.bazel.bzlmod.Selection.SelectionResult;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.BazelCompatibilityMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.lib.vfs.PathFragment;
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
 * Runs Bazel module resolution. This function produces the dependency graph containing all Bazel
 * modules, along with a few lookup maps that help with further usage. By this stage, module
 * extensions are not evaluated yet.
 */
public class BazelModuleResolutionFunction implements SkyFunction {

  public static final Precomputed<CheckDirectDepsMode> CHECK_DIRECT_DEPENDENCIES =
      new Precomputed<>("check_direct_dependency");
  public static final Precomputed<BazelCompatibilityMode> BAZEL_COMPATIBILITY_MODE =
      new Precomputed<>("bazel_compatibility_mode");

  public static final Precomputed<List<String>> ALLOWED_YANKED_VERSIONS =
      new Precomputed<>("allowed_yanked_versions");
  private static final String BZLMOD_ALLOWED_YANKED_VERSIONS_ENV = "BZLMOD_ALLOW_YANKED_VERSIONS";

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
    ImmutableMap<ModuleKey, Module> initialDepGraph = Discovery.run(env, root);
    if (initialDepGraph == null) {
      return null;
    }
    ImmutableMap<String, ModuleOverride> overrides = root.getOverrides();
    SelectionResult selectionResult;
    try {
      selectionResult = Selection.run(initialDepGraph, overrides);
    } catch (ExternalDepsException e) {
      throw new BazelModuleResolutionFunctionException(e, Transience.PERSISTENT);
    }
    ImmutableMap<ModuleKey, Module> resolvedDepGraph = selectionResult.getResolvedDepGraph();

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
    verifyRootModuleDirectDepsAreAccurate(
        env, initialDepGraph.get(ModuleKey.ROOT), resolvedDepGraph.get(ModuleKey.ROOT));
    return createValue(resolvedDepGraph, selectionResult.getUnprunedDepGraph(), overrides);
  }

  public static void checkBazelCompatibility(
      ImmutableCollection<Module> modules, BazelCompatibilityMode mode, EventHandler eventHandler)
      throws BazelModuleResolutionFunctionException {
    if (mode == BazelCompatibilityMode.OFF) {
      return;
    }

    String currentBazelVersion = BlazeVersionInfo.instance().getVersion();
    if (Strings.isNullOrEmpty(currentBazelVersion)) {
      return;
    }

    BazelVersion curVersion = BazelVersion.parse(currentBazelVersion);
    for (Module module : modules) {
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

  private void verifyYankedVersions(
      ImmutableMap<ModuleKey, Module> depGraph,
      Optional<ImmutableSet<ModuleKey>> allowedYankedVersions,
      ExtendedEventHandler eventHandler)
      throws BazelModuleResolutionFunctionException, InterruptedException {
    // Check whether all resolved modules are either not yanked or allowed. Modules with a
    // NonRegistryOverride are ignored as their metadata is not available whatsoever.
    for (Module m : depGraph.values()) {
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

  private static void verifyRootModuleDirectDepsAreAccurate(
      Environment env, Module discoveredRootModule, Module resolvedRootModule)
      throws InterruptedException, BazelModuleResolutionFunctionException {
    CheckDirectDepsMode mode = Objects.requireNonNull(CHECK_DIRECT_DEPENDENCIES.get(env));
    if (mode == CheckDirectDepsMode.OFF) {
      return;
    }
    boolean failure = false;
    for (Map.Entry<String, ModuleKey> dep : discoveredRootModule.getDeps().entrySet()) {
      ModuleKey resolved = resolvedRootModule.getDeps().get(dep.getKey());
      if (!dep.getValue().equals(resolved)) {
        String message =
            String.format(
                "For repository '%s', the root module requires module version %s, but got %s in the"
                    + " resolved dependency graph.",
                dep.getKey(), dep.getValue(), resolved);
        if (mode == CheckDirectDepsMode.WARNING) {
          env.getListener().handle(Event.warn(message));
        } else {
          env.getListener().handle(Event.error(message));
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

  @VisibleForTesting
  static BazelModuleResolutionValue createValue(
      ImmutableMap<ModuleKey, Module> depGraph,
      ImmutableMap<ModuleKey, Module> unprunedDepGraph,
      ImmutableMap<String, ModuleOverride> overrides)
      throws BazelModuleResolutionFunctionException {
    // Build some reverse lookups for later use.
    ImmutableMap<RepositoryName, ModuleKey> canonicalRepoNameLookup =
        depGraph.keySet().stream()
            .collect(toImmutableMap(ModuleKey::getCanonicalRepoName, key -> key));
    ImmutableMap<String, ModuleKey> moduleNameLookup =
        depGraph.keySet().stream()
            // The root module is not meaningfully used by this lookup so we skip it (it's
            // guaranteed to be the first in iteration order).
            .skip(1)
            .filter(key -> !(overrides.get(key.getName()) instanceof MultipleVersionOverride))
            .collect(toImmutableMap(ModuleKey::getName, key -> key));

    // For each extension usage, we resolve (i.e. canonicalize) its bzl file label. Then we can
    // group all usages by the label + name (the ModuleExtensionId).
    ImmutableTable.Builder<ModuleExtensionId, ModuleKey, ModuleExtensionUsage>
        extensionUsagesTableBuilder = ImmutableTable.builder();
    for (Module module : depGraph.values()) {
      LabelConverter labelConverter =
          new LabelConverter(
              PackageIdentifier.create(module.getCanonicalRepoName(), PathFragment.EMPTY_FRAGMENT),
              module.getRepoMappingWithBazelDepsOnly());
      for (ModuleExtensionUsage usage : module.getExtensionUsages()) {
        try {
          ModuleExtensionId moduleExtensionId =
              ModuleExtensionId.create(
                  labelConverter.convert(usage.getExtensionBzlFile()), usage.getExtensionName());
          extensionUsagesTableBuilder.put(moduleExtensionId, module.getKey(), usage);
        } catch (LabelSyntaxException e) {
          throw new BazelModuleResolutionFunctionException(
              ExternalDepsException.withCauseAndMessage(
                  Code.BAD_MODULE,
                  e,
                  "invalid label for module extension found at %s",
                  usage.getLocation()),
              Transience.PERSISTENT);
        }
      }
    }
    ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage> extensionUsagesById =
        extensionUsagesTableBuilder.buildOrThrow();

    // Calculate a unique name for each used extension id.
    BiMap<String, ModuleExtensionId> extensionUniqueNames = HashBiMap.create();
    for (ModuleExtensionId id : extensionUsagesById.rowKeySet()) {
      String bestName =
          id.getBzlFileLabel().getRepository().getName() + "~" + id.getExtensionName();
      if (extensionUniqueNames.putIfAbsent(bestName, id) == null) {
        continue;
      }
      int suffix = 2;
      while (extensionUniqueNames.putIfAbsent(bestName + suffix, id) != null) {
        suffix++;
      }
    }

    return BazelModuleResolutionValue.create(
        depGraph,
        unprunedDepGraph,
        canonicalRepoNameLookup,
        moduleNameLookup,
        depGraph.values().stream().map(AbridgedModule::from).collect(toImmutableList()),
        extensionUsagesById,
        ImmutableMap.copyOf(extensionUniqueNames.inverse()));
  }

  static class BazelModuleResolutionFunctionException extends SkyFunctionException {
    BazelModuleResolutionFunctionException(ExternalDepsException e, Transience transience) {
      super(e, transience);
    }
  }
}
