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

import static com.google.common.base.Strings.nullToEmpty;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.Map.Entry;
import javax.annotation.Nullable;

/**
 * This function runs Bazel module resolution, extracts the dependency graph from it and creates a
 * value containing all Bazel modules, along with a few lookup maps that help with further usage. By
 * this stage, module extensions are not evaluated yet.
 */
public class BazelDepGraphFunction implements SkyFunction {

  public BazelDepGraphFunction() {}

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws BazelDepGraphFunctionException, InterruptedException {
    RootModuleFileValue root =
        (RootModuleFileValue) env.getValue(ModuleFileValue.KEY_FOR_ROOT_MODULE);
    if (root == null) {
      return null;
    }
    LockfileMode lockfileMode = BazelLockFileFunction.LOCKFILE_MODE.get(env);

    ImmutableMap<String, String> localOverrideHashes = null;
    ImmutableMap<ModuleKey, Module> depGraph = null;
    BzlmodFlagsAndEnvVars flags = null;
    BazelLockFileValue lockfile = null;

    // If the module has not changed (has the same contents and flags as the lockfile),
    // read the dependency graph from the lock file, else run resolution and update lockfile
    if (!lockfileMode.equals(LockfileMode.OFF)) {
      lockfile = (BazelLockFileValue) env.getValue(BazelLockFileValue.KEY);
      if (lockfile == null) {
        return null;
      }
      flags = getFlagsAndEnvVars(env);
      if (flags == null) { // unable to read environment variables
        return null;
      }
      localOverrideHashes = getLocalOverridesHashes(root.getOverrides(), env);
      if (localOverrideHashes == null) { // still reading an override "module"
        return null;
      }

      if (root.getModuleFileHash().equals(lockfile.getModuleFileHash())
          && flags.equals(lockfile.getFlags())
          && localOverrideHashes.equals(lockfile.getLocalOverrideHashes())) {
        depGraph = lockfile.getModuleDepGraph();
      } else if (lockfileMode.equals(LockfileMode.ERROR)) {
        ImmutableList<String> diffLockfile =
            lockfile.getModuleAndFlagsDiff(root.getModuleFileHash(), localOverrideHashes, flags);
        throw new BazelDepGraphFunctionException(
            ExternalDepsException.withMessage(
                Code.BAD_MODULE,
                "Lock file is no longer up-to-date because: %s",
                String.join(", ", diffLockfile)),
            Transience.PERSISTENT);
      }
    }

    if (depGraph == null) {
      BazelModuleResolutionValue selectionResult =
          (BazelModuleResolutionValue) env.getValue(BazelModuleResolutionValue.KEY);
      if (env.valuesMissing()) {
        return null;
      }
      depGraph = selectionResult.getResolvedDepGraph();
    }

    ImmutableMap<RepositoryName, ModuleKey> canonicalRepoNameLookup =
        depGraph.keySet().stream()
            .collect(toImmutableMap(ModuleKey::getCanonicalRepoName, key -> key));

    ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage> extensionUsagesById;
    try {
      extensionUsagesById = getExtensionUsagesById(depGraph);
    } catch (ExternalDepsException e) {
      throw new BazelDepGraphFunctionException(e, Transience.PERSISTENT);
    }

    ImmutableBiMap<String, ModuleExtensionId> extensionUniqueNames =
        calculateUniqueNameForUsedExtensionId(extensionUsagesById);

    if (!lockfileMode.equals(LockfileMode.OFF)) {
      BazelLockFileValue updateLockfile =
          lockfile.toBuilder()
              .setLockFileVersion(BazelLockFileValue.LOCK_FILE_VERSION)
              .setModuleFileHash(root.getModuleFileHash())
              .setFlags(flags)
              .setLocalOverrideHashes(localOverrideHashes)
              .setModuleDepGraph(depGraph)
              .build();
      env.getListener()
          .post(BazelModuleResolutionEvent.create(updateLockfile, extensionUsagesById));
    }

    return BazelDepGraphValue.create(
        depGraph,
        canonicalRepoNameLookup,
        depGraph.values().stream().map(AbridgedModule::from).collect(toImmutableList()),
        extensionUsagesById,
        extensionUniqueNames.inverse());
  }

  @Nullable
  @VisibleForTesting
  static ImmutableMap<String, String> getLocalOverridesHashes(
      Map<String, ModuleOverride> overrides, Environment env) throws InterruptedException {
    ImmutableMap.Builder<String, String> localOverrideHashes = new ImmutableMap.Builder<>();
    for (Entry<String, ModuleOverride> entry : overrides.entrySet()) {
      if (entry.getValue() instanceof LocalPathOverride) {
        ModuleFileValue moduleValue =
            (ModuleFileValue)
                env.getValue(
                    ModuleFileValue.key(
                        ModuleKey.create(entry.getKey(), Version.EMPTY), entry.getValue()));
        if (moduleValue == null) {
          return null;
        }
        localOverrideHashes.put(entry.getKey(), moduleValue.getModuleFileHash());
      }
    }
    return localOverrideHashes.buildOrThrow();
  }

  @VisibleForTesting
  @Nullable
  static BzlmodFlagsAndEnvVars getFlagsAndEnvVars(Environment env) throws InterruptedException {
    ClientEnvironmentValue allowedYankedVersionsFromEnv =
        (ClientEnvironmentValue)
            env.getValue(
                ClientEnvironmentFunction.key(
                    YankedVersionsUtil.BZLMOD_ALLOWED_YANKED_VERSIONS_ENV));
    if (allowedYankedVersionsFromEnv == null) {
      return null;
    }

    ImmutableList<String> registries = ImmutableList.copyOf(ModuleFileFunction.REGISTRIES.get(env));
    ImmutableMap<String, String> moduleOverrides =
        ModuleFileFunction.MODULE_OVERRIDES.get(env).entrySet().stream()
            .collect(
                toImmutableMap(Entry::getKey, e -> ((LocalPathOverride) e.getValue()).getPath()));

    ImmutableList<String> yankedVersions =
        ImmutableList.copyOf(YankedVersionsUtil.ALLOWED_YANKED_VERSIONS.get(env));
    Boolean ignoreDevDeps = ModuleFileFunction.IGNORE_DEV_DEPS.get(env);
    String compatabilityMode =
        BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE.get(env).name();
    String directDepsMode = BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES.get(env).name();

    String envYanked = allowedYankedVersionsFromEnv.getValue();

    return BzlmodFlagsAndEnvVars.create(
        registries,
        moduleOverrides,
        yankedVersions,
        nullToEmpty(envYanked),
        ignoreDevDeps,
        directDepsMode,
        compatabilityMode);
  }

  /**
   * For each extension usage, we resolve (i.e. canonicalize) its bzl file label. Then we can group
   * all usages by the label + name (the ModuleExtensionId).
   */
  public static ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage>
      getExtensionUsagesById(ImmutableMap<ModuleKey, Module> depGraph)
          throws ExternalDepsException {
    ImmutableTable.Builder<ModuleExtensionId, ModuleKey, ModuleExtensionUsage>
        extensionUsagesTableBuilder = ImmutableTable.builder();
    for (Module module : depGraph.values()) {
      LabelConverter labelConverter =
          new LabelConverter(
              PackageIdentifier.create(module.getCanonicalRepoName(), PathFragment.EMPTY_FRAGMENT),
              module.getRepoMappingWithBazelDepsOnly());
      for (ModuleExtensionUsage usage : module.getExtensionUsages()) {
        ModuleExtensionId moduleExtensionId;
        try {
          moduleExtensionId =
              ModuleExtensionId.create(
                  labelConverter.convert(usage.getExtensionBzlFile()),
                  usage.getExtensionName(),
                  usage.getIsolationKey());
        } catch (LabelSyntaxException e) {
          throw ExternalDepsException.withCauseAndMessage(
              Code.BAD_MODULE,
              e,
              "invalid label for module extension found at %s",
              usage.getLocation());
        }
        if (!moduleExtensionId.getBzlFileLabel().getRepository().isVisible()) {
          throw ExternalDepsException.withMessage(
              Code.BAD_MODULE,
              "invalid label for module extension found at %s: no repo visible as '@%s' here",
              usage.getLocation(),
              moduleExtensionId.getBzlFileLabel().getRepository().getName());
        }
        extensionUsagesTableBuilder.put(moduleExtensionId, module.getKey(), usage);
      }
    }
    return extensionUsagesTableBuilder.buildOrThrow();
  }

  private ImmutableBiMap<String, ModuleExtensionId> calculateUniqueNameForUsedExtensionId(
      ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage> extensionUsagesById) {
    // Calculate a unique name for each used extension id with the following property that is
    // required for BzlmodRepoRuleFunction to unambiguously identify the extension that generates a
    // given repo:
    // After appending a single `~` to each such name, none of the resulting strings is a prefix of
    // any other such string.
    BiMap<String, ModuleExtensionId> extensionUniqueNames = HashBiMap.create();
    for (ModuleExtensionId id : extensionUsagesById.rowKeySet()) {
      int attempt = 1;
      while (extensionUniqueNames.putIfAbsent(makeUniqueNameCandidate(id, attempt), id) != null) {
        attempt++;
      }
    }
    return ImmutableBiMap.copyOf(extensionUniqueNames);
  }

  private static String makeUniqueNameCandidate(ModuleExtensionId id, int attempt) {
    // Ensure that the resulting extension name (and thus the repository names derived from it) do
    // not start with a tilde.
    RepositoryName repository = id.getBzlFileLabel().getRepository();
    String nonEmptyRepoPart = repository.isMain() ? "_main" : repository.getName();
    // When using a namespace, prefix the extension name with "_" to distinguish the prefix from
    // those generated by non-namespaced extension usages. Extension names are identified by their
    // Starlark identifier, which in the case of an exported symbol cannot start with "_".
    Preconditions.checkArgument(attempt >= 1);
    String extensionNameDisambiguator = attempt == 1 ? "" : String.valueOf(attempt);
    return id.getIsolationKey()
        .map(
            namespace ->
                String.format(
                    "%s~_%s%s~%s~%s~%s",
                    nonEmptyRepoPart,
                    id.getExtensionName(),
                    extensionNameDisambiguator,
                    namespace.getModule().getName(),
                    namespace.getModule().getVersion(),
                    namespace.getUsageExportedName()))
        .orElse(nonEmptyRepoPart + "~" + id.getExtensionName() + extensionNameDisambiguator);
  }

  static class BazelDepGraphFunctionException extends SkyFunctionException {
    BazelDepGraphFunctionException(ExternalDepsException e, Transience transience) {
      super(e, transience);
    }
  }
}
