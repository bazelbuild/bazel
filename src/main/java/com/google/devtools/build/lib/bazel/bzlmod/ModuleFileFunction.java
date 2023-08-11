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

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.NonRootModuleFileValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.DotBazelFileSyntaxChecker;
import com.google.devtools.build.lib.packages.StarlarkExportable;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.errorprone.annotations.FormatMethod;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.SyntaxError;

/**
 * Takes a {@link ModuleKey} and its override (if any), retrieves the module file from a registry or
 * as directed by the override, and evaluates the module file.
 */
public class ModuleFileFunction implements SkyFunction {

  public static final Precomputed<List<String>> REGISTRIES = new Precomputed<>("registries");
  public static final Precomputed<Boolean> IGNORE_DEV_DEPS =
      new Precomputed<>("ignore_dev_dependency");

  public static final Precomputed<Map<String, ModuleOverride>> MODULE_OVERRIDES =
      new Precomputed<>("module_overrides");

  private final RegistryFactory registryFactory;
  private final Path workspaceRoot;
  private final ImmutableMap<String, NonRegistryOverride> builtinModules;

  /**
   * @param builtinModules A list of "built-in" modules that are treated as implicit dependencies of
   *     every other module (including other built-in modules). These modules are defined as
   *     non-registry overrides.
   */
  public ModuleFileFunction(
      RegistryFactory registryFactory,
      Path workspaceRoot,
      ImmutableMap<String, NonRegistryOverride> builtinModules) {
    this.registryFactory = registryFactory;
    this.workspaceRoot = workspaceRoot;
    this.builtinModules = builtinModules;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ModuleFileFunctionException, InterruptedException {
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    if (skyKey.equals(ModuleFileValue.KEY_FOR_ROOT_MODULE)) {
      return computeForRootModule(starlarkSemantics, env);
    }

    ClientEnvironmentValue allowedYankedVersionsFromEnv =
        (ClientEnvironmentValue)
            env.getValue(
                ClientEnvironmentFunction.key(
                    YankedVersionsUtil.BZLMOD_ALLOWED_YANKED_VERSIONS_ENV));
    if (allowedYankedVersionsFromEnv == null) {
      return null;
    }

    Optional<ImmutableSet<ModuleKey>> allowedYankedVersions;
    try {
      allowedYankedVersions =
          YankedVersionsUtil.parseAllowedYankedVersions(
              allowedYankedVersionsFromEnv.getValue(),
              Objects.requireNonNull(YankedVersionsUtil.ALLOWED_YANKED_VERSIONS.get(env)));
    } catch (ExternalDepsException e) {
      throw new ModuleFileFunctionException(e, SkyFunctionException.Transience.PERSISTENT);
    }

    ModuleFileValue.Key moduleFileKey = (ModuleFileValue.Key) skyKey;
    ModuleKey moduleKey = moduleFileKey.getModuleKey();
    GetModuleFileResult getModuleFileResult =
        getModuleFile(moduleKey, moduleFileKey.getOverride(), allowedYankedVersions, env);
    if (getModuleFileResult == null) {
      return null;
    }
    String moduleFileHash =
        new Fingerprint().addBytes(getModuleFileResult.moduleFile.getContent()).hexDigestAndReset();

    ModuleFileGlobals moduleFileGlobals =
        execModuleFile(
            getModuleFileResult.moduleFile,
            getModuleFileResult.registry,
            moduleKey,
            // Dev dependencies should always be ignored if the current module isn't the root module
            /* ignoreDevDeps= */ true,
            // We try to prevent most side effects of yanked modules, in particular print().
            /* printIsNoop= */ getModuleFileResult.yankedInfo != null,
            starlarkSemantics,
            env);

    // Perform some sanity checks.
    InterimModule module;
    try {
      module = moduleFileGlobals.buildModule();
    } catch (EvalException e) {
      env.getListener().handle(Event.error(e.getMessageWithStack()));
      throw errorf(Code.BAD_MODULE, "error executing MODULE.bazel file for %s", moduleKey);
    }
    if (!module.getName().equals(moduleKey.getName())) {
      throw errorf(
          Code.BAD_MODULE,
          "the MODULE.bazel file of %s declares a different name (%s)",
          moduleKey,
          module.getName());
    }
    if (!moduleKey.getVersion().isEmpty() && !module.getVersion().equals(moduleKey.getVersion())) {
      throw errorf(
          Code.BAD_MODULE,
          "the MODULE.bazel file of %s declares a different version (%s)",
          moduleKey,
          module.getVersion());
    }

    if (getModuleFileResult.yankedInfo != null) {
      // Yanked modules should not have observable side effects such as adding dependency
      // requirements, so we drop those from the constructed module. We do have to preserve the
      // compatibility level as it influences the set of versions the yanked version can be updated
      // to during selection.
      return NonRootModuleFileValue.create(
          InterimModule.builder()
              .setKey(module.getKey())
              .setName(module.getName())
              .setVersion(module.getVersion())
              .setCompatibilityLevel(module.getCompatibilityLevel())
              .setRegistry(module.getRegistry())
              .setYankedInfo(Optional.of(getModuleFileResult.yankedInfo))
              .build(),
          moduleFileHash);
    }

    return NonRootModuleFileValue.create(module, moduleFileHash);
  }

  @Nullable
  private SkyValue computeForRootModule(StarlarkSemantics starlarkSemantics, Environment env)
      throws ModuleFileFunctionException, InterruptedException {
    RootedPath moduleFilePath =
        RootedPath.toRootedPath(
            Root.fromPath(workspaceRoot), LabelConstants.MODULE_DOT_BAZEL_FILE_NAME);
    if (env.getValue(FileValue.key(moduleFilePath)) == null) {
      return null;
    }
    byte[] moduleFileContents = readModuleFile(moduleFilePath.asPath());
    String moduleFileHash = new Fingerprint().addBytes(moduleFileContents).hexDigestAndReset();
    ModuleFileGlobals moduleFileGlobals =
        execModuleFile(
            ModuleFile.create(moduleFileContents, moduleFilePath.asPath().toString()),
            /* registry= */ null,
            ModuleKey.ROOT,
            /* ignoreDevDeps= */ Objects.requireNonNull(IGNORE_DEV_DEPS.get(env)),
            /* printIsNoop= */ false,
            starlarkSemantics,
            env);
    InterimModule module;
    try {
      module = moduleFileGlobals.buildModule();
    } catch (EvalException e) {
      env.getListener().handle(Event.error(e.getMessageWithStack()));
      throw errorf(Code.BAD_MODULE, "error executing MODULE.bazel file for the root module");
    }
    for (ModuleExtensionUsage usage : module.getExtensionUsages()) {
      if (usage.getIsolationKey().isPresent() && usage.getImports().isEmpty()) {
        throw errorf(
            Code.BAD_MODULE,
            "the isolated usage at %s of extension %s defined in %s has no effect as no "
                + "repositories are imported from it. Either import one or more repositories "
                + "generated by the extension with use_repo or remove the usage.",
            usage.getLocation(),
            usage.getExtensionName(),
            usage.getExtensionBzlFile());
      }
    }

    ImmutableMap<String, ModuleOverride> moduleOverrides = moduleFileGlobals.buildOverrides();
    Map<String, ModuleOverride> commandOverrides = MODULE_OVERRIDES.get(env);
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.<String, ModuleOverride>builder()
            .putAll(moduleOverrides)
            .putAll(commandOverrides)
            .buildKeepingLast();

    // Check that overrides don't contain the root module itself.
    ModuleOverride rootOverride = overrides.get(module.getName());
    if (rootOverride != null) {
      throw errorf(Code.BAD_MODULE, "invalid override for the root module found: %s", rootOverride);
    }
    ImmutableMap<RepositoryName, String> nonRegistryOverrideCanonicalRepoNameLookup =
        Maps.filterValues(overrides, override -> override instanceof NonRegistryOverride)
            .keySet()
            .stream()
            .collect(
                toImmutableMap(
                    name -> ModuleKey.create(name, Version.EMPTY).getCanonicalRepoName(),
                    name -> name));
    return RootModuleFileValue.create(
        module, moduleFileHash, overrides, nonRegistryOverrideCanonicalRepoNameLookup);
  }

  private ModuleFileGlobals execModuleFile(
      ModuleFile moduleFile,
      @Nullable Registry registry,
      ModuleKey moduleKey,
      boolean ignoreDevDeps,
      boolean printIsNoop,
      StarlarkSemantics starlarkSemantics,
      Environment env)
      throws ModuleFileFunctionException, InterruptedException {
    StarlarkFile starlarkFile =
        StarlarkFile.parse(ParserInput.fromUTF8(moduleFile.getContent(), moduleFile.getLocation()));
    if (!starlarkFile.ok()) {
      Event.replayEventsOn(env.getListener(), starlarkFile.errors());
      throw errorf(Code.BAD_MODULE, "error parsing MODULE.bazel file for %s", moduleKey);
    }

    ModuleFileGlobals moduleFileGlobals =
        new ModuleFileGlobals(builtinModules, moduleKey, registry, ignoreDevDeps);
    try (Mutability mu = Mutability.create("module file", moduleKey)) {
      new DotBazelFileSyntaxChecker("MODULE.bazel files", /* canLoadBzl= */ false)
          .check(starlarkFile);
      net.starlark.java.eval.Module predeclaredEnv =
          getPredeclaredEnv(moduleFileGlobals, starlarkSemantics);
      Program program = Program.compileFile(starlarkFile, predeclaredEnv);
      StarlarkThread thread = new StarlarkThread(mu, starlarkSemantics);
      if (printIsNoop) {
        thread.setPrintHandler((t, msg) -> {});
      } else {
        thread.setPrintHandler(Event.makeDebugPrintHandler(env.getListener()));
      }
      thread.setPostAssignHook(
          (name, value) -> {
            if (value instanceof StarlarkExportable) {
              StarlarkExportable exportable = (StarlarkExportable) value;
              if (!exportable.isExported()) {
                exportable.export(env.getListener(), null, name);
              }
            }
          });
      Starlark.execFileProgram(program, predeclaredEnv, thread);
    } catch (SyntaxError.Exception e) {
      Event.replayEventsOn(env.getListener(), e.errors());
      throw errorf(Code.BAD_MODULE, "error executing MODULE.bazel file for %s", moduleKey);
    } catch (EvalException e) {
      env.getListener().handle(Event.error(e.getMessageWithStack()));
      throw errorf(Code.BAD_MODULE, "error executing MODULE.bazel file for %s", moduleKey);
    }
    return moduleFileGlobals;
  }

  private static class GetModuleFileResult {
    ModuleFile moduleFile;
    // `yankedInfo` is non-null if and only if the module has been yanked and hasn't been
    // allowlisted.
    @Nullable String yankedInfo;
    // `registry` can be null if this module has a non-registry override.
    @Nullable Registry registry;
  }

  @Nullable
  private GetModuleFileResult getModuleFile(
      ModuleKey key,
      @Nullable ModuleOverride override,
      Optional<ImmutableSet<ModuleKey>> allowedYankedVersions,
      Environment env)
      throws ModuleFileFunctionException, InterruptedException {
    // If there is a non-registry override for this module, we need to fetch the corresponding repo
    // first and read the module file from there.
    if (override instanceof NonRegistryOverride) {
      RepositoryName canonicalRepoName = key.getCanonicalRepoName();
      RepositoryDirectoryValue repoDir =
          (RepositoryDirectoryValue) env.getValue(RepositoryDirectoryValue.key(canonicalRepoName));
      if (repoDir == null) {
        return null;
      }
      RootedPath moduleFilePath =
          RootedPath.toRootedPath(
              Root.fromPath(repoDir.getPath()), LabelConstants.MODULE_DOT_BAZEL_FILE_NAME);
      if (env.getValue(FileValue.key(moduleFilePath)) == null) {
        return null;
      }
      Label moduleFileLabel =
          Label.createUnvalidated(
              PackageIdentifier.create(key.getCanonicalRepoName(), PathFragment.EMPTY_FRAGMENT),
              LabelConstants.MODULE_DOT_BAZEL_FILE_NAME.getBaseName());
      GetModuleFileResult result = new GetModuleFileResult();
      result.moduleFile =
          ModuleFile.create(
              readModuleFile(moduleFilePath.asPath()),
              moduleFileLabel.getUnambiguousCanonicalForm());
      return result;
    }

    // Otherwise, we should get the module file from a registry.
    if (key.getVersion().isEmpty()) {
      // Print a friendlier error message if the user forgets to specify a version *and* doesn't
      // have a non-registry override.
      throw errorf(
          Code.MODULE_NOT_FOUND,
          "bad bazel_dep on module '%s' with no version. Did you forget to specify a version, or a"
              + " non-registry override?",
          key.getName());
    }
    // TODO(wyv): Move registry object creation to BazelRepositoryModule so we don't repeatedly
    //   create them, and we can better report the error (is it a flag error or override error?).
    List<String> registries = Objects.requireNonNull(REGISTRIES.get(env));
    if (override instanceof RegistryOverride) {
      String overrideRegistry = ((RegistryOverride) override).getRegistry();
      if (!overrideRegistry.isEmpty()) {
        registries = ImmutableList.of(overrideRegistry);
      }
    } else if (override != null) {
      // This should never happen.
      // TODO(wyv): make ModuleOverride a sealed interface so this is checked at compile time.
      throw new IllegalStateException(
          String.format(
              "unrecognized override type %s for module %s",
              override.getClass().getSimpleName(), key));
    }
    List<Registry> registryObjects = new ArrayList<>(registries.size());
    for (String registryUrl : registries) {
      try {
        registryObjects.add(
            registryFactory.getRegistryWithUrl(
                registryUrl.replace("%workspace%", workspaceRoot.getPathString())));
      } catch (URISyntaxException e) {
        throw errorf(Code.INVALID_REGISTRY_URL, e, "Invalid registry URL");
      }
    }

    // Now go through the list of registries and use the first one that contains the requested
    // module.
    GetModuleFileResult result = new GetModuleFileResult();
    for (Registry registry : registryObjects) {
      try {
        Optional<ModuleFile> moduleFile = registry.getModuleFile(key, env.getListener());
        if (moduleFile.isEmpty()) {
          continue;
        }
        result.moduleFile = moduleFile.get();
        result.registry = registry;
        result.yankedInfo =
            YankedVersionsUtil.getYankedInfo(
                    registry, key, allowedYankedVersions, env.getListener())
                .orElse(null);
        return result;
      } catch (IOException e) {
        throw errorf(
            Code.ERROR_ACCESSING_REGISTRY, e, "Error accessing registry %s", registry.getUrl());
      }
    }

    throw errorf(Code.MODULE_NOT_FOUND, "module not found in registries: %s", key);
  }

  private static byte[] readModuleFile(Path path) throws ModuleFileFunctionException {
    try {
      return FileSystemUtils.readWithKnownFileSize(path, path.getFileSize());
    } catch (IOException e) {
      throw errorf(Code.MODULE_NOT_FOUND, "MODULE.bazel expected but not found at %s", path);
    }
  }

  private net.starlark.java.eval.Module getPredeclaredEnv(
      ModuleFileGlobals moduleFileGlobals, StarlarkSemantics starlarkSemantics) {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, moduleFileGlobals, starlarkSemantics);
    return net.starlark.java.eval.Module.withPredeclared(starlarkSemantics, env.buildOrThrow());
  }

  @FormatMethod
  private static ModuleFileFunctionException errorf(Code code, String format, Object... args) {
    return new ModuleFileFunctionException(ExternalDepsException.withMessage(code, format, args));
  }

  @FormatMethod
  private static ModuleFileFunctionException errorf(
      Code code, Throwable cause, String format, Object... args) {
    return new ModuleFileFunctionException(
        ExternalDepsException.withCauseAndMessage(code, cause, format, args));
  }

  static final class ModuleFileFunctionException extends SkyFunctionException {

    ModuleFileFunctionException(ExternalDepsException cause) {
      super(cause, Transience.TRANSIENT);
    }

    ModuleFileFunctionException(ExternalDepsException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
