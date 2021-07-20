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
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.NonRootModuleFileValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
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

  private final RegistryFactory registryFactory;
  private final Path workspaceRoot;

  public ModuleFileFunction(RegistryFactory registryFactory, Path workspaceRoot) {
    this.registryFactory = registryFactory;
    this.workspaceRoot = workspaceRoot;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    if (skyKey.equals(ModuleFileValue.keyForRootModule())) {
      return computeForRootModule(starlarkSemantics, env);
    }

    ModuleFileValue.Key moduleFileKey = (ModuleFileValue.Key) skyKey;
    ModuleKey moduleKey = moduleFileKey.getModuleKey();
    GetModuleFileResult getModuleFileResult =
        getModuleFile(moduleKey, moduleFileKey.getOverride(), env);
    if (getModuleFileResult == null) {
      return null;
    }

    ModuleFileGlobals moduleFileGlobals =
        execModuleFile(getModuleFileResult.moduleFileContents, moduleKey, starlarkSemantics, env);

    // Perform some sanity checks.
    Module module = moduleFileGlobals.buildModule(getModuleFileResult.registry);
    if (!module.getName().equals(moduleKey.getName())) {
      throw errorf(
          "the MODULE.bazel file of %s declares a different name (%s)",
          moduleKey, module.getName());
    }
    if (!moduleKey.getVersion().isEmpty() && !module.getVersion().equals(moduleKey.getVersion())) {
      throw errorf(
          "the MODULE.bazel file of %s declares a different version (%s)",
          moduleKey, module.getVersion());
    }
    if (!moduleFileGlobals.buildOverrides().isEmpty()) {
      throw errorf("The MODULE.bazel file of %s declares overrides", moduleKey);
    }

    return NonRootModuleFileValue.create(module);
  }

  private SkyValue computeForRootModule(StarlarkSemantics starlarkSemantics, Environment env)
      throws SkyFunctionException, InterruptedException {
    RootedPath moduleFilePath =
        RootedPath.toRootedPath(
            Root.fromPath(workspaceRoot), LabelConstants.MODULE_DOT_BAZEL_FILE_NAME);
    if (env.getValue(FileValue.key(moduleFilePath)) == null) {
      return null;
    }
    byte[] moduleFile = readFile(moduleFilePath.asPath());
    ModuleFileGlobals moduleFileGlobals =
        execModuleFile(moduleFile, ModuleFileValue.ROOT_MODULE_KEY, starlarkSemantics, env);
    Module module = moduleFileGlobals.buildModule(null);

    // Check that overrides don't contain the root module itself.
    ImmutableMap<String, ModuleOverride> overrides = moduleFileGlobals.buildOverrides();
    ModuleOverride rootOverride = overrides.get(module.getName());
    if (rootOverride != null) {
      throw errorf("invalid override for the root module found: %s", rootOverride);
    }
    ImmutableMap<String, String> nonRegistryOverrideCanonicalRepoNameLookup =
        Maps.filterValues(overrides, override -> override instanceof NonRegistryOverride)
            .keySet()
            .stream()
            .collect(
                toImmutableMap(
                    name -> ModuleKey.create(name, Version.EMPTY).getCanonicalRepoName(),
                    name -> name));
    return RootModuleFileValue.create(
        module, overrides, nonRegistryOverrideCanonicalRepoNameLookup);
  }

  private ModuleFileGlobals execModuleFile(
      byte[] moduleFile, ModuleKey moduleKey, StarlarkSemantics starlarkSemantics, Environment env)
      throws ModuleFileFunctionException, InterruptedException {
    StarlarkFile starlarkFile =
        StarlarkFile.parse(ParserInput.fromUTF8(moduleFile, moduleKey + "/MODULE.bazel"));
    if (!starlarkFile.ok()) {
      Event.replayEventsOn(env.getListener(), starlarkFile.errors());
      throw errorf("error parsing MODULE.bazel file for %s", moduleKey);
    }

    ModuleFileGlobals moduleFileGlobals = new ModuleFileGlobals();
    try (Mutability mu = Mutability.create("module file", moduleKey)) {
      net.starlark.java.eval.Module predeclaredEnv =
          getPredeclaredEnv(moduleFileGlobals, starlarkSemantics);
      Program program = Program.compileFile(starlarkFile, predeclaredEnv);
      // TODO(wyv): check that `program` has no `def`, `if`, etc
      StarlarkThread thread = new StarlarkThread(mu, starlarkSemantics);
      thread.setPrintHandler(Event.makeDebugPrintHandler(env.getListener()));
      Starlark.execFileProgram(program, predeclaredEnv, thread);
    } catch (SyntaxError.Exception | EvalException e) {
      throw new ModuleFileFunctionException(e);
    }
    return moduleFileGlobals;
  }

  private static class GetModuleFileResult {
    byte[] moduleFileContents;
    // `registry` can be null if this module has a non-registry override.
    @Nullable Registry registry;
  }

  @Nullable
  private GetModuleFileResult getModuleFile(
      ModuleKey key, @Nullable ModuleOverride override, Environment env)
      throws ModuleFileFunctionException, InterruptedException {
    // If there is a non-registry override for this module, we need to fetch the corresponding repo
    // first and read the module file from there.
    if (override instanceof NonRegistryOverride) {
      String canonicalRepoName = key.getCanonicalRepoName();
      RepositoryDirectoryValue repoDir =
          (RepositoryDirectoryValue)
              env.getValue(
                  RepositoryDirectoryValue.key(
                      RepositoryName.createFromValidStrippedName(canonicalRepoName)));
      if (repoDir == null) {
        return null;
      }
      RootedPath moduleFilePath =
          RootedPath.toRootedPath(
              Root.fromPath(repoDir.getPath()), LabelConstants.MODULE_DOT_BAZEL_FILE_NAME);
      if (env.getValue(FileValue.key(moduleFilePath)) == null) {
        return null;
      }
      GetModuleFileResult result = new GetModuleFileResult();
      result.moduleFileContents = readFile(moduleFilePath.asPath());
      return result;
    }

    // Otherwise, we should get the module file from a registry.
    List<String> registries = Objects.requireNonNull(REGISTRIES.get(env));
    if (override instanceof RegistryOverride) {
      String overrideRegistry = ((RegistryOverride) override).getRegistry();
      if (!overrideRegistry.isEmpty()) {
        registries = ImmutableList.of(overrideRegistry);
      }
    } else if (override != null) {
      throw errorf("unrecognized override type %s", override.getClass().getName());
    }
    List<Registry> registryObjects = new ArrayList<>(registries.size());
    for (String registryUrl : registries) {
      try {
        registryObjects.add(
            registryFactory.getRegistryWithUrl(
                registryUrl.replace("%workspace%", workspaceRoot.getPathString())));
      } catch (URISyntaxException e) {
        throw new ModuleFileFunctionException(e);
      }
    }

    // Now go through the list of registries and use the first one that contains the requested
    // module.
    try {
      GetModuleFileResult result = new GetModuleFileResult();
      for (Registry registry : registryObjects) {
        Optional<byte[]> moduleFile = registry.getModuleFile(key, env.getListener());
        if (!moduleFile.isPresent()) {
          continue;
        }
        result.moduleFileContents = moduleFile.get();
        result.registry = registry;
        return result;
      }
    } catch (IOException e) {
      throw new ModuleFileFunctionException(e);
    }

    throw errorf("module not found in registries: %s", key);
  }

  private static byte[] readFile(Path path) throws ModuleFileFunctionException {
    try {
      return FileSystemUtils.readWithKnownFileSize(path, path.getFileSize());
    } catch (IOException e) {
      throw new ModuleFileFunctionException(e);
    }
  }

  private net.starlark.java.eval.Module getPredeclaredEnv(
      ModuleFileGlobals moduleFileGlobals, StarlarkSemantics starlarkSemantics) {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, moduleFileGlobals, starlarkSemantics);
    return net.starlark.java.eval.Module.withPredeclared(starlarkSemantics, env.build());
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  @FormatMethod
  private static ModuleFileFunctionException errorf(String format, Object... args) {
    return new ModuleFileFunctionException(new NoSuchThingException(String.format(format, args)));
  }

  static final class ModuleFileFunctionException extends SkyFunctionException {

    ModuleFileFunctionException(Exception cause) {
      super(cause, Transience.TRANSIENT);
    }
  }
}
