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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.DocumentMethods;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;

/** A collection of global Starlark build API functions that apply to MODULE.bazel files. */
@DocumentMethods
public class ModuleFileGlobals {
  private boolean moduleCalled = false;
  private final Module.Builder module = Module.builder();
  private final Map<String, ModuleKey> deps = new LinkedHashMap<>();
  private final Map<String, ModuleOverride> overrides = new HashMap<>();

  public ModuleFileGlobals() {}

  @StarlarkMethod(
      name = "module",
      doc =
          "Declares certain properties of the Bazel module represented by the current Bazel repo."
              + " These properties are either essential metadata of the module (such as the name"
              + " and version), or affect behavior of the current module and its dependents.  <p>It"
              + " should be called at most once. It can be omitted only if this module is the root"
              + " module (as in, if it's not going to be depended on by another module).",
      parameters = {
        @Param(
            name = "name",
            // TODO(wyv): explain module name format
            doc =
                "The name of the module. Can be omitted only if this module is the root module (as"
                    + " in, if it's not going to be depended on by another module).",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "version",
            // TODO(wyv): explain version format
            doc =
                "The version of the module. Can be omitted only if this module is the root module"
                    + " (as in, if it's not going to be depended on by another module).",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "compatibility_level",
            // TODO(wyv): See X for more details
            doc =
                "The compatibility level of the module; this should be changed every time a major"
                    + " incompatible change is introduced. This is essentially the \"major"
                    + " version\" of the module in terms of SemVer, except that it's not embedded"
                    + " in the version string itself, but exists as a separate field. Modules with"
                    + " different compatibility levels participate in version resolution as if"
                    + " they're modules with different names, but the final dependency graph"
                    + " cannot contain multiple modules with the same name but different"
                    + " compatibility levels (unless <code>multiple_version_override</code> is in"
                    + " effect; see there for more details).",
            named = true,
            positional = false,
            defaultValue = "0"),
        @Param(
            name = "execution_platforms_to_register",
            doc =
                "A list of already-defined execution platforms to be registered when this module is"
                    + " selected. Should be a list of absolute target patterns (ie. beginning with"
                    + " either <code>@</code> or <code>//</code>). See <a"
                    + " href=\"../../toolchains.html\">toolchain resolution</a> for more"
                    + " information.",
            named = true,
            positional = false,
            allowedTypes = {@ParamType(type = Iterable.class, generic1 = String.class)},
            defaultValue = "[]"),
        @Param(
            name = "toolchains_to_register",
            doc =
                "A list of already-defined toolchains to be registered when this module is"
                    + " selected. Should be a list of absolute target patterns (ie. beginning with"
                    + " either <code>@</code> or <code>//</code>). See <a"
                    + " href=\"../../toolchains.html\">toolchain resolution</a> for more"
                    + " information.",
            named = true,
            positional = false,
            allowedTypes = {@ParamType(type = Iterable.class, generic1 = String.class)},
            defaultValue = "[]"),
        // TODO(wyv): module_rule_exports
      })
  public void module(
      String name,
      String version,
      StarlarkInt compatibilityLevel,
      Iterable<?> executionPlatformsToRegister,
      Iterable<?> toolchainsToRegister)
      throws EvalException {
    if (moduleCalled) {
      throw Starlark.errorf("the module() directive can only be called once");
    }
    moduleCalled = true;
    // TODO(wyv): add validation logic for name (alphanumerical) & others in the future
    Version parsedVersion;
    try {
      parsedVersion = Version.parse(version);
    } catch (ParseException e) {
      throw new EvalException("Invalid version in module()", e);
    }
    module
        .setName(name)
        .setVersion(parsedVersion)
        .setCompatibilityLevel(compatibilityLevel.toInt("compatibility_level"))
        .setExecutionPlatformsToRegister(
            checkAllAbsolutePatterns(
                executionPlatformsToRegister, "execution_platforms_to_register"))
        .setToolchainsToRegister(
            checkAllAbsolutePatterns(toolchainsToRegister, "toolchains_to_register"));
  }

  @StarlarkMethod(
      name = "bazel_dep",
      doc = "Declares a direct dependency on another Bazel module.",
      parameters = {
        @Param(
            name = "name",
            doc = "The name of the module to be added as a direct dependency.",
            named = true,
            positional = false),
        @Param(
            name = "version",
            doc = "The version of the module to be added as a direct dependency.",
            named = true,
            positional = false),
        @Param(
            name = "repo_name",
            doc =
                "The name of the external repo representing this dependency. This is by default the"
                    + " name of the module.",
            named = true,
            positional = false,
            defaultValue = "''"),
      })
  public void bazelDep(String name, String version, String repoName) throws EvalException {
    if (repoName.isEmpty()) {
      repoName = name;
    }
    // TODO(wyv): add validation logic for name (alphanumerical) and repoName (RepositoryName?)
    Version parsedVersion;
    try {
      parsedVersion = Version.parse(version);
    } catch (ParseException e) {
      throw new EvalException("Invalid version in bazel_dep()", e);
    }
    if (deps.putIfAbsent(repoName, ModuleKey.create(name, parsedVersion)) != null) {
      throw Starlark.errorf("a bazel_dep with the repo name %s already exists", repoName);
    }
  }

  private void addOverride(String moduleName, ModuleOverride override) throws EvalException {
    ModuleOverride existingOverride = overrides.putIfAbsent(moduleName, override);
    if (existingOverride != null) {
      throw Starlark.errorf("multiple overrides for dep %s found", moduleName);
    }
  }

  private static ImmutableList<String> checkAllStrings(Iterable<?> iterable, String where)
      throws EvalException {
    ImmutableList.Builder<String> result = ImmutableList.builder();

    for (Object o : iterable) {
      if (!(o instanceof String)) {
        throw Starlark.errorf(
            "Expected sequence of strings for '%s' argument, but got '%s' item in the sequence",
            where, Starlark.type(o));
      }
      result.add((String) o);
    }

    return result.build();
  }

  private static ImmutableList<String> checkAllAbsolutePatterns(Iterable<?> iterable, String where)
      throws EvalException {
    ImmutableList<String> list = checkAllStrings(iterable, where);
    for (String item : list) {
      if (!item.startsWith("//") && !item.startsWith("@")) {
        throw Starlark.errorf(
            "Expected absolute target patterns (must begin with '//' or '@') for '%s' argument, but"
                + " got '%s' in the sequence",
            where, item);
      }
    }
    return list;
  }

  @StarlarkMethod(
      name = "single_version_override",
      doc =
          "Specifies that a dependency should still come from a registry, but its version should"
              + " be pinned, or its registry overridden, or a list of patches applied. This"
              + " directive can only be used by the root module; in other words, if a module"
              + " specifies any overrides, it cannot be used as a dependency by others.",
      parameters = {
        @Param(
            name = "module_name",
            doc = "The name of the Bazel module dependency to apply this override to.",
            named = true,
            positional = false),
        @Param(
            name = "version",
            doc =
                "Overrides the declared version of this module in the dependency graph. In other"
                    + " words, this module will be \"pinned\" to this override version. This"
                    + " attribute can be omitted if all one wants to override is the registry or"
                    + " the patches. ",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "registry",
            doc =
                "Overrides the registry for this module; instead of finding this module from the"
                    + " default list of registries, the given registry should be used.",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "patches",
            doc =
                "A list of labels pointing to patch files to apply for this module. The patch files"
                    + " must exist in the source tree of the top level project. They are applied in"
                    + " the list order.",
            allowedTypes = {@ParamType(type = Iterable.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]"),
        @Param(
            name = "patch_strip",
            doc = "Same as the --strip argument of Unix patch.",
            named = true,
            positional = false,
            defaultValue = "0"),
      })
  public void singleVersionOverride(
      String moduleName,
      String version,
      String registry,
      Iterable<?> patches,
      StarlarkInt patchStrip)
      throws EvalException {
    Version parsedVersion;
    try {
      parsedVersion = Version.parse(version);
    } catch (ParseException e) {
      throw new EvalException("Invalid version in single_version_override()", e);
    }
    addOverride(
        moduleName,
        SingleVersionOverride.create(
            parsedVersion,
            registry,
            checkAllStrings(patches, "patches"),
            patchStrip.toInt("single_version_override.patch_strip")));
  }

  @StarlarkMethod(
      name = "multiple_version_override",
      doc =
          "Specifies that a dependency should still come from a registry, but multiple versions of"
              + " it should be allowed to coexist. This directive can only be used by the root"
              + " module; in other words, if a module specifies any overrides, it cannot be used"
              + " as a dependency by others.",
      parameters = {
        @Param(
            name = "module_name",
            doc = "The name of the Bazel module dependency to apply this override to.",
            named = true,
            positional = false),
        @Param(
            name = "versions",
            // TODO(wyv): See X for more details
            doc =
                "Explicitly specifies the versions allowed to coexist. These versions must already"
                    + " be present in the dependency graph pre-selection. Dependencies on this"
                    + " module will be \"upgraded\" to the nearest higher allowed version at the"
                    + " same compatibility level, whereas dependencies that have a higher version"
                    + " than any allowed versions at the same compatibility level will cause an"
                    + " error.",
            allowedTypes = {@ParamType(type = Iterable.class, generic1 = String.class)},
            named = true,
            positional = false),
        @Param(
            name = "registry",
            doc =
                "Overrides the registry for this module; instead of finding this module from the"
                    + " default list of registries, the given registry should be used.",
            named = true,
            positional = false,
            defaultValue = "''"),
      })
  public void multipleVersionOverride(String moduleName, Iterable<?> versions, String registry)
      throws EvalException {
    ImmutableList.Builder<Version> parsedVersionsBuilder = new ImmutableList.Builder<>();
    try {
      for (String version : checkAllStrings(versions, "versions")) {
        parsedVersionsBuilder.add(Version.parse(version));
      }
    } catch (ParseException e) {
      throw new EvalException("Invalid version in multiple_version_override()", e);
    }
    ImmutableList<Version> parsedVersions = parsedVersionsBuilder.build();
    if (parsedVersions.size() < 2) {
      throw new EvalException("multiple_version_override() must specify at least 2 versions");
    }
    addOverride(moduleName, MultipleVersionOverride.create(parsedVersions, registry));
  }

  @StarlarkMethod(
      name = "archive_override",
      doc =
          "Specifies that this dependency should come from an archive file (zip, gzip, etc) at a"
              + " certain location, instead of from a registry. This directive can only be used by"
              + " the root module; in other words, if a module specifies any overrides, it cannot"
              + " be used as a dependency by others.",
      parameters = {
        @Param(
            name = "module_name",
            doc = "The name of the Bazel module dependency to apply this override to.",
            named = true,
            positional = false),
        @Param(
            name = "urls",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Iterable.class, generic1 = String.class),
            },
            doc = "The URLs of the archive; can be http(s):// or file:// URLs.",
            named = true,
            positional = false),
        @Param(
            name = "integrity",
            doc = "The expected checksum of the archive file, in Subresource Integrity format.",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "strip_prefix",
            doc = "A directory prefix to strip from the extracted files.",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "patches",
            doc =
                "A list of labels pointing to patch files to apply for this module. The patch files"
                    + " must exist in the source tree of the top level project. They are applied in"
                    + " the list order.",
            allowedTypes = {@ParamType(type = Iterable.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]"),
        @Param(
            name = "patch_strip",
            doc = "Same as the --strip argument of Unix patch.",
            named = true,
            positional = false,
            defaultValue = "0"),
      })
  public void archiveOverride(
      String moduleName,
      Object urls,
      String integrity,
      String stripPrefix,
      Iterable<?> patches,
      StarlarkInt patchStrip)
      throws EvalException {
    ImmutableList<String> urlList =
        urls instanceof String
            ? ImmutableList.of((String) urls)
            : checkAllStrings((Iterable<?>) urls, "urls");
    addOverride(
        moduleName,
        ArchiveOverride.create(
            urlList,
            checkAllStrings(patches, "patches"),
            integrity,
            stripPrefix,
            patchStrip.toInt("archive_override.patch_strip")));
  }

  @StarlarkMethod(
      name = "git_override",
      doc =
          "Specifies that a dependency should come from a certain commit of a Git repository. This"
              + " directive can only be used by the root module; in other words, if a module"
              + " specifies any overrides, it cannot be used as a dependency by others.",
      parameters = {
        @Param(
            name = "module_name",
            doc = "The name of the Bazel module dependency to apply this override to.",
            named = true,
            positional = false),
        @Param(
            name = "remote",
            doc = "The URL of the remote Git repository.",
            named = true,
            positional = false),
        @Param(
            name = "commit",
            doc = "The commit that should be checked out.",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "patches",
            doc =
                "A list of labels pointing to patch files to apply for this module. The patch files"
                    + " must exist in the source tree of the top level project. They are applied in"
                    + " the list order.",
            allowedTypes = {@ParamType(type = Iterable.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]"),
        @Param(
            name = "patch_strip",
            doc = "Same as the --strip argument of Unix patch.",
            named = true,
            positional = false,
            defaultValue = "0"),
      })
  public void gitOverride(
      String moduleName, String remote, String commit, Iterable<?> patches, StarlarkInt patchStrip)
      throws EvalException {
    addOverride(
        moduleName,
        GitOverride.create(
            remote,
            commit,
            checkAllStrings(patches, "patches"),
            patchStrip.toInt("git_override.patch_strip")));
  }

  @StarlarkMethod(
      name = "local_path_override",
      doc =
          "Specifies that a dependency should come from a certain directory on local disk. This"
              + " directive can only be used by the root module; in other words, if a module"
              + " specifies any overrides, it cannot be used as a dependency by others.",
      parameters = {
        @Param(
            name = "module_name",
            doc = "The name of the Bazel module dependency to apply this override to.",
            named = true,
            positional = false),
        @Param(
            name = "path",
            doc = "The path to the directory where this module is.",
            named = true,
            positional = false),
      })
  public void localPathOverride(String moduleName, String path) throws EvalException {
    addOverride(moduleName, LocalPathOverride.create(path));
  }

  public Module buildModule(Registry registry) {
    return module.setDeps(ImmutableMap.copyOf(deps)).setRegistry(registry).build();
  }

  public ImmutableMap<String, ModuleOverride> buildOverrides() {
    return ImmutableMap.copyOf(overrides);
  }
}
