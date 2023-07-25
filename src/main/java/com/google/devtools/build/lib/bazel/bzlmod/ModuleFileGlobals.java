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
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.docgen.annot.GlobalMethods;
import com.google.devtools.build.docgen.annot.GlobalMethods.Environment;
import com.google.devtools.build.lib.bazel.bzlmod.InterimModule.DepSpec;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileGlobals.ModuleExtensionUsageBuilder.ModuleExtensionProxy;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.StarlarkExportable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.Structure;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.Location;

/** A collection of global Starlark build API functions that apply to MODULE.bazel files. */
@GlobalMethods(environment = Environment.MODULE)
public class ModuleFileGlobals {

  /* Valid bazel compatibility argument must 1) start with (<,<=,>,>=,-);
     2) then contain a version number in form of X.X.X where X has one or two digits
  */
  private static final Pattern VALID_BAZEL_COMPATIBILITY_VERSION =
      Pattern.compile("(>|<|-|<=|>=)(\\d+\\.){2}\\d+");

  private boolean moduleCalled = false;
  private boolean hadNonModuleCall = false;
  private final boolean ignoreDevDeps;
  private final InterimModule.Builder module;
  private final Map<String, DepSpec> deps = new LinkedHashMap<>();
  private final List<ModuleExtensionUsageBuilder> extensionUsageBuilders = new ArrayList<>();
  private final Map<String, ModuleOverride> overrides = new HashMap<>();
  private final Map<String, RepoNameUsage> repoNameUsages = new HashMap<>();

  public ModuleFileGlobals(
      ImmutableMap<String, NonRegistryOverride> builtinModules,
      ModuleKey key,
      @Nullable Registry registry,
      boolean ignoreDevDeps) {
    module = InterimModule.builder().setKey(key).setRegistry(registry);
    this.ignoreDevDeps = ignoreDevDeps;
    if (ModuleKey.ROOT.equals(key)) {
      overrides.putAll(builtinModules);
    }
    for (String builtinModule : builtinModules.keySet()) {
      if (key.getName().equals(builtinModule)) {
        // The built-in module does not depend on itself.
        continue;
      }
      deps.put(builtinModule, DepSpec.create(builtinModule, Version.EMPTY, -1));
      try {
        addRepoNameUsage(builtinModule, "as a built-in dependency", Location.BUILTIN);
      } catch (EvalException e) {
        throw new IllegalStateException(e);
      }
    }
  }

  @AutoValue
  abstract static class RepoNameUsage {
    abstract String getHow();

    abstract Location getWhere();
  }

  private void addRepoNameUsage(String repoName, String how, Location where) throws EvalException {
    RepoNameUsage collision =
        repoNameUsages.put(repoName, new AutoValue_ModuleFileGlobals_RepoNameUsage(how, where));
    if (collision != null) {
      throw Starlark.errorf(
          "The repo name '%s' is already being used %s at %s",
          repoName, collision.getHow(), collision.getWhere());
    }
  }

  @VisibleForTesting
  static void validateModuleName(String moduleName) throws EvalException {
    if (!RepositoryName.VALID_MODULE_NAME.matcher(moduleName).matches()) {
      throw Starlark.errorf(
          "invalid module name '%s': valid names must 1) only contain lowercase letters (a-z),"
              + " digits (0-9), dots (.), hyphens (-), and underscores (_); 2) begin with a"
              + " lowercase letter; 3) end with a lowercase letter or digit.",
          moduleName);
    }
  }

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
            doc =
                "The name of the module. Can be omitted only if this module is the root module (as"
                    + " in, if it's not going to be depended on by another module). A valid module"
                    + " name must: 1) only contain lowercase letters (a-z), digits (0-9), dots (.),"
                    + " hyphens (-), and underscores (_); 2) begin with a lowercase letter; 3) end"
                    + " with a lowercase letter or digit.",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "version",
            doc =
                "The version of the module. Can be omitted only if this module is the root module"
                    + " (as in, if it's not going to be depended on by another module). The version"
                    + " must be in a relaxed SemVer format; see <a"
                    + " href=\"/external/module#version_format\">the documentation</a> for more"
                    + " details.",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "compatibility_level",
            doc =
                "The compatibility level of the module; this should be changed every time a major"
                    + " incompatible change is introduced. This is essentially the \"major"
                    + " version\" of the module in terms of SemVer, except that it's not embedded"
                    + " in the version string itself, but exists as a separate field. Modules with"
                    + " different compatibility levels participate in version resolution as if"
                    + " they're modules with different names, but the final dependency graph cannot"
                    + " contain multiple modules with the same name but different compatibility"
                    + " levels (unless <code>multiple_version_override</code> is in effect). See <a"
                    + " href=\"/external/module#compatibility_level\">the documentation</a> for"
                    + " more details.",
            named = true,
            positional = false,
            defaultValue = "0"),
        @Param(
            name = "repo_name",
            doc =
                "The name of the repository representing this module, as seen by the module itself."
                    + " By default, the name of the repo is the name of the module. This can be"
                    + " specified to ease migration for projects that have been using a repo name"
                    + " for itself that differs from its module name.",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "bazel_compatibility",
            doc =
                "A list of bazel versions that allows users to declare which Bazel versions"
                    + " are compatible with this module. It does NOT affect dependency resolution,"
                    + " but bzlmod will use this information to check if your current Bazel version"
                    + " is compatible. The format of this value is a string of some constraint"
                    + " values separated by comma. Three constraints are supported: <=X.X.X: The"
                    + " Bazel version must be equal or older than X.X.X. Used when there is a known"
                    + " incompatible change in a newer version. >=X.X.X: The Bazel version must be"
                    + " equal or newer than X.X.X.Used when you depend on some features that are"
                    + " only available since X.X.X. -X.X.X: The Bazel version X.X.X is not"
                    + " compatible. Used when there is a bug in X.X.X that breaks you, but fixed in"
                    + " later versions.",
            named = true,
            positional = false,
            allowedTypes = {@ParamType(type = Iterable.class, generic1 = String.class)},
            defaultValue = "[]"),
      },
      useStarlarkThread = true)
  public void module(
      String name,
      String version,
      StarlarkInt compatibilityLevel,
      String repoName,
      Iterable<?> bazelCompatibility,
      StarlarkThread thread)
      throws EvalException {
    if (moduleCalled) {
      throw Starlark.errorf("the module() directive can only be called once");
    }
    if (hadNonModuleCall) {
      throw Starlark.errorf("if module() is called, it must be called before any other functions");
    }
    moduleCalled = true;
    if (!name.isEmpty()) {
      validateModuleName(name);
    }
    if (repoName.isEmpty()) {
      repoName = name;
      addRepoNameUsage(name, "as the current module name", thread.getCallerLocation());
    } else {
      RepositoryName.validateUserProvidedRepoName(repoName);
      addRepoNameUsage(repoName, "as the module's own repo name", thread.getCallerLocation());
    }
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
        .addBazelCompatibilityValues(
            checkAllCompatibilityVersions(bazelCompatibility, "bazel_compatibility"))
        .setRepoName(repoName);
  }

  private static ImmutableList<String> checkAllAbsolutePatterns(Iterable<?> iterable, String where)
      throws EvalException {
    Sequence<String> list = Sequence.cast(iterable, String.class, where);
    for (String item : list) {
      if (!item.startsWith("//") && !item.startsWith("@")) {
        throw Starlark.errorf(
            "Expected absolute target patterns (must begin with '//' or '@') for '%s' argument, but"
                + " got '%s' as an argument",
            where, item);
      }
    }
    return list.getImmutableList();
  }

  private static ImmutableList<String> checkAllCompatibilityVersions(
      Iterable<?> iterable, String where) throws EvalException {
    Sequence<String> list = Sequence.cast(iterable, String.class, where);
    for (String version : list) {
      if (!VALID_BAZEL_COMPATIBILITY_VERSION.matcher(version).matches()) {
        throw Starlark.errorf(
            "invalid version argument '%s': valid argument must 1) start with (<,<=,>,>=,-); "
                + "2) contain a version number in form of X.X.X where X is a number",
            version);
      }
    }
    return list.getImmutableList();
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
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "max_compatibility_level",
            doc =
                "The maximum <code>compatibility_level</code> supported for the module to be added"
                    + " as a direct dependency. The version of the module implies the minimum"
                    + " compatibility_level supported, as well as the maximum if this attribute is"
                    + " not specified.",
            named = true,
            positional = false,
            defaultValue = "-1"),
        @Param(
            name = "repo_name",
            doc =
                "The name of the external repo representing this dependency. This is by default the"
                    + " name of the module.",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "dev_dependency",
            doc =
                "If true, this dependency will be ignored if the current module is not the root"
                    + " module or `--ignore_dev_dependency` is enabled.",
            named = true,
            positional = false,
            defaultValue = "False"),
      },
      useStarlarkThread = true)
  public void bazelDep(
      String name,
      String version,
      StarlarkInt maxCompatibilityLevel,
      String repoName,
      boolean devDependency,
      StarlarkThread thread)
      throws EvalException {
    hadNonModuleCall = true;
    if (repoName.isEmpty()) {
      repoName = name;
    }
    validateModuleName(name);
    Version parsedVersion;
    try {
      parsedVersion = Version.parse(version);
    } catch (ParseException e) {
      throw new EvalException("Invalid version in bazel_dep()", e);
    }
    RepositoryName.validateUserProvidedRepoName(repoName);

    if (!(ignoreDevDeps && devDependency)) {
      deps.put(
          repoName,
          DepSpec.create(
              name, parsedVersion, maxCompatibilityLevel.toInt("max_compatibility_level")));
    }

    addRepoNameUsage(repoName, "by a bazel_dep", thread.getCallerLocation());
  }

  @StarlarkMethod(
      name = "register_execution_platforms",
      doc =
          "Specifies already-defined execution platforms to be registered when this module is"
              + " selected. Should be absolute target patterns (ie. beginning with either"
              + " <code>@</code> or <code>//</code>). See <a href=\"${link toolchains}\">toolchain"
              + " resolution</a> for more information.",
      parameters = {
        @Param(
            name = "dev_dependency",
            doc =
                "If true, the execution platforms will not be registered if the current module is"
                    + " not the root module or `--ignore_dev_dependency` is enabled.",
            named = true,
            positional = false,
            defaultValue = "False"),
      },
      extraPositionals =
          @Param(
              name = "platform_labels",
              allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
              doc = "The labels of the platforms to register."))
  public void registerExecutionPlatforms(boolean devDependency, Sequence<?> platformLabels)
      throws EvalException {
    hadNonModuleCall = true;
    if (ignoreDevDeps && devDependency) {
      return;
    }
    module.addExecutionPlatformsToRegister(
        checkAllAbsolutePatterns(platformLabels, "register_execution_platforms"));
  }

  @StarlarkMethod(
      name = "register_toolchains",
      doc =
          "Specifies already-defined toolchains to be registered when this module is selected."
              + " Should be absolute target patterns (ie. beginning with either <code>@</code> or"
              + " <code>//</code>). See <a href=\"${link toolchains}\">toolchain resolution</a> for"
              + " more information.",
      parameters = {
        @Param(
            name = "dev_dependency",
            doc =
                "If true, the toolchains will not be registered if the current module is not the"
                    + " root module or `--ignore_dev_dependency` is enabled.",
            named = true,
            positional = false,
            defaultValue = "False"),
      },
      extraPositionals =
          @Param(
              name = "toolchain_labels",
              allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
              doc =
                  "The labels of the toolchains to register. Labels can include "
                      + "<code>:all</code>, in which case, all toolchain-providing targets in the "
                      + "package will be registered in lexicographical order by name."))
  public void registerToolchains(boolean devDependency, Sequence<?> toolchainLabels)
      throws EvalException {
    hadNonModuleCall = true;
    if (ignoreDevDeps && devDependency) {
      return;
    }
    module.addToolchainsToRegister(
        checkAllAbsolutePatterns(toolchainLabels, "register_toolchains"));
  }

  @StarlarkMethod(
      name = "use_extension",
      doc =
          "Returns a proxy object representing a module extension; its methods can be invoked to"
              + " create module extension tags.",
      parameters = {
        @Param(
            name = "extension_bzl_file",
            doc = "A label to the Starlark file defining the module extension."),
        @Param(
            name = "extension_name",
            doc =
                "The name of the module extension to use. A symbol with this name must be exported"
                    + " by the Starlark file."),
        @Param(
            name = "dev_dependency",
            doc =
                "If true, this usage of the module extension will be ignored if the current module"
                    + " is not the root module or `--ignore_dev_dependency` is enabled.",
            named = true,
            positional = false,
            defaultValue = "False"),
        @Param(
            name = "isolate",
            doc =
                "If true, this usage of the module extension will be isolated from all other "
                    + "usages, both in this and other modules. Tags created for this usage do not "
                    + "affect other usages and the repositories generated by the extension for "
                    + "this usage will be distinct from all other repositories generated by the "
                    + "extension."
                    + "<p>This parameter is currently experimental and only available with the "
                    + "flag <code>--experimental_isolated_extension_usages</code>.",
            named = true,
            positional = false,
            defaultValue = "False",
            enableOnlyWithFlag = "-experimental_isolated_extension_usages",
            valueWhenDisabled = "False"),
      },
      useStarlarkThread = true)
  public ModuleExtensionProxy useExtension(
      String rawExtensionBzlFile,
      String extensionName,
      boolean devDependency,
      boolean isolate,
      StarlarkThread thread) {
    hadNonModuleCall = true;

    String extensionBzlFile = normalizeLabelString(rawExtensionBzlFile);
    ModuleExtensionUsageBuilder newUsageBuilder =
        new ModuleExtensionUsageBuilder(
            extensionBzlFile, extensionName, isolate, thread.getCallerLocation());

    if (ignoreDevDeps && devDependency) {
      // This is a no-op proxy.
      return newUsageBuilder.getProxy(devDependency);
    }

    // Find an existing usage builder corresponding to this extension. Isolated usages need to get
    // their own proxy.
    if (!isolate) {
      for (ModuleExtensionUsageBuilder usageBuilder : extensionUsageBuilders) {
        if (usageBuilder.extensionBzlFile.equals(extensionBzlFile)
            && usageBuilder.extensionName.equals(extensionName)
            && !usageBuilder.isolate) {
          return usageBuilder.getProxy(devDependency);
        }
      }
    }

    // If no such proxy exists, we can just use a new one.
    extensionUsageBuilders.add(newUsageBuilder);
    return newUsageBuilder.getProxy(devDependency);
  }

  private String normalizeLabelString(String rawExtensionBzlFile) {
    // Normalize the label by adding the current module's repo_name if the label doesn't specify a
    // repository name. This is necessary as ModuleExtensionUsages are grouped by the string value
    // of this label, but later mapped to their Label representation. If multiple strings map to the
    // same Label, this would result in a crash.
    // ownName can't change anymore as calling module() after this results in an error.
    String ownName = module.getRepoName().orElse(module.getName());
    if (module.getKey().equals(ModuleKey.ROOT) && rawExtensionBzlFile.startsWith("@//")) {
      return "@" + ownName + rawExtensionBzlFile.substring(1);
    } else if (rawExtensionBzlFile.startsWith("//")) {
      return "@" + ownName + rawExtensionBzlFile;
    } else {
      return rawExtensionBzlFile;
    }
  }

  class ModuleExtensionUsageBuilder {
    private final String extensionBzlFile;
    private final String extensionName;
    private final boolean isolate;
    private final Location location;
    private final HashBiMap<String, String> imports;
    private final ImmutableSet.Builder<String> devImports;
    private final ImmutableList.Builder<Tag> tags;

    private boolean hasNonDevUseExtension;
    private boolean hasDevUseExtension;
    private String exportedName;

    ModuleExtensionUsageBuilder(
        String extensionBzlFile, String extensionName, boolean isolate, Location location) {
      this.extensionBzlFile = extensionBzlFile;
      this.extensionName = extensionName;
      this.isolate = isolate;
      this.location = location;
      this.imports = HashBiMap.create();
      this.devImports = ImmutableSet.builder();
      this.tags = ImmutableList.builder();
    }

    ModuleExtensionUsage buildUsage() throws EvalException {
      var builder =
          ModuleExtensionUsage.builder()
              .setExtensionBzlFile(extensionBzlFile)
              .setExtensionName(extensionName)
              .setUsingModule(module.getKey())
              .setLocation(location)
              .setImports(ImmutableBiMap.copyOf(imports))
              .setDevImports(devImports.build())
              .setHasDevUseExtension(hasDevUseExtension)
              .setHasNonDevUseExtension(hasNonDevUseExtension)
              .setTags(tags.build());
      if (isolate) {
        if (exportedName == null) {
          throw Starlark.errorf(
              "Isolated extension usage at %s must be assigned to a top-level variable", location);
        }
        builder.setIsolationKey(
            Optional.of(ModuleExtensionId.IsolationKey.create(module.getKey(), exportedName)));
      } else {
        builder.setIsolationKey(Optional.empty());
      }
      return builder.build();
    }

    /**
     * Creates a proxy with the specified dev_dependency bit that shares accumulated imports and
     * tags with all other such proxies, thus preserving their order across dev/non-dev deps.
     */
    ModuleExtensionProxy getProxy(boolean devDependency) {
      if (devDependency) {
        hasDevUseExtension = true;
      } else {
        hasNonDevUseExtension = true;
      }
      return new ModuleExtensionProxy(devDependency);
    }

    @StarlarkBuiltin(name = "module_extension_proxy", documented = false)
    class ModuleExtensionProxy implements Structure, StarlarkExportable {

      private final boolean devDependency;

      private ModuleExtensionProxy(boolean devDependency) {
        this.devDependency = devDependency;
      }

      void addImport(String localRepoName, String exportedName, Location location)
          throws EvalException {
        RepositoryName.validateUserProvidedRepoName(localRepoName);
        RepositoryName.validateUserProvidedRepoName(exportedName);
        addRepoNameUsage(localRepoName, "by a use_repo() call", location);
        if (imports.containsValue(exportedName)) {
          String collisionRepoName = imports.inverse().get(exportedName);
          throw Starlark.errorf(
              "The repo exported as '%s' by module extension '%s' is already imported at %s",
              exportedName, extensionName, repoNameUsages.get(collisionRepoName).getWhere());
        }
        imports.put(localRepoName, exportedName);
        if (devDependency) {
          devImports.add(exportedName);
        }
      }

      @Nullable
      @Override
      public Object getValue(String tagName) throws EvalException {
        return new StarlarkValue() {
          @StarlarkMethod(
              name = "call",
              selfCall = true,
              documented = false,
              extraKeywords = @Param(name = "kwargs"),
              useStarlarkThread = true)
          public void call(Dict<String, Object> kwargs, StarlarkThread thread) {
            tags.add(
                Tag.builder()
                    .setTagName(tagName)
                    .setAttributeValues(AttributeValues.create(kwargs))
                    .setDevDependency(devDependency)
                    .setLocation(thread.getCallerLocation())
                    .build());
          }
        };
      }

      @Override
      public ImmutableCollection<String> getFieldNames() {
        return ImmutableList.of();
      }

      @Nullable
      @Override
      public String getErrorMessageForUnknownField(String field) {
        return null;
      }

      @Override
      public boolean isExported() {
        return exportedName != null;
      }

      @Override
      public void export(EventHandler handler, Label bzlFileLabel, String name) {
        exportedName = name;
      }
    }
  }

  @StarlarkMethod(
      name = "use_repo",
      doc =
          "Imports one or more repos generated by the given module extension into the scope of the"
              + " current module.",
      parameters = {
        @Param(
            name = "extension_proxy",
            doc = "A module extension proxy object returned by a <code>use_extension</code> call."),
      },
      extraPositionals = @Param(name = "args", doc = "The names of the repos to import."),
      extraKeywords =
          @Param(
              name = "kwargs",
              doc =
                  "Specifies certain repos to import into the scope of the current module with"
                      + " different names. The keys should be the name to use in the current scope,"
                      + " whereas the values should be the original names exported by the module"
                      + " extension."),
      useStarlarkThread = true)
  public void useRepo(
      ModuleExtensionProxy extensionProxy,
      Tuple args,
      Dict<String, Object> kwargs,
      StarlarkThread thread)
      throws EvalException {
    hadNonModuleCall = true;
    Location location = thread.getCallerLocation();
    for (String arg : Sequence.cast(args, String.class, "args")) {
      extensionProxy.addImport(arg, arg, location);
    }
    for (Map.Entry<String, String> entry :
        Dict.cast(kwargs, String.class, String.class, "kwargs").entrySet()) {
      extensionProxy.addImport(entry.getKey(), entry.getValue(), location);
    }
  }

  private void addOverride(String moduleName, ModuleOverride override) throws EvalException {
    validateModuleName(moduleName);
    ModuleOverride existingOverride = overrides.putIfAbsent(moduleName, override);
    if (existingOverride != null) {
      throw Starlark.errorf("multiple overrides for dep %s found", moduleName);
    }
  }

  @StarlarkMethod(
      name = "single_version_override",
      doc =
          "Specifies that a dependency should still come from a registry, but its version should"
              + " be pinned, or its registry overridden, or a list of patches applied. This"
              + " directive only takes effect in the root module; in other words, if a module"
              + " is used as a dependency by others, its own overrides are ignored.",
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
            name = "patch_cmds",
            doc =
                "Sequence of Bash commands to be applied on Linux/Macos after patches are applied.",
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
      Iterable<?> patchCmds,
      StarlarkInt patchStrip)
      throws EvalException {
    hadNonModuleCall = true;
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
            Sequence.cast(patches, String.class, "patches").getImmutableList(),
            Sequence.cast(patchCmds, String.class, "patchCmds").getImmutableList(),
            patchStrip.toInt("single_version_override.patch_strip")));
  }

  @StarlarkMethod(
      name = "multiple_version_override",
      doc =
          "Specifies that a dependency should still come from a registry, but multiple versions of"
              + " it should be allowed to coexist. See <a"
              + " href=\"/external/module#multiple-version_override\">the documentation</a> for"
              + " more details. This"
              + " directive only takes effect in the root module; in other words, if a module"
              + " is used as a dependency by others, its own overrides are ignored.",
      parameters = {
        @Param(
            name = "module_name",
            doc = "The name of the Bazel module dependency to apply this override to.",
            named = true,
            positional = false),
        @Param(
            name = "versions",
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
    hadNonModuleCall = true;
    ImmutableList.Builder<Version> parsedVersionsBuilder = new ImmutableList.Builder<>();
    try {
      for (String version : Sequence.cast(versions, String.class, "versions").getImmutableList()) {
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
              + " certain location, instead of from a registry. This"
              + " directive only takes effect in the root module; in other words, if a module"
              + " is used as a dependency by others, its own overrides are ignored.",
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
            name = "patch_cmds",
            doc =
                "Sequence of Bash commands to be applied on Linux/Macos after patches are applied.",
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
      Iterable<?> patchCmds,
      StarlarkInt patchStrip)
      throws EvalException {
    hadNonModuleCall = true;
    ImmutableList<String> urlList =
        urls instanceof String
            ? ImmutableList.of((String) urls)
            : Sequence.cast(urls, String.class, "urls").getImmutableList();
    addOverride(
        moduleName,
        ArchiveOverride.create(
            urlList,
            Sequence.cast(patches, String.class, "patches").getImmutableList(),
            Sequence.cast(patchCmds, String.class, "patchCmds").getImmutableList(),
            integrity,
            stripPrefix,
            patchStrip.toInt("archive_override.patch_strip")));
  }

  @StarlarkMethod(
      name = "git_override",
      doc =
          "Specifies that a dependency should come from a certain commit of a Git repository. This"
              + " directive only takes effect in the root module; in other words, if a module"
              + " is used as a dependency by others, its own overrides are ignored.",
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
            name = "patch_cmds",
            doc =
                "Sequence of Bash commands to be applied on Linux/Macos after patches are applied.",
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
      String moduleName,
      String remote,
      String commit,
      Iterable<?> patches,
      Iterable<?> patchCmds,
      StarlarkInt patchStrip)
      throws EvalException {
    hadNonModuleCall = true;
    addOverride(
        moduleName,
        GitOverride.create(
            remote,
            commit,
            Sequence.cast(patches, String.class, "patches").getImmutableList(),
            Sequence.cast(patchCmds, String.class, "patchCmds").getImmutableList(),
            patchStrip.toInt("git_override.patch_strip")));
  }

  @StarlarkMethod(
      name = "local_path_override",
      doc =
          "Specifies that a dependency should come from a certain directory on local disk. This"
              + " directive only takes effect in the root module; in other words, if a module"
              + " is used as a dependency by others, its own overrides are ignored.",
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
    hadNonModuleCall = true;
    addOverride(moduleName, LocalPathOverride.create(path));
  }

  public InterimModule buildModule() throws EvalException {
    var extensionUsages = ImmutableList.<ModuleExtensionUsage>builder();
    for (var extensionUsageBuilder : extensionUsageBuilders) {
      extensionUsages.add(extensionUsageBuilder.buildUsage());
    }
    return module
        .setDeps(ImmutableMap.copyOf(deps))
        .setOriginalDeps(ImmutableMap.copyOf(deps))
        .setExtensionUsages(extensionUsages.build())
        .build();
  }

  public ImmutableMap<String, ModuleOverride> buildOverrides() {
    return ImmutableMap.copyOf(overrides);
  }
}
