// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.InterimModule.DepSpec;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/** Context object for a Starlark thread evaluating the MODULE.bazel file and its imports. */
public class ModuleThreadContext {
  private boolean moduleCalled = false;
  private boolean hadNonModuleCall = false;
  private final boolean ignoreDevDeps;
  private final InterimModule.Builder module;
  private final ImmutableMap<String, NonRegistryOverride> builtinModules;
  private final Map<String, DepSpec> deps = new LinkedHashMap<>();
  private final List<ModuleExtensionUsageBuilder> extensionUsageBuilders = new ArrayList<>();
  private final Map<String, ModuleOverride> overrides = new HashMap<>();
  private final Map<String, RepoNameUsage> repoNameUsages = new HashMap<>();

  public static ModuleThreadContext fromOrFail(StarlarkThread thread, String what)
      throws EvalException {
    ModuleThreadContext context = thread.getThreadLocal(ModuleThreadContext.class);
    if (context == null) {
      throw Starlark.errorf("%s can only be called from MODULE.bazel and its imports", what);
    }
    return context;
  }

  public void storeInThread(StarlarkThread thread) {
    thread.setThreadLocal(ModuleThreadContext.class, this);
  }

  public ModuleThreadContext(
      ImmutableMap<String, NonRegistryOverride> builtinModules,
      ModuleKey key,
      @Nullable Registry registry,
      boolean ignoreDevDeps) {
    module = InterimModule.builder().setKey(key).setRegistry(registry);
    this.ignoreDevDeps = ignoreDevDeps;
    this.builtinModules = builtinModules;
  }

  record RepoNameUsage(String how, Location where) {}

  public void addRepoNameUsage(String repoName, String how, Location where) throws EvalException {
    RepoNameUsage collision = repoNameUsages.put(repoName, new RepoNameUsage(how, where));
    if (collision != null) {
      throw Starlark.errorf(
          "The repo name '%s' is already being used %s at %s",
          repoName, collision.how(), collision.where());
    }
  }

  /** Whether the {@code module()} directive has been called. */
  public boolean isModuleCalled() {
    return moduleCalled;
  }

  public void setModuleCalled() {
    moduleCalled = true;
  }

  /** Whether any directives other than {@code module()} have been called. */
  public boolean hadNonModuleCall() {
    return hadNonModuleCall;
  }

  public void setNonModuleCalled() {
    hadNonModuleCall = true;
  }

  public InterimModule.Builder getModuleBuilder() {
    return module;
  }

  public boolean shouldIgnoreDevDeps() {
    return ignoreDevDeps;
  }

  public void addDep(String repoName, DepSpec depSpec) {
    deps.put(repoName, depSpec);
  }

  List<ModuleExtensionUsageBuilder> getExtensionUsageBuilders() {
    return extensionUsageBuilders;
  }

  static class ModuleExtensionUsageBuilder {
    private final ModuleThreadContext context;
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
        ModuleThreadContext context,
        String extensionBzlFile,
        String extensionName,
        boolean isolate,
        Location location) {
      this.context = context;
      this.extensionBzlFile = extensionBzlFile;
      this.extensionName = extensionName;
      this.isolate = isolate;
      this.location = location;
      this.imports = HashBiMap.create();
      this.devImports = ImmutableSet.builder();
      this.tags = ImmutableList.builder();
    }

    ModuleThreadContext getContext() {
      return context;
    }

    void setHasNonDevUseExtension() {
      hasNonDevUseExtension = true;
    }

    void setHasDevUseExtension() {
      hasDevUseExtension = true;
    }

    void setExportedName(String exportedName) {
      this.exportedName = exportedName;
    }

    boolean isExported() {
      return exportedName != null;
    }

    boolean isForExtension(String extensionBzlFile, String extensionName) {
      return this.extensionBzlFile.equals(extensionBzlFile)
          && this.extensionName.equals(extensionName)
          && !this.isolate;
    }

    void addImport(
        String localRepoName,
        String exportedName,
        boolean devDependency,
        String byWhat,
        Location location)
        throws EvalException {
      RepositoryName.validateUserProvidedRepoName(localRepoName);
      RepositoryName.validateUserProvidedRepoName(exportedName);
      context.addRepoNameUsage(localRepoName, byWhat, location);
      if (imports.containsValue(exportedName)) {
        String collisionRepoName = imports.inverse().get(exportedName);
        throw Starlark.errorf(
            "The repo exported as '%s' by module extension '%s' is already imported at %s",
            exportedName, extensionName, context.repoNameUsages.get(collisionRepoName).where());
      }
      imports.put(localRepoName, exportedName);
      if (devDependency) {
        devImports.add(exportedName);
      }
    }

    void addTag(Tag tag) {
      tags.add(tag);
    }

    ModuleExtensionUsage buildUsage() throws EvalException {
      var builder =
          ModuleExtensionUsage.builder()
              .setExtensionBzlFile(extensionBzlFile)
              .setExtensionName(extensionName)
              .setUsingModule(context.getModuleBuilder().getKey())
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
            Optional.of(
                ModuleExtensionId.IsolationKey.create(
                    context.getModuleBuilder().getKey(), exportedName)));
      } else {
        builder.setIsolationKey(Optional.empty());
      }
      return builder.build();
    }
  }

  public void addOverride(String moduleName, ModuleOverride override) throws EvalException {
    ModuleOverride existingOverride = overrides.putIfAbsent(moduleName, override);
    if (existingOverride != null) {
      throw Starlark.errorf("multiple overrides for dep %s found", moduleName);
    }
  }

  public InterimModule buildModule() throws EvalException {
    // Add builtin modules as default deps of the current module.
    for (String builtinModule : builtinModules.keySet()) {
      if (module.getKey().getName().equals(builtinModule)) {
        // The built-in module does not depend on itself.
        continue;
      }
      deps.put(builtinModule, DepSpec.create(builtinModule, Version.EMPTY, -1));
      try {
        addRepoNameUsage(builtinModule, "as a built-in dependency", Location.BUILTIN);
      } catch (EvalException e) {
        throw new EvalException(
            e.getMessage()
                + String.format(
                    ", '%s' is a built-in dependency and cannot be used by any 'bazel_dep' or"
                        + " 'use_repo' directive",
                    builtinModule),
            e);
      }
    }
    // Build module extension usages and the rest of the module.
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
    // Add overrides for builtin modules if there is no existing override for them.
    if (ModuleKey.ROOT.equals(module.getKey())) {
      for (String moduleName : builtinModules.keySet()) {
        overrides.putIfAbsent(moduleName, builtinModules.get(moduleName));
      }
    }
    return ImmutableMap.copyOf(overrides);
  }
}
