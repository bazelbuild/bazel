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

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionUsage.Proxy;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * A trimmed down version of {@link ModuleExtensionMetadata} that is persisted in the lockfile.
 *
 * <p>The difference between this and {@link ModuleExtensionMetadata} is that this class does not
 * include the {@link Facts} field. Instead, the facts are stored in a dedicated top-level field in
 * the lockfile, for two reasons:
 *
 * <ul>
 *   <li>Reproducible extensions, which do not store a lockfile entry and thus no instance of this
 *       class, can have facts and should have them persisted - they may only be reproducible given
 *       these facts.
 *   <li>Lockfile entries are stored per OS/CPU if the extension declares a dependency on those, but
 *       facts are always cross-platform.
 * </ul>
 */
@AutoValue
@GenerateTypeAdapter
public abstract class LockfileModuleExtensionMetadata {

  /**
   * Helper record to track imports with their extension repo name mappings.
   *
   * @param imports The module-local names (keys from the map)
   * @param mappings Full map: module-local name -> extension name
   */
  private record ImportsWithMappings(
      ImmutableSet<String> imports, ImmutableMap<String, String> mappings) {}

  @Nullable
  abstract ImmutableMap<String, String> getExplicitRootModuleDirectDeps();

  @Nullable
  abstract ImmutableMap<String, String> getExplicitRootModuleDirectDevDeps();

  abstract ModuleExtensionMetadata.UseAllRepos getUseAllRepos();

  abstract boolean getReproducible();

  public static Optional<LockfileModuleExtensionMetadata> of(
      ModuleExtensionMetadata moduleExtensionMetadata) {
    if (moduleExtensionMetadata.equals(ModuleExtensionMetadata.DEFAULT)) {
      return Optional.empty();
    }
    return Optional.of(
        new AutoValue_LockfileModuleExtensionMetadata(
            moduleExtensionMetadata.getExplicitRootModuleDirectDeps(),
            moduleExtensionMetadata.getExplicitRootModuleDirectDevDeps(),
            moduleExtensionMetadata.getUseAllRepos(),
            moduleExtensionMetadata.getReproducible()));
  }

  public Optional<RootModuleFileFixup> generateFixup(
      ModuleExtensionUsage rootUsage, Set<String> allRepos) throws EvalException {
    var rootModuleDirectDevDepsResult = getRootModuleDirectDevDepsWithMappings(allRepos);
    var rootModuleDirectDepsResult = getRootModuleDirectDepsWithMappings(allRepos);
    if (rootModuleDirectDevDepsResult.isEmpty() && rootModuleDirectDepsResult.isEmpty()) {
      return Optional.empty();
    }
    Preconditions.checkState(
        rootModuleDirectDevDepsResult.isPresent() && rootModuleDirectDepsResult.isPresent());

    var rootModuleDirectDevDeps = rootModuleDirectDevDepsResult.get();
    var rootModuleDirectDeps = rootModuleDirectDepsResult.get();

    if (!rootUsage.getHasNonDevUseExtension() && !rootModuleDirectDeps.imports.isEmpty()) {
      throw Starlark.errorf(
          "root_module_direct_deps must be empty if the root module contains no "
              + "usages with dev_dependency = False");
    }
    if (!rootUsage.getHasDevUseExtension() && !rootModuleDirectDevDeps.imports.isEmpty()) {
      throw Starlark.errorf(
          "root_module_direct_dev_deps must be empty if the root module contains no "
              + "usages with dev_dependency = True");
    }

    return generateFixup(rootUsage, allRepos, rootModuleDirectDeps, rootModuleDirectDevDeps);
  }

  private static Optional<RootModuleFileFixup> generateFixup(
      ModuleExtensionUsage rootUsage,
      Set<String> allRepos,
      ImportsWithMappings expectedImports,
      ImportsWithMappings expectedDevImports) throws EvalException {
    // Build actual imports as maps: module_local_name -> extension_name
    // Use manual iteration instead of toImmutableMap() to provide better error messages
    // if duplicate keys are encountered (though this should be prevented by earlier validation)
    var actualImportsMap = new java.util.LinkedHashMap<String, String>();
    var actualDevImportsMap = new java.util.LinkedHashMap<String, String>();
    for (var proxy : rootUsage.getProxies()) {
      var map = proxy.isDevDependency() ? actualDevImportsMap : actualImportsMap;
      var depType = proxy.isDevDependency() ? "True" : "False";
      for (var entry : proxy.getImports().entrySet()) {
        String previousValue = map.putIfAbsent(entry.getKey(), entry.getValue());
        if (previousValue != null) {
          throw Starlark.errorf(
              "Repository '%s' is imported multiple times with dev_dependency = %s",
              entry.getKey(), depType);
        }
      }
    }

    String extensionBzlFile = rootUsage.getExtensionBzlFile();
    String extensionName = rootUsage.getExtensionName();

    // Calculate imports to add/remove based on extension names (values), not module-local names
    // (keys). This handles cases where a repo is imported with a different name than expected.
    var expectedExtensionNames = ImmutableSet.copyOf(expectedImports.mappings.values());
    var actualExtensionNames = ImmutableSet.copyOf(actualImportsMap.values());
    var expectedDevExtensionNames = ImmutableSet.copyOf(expectedDevImports.mappings.values());
    var actualDevExtensionNames = ImmutableSet.copyOf(actualDevImportsMap.values());

    // Find extension names that need to be added (expected but not imported)
    var missingExtensionNames = Sets.difference(expectedExtensionNames, actualExtensionNames);
    var missingDevExtensionNames =
        Sets.difference(expectedDevExtensionNames, actualDevExtensionNames);

    // Map back to module-local names for the buildozer commands
    var importsToAdd =
        ImmutableSortedSet.copyOf(
            expectedImports.mappings.entrySet().stream()
                .filter(e -> missingExtensionNames.contains(e.getValue()))
                .map(e -> e.getKey())
                .collect(toImmutableSet()));
    var devImportsToAdd =
        ImmutableSortedSet.copyOf(
            expectedDevImports.mappings.entrySet().stream()
                .filter(e -> missingDevExtensionNames.contains(e.getValue()))
                .map(e -> e.getKey())
                .collect(toImmutableSet()));

    // Find extension names that need to be removed (imported but not expected)
    // Note: we keep these as extension names (not module-local names) because buildozer's
    // use_repo_remove command expects extension-exported names
    var importsToRemove =
        ImmutableSortedSet.copyOf(Sets.difference(actualExtensionNames, expectedExtensionNames));
    var devImportsToRemove =
        ImmutableSortedSet.copyOf(
            Sets.difference(actualDevExtensionNames, expectedDevExtensionNames));

    if (importsToAdd.isEmpty()
        && importsToRemove.isEmpty()
        && devImportsToAdd.isEmpty()
        && devImportsToRemove.isEmpty()) {
      return Optional.empty();
    }

    var message =
        String.format(
            "The module extension %s defined in %s reported incorrect imports "
                + "of repositories via use_repo():\n\n",
            extensionName, extensionBzlFile);

    // For validation, we need to use extension names (values), not module-local names (keys)
    var allActualExtensionNames =
        ImmutableSortedSet.copyOf(
            Sets.union(
                ImmutableSet.copyOf(actualImportsMap.values()),
                ImmutableSet.copyOf(actualDevImportsMap.values())));
    var allExpectedExtensionNames =
        ImmutableSortedSet.copyOf(
            Sets.union(
                ImmutableSet.copyOf(expectedImports.mappings.values()),
                ImmutableSet.copyOf(expectedDevImports.mappings.values())));

    var invalidImports =
        ImmutableSortedSet.copyOf(Sets.difference(allActualExtensionNames, allRepos));
    if (!invalidImports.isEmpty()) {
      message +=
          String.format(
              "Imported, but not created by the extension (will cause the build to fail):\n"
                  + "    %s\n\n",
              String.join(", ", invalidImports));
    }

    var missingImports =
        ImmutableSortedSet.copyOf(Sets.difference(allExpectedExtensionNames, allActualExtensionNames));
    if (!missingImports.isEmpty()) {
      message +=
          String.format(
              "Not imported, but reported as direct dependencies by the extension (may cause the"
                  + " build to fail):\n"
                  + "    %s\n\n",
              String.join(", ", missingImports));
    }

    // Find repos imported as non-dev but expected as dev (by checking extension names)
    var nonDevImportsOfDevDepsExtNames =
        Sets.intersection(expectedDevExtensionNames, actualExtensionNames);
    // Map back to module-local names for the warning message
    var nonDevImportsOfDevDeps =
        ImmutableSortedSet.copyOf(
            actualImportsMap.entrySet().stream()
                .filter(e -> nonDevImportsOfDevDepsExtNames.contains(e.getValue()))
                .map(e -> e.getKey())
                .collect(toImmutableSet()));
    if (!nonDevImportsOfDevDeps.isEmpty()) {
      message +=
          String.format(
              "Imported as a regular dependency, but reported as a dev dependency by the "
                  + "extension (may cause the build to fail when used by other modules):\n"
                  + "    %s\n\n",
              String.join(", ", nonDevImportsOfDevDeps));
    }

    // Find repos imported as dev but expected as non-dev (by checking extension names)
    var devImportsOfNonDevDepsExtNames =
        Sets.intersection(expectedExtensionNames, actualDevExtensionNames);
    // Map back to module-local names for the warning message
    var devImportsOfNonDevDeps =
        ImmutableSortedSet.copyOf(
            actualDevImportsMap.entrySet().stream()
                .filter(e -> devImportsOfNonDevDepsExtNames.contains(e.getValue()))
                .map(e -> e.getKey())
                .collect(toImmutableSet()));
    if (!devImportsOfNonDevDeps.isEmpty()) {
      message +=
          String.format(
              "Imported as a dev dependency, but reported as a regular dependency by the "
                  + "extension (may cause the build to fail when used by other modules):\n"
                  + "    %s\n\n",
              String.join(", ", devImportsOfNonDevDeps));
    }

    var indirectDepImports =
        ImmutableSortedSet.copyOf(
            Sets.difference(
                Sets.intersection(allActualExtensionNames, allRepos), allExpectedExtensionNames));
    if (!indirectDepImports.isEmpty()) {
      message +=
          String.format(
              "Imported, but reported as indirect dependencies by the extension:\n    %s\n\n",
              String.join(", ", indirectDepImports));
    }

    message += "Fix the use_repo calls by running 'bazel mod tidy'.";

    var moduleFilePathToCommandsBuilder = ImmutableListMultimap.<PathFragment, String>builder();
    // Repos to add are easy: always add them to the first proxy of the correct type.
    if (!importsToAdd.isEmpty()) {
      Proxy firstNonDevProxy =
          rootUsage.getProxies().stream().filter(p -> !p.isDevDependency()).findFirst().get();
      moduleFilePathToCommandsBuilder.put(
          firstNonDevProxy.getContainingModuleFilePath(),
          makeUseRepoCommandWithMappings(
              "use_repo_add",
              firstNonDevProxy.getProxyName(),
              importsToAdd,
              expectedImports.mappings));
    }
    if (!devImportsToAdd.isEmpty()) {
      Proxy firstDevProxy =
          rootUsage.getProxies().stream().filter(p -> p.isDevDependency()).findFirst().get();
      moduleFilePathToCommandsBuilder.put(
          firstDevProxy.getContainingModuleFilePath(),
          makeUseRepoCommandWithMappings(
              "use_repo_add",
              firstDevProxy.getProxyName(),
              devImportsToAdd,
              expectedDevImports.mappings));
    }
    // Repos to remove are a bit trickier: remove them from the proxy that actually imported them.
    // Note: we use .values() (extension names) here because buildozer's use_repo_remove command
    // expects the extension-exported name, not the module-local name
    for (Proxy proxy : rootUsage.getProxies()) {
      var toRemove =
          ImmutableSortedSet.copyOf(
              Sets.intersection(
                  proxy.getImports().values(),
                  proxy.isDevDependency() ? devImportsToRemove : importsToRemove));
      if (!toRemove.isEmpty()) {
        moduleFilePathToCommandsBuilder.put(
            proxy.getContainingModuleFilePath(),
            makeUseRepoCommand("use_repo_remove", proxy.getProxyName(), toRemove));
      }
    }

    return Optional.of(
        new RootModuleFileFixup(
            moduleFilePathToCommandsBuilder.build(),
            rootUsage,
            Event.warn(rootUsage.getProxies().getFirst().getLocation(), message)));
  }

  private static String makeUseRepoCommand(String cmd, String proxyName, Collection<String> repos) {
    var commandParts = new ArrayList<String>();
    commandParts.add(cmd);
    commandParts.add(proxyName.isEmpty() ? "_unnamed_usage" : proxyName);
    commandParts.addAll(repos);
    return String.join(" ", commandParts);
  }

  /**
   * Creates a use_repo command with support for repository name mappings.
   *
   * <p>For imports that map to themselves (identity mappings), uses simple syntax: "repo_name"
   *
   * <p>For imports with custom mappings, uses equals syntax: "module_name=extension_name"
   *
   * @param cmd the command name (e.g., "use_repo_add")
   * @param proxyName the proxy name for the extension
   * @param importsToAdd the module-local names to add
   * @param mappings the full mapping from module-local names to extension names
   */
  private static String makeUseRepoCommandWithMappings(
      String cmd,
      String proxyName,
      Collection<String> importsToAdd,
      ImmutableMap<String, String> mappings) {
    var commandParts = new ArrayList<String>();
    commandParts.add(cmd);
    commandParts.add(proxyName.isEmpty() ? "_unnamed_usage" : proxyName);

    for (String moduleLocalName : importsToAdd) {
      String extensionName = mappings.get(moduleLocalName);
      Preconditions.checkState(extensionName != null, "Missing mapping for %s", moduleLocalName);
      if (extensionName.equals(moduleLocalName)) {
        // Identity mapping: use simple syntax
        commandParts.add(moduleLocalName);
      } else {
        // Custom mapping: use equals syntax "module_name=extension_name"
        commandParts.add(moduleLocalName + "=" + extensionName);
      }
    }

    return String.join(" ", commandParts);
  }

  private Optional<ImmutableSet<String>> getRootModuleDirectDeps(Set<String> allRepos)
      throws EvalException {
    return getRootModuleDirectDepsWithMappings(allRepos)
        .map(importsWithMappings -> importsWithMappings.imports);
  }

  private Optional<ImmutableSet<String>> getRootModuleDirectDevDeps(Set<String> allRepos)
      throws EvalException {
    return getRootModuleDirectDevDepsWithMappings(allRepos)
        .map(importsWithMappings -> importsWithMappings.imports);
  }

  private Optional<ImportsWithMappings> getRootModuleDirectDepsWithMappings(Set<String> allRepos)
      throws EvalException {
    return switch (getUseAllRepos()) {
      case NO -> {
        if (getExplicitRootModuleDirectDeps() != null) {
          // Check that all values (extension names) are in allRepos
          Set<String> invalidRepos =
              Sets.difference(
                  ImmutableSet.copyOf(getExplicitRootModuleDirectDeps().values()), allRepos);
          if (!invalidRepos.isEmpty()) {
            throw Starlark.errorf(
                "root_module_direct_deps contained the following repositories "
                    + "not generated by the extension: %s",
                String.join(", ", invalidRepos));
          }
          // Return both the imports (keys = module-local names) and the full mappings
          yield Optional.of(
              new ImportsWithMappings(
                  getExplicitRootModuleDirectDeps().keySet(),
                  getExplicitRootModuleDirectDeps()));
        }
        yield Optional.empty();
      }
      case REGULAR -> {
        // For "all" repos, create identity mappings
        ImmutableMap<String, String> identityMap =
            allRepos.stream().collect(ImmutableMap.toImmutableMap(r -> r, r -> r));
        yield Optional.of(
            new ImportsWithMappings(identityMap.keySet(), identityMap));
      }
      case DEV -> Optional.of(
          new ImportsWithMappings(ImmutableSet.of(), ImmutableMap.of()));
    };
  }

  private Optional<ImportsWithMappings> getRootModuleDirectDevDepsWithMappings(
      Set<String> allRepos) throws EvalException {
    return switch (getUseAllRepos()) {
      case NO -> {
        if (getExplicitRootModuleDirectDevDeps() != null) {
          // Check that all values (extension names) are in allRepos
          Set<String> invalidRepos =
              Sets.difference(
                  ImmutableSet.copyOf(getExplicitRootModuleDirectDevDeps().values()), allRepos);
          if (!invalidRepos.isEmpty()) {
            throw Starlark.errorf(
                "root_module_direct_dev_deps contained the following "
                    + "repositories not generated by the extension: %s",
                String.join(", ", invalidRepos));
          }
          // Return both the imports (keys = module-local names) and the full mappings
          yield Optional.of(
              new ImportsWithMappings(
                  getExplicitRootModuleDirectDevDeps().keySet(),
                  getExplicitRootModuleDirectDevDeps()));
        }
        yield Optional.empty();
      }
      case REGULAR -> Optional.of(
          new ImportsWithMappings(ImmutableSet.of(), ImmutableMap.of()));
      case DEV -> {
        // For "all" repos, create identity mappings
        ImmutableMap<String, String> identityMap =
            allRepos.stream().collect(ImmutableMap.toImmutableMap(r -> r, r -> r));
        yield Optional.of(
            new ImportsWithMappings(identityMap.keySet(), identityMap));
      }
    };
  }
}
