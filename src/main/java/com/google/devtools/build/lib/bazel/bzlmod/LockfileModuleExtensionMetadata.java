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

  @Nullable
  abstract ImmutableSet<String> getExplicitRootModuleDirectDeps();

  @Nullable
  abstract ImmutableSet<String> getExplicitRootModuleDirectDevDeps();

  abstract ModuleExtensionMetadata.UseAllRepos getUseAllRepos();

  abstract boolean getReproducible();

  public static LockfileModuleExtensionMetadata of(
      ModuleExtensionMetadata moduleExtensionMetadata) {
    return new AutoValue_LockfileModuleExtensionMetadata(
        moduleExtensionMetadata.getExplicitRootModuleDirectDeps(),
        moduleExtensionMetadata.getExplicitRootModuleDirectDevDeps(),
        moduleExtensionMetadata.getUseAllRepos(),
        moduleExtensionMetadata.getReproducible());
  }

  public Optional<RootModuleFileFixup> generateFixup(
      ModuleExtensionUsage rootUsage, Set<String> allRepos) throws EvalException {
    var rootModuleDirectDevDeps = getRootModuleDirectDevDeps(allRepos);
    var rootModuleDirectDeps = getRootModuleDirectDeps(allRepos);
    if (rootModuleDirectDevDeps.isEmpty() && rootModuleDirectDeps.isEmpty()) {
      return Optional.empty();
    }
    Preconditions.checkState(
        rootModuleDirectDevDeps.isPresent() && rootModuleDirectDeps.isPresent());

    if (!rootUsage.getHasNonDevUseExtension() && !rootModuleDirectDeps.get().isEmpty()) {
      throw Starlark.errorf(
          "root_module_direct_deps must be empty if the root module contains no "
              + "usages with dev_dependency = False");
    }
    if (!rootUsage.getHasDevUseExtension() && !rootModuleDirectDevDeps.get().isEmpty()) {
      throw Starlark.errorf(
          "root_module_direct_dev_deps must be empty if the root module contains no "
              + "usages with dev_dependency = True");
    }

    return generateFixup(
        rootUsage, allRepos, rootModuleDirectDeps.get(), rootModuleDirectDevDeps.get());
  }

  private static Optional<RootModuleFileFixup> generateFixup(
      ModuleExtensionUsage rootUsage,
      Set<String> allRepos,
      Set<String> expectedImports,
      Set<String> expectedDevImports) {
    var actualDevImports =
        rootUsage.getProxies().stream()
            .filter(p -> p.isDevDependency())
            .flatMap(p -> p.getImports().values().stream())
            .collect(toImmutableSet());
    var actualImports =
        rootUsage.getProxies().stream()
            .filter(p -> !p.isDevDependency())
            .flatMap(p -> p.getImports().values().stream())
            .collect(toImmutableSet());

    String extensionBzlFile = rootUsage.getExtensionBzlFile();
    String extensionName = rootUsage.getExtensionName();

    var importsToAdd = ImmutableSortedSet.copyOf(Sets.difference(expectedImports, actualImports));
    var importsToRemove =
        ImmutableSortedSet.copyOf(Sets.difference(actualImports, expectedImports));
    var devImportsToAdd =
        ImmutableSortedSet.copyOf(Sets.difference(expectedDevImports, actualDevImports));
    var devImportsToRemove =
        ImmutableSortedSet.copyOf(Sets.difference(actualDevImports, expectedDevImports));

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

    var allActualImports = ImmutableSortedSet.copyOf(Sets.union(actualImports, actualDevImports));
    var allExpectedImports =
        ImmutableSortedSet.copyOf(Sets.union(expectedImports, expectedDevImports));

    var invalidImports = ImmutableSortedSet.copyOf(Sets.difference(allActualImports, allRepos));
    if (!invalidImports.isEmpty()) {
      message +=
          String.format(
              "Imported, but not created by the extension (will cause the build to fail):\n"
                  + "    %s\n\n",
              String.join(", ", invalidImports));
    }

    var missingImports =
        ImmutableSortedSet.copyOf(Sets.difference(allExpectedImports, allActualImports));
    if (!missingImports.isEmpty()) {
      message +=
          String.format(
              "Not imported, but reported as direct dependencies by the extension (may cause the"
                  + " build to fail):\n"
                  + "    %s\n\n",
              String.join(", ", missingImports));
    }

    var nonDevImportsOfDevDeps =
        ImmutableSortedSet.copyOf(Sets.intersection(expectedDevImports, actualImports));
    if (!nonDevImportsOfDevDeps.isEmpty()) {
      message +=
          String.format(
              "Imported as a regular dependency, but reported as a dev dependency by the "
                  + "extension (may cause the build to fail when used by other modules):\n"
                  + "    %s\n\n",
              String.join(", ", nonDevImportsOfDevDeps));
    }

    var devImportsOfNonDevDeps =
        ImmutableSortedSet.copyOf(Sets.intersection(expectedImports, actualDevImports));
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
            Sets.difference(Sets.intersection(allActualImports, allRepos), allExpectedImports));
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
          makeUseRepoCommand("use_repo_add", firstNonDevProxy.getProxyName(), importsToAdd));
    }
    if (!devImportsToAdd.isEmpty()) {
      Proxy firstDevProxy =
          rootUsage.getProxies().stream().filter(p -> p.isDevDependency()).findFirst().get();
      moduleFilePathToCommandsBuilder.put(
          firstDevProxy.getContainingModuleFilePath(),
          makeUseRepoCommand("use_repo_add", firstDevProxy.getProxyName(), devImportsToAdd));
    }
    // Repos to remove are a bit trickier: remove them from the proxy that actually imported them.
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

  private Optional<ImmutableSet<String>> getRootModuleDirectDeps(Set<String> allRepos)
      throws EvalException {
    return switch (getUseAllRepos()) {
      case NO -> {
        if (getExplicitRootModuleDirectDeps() != null) {
          Set<String> invalidRepos = Sets.difference(getExplicitRootModuleDirectDeps(), allRepos);
          if (!invalidRepos.isEmpty()) {
            throw Starlark.errorf(
                "root_module_direct_deps contained the following repositories "
                    + "not generated by the extension: %s",
                String.join(", ", invalidRepos));
          }
        }
        yield Optional.ofNullable(getExplicitRootModuleDirectDeps());
      }
      case REGULAR -> Optional.of(ImmutableSet.copyOf(allRepos));
      case DEV -> Optional.of(ImmutableSet.of());
    };
  }

  private Optional<ImmutableSet<String>> getRootModuleDirectDevDeps(Set<String> allRepos)
      throws EvalException {
    return switch (getUseAllRepos()) {
      case NO -> {
        if (getExplicitRootModuleDirectDevDeps() != null) {
          Set<String> invalidRepos =
              Sets.difference(getExplicitRootModuleDirectDevDeps(), allRepos);
          if (!invalidRepos.isEmpty()) {
            throw Starlark.errorf(
                "root_module_direct_dev_deps contained the following "
                    + "repositories not generated by the extension: %s",
                String.join(", ", invalidRepos));
          }
        }
        yield Optional.ofNullable(getExplicitRootModuleDirectDevDeps());
      }
      case REGULAR -> Optional.of(ImmutableSet.of());
      case DEV -> Optional.of(ImmutableSet.copyOf(allRepos));
    };
  }
}
