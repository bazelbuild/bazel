// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.analysis.constraints.ConstraintConstants.getOsFromConstraintsOrHost;

import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeArtifact;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * HeaderDiscovery checks whether all header files that a compile action uses are actually declared
 * as inputs.
 *
 * <p>Tree artifacts: a tree artifact with path P causes any header file prefixed by P to be
 * accepted. Testing whether a used header file is prefixed by any tree artifact is linear search,
 * but the result is cached. If all files in a tree artifact are at the root of the artifact, the
 * entire check is performed by hash lookups.
 */
final class HeaderDiscovery {

  private HeaderDiscovery() {}

  /**
   * Returns a collection with additional input artifacts relevant to the action by reading the
   * dynamically-discovered dependency information from the parsed dependency set after the action
   * has run.
   *
   * <p>Artifacts are considered inputs but not "mandatory" inputs.
   *
   * @param dependencies the files used by the compilation, either as paths relative to {@code
   *     execRoot} that don't start with an uplevel reference or as absolute paths
   * @param permittedSystemIncludePrefixes absolute paths of directories whose headers are allowed
   *     to be used without being declared
   * @throws ActionExecutionException iff the .d is missing (when required), malformed, or has
   *     unresolvable included artifacts.
   */
  static NestedSet<Artifact> discoverInputsFromDependencies(
      Action action,
      Artifact sourceFile,
      boolean shouldValidateInclusions,
      Collection<PathFragment> dependencies,
      List<PathFragment> permittedSystemIncludePrefixes,
      NestedSet<Artifact> allowedDerivedInputs,
      Path execRoot,
      ArtifactResolver artifactResolver,
      PathMapper pathMapper)
      throws ActionExecutionException {
    boolean possiblyCaseInsensitiveFileSystem =
        getOsFromConstraintsOrHost(action.getExecutionPlatform()) == OS.WINDOWS;
    IncludeProblems absolutePathProblems = new IncludeProblems();
    List<PathFragment> execPaths =
        toExecPaths(
            dependencies,
            permittedSystemIncludePrefixes,
            execRoot.asFragment(),
            possiblyCaseInsensitiveFileSystem,
            absolutePathProblems);

    // The allowed derived inputs contain all transitively declared generated headers, which can be
    // far more than the ones actually used. Rather than indexing all of them, only look for the
    // used ones: map each used path to null and fill in the matching artifacts below.
    Map<PathFragment, Artifact> regularDerivedArtifacts =
        Maps.newHashMapWithExpectedSize(execPaths.size());
    for (PathFragment execPath : execPaths) {
      regularDerivedArtifacts.put(execPath, null);
    }
    Map<PathFragment, TreeArtifact> treeArtifacts = new HashMap<>();
    for (Artifact a : allowedDerivedInputs.toList()) {
      if (a.isSourceArtifact()) {
        continue;
      }
      // We may encounter duplicate keys in the derived inputs if two artifacts have different
      // owners. Just use the first one. The two artifacts must be generated by equivalent
      // (shareable) actions in order to have not generated a conflict in Bazel. If on an
      // incremental build one changes without the other one changing, then if their paths remain
      // the same, that will trigger an action conflict and fail the build. If one path changes,
      // then this action will be re-analyzed, and will execute in Skyframe. It can legitimately get
      // an action cache hit in that case, since even if it previously depended on the artifact
      // whose path changed, that is not taken into account by the action cache, and it will get an
      // action cache hit using the remaining un-renamed artifact.
      // TODO: Consider using a multimap here so that header resolution depends on _all_ identical
      // files, not just a single representative.
      if (a instanceof TreeArtifact treeArtifact) {
        treeArtifacts.putIfAbsent(pathMapper.map(a.getExecPath()), treeArtifact);
      } else {
        // Only sets the artifact for a used path that hasn't been matched yet.
        regularDerivedArtifacts.replace(pathMapper.map(a.getExecPath()), null, a);
      }
    }

    return runDiscovery(
        action,
        sourceFile,
        shouldValidateInclusions,
        execPaths,
        absolutePathProblems,
        regularDerivedArtifacts,
        treeArtifacts,
        possiblyCaseInsensitiveFileSystem,
        artifactResolver,
        pathMapper);
  }

  /**
   * Returns the exec paths of the given dependencies, in order. Dependencies below a permitted
   * system include prefix are skipped and absolute paths outside the exec root are recorded in
   * {@code absolutePathProblems}.
   */
  private static List<PathFragment> toExecPaths(
      Collection<PathFragment> dependencies,
      List<PathFragment> permittedSystemIncludePrefixes,
      PathFragment execRoot,
      boolean possiblyCaseInsensitiveFileSystem,
      IncludeProblems absolutePathProblems) {
    List<PathFragment> execPaths = new ArrayList<>(dependencies.size());
    // A relative dependency is below the exec root, so it can only be below a permitted system
    // include prefix if that prefix is above or below the exec root. This is very unusual, but in
    // that case, relative dependencies are checked against the prefixes as absolute paths.
    boolean checkRelativeDependenciesAgainstSystemIncludePrefixes =
        permittedSystemIncludePrefixes.stream()
            .anyMatch(
                prefix ->
                    possiblyCaseInsensitiveFileSystem
                        ? prefix.startsWithIgnoringCase(execRoot)
                            || execRoot.startsWithIgnoringCase(prefix)
                        : prefix.startsWith(execRoot) || execRoot.startsWith(prefix));
    for (PathFragment dependency : dependencies) {
      PathFragment execPath = dependency;
      if (checkRelativeDependenciesAgainstSystemIncludePrefixes && !execPath.isAbsolute()) {
        execPath = execRoot.getRelative(execPath);
      }
      if (execPath.isAbsolute()) {
        if (possiblyCaseInsensitiveFileSystem) {
          // Absolute includes from system paths are ignored. With a case-insensitive file system,
          // the paths as reported by the compiler may differ in casing from those listed by the
          // toolchain.
          if (FileSystemUtils.startsWithAnyIgnoringCase(execPath, permittedSystemIncludePrefixes)) {
            continue;
          }
        } else {
          if (FileSystemUtils.startsWithAny(execPath, permittedSystemIncludePrefixes)) {
            continue;
          }
        }
        if (execPath.startsWith(execRoot)) {
          execPath = execPath.relativeTo(execRoot); // funky but tolerable path
        } else {
          // Since gcc is given only relative paths on the command line, non-builtin include paths
          // here should never be absolute. If they are, it's probably due to a non-hermetic
          // #include,
          // and we should stop the build with an error.
          absolutePathProblems.add(execPath.getPathString());
          continue;
        }
      }
      execPaths.add(execPath);
    }
    return execPaths;
  }

  private static NestedSet<Artifact> runDiscovery(
      Action action,
      Artifact sourceFile,
      boolean shouldValidateInclusions,
      List<PathFragment> execPaths,
      IncludeProblems absolutePathProblems,
      Map<PathFragment, Artifact> regularDerivedArtifacts,
      Map<PathFragment, TreeArtifact> treeArtifacts,
      boolean possiblyCaseInsensitiveFileSystem,
      ArtifactResolver artifactResolver,
      PathMapper pathMapper)
      throws ActionExecutionException {
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();

    // Check inclusions.
    IncludeProblems unresolvablePathProblems = new IncludeProblems();
    CompactHashSet<Artifact> sourceArtifactInputs = null;
    for (PathFragment execPathFragment : execPaths) {
      Collection<? extends Artifact> resolvedArtifacts = ImmutableList.of();
      Artifact derivedArtifact = regularDerivedArtifacts.get(execPathFragment);
      if (derivedArtifact == null) {
        Optional<PackageIdentifier> pkgId =
            PackageIdentifier.discoverFromExecPath(execPathFragment, /* forFiles= */ false);
        if (pkgId.isPresent()) {
          if (possiblyCaseInsensitiveFileSystem) {
            resolvedArtifacts =
                artifactResolver.resolveSourceArtifactsAsciiCaseInsensitively(
                    execPathFragment, pkgId.get().getRepository());
          } else {
            var sourceArtifact =
                artifactResolver.resolveSourceArtifact(
                    execPathFragment, pkgId.get().getRepository());
            if (sourceArtifact != null) {
              resolvedArtifacts = ImmutableList.of(sourceArtifact);
            }
          }
        }
      } else {
        resolvedArtifacts = ImmutableList.of(derivedArtifact);
      }
      if (!resolvedArtifacts.isEmpty()) {
        // We don't need to add the sourceFile itself as it is a mandatory input.
        resolvedArtifacts = Collections2.filter(resolvedArtifacts, a -> !a.equals(sourceFile));
        switch (resolvedArtifacts.size()) {
          case 0 -> {}
          case 1 -> inputs.add(Iterables.getOnlyElement(resolvedArtifacts));
          default -> {
            if (sourceArtifactInputs == null) {
              sourceArtifactInputs = CompactHashSet.create();
              for (Artifact input : action.getInputs().toList()) {
                if (input.isSourceArtifact()) {
                  sourceArtifactInputs.add(input);
                }
              }
            }
            if (Collections.disjoint(resolvedArtifacts, sourceArtifactInputs)) {
              inputs.addAll(resolvedArtifacts);
            } else {
              for (Artifact resolvedArtifact : resolvedArtifacts) {
                if (sourceArtifactInputs.contains(resolvedArtifact)) {
                  inputs.add(resolvedArtifact);
                }
              }
            }
          }
        }
        continue;
      } else if (execPathFragment.getFileExtension().equals("cppmap")) {
        // Transitive cppmap files are added to the dotd files of compiles even
        // though they are not required for compilation. Since they're not
        // explicit inputs to the action this only happens when sandboxing is
        // disabled.
        continue;
      }

      TreeArtifact treeArtifact =
          findOwningTreeArtifact(execPathFragment, treeArtifacts, pathMapper);
      if (treeArtifact != null) {
        inputs.add(treeArtifact);
      } else {
        // Record a problem if we see files that we can't resolve, likely caused by undeclared
        // includes or illegal include constructs.
        unresolvablePathProblems.add(execPathFragment.getPathString());
      }
    }
    if (shouldValidateInclusions) {
      absolutePathProblems.assertProblemFree(
          "absolute path inclusion(s) found in rule '"
              + action.getOwner().getLabel()
              + "':\n"
              + "the source file '"
              + sourceFile.prettyPrint()
              + "' includes the following non-builtin files with absolute paths "
              + "(if these are builtin files, make sure these paths are in your toolchain):",
          action);
      unresolvablePathProblems.assertProblemFree(
          "undeclared inclusion(s) in rule '"
              + action.getOwner().getLabel()
              + "':\n"
              + "this rule is missing dependency declarations for the following files "
              + "included by '"
              + sourceFile.prettyPrint()
              + "':",
          action);
    }
    return inputs.build();
  }

  @Nullable
  private static TreeArtifact findOwningTreeArtifact(
      PathFragment execPath, Map<PathFragment, TreeArtifact> treeArtifacts, PathMapper pathMapper) {
    // Check the map for the exec path's parent directory first. If the exec path matches a direct
    // child of a tree artifact (a common case), we can skip the full iteration below.
    PathFragment dir = execPath.getParentDirectory();
    TreeArtifact tree = treeArtifacts.get(dir);
    if (tree != null) {
      return tree;
    }

    // Search for any tree artifact that encloses the exec path.
    return treeArtifacts.values().stream()
        .filter(a -> dir.startsWith(pathMapper.map(a.getExecPath())))
        .findAny()
        .orElse(null);
  }
}
