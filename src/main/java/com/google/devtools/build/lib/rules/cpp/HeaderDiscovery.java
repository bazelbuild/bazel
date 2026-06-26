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
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

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
   * @throws ActionExecutionException iff the .d is missing (when required), malformed, or has
   *     unresolvable included artifacts.
   */
  static NestedSet<Artifact> discoverInputsFromDependencies(
      Action action,
      Artifact sourceFile,
      boolean shouldValidateInclusions,
      Collection<Path> dependencies,
      List<Path> permittedSystemIncludePrefixes,
      NestedSet<Artifact> allowedDerivedInputs,
      Path execRoot,
      ArtifactResolver artifactResolver,
      boolean siblingRepositoryLayout,
      PathMapper pathMapper)
      throws ActionExecutionException {
    Multimap<PathFragment, Artifact> regularDerivedArtifacts = LinkedHashMultimap.create();
    Multimap<PathFragment, SpecialArtifact> treeArtifacts = LinkedHashMultimap.create();
    for (Artifact a : allowedDerivedInputs.toList()) {
      if (a.isSourceArtifact()) {
        continue;
      }
      if (a.isTreeArtifact()) {
        treeArtifacts.put(pathMapper.map(a.getExecPath()), (SpecialArtifact) a);
      } else {
        regularDerivedArtifacts.put(pathMapper.map(a.getExecPath()), a);
      }
    }

    return runDiscovery(
        action,
        sourceFile,
        shouldValidateInclusions,
        dependencies,
        permittedSystemIncludePrefixes,
        regularDerivedArtifacts,
        treeArtifacts,
        execRoot,
        artifactResolver,
        siblingRepositoryLayout,
        pathMapper);
  }

  private static NestedSet<Artifact> runDiscovery(
      Action action,
      Artifact sourceFile,
      boolean shouldValidateInclusions,
      Collection<Path> dependencies,
      List<Path> permittedSystemIncludePrefixes,
      Multimap<PathFragment, Artifact> regularDerivedArtifacts,
      Multimap<PathFragment, SpecialArtifact> treeArtifacts,
      Path execRoot,
      ArtifactResolver artifactResolver,
      boolean siblingRepositoryLayout,
      PathMapper pathMapper)
      throws ActionExecutionException {
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();

    // This is a very special case: in certain corner cases (notably, protobuf), the WORKSPACE file
    // contains a local_repository that has the same name and path as the main repository. In this
    // case, if sibling repository layout is active, files in that local repository have the same
    // absolute execpath as files in the main repository but not the same *relative* execpath (e.g.
    // a:b would be ../repo/a/b in the local_repository and a/b in the main repository)
    //
    // This would mean that artifacts coming from the local repository would be discovered as ones
    // in the main repository, thus resulting in "undeclared dependency" errors. The way this flag
    // hacks around this is by pretending that such artifacts are always in the local_repository if
    // the action is in it. This of course breaks horribly if there are artifacts from *both*
    // repositories on the inputs.
    //
    // Protobuf uses this to work around the fact that @bazel_tools depends on it (see
    // https://github.com/bazelbuild/bazel/issues/19973). The fix is either to cut that dependency
    // or to migrate to bzlmod.
    boolean ignoreMainRepository =
        siblingRepositoryLayout
            && action
                .getOwner()
                .getLabel()
                .getRepository()
                .getName()
                .equals(execRoot.getBaseName());

    // Check inclusions.
    IncludeProblems absolutePathProblems = new IncludeProblems();
    IncludeProblems unresolvablePathProblems = new IncludeProblems();
    boolean possiblyCaseInsensitiveFileSystem =
        getOsFromConstraintsOrHost(action.getExecutionPlatform()) == OS.WINDOWS;
    CompactHashSet<Artifact> sourceArtifactInputs = null;
    for (Path execPath : dependencies) {
      PathFragment execPathFragment = execPath.asFragment();
      if (execPathFragment.isAbsolute()) {
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
        if (execPath.startsWith(execRoot)
            && (!ignoreMainRepository
                || artifactResolver.isDerivedArtifact(execPath.relativeTo(execRoot)))) {
          execPathFragment = execPath.relativeTo(execRoot); // funky but tolerable path
        } else if (siblingRepositoryLayout && execPath.startsWith(execRoot.getParentDirectory())) {
          // for --experimental_sibling_repository_layout
          execPathFragment =
              LabelConstants.EXPERIMENTAL_EXTERNAL_PATH_PREFIX.getRelative(
                  execPath.relativeTo(execRoot.getParentDirectory()));
        } else {
          // Since gcc is given only relative paths on the command line, non-builtin include paths
          // here should never be absolute. If they are, it's probably due to a non-hermetic
          // #include,
          // and we should stop the build with an error.
          absolutePathProblems.add(execPathFragment.getPathString());
          continue;
        }
      }
      Collection<? extends Artifact> resolvedArtifacts = ImmutableList.of();
      Collection<Artifact> derivedArtifacts = regularDerivedArtifacts.get(execPathFragment);
      if (derivedArtifacts.isEmpty()) {
        Optional<PackageIdentifier> pkgId =
            PackageIdentifier.discoverFromExecPath(
                execPathFragment, false, siblingRepositoryLayout);
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
        resolvedArtifacts = derivedArtifacts;
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

      Collection<SpecialArtifact> owningTreeArtifacts =
          findOwningTreeArtifacts(execPathFragment, treeArtifacts, pathMapper);
      if (!owningTreeArtifacts.isEmpty()) {
        inputs.addAll(owningTreeArtifacts);
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

  private static Collection<SpecialArtifact> findOwningTreeArtifacts(
      PathFragment execPath,
      Multimap<PathFragment, SpecialArtifact> treeArtifacts,
      PathMapper pathMapper) {
    // Check the map for the exec path's parent directory first. If the exec path matches a direct
    // child of a tree artifact (a common case), we can skip the full iteration below.
    PathFragment dir = execPath.getParentDirectory();
    Collection<SpecialArtifact> trees = treeArtifacts.get(dir);
    if (!trees.isEmpty()) {
      return trees;
    }

    // Search for any tree artifact that encloses the exec path.
    return treeArtifacts.values().stream()
        .filter(a -> dir.startsWith(pathMapper.map(a.getExecPath())))
        .collect(ImmutableList.toImmutableList());
  }
}
