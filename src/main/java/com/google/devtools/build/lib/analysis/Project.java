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
package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.skyframe.PackageLookupFunction.PROJECT_FILE_NAME;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.ContainingPackageLookupValue;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Container for reading project metadata.
 *
 * <p>A "project" is a set of related packages that support a common piece of software. For example,
 * "bazel" is a project that includes packages {@code src/main/java/com/google/devtools/build/lib},
 * {@code src/main/java/com/google/devtools/build/lib/analysis}, {@code src/test/cpp}, and more.
 *
 * <p>"Project metadata" is any useful information that might be associated with a project. Possible
 * consumers include <a
 * href="https://github.com/bazelbuild/bazel/commit/693215317a6732085731809266f63ff0e7fc31a5">
 * Skyfocus</a>> and project-sanctioned build flags (i.e. "these are the correct flags to use with
 * this project").
 *
 * <p>Projects are defined in .scl files that are checked into source control with BUILD files and
 * code. scl stands for "Starlark configuration language". This is a limited subset of Starlark
 * intended to model generic configuration without requiring Bazel to parse it (similar to JSON).
 *
 * <p>This is not the same as {@link com.google.devtools.build.lib.runtime.ProjectFile}. That's an
 * older implementation of the same idea that was built before .scl and .bzl existed. The code here
 * is a rejuvenation of these ideas with more modern APIs.
 */
// TODO: b/324127050 - Make the co-existence of this and ProjectFile less confusing. ProjectFile is
//   an outdated API that should be removed.
public final class Project {
  private Project() {}

  /** Thrown when project data can't be read. */
  public static class ProjectParseException extends Exception {
    ProjectParseException(String msg, Throwable cause) {
      super(msg, cause);
    }
  }

  /**
   * Finds and returns the project files for a set of build targets.
   *
   * <p>This walks up each target's package path looking for {@link
   * PackageLookupFunction#PROJECT_FILE_NAME} files. For example, for {@code
   * //foo/bar/baz:mytarget}, this might look in {@code foo/bar/baz}, {@code foo/bar}, and {@code
   * foo} ("might" because it skips directories that don't have BUILD files - those directories
   * aren't packages).
   *
   * @return a map from each target to its set of project files, ordered by package depth. So a
   *     project file in {@code foo} appears before a project file in {@code foo/bar}. Paths are
   *     workspace-root-relative {@link PathFragment}s
   */
  // TODO: b/324127050 - Document resolution semantics when this is less experimental.
  public static ImmutableMultimap<Label, PathFragment> findProjectFiles(
      Collection<Label> targets,
      SkyframeExecutor skyframeExecutor,
      ExtendedEventHandler eventHandler)
      throws ProjectParseException {
    ImmutableSet<PackageIdentifier> targetPackages =
        targets.stream()
            .map(Label::getPackageFragment)
            // TODO: b/324127050 - Support other repos.
            .map(PackageIdentifier::createInMainRepo)
            .collect(toImmutableSet());
    // For every target package, we need to walk up its package path. This tracks where in the walk
    // we currently are, beginning with the targets' direct packages.
    //
    // Note that if we evaluate both a/b and a/b/c, both share the common subset a/b. In a naive
    // lookup, they walk independently: a/b evaluates a/b, then a. a/b/c evaluates a/b/c, then a/b/,
    // then a. Most of this overlaps, which wastes processing. We avoid this by keeping pointers to
    // ancestors. Using targetPkgToEvaluatedAncestor, we short-circuit a/b/c's walk once it gets
    // to a/b. Then as a post-processing step we add whatever results came from a/b's evaluation.
    Map<PackageIdentifier, PackageIdentifier> targetPkgToCurrentLookupPkg = new LinkedHashMap<>();
    Map<PackageIdentifier, PackageIdentifier> targetPkgToEvaluatedAncestor = new LinkedHashMap<>();
    for (PackageIdentifier pkg : targetPackages) {
      targetPkgToCurrentLookupPkg.put(pkg, pkg);
    }
    // Map of each package to its set of project files.
    ListMultimap<PackageIdentifier, PathFragment> projectFiles = LinkedListMultimap.create();

    while (!targetPkgToCurrentLookupPkg.isEmpty()) {
      ImmutableMap<PackageIdentifier, SkyKey> targetPkgToSkyKey =
          ImmutableMap.copyOf(
              // Maps.transformValues returns a view of the underlying map, which risks
              // ConcurrentModificationExceptions. Hence the ImmutableMap.copyOf.
              Maps.transformValues(targetPkgToCurrentLookupPkg, ContainingPackageLookupValue::key));
      var evalResult =
          skyframeExecutor.evaluateSkyKeys(
              eventHandler, targetPkgToSkyKey.values(), /* keepGoing= */ false);
      if (evalResult.hasError()) {
        throw new ProjectParseException(
            "Error finding project files", evalResult.getError().getException());
      }
      for (var entry : targetPkgToSkyKey.entrySet()) {
        PackageIdentifier targetPkg = entry.getKey();
        ContainingPackageLookupValue containingPkg =
            (ContainingPackageLookupValue) evalResult.get(entry.getValue());
        if (!containingPkg.hasContainingPackage()
            || Objects.equals(
                containingPkg.getContainingPackageName(), PackageIdentifier.EMPTY_PACKAGE_ID)) {
          // We've fully walked this target's package path.
          targetPkgToCurrentLookupPkg.remove(targetPkg);
          continue;
        }
        if (containingPkg.hasProjectFile()) {
          projectFiles.put(
              targetPkg,
              containingPkg
                  .getContainingPackageName()
                  .getPackageFragment()
                  .getRelative(PROJECT_FILE_NAME));
        }
        // TODO: b/324127050 - Support other repos.
        PackageIdentifier parentPkg =
            PackageIdentifier.createInMainRepo(
                containingPkg.getContainingPackageName().getPackageFragment().getParentDirectory());
        if (targetPkgToCurrentLookupPkg.containsKey(parentPkg)
            || projectFiles.containsKey(parentPkg)) {
          // If the parent directory is another package we're also evaluating, or the results of a
          // a package we finished evaluating, short-circuit the current package's evaluation and
          // refer to the results of the other package later. For example, if evaluating a/b/c and
          // a/b, both walk the common directories a/b and a. But we only have to visit those
          // directories once.
          targetPkgToEvaluatedAncestor.put(targetPkg, parentPkg);
          targetPkgToCurrentLookupPkg.remove(targetPkg);
        } else {
          targetPkgToCurrentLookupPkg.put(targetPkg, parentPkg);
        }
      }
    }

    // Add memoized results from ancestor evaluations. For example, if evaluating a/b/c and we also
    // evaluated a/b, add the results of the a/b evaluation here.
    for (Map.Entry<PackageIdentifier, PackageIdentifier> ancestorRef :
        targetPkgToEvaluatedAncestor.entrySet()) {
      projectFiles.putAll(ancestorRef.getKey(), projectFiles.get(ancestorRef.getValue()));
    }
    // projectFiles values are in [child.scl, parent.scl] order. Reverse this.
    ListMultimap<PackageIdentifier, PathFragment> parentFirstOrder = LinkedListMultimap.create();
    for (var entry : projectFiles.asMap().entrySet()) {
      parentFirstOrder.putAll(
          entry.getKey(), Lists.reverse(ImmutableList.copyOf(entry.getValue())));
    }
    ImmutableMultimap.Builder<Label, PathFragment> ans = ImmutableMultimap.builder();
    for (Label target : targets) {
      ans.putAll(target, parentFirstOrder.get(target.getPackageIdentifier()));
    }
    return ans.build();
  }
}
