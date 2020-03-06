// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.ninja.actions;


import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Helper class to create artifacts for {@link NinjaAction} to be used from {@link NinjaGraphRule}.
 * All created output artifacts are accumulated in the NestedSetBuilder.
 *
 * <p>Input and putput paths are interpreted relative to the working directory, see
 * working_directory property in {@link NinjaGraphRule}. All output artifact are created under the
 * derived artifacts root <execroot>/<outputRoot>, see output_root property in {@link
 * NinjaGraphRule}.
 */
class NinjaGraphArtifactsHelper {
  private final RuleContext ruleContext;
  private final PathFragment outputRootPath;
  private final PathFragment workingDirectory;
  private final ArtifactRoot derivedOutputRoot;

  private final ImmutableSortedMap<PathFragment, Artifact> depsNameToArtifact;
  private final ImmutableSortedMap<PathFragment, Artifact> symlinkPathToArtifact;
  private final ImmutableSortedSet<PathFragment> outputRootSymlinks;

  /**
   * Constructor
   *
   * @param ruleContext parent NinjaGraphRule rule context
   * @param outputRootPath name of output directory for Ninja actions under execroot
   * @param workingDirectory relative path under execroot, the root for interpreting all paths in
   *     Ninja file
   * @param depsNameToArtifact mapping between the path fragment in the Ninja file and prebuilt
   * @param symlinkPathToArtifact mapping of paths to artifacts for input symlinks under output_root
   * @param outputRootSymlinks list of output paths for which symlink artifacts should be created,
   *     paths are relative to the output_root.
   */
  NinjaGraphArtifactsHelper(
      RuleContext ruleContext,
      PathFragment outputRootPath,
      PathFragment workingDirectory,
      ImmutableSortedMap<PathFragment, Artifact> depsNameToArtifact,
      ImmutableSortedMap<PathFragment, Artifact> symlinkPathToArtifact,
      ImmutableSortedSet<PathFragment> outputRootSymlinks) {
    this.ruleContext = ruleContext;
    this.outputRootPath = outputRootPath;
    this.workingDirectory = workingDirectory;
    this.depsNameToArtifact = depsNameToArtifact;
    this.symlinkPathToArtifact = symlinkPathToArtifact;
    this.outputRootSymlinks = outputRootSymlinks;
    Path execRoot =
        Preconditions.checkNotNull(ruleContext.getConfiguration())
            .getDirectories()
            .getExecRoot(ruleContext.getWorkspaceName());
    this.derivedOutputRoot = ArtifactRoot.asDerivedRoot(execRoot, outputRootPath);
  }

  DerivedArtifact createOutputArtifact(PathFragment pathRelativeToWorkingDirectory)
      throws GenericParsingException {
    PathFragment execPath = workingDirectory.getRelative(pathRelativeToWorkingDirectory);

    if (!execPath.startsWith(outputRootPath)) {
      throw new GenericParsingException(
          String.format(
              "Ninja actions are allowed to create outputs only under output_root,"
                  + " path '%s' is not allowed.",
              pathRelativeToWorkingDirectory));
    }
    // If the path was declared as output symlink, create a symlink artifact.
    if (outputRootSymlinks.contains(execPath.relativeTo(outputRootPath))) {
      return ruleContext
          .getAnalysisEnvironment()
          .getSymlinkArtifact(execPath.relativeTo(outputRootPath), derivedOutputRoot);
    }
    return ruleContext.getDerivedArtifact(execPath.relativeTo(outputRootPath), derivedOutputRoot);
  }

  Artifact getInputArtifact(PathFragment workingDirectoryPath) throws GenericParsingException {
    if (depsNameToArtifact.containsKey(workingDirectoryPath)) {
      return depsNameToArtifact.get(workingDirectoryPath);
    }

    if (symlinkPathToArtifact.containsKey(workingDirectoryPath)) {
      return symlinkPathToArtifact.get(workingDirectoryPath);
    }

    PathFragment execPath = workingDirectory.getRelative(workingDirectoryPath);
    if (execPath.startsWith(outputRootPath)) {
      // In the output directory, so it is either specified via output_root_symlinks, or
      // it is a derived artifact.
      if (outputRootSymlinks.contains(execPath.relativeTo(outputRootPath))) {
        return ruleContext
            .getAnalysisEnvironment()
            .getSymlinkArtifact(execPath.relativeTo(outputRootPath), derivedOutputRoot);
      }
      return ruleContext.getDerivedArtifact(execPath.relativeTo(outputRootPath), derivedOutputRoot);
    }

    if (!execPath.startsWith(ruleContext.getPackageDirectory())) {
      throw new GenericParsingException(
          String.format(
              "Source artifact '%s' is not under the package of the ninja_build rule", execPath));
    }

    // Not a derived artifact. Create a corresponding source artifact. This isn't really great
    // because we have no guarantee that the artifact is not under a different package which
    // invalidates the guarantee of "bazel query" that the dependencies reported for a target are a
    // superset of all possible targets that are needed to build it, worse yet, there isn't even a
    // guarantee that there isn't a package on a different package path in between.
    //
    // For example, if the ninja_build rule is in a/BUILD and has a file a/b/c, it's possible that
    // there is a BUILD file a/b/BUILD and thus the source file a/b/c is created from the package
    // //a even though package //a/b exists (violating the above "bazel query" invariant) and it can
    // be that a/b/BUILD is on a different package path entry (is not correct because the other
    // package path entry can contain a *different* source file whose execpath is a/b/c)
    //
    // TODO(lberki): Check whether there is a package in between from another package path entry.
    // We probably can't prohibit packages in between, though. Schade.
    return ruleContext
        .getAnalysisEnvironment()
        .getSourceArtifactForNinjaBuild(
            execPath, ruleContext.getRule().getPackage().getSourceRoot());
  }

  public Artifact getDepsMappingArtifact(PathFragment fragment) {
    return depsNameToArtifact.get(fragment);
  }

  public PathFragment getOutputRootPath() {
    return outputRootPath;
  }

  public PathFragment getWorkingDirectory() {
    return workingDirectory;
  }
}
