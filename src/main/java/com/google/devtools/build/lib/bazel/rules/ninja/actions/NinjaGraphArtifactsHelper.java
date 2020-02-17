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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;

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
  private final Path outputRootInSources;
  private final PathFragment outputRootPath;
  private final PathFragment workingDirectory;
  private final ArtifactRoot derivedOutputRoot;

  private final ImmutableSortedMap<PathFragment, Artifact> depsNameToArtifact;
  private final ImmutableSortedMap<PathFragment, Artifact> srcsMap;

  /**
   * Constructor
   *
   * @param ruleContext parent NinjaGraphRule rule context
   * @param sourceRoot the source root, under which the main Ninja file resides.
   * @param outputRootPath name of output directory for Ninja actions under execroot
   * @param workingDirectory relative path under execroot, the root for interpreting all paths in
   *     Ninja file
   * @param srcsMap mapping between the path fragment and artifact for the files passed in 'srcs'
   *     attribute
   * @param depsNameToArtifact mapping between the path fragment in the Ninja file and prebuilt
   */
  NinjaGraphArtifactsHelper(
      RuleContext ruleContext,
      Root sourceRoot,
      PathFragment outputRootPath,
      PathFragment workingDirectory,
      ImmutableSortedMap<PathFragment, Artifact> srcsMap,
      ImmutableSortedMap<PathFragment, Artifact> depsNameToArtifact) {
    this.ruleContext = ruleContext;
    this.outputRootInSources =
        Preconditions.checkNotNull(sourceRoot.asPath()).getRelative(outputRootPath);
    this.outputRootPath = outputRootPath;
    this.workingDirectory = workingDirectory;
    this.srcsMap = srcsMap;
    this.depsNameToArtifact = depsNameToArtifact;
    Path execRoot =
        Preconditions.checkNotNull(ruleContext.getConfiguration())
            .getDirectories()
            .getExecRoot(ruleContext.getWorkspaceName());
    this.derivedOutputRoot = ArtifactRoot.asDerivedRoot(execRoot, outputRootPath);
  }

  PathFragment createAbsolutePathUnderOutputRoot(PathFragment pathUnderOutputRoot) {
    return outputRootInSources.getRelative(pathUnderOutputRoot).asFragment();
  }

  DerivedArtifact createOutputArtifact(PathFragment pathRelativeToWorkingDirectory)
      throws GenericParsingException {
    PathFragment pathRelativeToWorkspaceRoot =
        workingDirectory.getRelative(pathRelativeToWorkingDirectory);
    if (!pathRelativeToWorkspaceRoot.startsWith(outputRootPath)) {
      throw new GenericParsingException(
          String.format(
              "Ninja actions are allowed to create outputs only under output_root,"
                  + " path '%s' is not allowed.",
              pathRelativeToWorkingDirectory));
    }
    DerivedArtifact derivedArtifact =
        ruleContext.getDerivedArtifact(
            pathRelativeToWorkspaceRoot.relativeTo(outputRootPath), derivedOutputRoot);
    return derivedArtifact;
  }

  Artifact getInputArtifact(PathFragment pathRelativeToWorkingDirectory)
      throws GenericParsingException {
    Preconditions.checkNotNull(srcsMap);
    PathFragment pathRelativeToWorkspaceRoot =
        workingDirectory.getRelative(pathRelativeToWorkingDirectory);
    Artifact asInput = srcsMap.get(pathRelativeToWorkspaceRoot);
    Artifact depsMappingArtifact = depsNameToArtifact.get(pathRelativeToWorkingDirectory);
    if (asInput != null && depsMappingArtifact != null) {
      throw new GenericParsingException(
          String.format(
              "Source file '%s' is passed both in 'srcs' " + "and 'deps_mapping' attributes.",
              pathRelativeToWorkingDirectory));
    }
    if (asInput != null) {
      return asInput;
    }
    if (depsMappingArtifact != null) {
      return depsMappingArtifact;
    }
    return createOutputArtifact(pathRelativeToWorkingDirectory);
  }

  public PathFragment getOutputRootPath() {
    return outputRootPath;
  }

  public PathFragment getWorkingDirectory() {
    return workingDirectory;
  }
}
