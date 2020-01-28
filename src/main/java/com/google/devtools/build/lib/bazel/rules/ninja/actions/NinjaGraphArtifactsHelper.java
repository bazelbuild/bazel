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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;

public class NinjaGraphArtifactsHelper {
  private final RuleContext ruleContext;
  private final Path outputRootInSources;
  private final PathFragment outputRootPath;
  private final PathFragment workingDirectory;
  private final ArtifactRoot derivedOutputRoot;

  private final NestedSetBuilder<Artifact> outputFilesBuilder;
  private ImmutableSortedMap<PathFragment, Artifact> srcsMap;

  public NinjaGraphArtifactsHelper(
      RuleContext ruleContext,
      Root sourceRoot,
      PathFragment outputRootPath,
      PathFragment workingDirectory) {
    this.ruleContext = ruleContext;
    this.outputRootInSources = Preconditions.checkNotNull(sourceRoot.asPath())
        .getRelative(outputRootPath);
    this.outputRootPath = outputRootPath;
    this.workingDirectory = workingDirectory;
    this.outputFilesBuilder = NestedSetBuilder.stableOrder();
    Path execRoot = Preconditions.checkNotNull(ruleContext.getConfiguration()).getDirectories()
        .getExecRoot(ruleContext.getWorkspaceName());
    this.derivedOutputRoot = ArtifactRoot.asDerivedRoot(execRoot, outputRootPath);
  }

  public void createInputsMap() {
    ImmutableList<Artifact> srcs = ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list();
    ImmutableSortedMap.Builder<PathFragment, Artifact> inputsMapBuilder =
        ImmutableSortedMap.naturalOrder();
    srcs.forEach(a -> inputsMapBuilder.put(a.getRootRelativePath(), a));
    srcsMap = inputsMapBuilder.build();
  }

  public PathFragment createAbsolutePathUnderOutputRoot(PathFragment pathUnderOutputRoot) {
    return outputRootInSources.getRelative(pathUnderOutputRoot).asFragment();
  }

  public DerivedArtifact createDerivedArtifactUnderOutputRoot(PathFragment pathRelativeToWorkingDirectory)
      throws GenericParsingException {
    PathFragment pathRelativeToWorkspaceRoot =
        workingDirectory.getRelative(pathRelativeToWorkingDirectory);
    if (!pathRelativeToWorkspaceRoot.startsWith(outputRootPath)) {
      throw new GenericParsingException(
          String.format("Ninja actions are allowed to create outputs only under output_root,"
              + " path '%s' is not allowed.", pathRelativeToWorkingDirectory));
    }
    DerivedArtifact derivedArtifact = ruleContext.getDerivedArtifact(
        pathRelativeToWorkspaceRoot, derivedOutputRoot);
    outputFilesBuilder.add(derivedArtifact);
    return derivedArtifact;
  }

  public Artifact createInputArtifact(PathFragment pathRelativeToWorkingDirectory)
      throws GenericParsingException {
    Preconditions.checkNotNull(srcsMap);
    PathFragment pathRelativeToWorkspaceRoot =
        workingDirectory.getRelative(pathRelativeToWorkingDirectory);
    Artifact asInput = srcsMap.get(pathRelativeToWorkspaceRoot);
    if (asInput != null) {
      return asInput;
    }
    return createDerivedArtifactUnderOutputRoot(pathRelativeToWorkingDirectory);
  }

  public NestedSet<Artifact> getOutputFiles() {
    return outputFilesBuilder.build();
  }
}
