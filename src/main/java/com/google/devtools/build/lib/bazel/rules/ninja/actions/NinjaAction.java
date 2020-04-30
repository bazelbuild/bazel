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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.NinjaMysteryArtifact;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.rules.cpp.CppIncludeExtractionContext;
import com.google.devtools.build.lib.skyframe.TrackSourceDirectoriesFlag;
import com.google.devtools.build.lib.util.DependencySet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/** Generic class for Ninja actions. Corresponds to the {@link NinjaTarget} in the Ninja file. */
public class NinjaAction extends SpawnAction {
  private static final String MNEMONIC = "NinjaGenericAction";

  private final Root sourceRoot;
  @Nullable private final Artifact depFile;
  private final ImmutableMap<PathFragment, Artifact> allowedDerivedInputs;
  private final ArtifactRoot derivedOutputRoot;

  public NinjaAction(
      ActionOwner owner,
      Root sourceRoot,
      NestedSet<Artifact> tools,
      NestedSet<Artifact> inputs,
      List<? extends Artifact> outputs,
      CommandLines commandLines,
      ActionEnvironment env,
      ImmutableMap<String, String> executionInfo,
      CharSequence progressMessage,
      RunfilesSupplier runfilesSupplier,
      boolean executeUnconditionally,
      @Nullable Artifact depFile,
      ArtifactRoot derivedOutputRoot) {
    super(
        /* owner= */ owner,
        /* tools= */ tools,
        /* inputs= */ inputs,
        /* outputs= */ outputs,
        /* primaryOutput= */ Iterables.getFirst(outputs, null),
        /* resourceSet= */ AbstractAction.DEFAULT_RESOURCE_SET,
        /* commandLines= */ commandLines,
        /* commandLineLimits= */ CommandLineLimits.UNLIMITED,
        /* isShellCommand= */ true,
        /* env= */ env,
        /* executionInfo= */ executionInfo,
        /* progressMessage= */ progressMessage,
        /* runfilesSupplier= */ runfilesSupplier,
        /* mnemonic= */ MNEMONIC,
        /* executeUnconditionally= */ executeUnconditionally,
        /* extraActionInfoSupplier= */ null,
        /* resultConsumer= */ null);
    this.sourceRoot = sourceRoot;
    this.depFile = depFile;
    ImmutableMap.Builder<PathFragment, Artifact> allowedDerivedInputsBuilder =
        ImmutableMap.builder();
    for (Artifact input : inputs.toList()) {
      if (!input.isSourceArtifact()) {
        allowedDerivedInputsBuilder.put(input.getExecPath(), input);
      }
    }
    this.allowedDerivedInputs = allowedDerivedInputsBuilder.build();
    this.derivedOutputRoot = derivedOutputRoot;
  }

  @Override
  protected void beforeExecute(ActionExecutionContext actionExecutionContext) throws IOException {
    if (!TrackSourceDirectoriesFlag.trackSourceDirectories()) {
      checkInputsForDirectories(
          actionExecutionContext.getEventHandler(), actionExecutionContext.getMetadataProvider());
    }
  }

  @Override
  protected void afterExecute(
      ActionExecutionContext actionExecutionContext, List<SpawnResult> spawnResults)
      throws EnvironmentalExecException {
    checkOutputsForDirectories(actionExecutionContext);

    if (depFile != null) {
      updateInputsFromDepfile(actionExecutionContext);
    }
  }

  private void updateInputsFromDepfile(ActionExecutionContext actionExecutionContext)
      throws EnvironmentalExecException {
    boolean siblingRepositoryLayout =
        actionExecutionContext
            .getOptions()
            .getOptions(StarlarkSemanticsOptions.class)
            .experimentalSiblingRepositoryLayout;
    CppIncludeExtractionContext scanningContext =
        actionExecutionContext.getContext(CppIncludeExtractionContext.class);
    ArtifactResolver artifactResolver = scanningContext.getArtifactResolver();
    Path execRoot = actionExecutionContext.getExecRoot();
    try {
      DependencySet depSet =
          new DependencySet(execRoot).read(actionExecutionContext.getInputPath(depFile));
      NestedSetBuilder<Artifact> inputsBuilder = NestedSetBuilder.stableOrder();
      for (Path inputPath : depSet.getDependencies()) {
        PathFragment execRelativePath = null;

        // This branch needed in case the depfile contains an absolute path to a source file.
        if (sourceRoot.contains(inputPath)) {
          execRelativePath = inputPath.asFragment().relativeTo(sourceRoot.asPath().asFragment());
        } else {
          execRelativePath = inputPath.asFragment().relativeTo(execRoot.asFragment());
        }
        Artifact inputArtifact = null;
        if (allowedDerivedInputs.containsKey(execRelativePath)) {
          // Predeclared generated input.
          inputArtifact = allowedDerivedInputs.get(execRelativePath);
        }
        if (inputArtifact == null) {
          RepositoryName repository =
              PackageIdentifier.discoverFromExecPath(
                      execRelativePath, false, siblingRepositoryLayout)
                  .getRepository();
          if (execRelativePath.startsWith(derivedOutputRoot.getExecPath())) {
            // This input is a generated file which was not declared in the original inputs for
            // this action.
            inputArtifact = new NinjaMysteryArtifact(derivedOutputRoot, execRelativePath);
          } else {
            // Source file input.
            inputArtifact = artifactResolver.resolveSourceArtifact(execRelativePath, repository);
          }
        }

        if (inputArtifact == null) {
          throw new EnvironmentalExecException(
              String.format(
                  "depfile-declared dependency '%s' is invalid: it must either be "
                      + "a source input, or a pre-declared generated input",
                  execRelativePath));
        }
        inputsBuilder.add(inputArtifact);
      }
      updateInputs(inputsBuilder.build());
    } catch (IOException e) {
      // Some kind of IO or parse exception--wrap & rethrow it to stop the build.
      throw new EnvironmentalExecException("error while parsing .d file: " + e.getMessage(), e);
    }
  }

  @Override
  public boolean discoversInputs() {
    return depFile != null;
  }

  @Override
  public NestedSet<Artifact> getAllowedDerivedInputs() {
    return getInputs();
  }

  @Override
  public NestedSet<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext) {
    updateInputs(getInputs());
    return getInputs();
  }
}
