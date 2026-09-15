// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractCommandLine;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/** Remaining code that is needed because {@link LtoBackendActionTemplate} is still native. */
public final class LtoBackendArtifacts {
  private LtoBackendArtifacts() {}

  // LINT.IfChange(lto_backends)
  private static void addPathsToBuildVariablesBuilder(
      CcToolchainVariables.Builder buildVariablesBuilder,
      String indexPath,
      String objectFilePath,
      String dwoFilePath,
      String bitcodeFilePath) {
    // Ideally, those strings would come directly from the execPath of the Artifacts of
    // the LtoBackendAction.Builder; however, in order to support tree artifacts, we need
    // the bitcodeFilePath to be different from the bitcodeTreeArtifact execPath.
    // The former is a file path and the latter is the directory path.
    // Therefore we accept strings as inputs rather than artifacts.
    if (indexPath != null) {
      buildVariablesBuilder.addVariable("thinlto_index", indexPath);
    } else {
      // An empty input indicates not to perform cross-module optimization.
      buildVariablesBuilder.addVariable("thinlto_index", "/dev/null");
    }
    // The output from the LTO backend step is a native object file.
    buildVariablesBuilder.addVariable("thinlto_output_object_file", objectFilePath);
    // The input to the LTO backend step is the bitcode file.
    buildVariablesBuilder.addVariable("thinlto_input_bitcode_file", bitcodeFilePath);
    // Add the context sensitive instrument path to the backend.

    if (dwoFilePath != null) {
      buildVariablesBuilder.addVariable(
          CompileBuildVariables.PER_OBJECT_DEBUG_INFO_FILE.getVariableName(), dwoFilePath);
      buildVariablesBuilder.addVariable(
          CompileBuildVariables.IS_USING_FISSION.getVariableName(), "");
    }
  }

  private static NestedSet<Artifact> getLtoBackendActionInputs(
      @Nullable Artifact index,
      @Nullable Artifact imports,
      Artifact bitcodeFile,
      NestedSet<Artifact> additionalInputs) {
    NestedSetBuilder<Artifact> inputsBuilder = NestedSetBuilder.stableOrder();
    inputsBuilder.addTransitive(additionalInputs);
    inputsBuilder.add(bitcodeFile);
    if (imports != null) {
      // Although the imports file is not used by the LTOBackendAction while the action is
      // executing, it is needed during the input discovery phase, and we must list it as an input
      // to the action in order for it to be preserved under --discard_orphaned_artifacts.
      inputsBuilder.add(imports);
    }
    if (index != null) {
      inputsBuilder.add(index);
    }
    return inputsBuilder.build();
  }

  private static ImmutableSet<Artifact> getLtoBackendActionOutputs(
      Artifact objectFile, Artifact dwoFile) {
    ImmutableSet.Builder<Artifact> builder = ImmutableSet.builder();
    builder.add(objectFile);
    // Add the context sensitive instrument path to the backend.
    if (dwoFile != null) {
      builder.add(dwoFile);
    }
    return builder.build();
  }

  private static CommandLines getLtoBackendCommandLine(
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables buildVariables,
      boolean usePic) {
    CommandLine ltoCommandLine =
        new AbstractCommandLine() {

          @Override
          public Iterable<String> arguments() throws CommandLineExpansionException {
            return arguments(/* inputMetadataProvider= */ null, PathMapper.NOOP);
          }

          @Override
          public ImmutableList<String> arguments(
              InputMetadataProvider inputMetadataProvider, PathMapper pathMapper)
              throws CommandLineExpansionException {
            ImmutableList.Builder<String> args = ImmutableList.builder();
            try {
              args.addAll(
                  featureConfiguration.getCommandLine(
                      CppActionNames.LTO_BACKEND,
                      buildVariables,
                      inputMetadataProvider,
                      pathMapper));
            } catch (ExpansionException e) {
              throw new CommandLineExpansionException(e.getMessage());
            }
            // If this is a PIC compile (set based on the CppConfiguration), the PIC
            // option should be added after the rest of the command line so that it
            // cannot be overridden. This is consistent with the ordering in the
            // CppCompileAction's compiler options.
            if (usePic) {
              args.add("-fPIC");
            }
            return args.build();
          }
        };
    PathFragment compiler =
        PathFragment.create(featureConfiguration.getToolPathForAction(CppActionNames.LTO_BACKEND));
    return CommandLines.builder()
        .addSingleArgument(compiler)
        .addCommandLine(ltoCommandLine)
        .build();
  }

  public static LtoBackendAction createLtoBackendActionForStarlark(
      ActionOwner owner,
      @Nullable BuildConfigurationValue configuration,
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables buildVariables,
      boolean usePic,
      NestedSet<Artifact> inputs,
      BitcodeFiles allBitcodeFiles,
      Artifact imports,
      ImmutableSet<Artifact> outputs,
      ActionEnvironment env) {

    CommandLines commandLines =
        LtoBackendArtifacts.getLtoBackendCommandLine(featureConfiguration, buildVariables, usePic);
    return LtoBackendAction.create(
        owner, configuration, inputs, allBitcodeFiles, imports, outputs, commandLines, env);
  }

  /**
   * Adds artifact to builder. The resulting builder can be built into a valid ltoBackendAction.
   *
   * <p>Assumes that buildVariables have been initialized previously. If this is not true, the
   * action will be wrong.
   *
   * @param buildVariables preinitialized CcToolchainVariables.
   * @param featureConfiguration the feature configuration to get the command line for the builder.
   * @param index the index artifact to add. Can be a TreeFileArtifact but cannot be a Tree
   *     Artifact.
   * @param imports the imports artifact to add. Can be a TreeFileArtifact but cannot be a Tree
   *     Artifact.
   * @param bitcodeArtifact the bitcode artifact to add. If it is a Tree Artifact, bitcodeFilePath
   *     must be set.
   * @param objectFile the object file to add. Can be a TreeFileArtifact but cannot be a Tree
   *     Artifact.
   * @param bitcodeFiles the bitcode files to add.
   * @param dwoFile the dwo file to add.
   * @param usePic whether to add the PIC option to the command line.
   * @param bitcodeFilePath the path of the bitcode object we are compiling. Only used if
   *     bitcodeArtifact is a tree artifact.
   * @param isDummyAction if true then ignores the preconditions, because it is generating a dummy
   *     action, not a valid action.
   */
  public static LtoBackendAction createLtoBackendActionForTemplate(
      ActionOwner actionOwner,
      NestedSet<Artifact> additionalInputs,
      ActionEnvironment env,
      CcToolchainVariables buildVariables,
      FeatureConfiguration featureConfiguration,
      @Nullable Artifact index,
      @Nullable Artifact imports,
      Artifact bitcodeArtifact,
      Artifact objectFile,
      @Nullable BitcodeFiles bitcodeFiles,
      @Nullable Artifact dwoFile,
      boolean usePic,
      @Nullable String bitcodeFilePath,
      boolean isDummyAction) {
    Preconditions.checkState(
        isDummyAction
            || ((index == null || !index.isTreeArtifact())
                && (imports == null || !imports.isTreeArtifact())
                && (dwoFile == null || !dwoFile.isTreeArtifact())
                && !objectFile.isTreeArtifact()),
        "index, imports, object and dwo files cannot be TreeArtifacts. We need to know their exact"
            + " path not just directory path.");
    Preconditions.checkState(
        isDummyAction || (bitcodeArtifact.isTreeArtifact() ^ bitcodeFilePath == null),
        "If bitcode file is a tree artifact, the bitcode file path must contain the path. If it is"
            + " not a tree artifact, then bitcode file path should be null to not override the"
            + " path.");
    CcToolchainVariables.Builder buildVariablesBuilder =
        CcToolchainVariables.builder(buildVariables);
    NestedSet<Artifact> inputs =
        getLtoBackendActionInputs(index, imports, bitcodeArtifact, additionalInputs);
    ImmutableSet<Artifact> outputs = getLtoBackendActionOutputs(objectFile, dwoFile);

    String indexPath = index == null ? null : index.getExecPathString();
    String dwoFilePath = dwoFile == null ? null : dwoFile.getExecPathString();
    addPathsToBuildVariablesBuilder(
        buildVariablesBuilder,
        indexPath,
        objectFile.getExecPathString(),
        dwoFilePath,
        bitcodeFilePath != null ? bitcodeFilePath : bitcodeArtifact.getExecPathString());
    CcToolchainVariables buildVariablesWithFiles = buildVariablesBuilder.build();

    return createLtoBackendActionForStarlark(
        actionOwner,
        /* configuration= */ null,
        featureConfiguration,
        buildVariablesWithFiles,
        usePic,
        inputs,
        bitcodeFiles,
        imports,
        outputs,
        env);
  }
  // LINT.ThenChange(@rules_cc//cc/private/link/lto_backends.bzl:lto_backends)

}
