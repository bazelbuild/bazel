// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.analysis.actions.ActionTemplate;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;

/**
 * An {@link ActionTemplate} that expands into {@link CppCompileAction}s at execution time.
 */
public final class CppCompileActionTemplate implements ActionTemplate<CppCompileAction> {
  private final CppCompileActionBuilder cppCompileActionBuilder;
  private final Artifact sourceTreeArtifact;
  private final Artifact outputTreeArtifact;
  private final CppConfiguration cppConfiguration;
  private final Iterable<ArtifactCategory> categories;
  private final ActionOwner actionOwner;
  private final NestedSet<Artifact> mandatoryInputs;
  private final NestedSet<Artifact> allInputs;

  /**
   * Creates an CppCompileActionTemplate.
   * @param sourceTreeArtifact the TreeArtifact that contains source files to compile.
   * @param outputTreeArtifact the TreeArtifact that contains compilation outputs.
   * @param cppCompileActionBuilder An almost completely configured  {@link CppCompileActionBuilder}
   *     without the input and output files set. It is used as a template to instantiate expanded
   *     {CppCompileAction}s.
   * @param cppConfiguration configuration for cpp.
   * @param categories A list of {@link ArtifactCategory} used to calculate output file name from
   *     a source file name.
   * @param actionOwner the owner of this {@link ActionTemplate}.
   */
  CppCompileActionTemplate(
      Artifact sourceTreeArtifact,
      Artifact outputTreeArtifact,
      CppCompileActionBuilder cppCompileActionBuilder,
      CppConfiguration cppConfiguration,
      Iterable<ArtifactCategory> categories,
      ActionOwner actionOwner) {
    this.cppCompileActionBuilder = cppCompileActionBuilder;
    this.sourceTreeArtifact = sourceTreeArtifact;
    this.outputTreeArtifact = outputTreeArtifact;
    this.cppConfiguration = cppConfiguration;
    this.categories = categories;
    this.actionOwner = actionOwner;
    this.mandatoryInputs = cppCompileActionBuilder.buildMandatoryInputs();
    this.allInputs = cppCompileActionBuilder.buildAllInputs(this.mandatoryInputs);
  }

  @Override
  public Iterable<CppCompileAction> generateActionForInputArtifacts(
      Iterable<TreeFileArtifact> inputTreeFileArtifacts, ArtifactOwner artifactOwner)
      throws ActionTemplateExpansionException {
    ImmutableList.Builder<CppCompileAction> expandedActions = new ImmutableList.Builder<>();
    for (TreeFileArtifact inputTreeFileArtifact : inputTreeFileArtifacts) {
      String outputName = outputTreeFileArtifactName(inputTreeFileArtifact);
      TreeFileArtifact outputTreeFileArtifact = ActionInputHelper.treeFileArtifact(
          outputTreeArtifact,
          PathFragment.create(outputName),
          artifactOwner);

      expandedActions.add(createAction(inputTreeFileArtifact, outputTreeFileArtifact));
    }

    return expandedActions.build();
  }

  private CppCompileAction createAction(
      Artifact sourceTreeFileArtifact, Artifact outputTreeFileArtifact)
      throws ActionTemplateExpansionException {
    CppCompileActionBuilder builder = new CppCompileActionBuilder(cppCompileActionBuilder);
    builder.setSourceFile(sourceTreeFileArtifact);
    builder.setOutputs(outputTreeFileArtifact, null);

    CcToolchainFeatures.Variables.Builder buildVariables =
        new CcToolchainFeatures.Variables.Builder(cppCompileActionBuilder.getVariables());
    buildVariables.overrideStringVariable(
        "source_file", sourceTreeFileArtifact.getExecPathString());
    buildVariables.overrideStringVariable(
        "output_file", outputTreeFileArtifact.getExecPathString());
    buildVariables.overrideStringVariable(
        "output_object_file", outputTreeFileArtifact.getExecPathString());

    builder.setVariables(buildVariables.build());

    List<String> errors = new ArrayList<>();
    CppCompileAction result =
        builder.buildAndVerify((String errorMessage) -> errors.add(errorMessage));
    if (!errors.isEmpty()) {
      throw new ActionTemplateExpansionException(Joiner.on(".\n").join(errors));
    }

    return result;
  }

  private String outputTreeFileArtifactName(TreeFileArtifact inputTreeFileArtifact) {
    String outputName = FileSystemUtils.removeExtension(
        inputTreeFileArtifact.getParentRelativePath().getPathString());
    for (ArtifactCategory category : categories) {
      outputName = cppConfiguration.getFeatures().getArtifactNameForCategory(category, outputName);
    }
    return outputName;
  }

  @Override
  public Artifact getInputTreeArtifact() {
    return sourceTreeArtifact;
  }

  @Override
  public Artifact getOutputTreeArtifact() {
    return outputTreeArtifact;
  }

  @Override
  public ActionOwner getOwner() {
    return actionOwner;
  }

  @Override
  public final String getMnemonic() {
    return "CppCompileActionTemplate";
  }

  @Override
  public Iterable<Artifact> getMandatoryInputs() {
    return NestedSetBuilder.<Artifact>compileOrder()
        .add(sourceTreeArtifact)
        .addTransitive(mandatoryInputs)
        .build();
  }

  @Override
  public Iterable<Artifact> getInputFilesForExtraAction(
      ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    return ImmutableList.of();
  }

  @Override
  public ImmutableSet<Artifact> getMandatoryOutputs() {
    return ImmutableSet.<Artifact>of();
  }

  @Override
  public Iterable<Artifact> getTools() {
    return ImmutableList.<Artifact>of();
  }

  @Override
  public Iterable<Artifact> getInputs() {
    return NestedSetBuilder.<Artifact>stableOrder()
        .add(sourceTreeArtifact)
        .addTransitive(allInputs)
        .build();
  }

  @Override
  public ImmutableSet<Artifact> getOutputs() {
    return ImmutableSet.of(outputTreeArtifact);
  }

  @Override
  public Iterable<String> getClientEnvironmentVariables() {
    return ImmutableList.<String>of();
  }

  @Override
  public Artifact getPrimaryInput() {
    return sourceTreeArtifact;
  }

  @Override
  public Artifact getPrimaryOutput() {
    return outputTreeArtifact;
  }

  @Override
  public boolean shouldReportPathPrefixConflict(ActionAnalysisMetadata action) {
    return this != action;
  }

  @Override
  public MiddlemanType getActionType() {
    return MiddlemanType.NORMAL;
  }

  @Override
  public String prettyPrint() {
    return String.format(
        "CppCompileActionTemplate compiling " + sourceTreeArtifact.getExecPathString());
  }
}
