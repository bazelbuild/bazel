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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyCacher;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.MiddlemanType;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.SourceCategory;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** An {@link ActionTemplate} that expands into {@link CppCompileAction}s at execution time. */
public final class CppCompileActionTemplate extends ActionKeyCacher
    implements ActionTemplate<CppCompileAction> {
  private final CppCompileActionBuilder cppCompileActionBuilder;
  private final SpecialArtifact sourceTreeArtifact;
  private final SpecialArtifact outputTreeArtifact;
  private final SpecialArtifact dotdTreeArtifact;
  private final CcToolchainProvider toolchain;
  private final Iterable<ArtifactCategory> categories;
  private final ActionOwner actionOwner;
  private final NestedSet<Artifact> mandatoryInputs;
  private final NestedSet<Artifact> allInputs;

  /**
   * Creates an CppCompileActionTemplate.
   *
   * @param sourceTreeArtifact the TreeArtifact that contains source files to compile.
   * @param outputTreeArtifact the TreeArtifact that contains compilation outputs.
   * @param dotdTreeArtifact the TreeArtifact that contains dotd files.
   * @param cppCompileActionBuilder An almost completely configured {@link CppCompileActionBuilder}
   *     without the input and output files set. It is used as a template to instantiate expanded
   *     {CppCompileAction}s.
   * @param toolchain the CcToolchainProvider representing the c++ toolchain for this action
   * @param categories A list of {@link ArtifactCategory} used to calculate output file name from a
   *     source file name.
   * @param actionOwner the owner of this {@link ActionTemplate}.
   */
  CppCompileActionTemplate(
      SpecialArtifact sourceTreeArtifact,
      SpecialArtifact outputTreeArtifact,
      SpecialArtifact dotdTreeArtifact,
      CppCompileActionBuilder cppCompileActionBuilder,
      CcToolchainProvider toolchain,
      Iterable<ArtifactCategory> categories,
      ActionOwner actionOwner) {
    this.cppCompileActionBuilder = cppCompileActionBuilder;
    this.sourceTreeArtifact = sourceTreeArtifact;
    this.outputTreeArtifact = outputTreeArtifact;
    this.dotdTreeArtifact = dotdTreeArtifact;
    this.toolchain = toolchain;
    this.categories = categories;
    this.actionOwner = checkNotNull(actionOwner, outputTreeArtifact);
    this.mandatoryInputs = cppCompileActionBuilder.buildMandatoryInputs();
    this.allInputs =
        NestedSetBuilder.fromNestedSet(mandatoryInputs)
            .addTransitive(cppCompileActionBuilder.buildInputsForInvalidation())
            .build();
  }

  @Override
  public ImmutableList<CppCompileAction> generateActionsForInputArtifacts(
      ImmutableSet<TreeFileArtifact> inputTreeFileArtifacts, ActionLookupKey artifactOwner)
      throws ActionTemplateExpansionException {
    ImmutableList.Builder<CppCompileAction> expandedActions = new ImmutableList.Builder<>();

    ImmutableList.Builder<TreeFileArtifact> sourcesBuilder = ImmutableList.builder();
    NestedSetBuilder<Artifact> privateHeadersBuilder = NestedSetBuilder.<Artifact>stableOrder();
    for (TreeFileArtifact inputTreeFileArtifact : inputTreeFileArtifacts) {
      boolean isHeader = CppFileTypes.CPP_HEADER.matches(inputTreeFileArtifact.getExecPath());
      boolean isTextualInclude =
          CppFileTypes.CPP_TEXTUAL_INCLUDE.matches(inputTreeFileArtifact.getExecPath());
      boolean isSource =
          SourceCategory.CC_AND_OBJC
                  .getSourceTypes()
                  .matches(inputTreeFileArtifact.getExecPathString())
              && !isHeader;

      if (isHeader) {
        privateHeadersBuilder.add(inputTreeFileArtifact);
      }
      if (isSource || (isHeader && shouldCompileHeaders() && !isTextualInclude)) {
        sourcesBuilder.add(inputTreeFileArtifact);
      } else if (!isHeader) {
        throw new ActionTemplateExpansionException(
            String.format(
                "Artifact '%s' expanded from the directory artifact '%s' is neither header "
                    + "nor source file.",
                inputTreeFileArtifact.getExecPathString(), sourceTreeArtifact.getExecPathString()));
      }
    }
    ImmutableList<TreeFileArtifact> sources = sourcesBuilder.build();
    NestedSet<Artifact> privateHeaders = privateHeadersBuilder.build();

    for (TreeFileArtifact inputTreeFileArtifact : sources) {
      try {
        String outputName = outputTreeFileArtifactName(inputTreeFileArtifact);
        TreeFileArtifact outputTreeFileArtifact =
            TreeFileArtifact.createTemplateExpansionOutput(
                outputTreeArtifact, outputName, artifactOwner);
        TreeFileArtifact dotdFileArtifact = null;
        if (dotdTreeArtifact != null
            && cppCompileActionBuilder.useDotdFile(inputTreeFileArtifact)) {
          dotdFileArtifact =
              TreeFileArtifact.createTemplateExpansionOutput(
                  dotdTreeArtifact, outputName + ".d", artifactOwner);
        }
        expandedActions.add(
            createAction(
                inputTreeFileArtifact, outputTreeFileArtifact, dotdFileArtifact, privateHeaders));
      } catch (EvalException e) {
        throw new ActionTemplateExpansionException(e);
      }
    }

    return expandedActions.build();
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable Artifact.ArtifactExpander artifactExpander,
      Fingerprint fp)
      throws CommandLineExpansionException, InterruptedException {
    CompileCommandLine commandLine =
        CppCompileAction.buildCommandLine(
            sourceTreeArtifact,
            cppCompileActionBuilder.getCoptsFilter(),
            CppActionNames.CPP_COMPILE,
            dotdTreeArtifact,
            cppCompileActionBuilder.getFeatureConfiguration(),
            cppCompileActionBuilder.getVariables());
    CppCompileAction.computeKey(
        actionKeyContext,
        fp,
        cppCompileActionBuilder.getActionClassId(),
        cppCompileActionBuilder.getActionEnvironment(),
        commandLine.getEnvironment(),
        cppCompileActionBuilder.getExecutionInfo(),
        CppCompileAction.computeCommandLineKey(
            commandLine.getCompilerOptions(/*overwrittenVariables=*/ null)),
        cppCompileActionBuilder.getCcCompilationContext().getDeclaredIncludeSrcs(),
        cppCompileActionBuilder.buildMandatoryInputs(),
        cppCompileActionBuilder.getPrunableHeaders(),
        cppCompileActionBuilder.getCcCompilationContext().getLooseHdrsDirs(),
        cppCompileActionBuilder.getBuiltinIncludeDirectories(),
        cppCompileActionBuilder.buildInputsForInvalidation(),
        toolchain
            .getCppConfigurationEvenThoughItCanBeDifferentThanWhatTargetHas()
            .validateTopLevelHeaderInclusions());
  }

  private boolean shouldCompileHeaders() {
    return cppCompileActionBuilder.shouldCompileHeaders();
  }

  private CppCompileAction createAction(
      Artifact sourceTreeFileArtifact,
      Artifact outputTreeFileArtifact,
      @Nullable Artifact dotdFileArtifact,
      NestedSet<Artifact> privateHeaders)
      throws ActionTemplateExpansionException {
    CppCompileActionBuilder builder = new CppCompileActionBuilder(cppCompileActionBuilder);
    builder.setAdditionalPrunableHeaders(privateHeaders);
    builder.setSourceFile(sourceTreeFileArtifact);
    builder.setOutputs(outputTreeFileArtifact, dotdFileArtifact);

    CcToolchainVariables.Builder buildVariables =
        CcToolchainVariables.builder(cppCompileActionBuilder.getVariables());
    buildVariables.overrideStringVariable(
        CompileBuildVariables.SOURCE_FILE.getVariableName(),
        sourceTreeFileArtifact.getExecPathString());
    buildVariables.overrideStringVariable(
        CompileBuildVariables.OUTPUT_FILE.getVariableName(),
        outputTreeFileArtifact.getExecPathString());
    if (dotdFileArtifact != null) {
      buildVariables.overrideStringVariable(
          CompileBuildVariables.DEPENDENCY_FILE.getVariableName(),
          dotdFileArtifact.getExecPathString());
    }

    builder.setVariables(buildVariables.build());

    List<String> errors = new ArrayList<>();
    CppCompileAction result =
        builder.buildAndVerify((String errorMessage) -> errors.add(errorMessage));
    if (!errors.isEmpty()) {
      throw new ActionTemplateExpansionException(Joiner.on(".\n").join(errors));
    }

    return result;
  }

  private String outputTreeFileArtifactName(TreeFileArtifact inputTreeFileArtifact)
      throws EvalException {
    String outputName = FileSystemUtils.removeExtension(
        inputTreeFileArtifact.getParentRelativePath().getPathString());
    for (ArtifactCategory category : categories) {
      outputName = toolchain.getFeatures().getArtifactNameForCategory(category, outputName);
    }
    return outputName;
  }

  @Override
  public SpecialArtifact getInputTreeArtifact() {
    return sourceTreeArtifact;
  }

  @Override
  public SpecialArtifact getOutputTreeArtifact() {
    return outputTreeArtifact;
  }

  @Override
  public ActionOwner getOwner() {
    return actionOwner;
  }

  @Override
  public boolean isShareable() {
    return false;
  }

  @Override
  public final String getMnemonic() {
    return "CppCompileActionTemplate";
  }

  @Override
  public NestedSet<Artifact> getMandatoryInputs() {
    return NestedSetBuilder.<Artifact>compileOrder()
        .add(sourceTreeArtifact)
        .addTransitive(mandatoryInputs)
        .build();
  }

  @Override
  public NestedSet<Artifact> getInputFilesForExtraAction(
      ActionExecutionContext actionExecutionContext) {
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  @Override
  public ImmutableSet<Artifact> getMandatoryOutputs() {
    return ImmutableSet.of();
  }

  @Override
  public NestedSet<Artifact> getTools() {
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  @Override
  public NestedSet<Artifact> getInputs() {
    return NestedSetBuilder.<Artifact>stableOrder()
        .add(sourceTreeArtifact)
        .addTransitive(allInputs)
        .build();
  }

  @Override
  public ImmutableSet<Artifact> getOutputs() {
    if (dotdTreeArtifact == null) {
      return ImmutableSet.of(outputTreeArtifact);
    }
    return ImmutableSet.of(outputTreeArtifact, dotdTreeArtifact);
  }

  @Override
  public Iterable<String> getClientEnvironmentVariables() {
    return ImmutableList.of();
  }

  @Override
  public boolean hasLooseHeaders() {
    return CppCompileAction.hasLooseHeaders(
        cppCompileActionBuilder.getCcCompilationContext(),
        cppCompileActionBuilder.getFeatureConfiguration());
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
    return "CppCompileActionTemplate compiling " + sourceTreeArtifact.getExecPathString();
  }

  @Override
  public String describe() {
    return "Compiling all C++ files in " + sourceTreeArtifact.prettyPrint();
  }

  @Override
  public String toString() {
    return prettyPrint();
  }
}
