// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CoptsFilter;
import com.google.devtools.build.lib.rules.cpp.CcCommon.Language;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariablesExtension;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * A class to create C/C++ compile actions in a way that is consistent with cc_library. Rules that
 * generate source files and emulate cc_library on top of that should use this class instead of the
 * lower-level APIs in CppHelper and CppCompileActionBuilder.
 *
 * <p>Rules that want to use this class are required to have implicit dependencies on the toolchain,
 * the STL, and so on. Optionally, they can also have copts, and malloc attributes, but note that
 * these require explicit calls to the corresponding setter methods.
 */
public final class CcStaticCompilationHelper {

  private CcStaticCompilationHelper() {}

  static String computeOutputNamePrefixDir(BuildConfigurationValue configuration, String purpose) {
    String outputNamePrefixDir = null;
    // purpose is only used by objc rules; if set it ends with either "_non_objc_arc" or
    // "_objc_arc", and it is used to override configuration.getMnemonic() to prefix the output
    // dir with "non_arc" or "arc".
    String mnemonic = configuration.getMnemonic();
    if (purpose != null) {
      mnemonic = purpose;
    }
    if (mnemonic.endsWith("_objc_arc")) {
      outputNamePrefixDir = mnemonic.endsWith("_non_objc_arc") ? "non_arc" : "arc";
    }
    return outputNamePrefixDir;
  }

  // Methods setting up compile build variables:

  static CcToolchainVariables setupCommonCompileBuildVariables(
      CcCompilationContext ccCompilationContext,
      CcToolchainProvider ccToolchain,
      CppConfiguration cppConfiguration,
      FdoContext fdoContext,
      FeatureConfiguration featureConfiguration,
      List<VariablesExtension> variablesExtensions)
      throws RuleErrorException, EvalException {
    Map<String, String> genericAdditionalBuildVariables = new LinkedHashMap<>();
    CcToolchainVariables cctoolchainVariables;
    try {
      cctoolchainVariables = ccToolchain.getBuildVars();
    } catch (EvalException e) {
      throw new RuleErrorException(e.getMessage());
    }
    boolean isUsingMemProf = false;
    if (fdoContext != null && fdoContext.getMemProfProfileArtifact() != null) {
      isUsingMemProf = true;
    }
    CcToolchainVariables.Builder buildVariables =
        CcToolchainVariables.builder(cctoolchainVariables);
    CompileBuildVariables.setupCommonVariables(
        buildVariables,
        featureConfiguration,
        ImmutableList.of(),
        CppHelper.getFdoBuildStamp(cppConfiguration, fdoContext, featureConfiguration),
        isUsingMemProf,
        variablesExtensions,
        genericAdditionalBuildVariables,
        ccCompilationContext.getIncludeDirs(),
        ccCompilationContext.getQuoteIncludeDirs(),
        ccCompilationContext.getSystemIncludeDirs(),
        ccCompilationContext.getFrameworkIncludeDirs(),
        ccCompilationContext.getDefines(),
        ccCompilationContext.getNonTransitiveDefines(),
        ccCompilationContext.getExternalIncludeDirs());

    return buildVariables.build();
  }

  private static CcToolchainVariables setupSpecificCompileBuildVariables(
      CcToolchainVariables commonToolchainVariables,
      CcCompilationContext ccCompilationContext,
      ImmutableList<String> conlyopts,
      ImmutableList<String> copts,
      CppConfiguration cppConfiguration,
      ImmutableList<String> cxxopts,
      FdoContext fdoContext,
      NestedSet<Artifact> auxiliaryFdoInputs,
      FeatureConfiguration featureConfiguration,
      CppSemantics semantics,
      CppCompileActionBuilder ccCompileActionBuilder,
      Label sourceLabel,
      boolean usePic,
      boolean needsFdoBuildVariables,
      ImmutableMap<String, String> fdoBuildVariables,
      CppModuleMap cppModuleMap,
      boolean enableCoverage,
      Artifact gcnoFile,
      boolean isUsingFission,
      Artifact dwoFile,
      Artifact ltoIndexingFile,
      ImmutableMap<String, String> additionalBuildVariables)
      throws EvalException {
    Artifact sourceFile = ccCompileActionBuilder.getSourceFile();
    if (needsFdoBuildVariables && fdoContext.hasArtifacts()) {
      // This modifies the passed-in builder, which is a surprising side-effect, and makes it unsafe
      // to call this method multiple times for the same builder.
      ccCompileActionBuilder.addMandatoryInputs(auxiliaryFdoInputs);
    }
    CcToolchainVariables.Builder buildVariables =
        CcToolchainVariables.builder(commonToolchainVariables);
    CompileBuildVariables.setupSpecificVariables(
        buildVariables,
        sourceFile,
        ccCompileActionBuilder.getOutputFile(),
        enableCoverage,
        gcnoFile,
        dwoFile,
        isUsingFission,
        ltoIndexingFile,
        getCopts(
            conlyopts,
            copts,
            cppConfiguration,
            cxxopts,
            semantics,
            ccCompileActionBuilder.getSourceFile(),
            sourceLabel),
        ccCompileActionBuilder.getDotdFile(),
        ccCompileActionBuilder.getDiagnosticsFile(),
        usePic,
        featureConfiguration,
        cppModuleMap,
        ccCompilationContext.getDirectModuleMaps(),
        additionalBuildVariables);
    if (needsFdoBuildVariables) {
      buildVariables.addAllStringVariables(fdoBuildVariables);
    }
    return buildVariables.build();
  }

  // FDO related methods:

  /**
   * Configures a compile action builder by setting up command line options and auxiliary inputs
   * according to the FDO configuration. This method does nothing If FDO is disabled.
   */
  static ImmutableMap<String, String> setupFdoBuildVariables(
      CcToolchainProvider ccToolchain,
      FdoContext fdoContext,
      NestedSet<Artifact> auxiliaryFdoInputs,
      FeatureConfiguration featureConfiguration,
      String fdoInstrument,
      String csFdoInstrument)
      throws EvalException {
    ImmutableMap.Builder<String, String> variablesBuilder = ImmutableMap.builder();
    if (featureConfiguration.isEnabled(CppRuleClasses.FDO_INSTRUMENT)) {
      variablesBuilder.put(
          CompileBuildVariables.FDO_INSTRUMENT_PATH.getVariableName(), fdoInstrument);
    }
    if (featureConfiguration.isEnabled(CppRuleClasses.CS_FDO_INSTRUMENT)) {
      variablesBuilder.put(
          CompileBuildVariables.CS_FDO_INSTRUMENT_PATH.getVariableName(), csFdoInstrument);
    }

    // FDO is disabled -> do nothing.
    Preconditions.checkNotNull(fdoContext);
    if (!fdoContext.hasArtifacts()) {
      return variablesBuilder.buildOrThrow();
    }

    if (fdoContext.getPrefetchHintsArtifact() != null) {
      variablesBuilder.put(
          CompileBuildVariables.FDO_PREFETCH_HINTS_PATH.getVariableName(),
          fdoContext.getPrefetchHintsArtifact().getExecPathString());
    }

    if (shouldPassPropellerProfiles(ccToolchain, fdoContext, featureConfiguration)) {
      if (fdoContext.getPropellerOptimizeInputFile().getCcArtifact() != null) {
        variablesBuilder.put(
            CompileBuildVariables.PROPELLER_OPTIMIZE_CC_PATH.getVariableName(),
            fdoContext.getPropellerOptimizeInputFile().getCcArtifact().getExecPathString());
      }

      if (fdoContext.getPropellerOptimizeInputFile().getLdArtifact() != null) {
        variablesBuilder.put(
            CompileBuildVariables.PROPELLER_OPTIMIZE_LD_PATH.getVariableName(),
            fdoContext.getPropellerOptimizeInputFile().getLdArtifact().getExecPathString());
      }
    }

    if (fdoContext.getMemProfProfileArtifact() != null) {
      variablesBuilder.put(
          CompileBuildVariables.MEMPROF_PROFILE_PATH.getVariableName(),
          fdoContext.getMemProfProfileArtifact().getExecPathString());
    }

    FdoContext.BranchFdoProfile branchFdoProfile = fdoContext.getBranchFdoProfile();
    // Optimization phase
    if (branchFdoProfile != null) {
      if (!auxiliaryFdoInputs.isEmpty()) {
        if (featureConfiguration.isEnabled(CppRuleClasses.AUTOFDO)
            || featureConfiguration.isEnabled(CppRuleClasses.XBINARYFDO)) {
          variablesBuilder.put(
              CompileBuildVariables.FDO_PROFILE_PATH.getVariableName(),
              branchFdoProfile.getProfileArtifact().getExecPathString());
        }
        if (featureConfiguration.isEnabled(CppRuleClasses.FDO_OPTIMIZE)) {
          if (branchFdoProfile.isLlvmFdo() || branchFdoProfile.isLlvmCSFdo()) {
            variablesBuilder.put(
                CompileBuildVariables.FDO_PROFILE_PATH.getVariableName(),
                branchFdoProfile.getProfileArtifact().getExecPathString());
          }
        }
      }
    }
    return variablesBuilder.buildOrThrow();
  }

  /** Returns whether Propeller profiles should be passed to a compile action. */
  private static boolean shouldPassPropellerProfiles(
      CcToolchainProvider ccToolchain,
      FdoContext fdoContext,
      FeatureConfiguration featureConfiguration)
      throws EvalException {
    if (ccToolchain.isToolConfiguration()) {
      // Propeller doesn't make much sense for host builds.
      return false;
    }

    if (fdoContext.getPropellerOptimizeInputFile() == null) {
      // No Propeller profiles to pass.
      return false;
    }
    // Don't pass Propeller input files if they have no effect (i.e. for ThinLTO).
    return !featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
        || featureConfiguration.isEnabled(
            CppRuleClasses.PROPELLER_OPTIMIZE_THINLTO_COMPILE_ACTIONS);
  }

  /** Returns the auxiliary files that need to be added to the {@link CppCompileAction}. */
  static NestedSet<Artifact> getAuxiliaryFdoInputs(
      CcToolchainProvider ccToolchain,
      FdoContext fdoContext,
      FeatureConfiguration featureConfiguration)
      throws EvalException {
    NestedSetBuilder<Artifact> auxiliaryInputs = NestedSetBuilder.stableOrder();

    if (fdoContext.getPrefetchHintsArtifact() != null) {
      auxiliaryInputs.add(fdoContext.getPrefetchHintsArtifact());
    }
    if (shouldPassPropellerProfiles(ccToolchain, fdoContext, featureConfiguration)) {
      if (fdoContext.getPropellerOptimizeInputFile().getCcArtifact() != null) {
        auxiliaryInputs.add(fdoContext.getPropellerOptimizeInputFile().getCcArtifact());
      }
      if (fdoContext.getPropellerOptimizeInputFile().getLdArtifact() != null) {
        auxiliaryInputs.add(fdoContext.getPropellerOptimizeInputFile().getLdArtifact());
      }
    }
    if (fdoContext.getMemProfProfileArtifact() != null) {
      auxiliaryInputs.add(fdoContext.getMemProfProfileArtifact());
    }
    FdoContext.BranchFdoProfile branchFdoProfile = fdoContext.getBranchFdoProfile();
    // If --fdo_optimize was not specified, we don't have any additional inputs.
    if (branchFdoProfile != null) {
      auxiliaryInputs.add(branchFdoProfile.getProfileArtifact());
    }

    return auxiliaryInputs.build();
  }

  // Methods creating actions:

  static Artifact createCompileActionTemplate(
      ActionConstructionContext actionConstructionContext,
      CcCompilationContext ccCompilationContext,
      CcToolchainProvider ccToolchain,
      BuildConfigurationValue configuration,
      ImmutableList<String> conlyopts,
      ImmutableList<String> copts,
      CppConfiguration cppConfiguration,
      ImmutableList<String> cxxopts,
      FdoContext fdoContext,
      NestedSet<Artifact> auxiliaryFdoInputs,
      FeatureConfiguration featureConfiguration,
      Label label,
      CcToolchainVariables commonToolchainVariables,
      ImmutableMap<String, String> fdoBuildVariables,
      RuleErrorConsumer unusedRuleErrorConsumer,
      CppSemantics semantics,
      CppSource source,
      String outputName,
      CppCompileActionBuilder builder,
      CcCompilationOutputs.Builder result,
      ImmutableList<ArtifactCategory> outputCategories,
      boolean usePic,
      boolean bitcodeOutput)
      throws RuleErrorException, EvalException {
    if (usePic) {
      builder = new CppCompileActionBuilder(builder).setPicMode(true);
    }
    SpecialArtifact sourceArtifact = (SpecialArtifact) source.getSource();
    SpecialArtifact outputFiles =
        CppHelper.getCompileOutputTreeArtifact(
            actionConstructionContext, label, sourceArtifact, outputName, usePic);
    // Dotd and dia file outputs are specified in the execution phase.
    builder.setOutputs(outputFiles, /* dotdFile= */ null, /* diagnosticsFile= */ null);
    builder.setVariables(
        setupSpecificCompileBuildVariables(
            commonToolchainVariables,
            ccCompilationContext,
            conlyopts,
            copts,
            cppConfiguration,
            cxxopts,
            fdoContext,
            auxiliaryFdoInputs,
            featureConfiguration,
            semantics,
            builder,
            /* sourceLabel= */ null,
            usePic,
            /* needsFdoBuildVariables= */ false,
            fdoBuildVariables,
            ccCompilationContext.getCppModuleMap(),
            /* enableCoverage= */ false,
            /* gcnoFile= */ null,
            /* isUsingFission= */ false,
            /* dwoFile= */ null,
            /* ltoIndexingFile= */ null,
            /* additionalBuildVariables= */ ImmutableMap.of()));
    semantics.finalizeCompileActionBuilder(configuration, featureConfiguration, builder);
    // Make sure this builder doesn't reference ruleContext outside of analysis phase.
    SpecialArtifact dotdTreeArtifact = null;
    if (builder.dotdFilesEnabled()) {
      dotdTreeArtifact =
          CppHelper.getDotdOutputTreeArtifact(
              actionConstructionContext, label, sourceArtifact, outputName, usePic);
    }
    SpecialArtifact diagnosticsTreeArtifact = null;
    if (builder.serializedDiagnosticsFilesEnabled()) {
      diagnosticsTreeArtifact =
          CppHelper.getDiagnosticsOutputTreeArtifact(
              actionConstructionContext, label, sourceArtifact, outputName, usePic);
    }

    // Currently we do not generate minimized bitcode files for tree artifacts because of issues
    // with the indexing step.
    // If ltoIndexTreeArtifact is set to a tree artifact, the minimized bitcode files will be
    // properly generated and will be an input to the indexing step. However, the lto indexing step
    // fails. The indexing step finds the full bitcode file by replacing the suffix of the
    // minimized bitcode file, therefore they have to be in the same directory.
    // Since the files are in the same directory, the command line artifact expander expands the
    // tree artifact to both the minimized bitcode files and the full bitcode files, causing an
    // error that functions are defined twice.
    // TODO(b/289071777): support for minimized bitcode files.
    SpecialArtifact ltoIndexTreeArtifact = null;

    if (bitcodeOutput) {
      Label sourceLabel = source.getLabel();
      result.addLtoBitcodeFile(
          outputFiles,
          ltoIndexTreeArtifact,
          getCopts(
              conlyopts, copts, cppConfiguration, cxxopts, semantics, sourceArtifact, sourceLabel));
    }

    ActionOwner actionOwner = null;
    if (actionConstructionContext instanceof RuleContext ruleContext
        && ruleContext.useAutoExecGroups()) {
      actionOwner =
          actionConstructionContext.getActionOwner(semantics.getCppToolchainType().toString());
    }
    try {
      CppCompileActionTemplate actionTemplate =
          new CppCompileActionTemplate(
              sourceArtifact,
              outputFiles,
              dotdTreeArtifact,
              diagnosticsTreeArtifact,
              ltoIndexTreeArtifact,
              builder,
              ccToolchain,
              outputCategories,
              actionOwner == null ? actionConstructionContext.getActionOwner() : actionOwner);
      actionConstructionContext.registerAction(actionTemplate);
    } catch (EvalException e) {
      throw new RuleErrorException(e.getMessage());
    }

    return outputFiles;
  }

  static void createModuleCodegenAction(
      ActionConstructionContext actionConstructionContext,
      CcCompilationContext ccCompilationContext,
      CcToolchainProvider ccToolchain,
      BuildConfigurationValue configuration,
      ImmutableList<String> conlyopts,
      ImmutableList<String> copts,
      CoptsFilter coptsFilter,
      CppConfiguration cppConfiguration,
      ImmutableList<String> cxxopts,
      ImmutableMap<String, String> executionInfo,
      FdoContext fdoContext,
      NestedSet<Artifact> auxiliaryFdoInputs,
      FeatureConfiguration featureConfiguration,
      boolean isCodeCoverageEnabled,
      Label label,
      CcToolchainVariables commonToolchainVariables,
      ImmutableMap<String, String> fdoBuildVariables,
      RuleErrorConsumer ruleErrorConsumer,
      CppSemantics semantics,
      CcCompilationOutputs.Builder result,
      Label sourceLabel,
      Artifact module,
      CppCompileActionBuilder builder)
      throws RuleErrorException, EvalException {
    String outputName = module.getRootRelativePath().getBaseName();

    // TODO(djasper): Make this less hacky after refactoring how the PIC/noPIC actions are created.
    boolean pic = module.getFilename().contains(".pic.");
    builder.setPicMode(pic);
    builder.setOutputs(
        actionConstructionContext,
        ruleErrorConsumer,
        label,
        ArtifactCategory.OBJECT_FILE,
        outputName);
    PathFragment ccRelativeName = module.getRootRelativePath();

    String gcnoFileName =
        CppHelper.getArtifactNameForCategory(
            ccToolchain, ArtifactCategory.COVERAGE_DATA_FILE, outputName);
    // TODO(djasper): This is now duplicated. Refactor the various create..Action functions.
    Artifact gcnoFile =
        isCodeCoverageEnabled && !cppConfiguration.useLLVMCoverageMapFormat()
            ? CppHelper.getCompileOutputArtifact(
                actionConstructionContext, label, gcnoFileName, configuration)
            : null;

    boolean generateDwo =
        CcToolchainProvider.shouldCreatePerObjectDebugInfo(featureConfiguration, cppConfiguration);
    Artifact dwoFile =
        generateDwo
            ? getDwoFile(actionConstructionContext, configuration, builder.getOutputFile())
            : null;
    // TODO(tejohnson): Add support for ThinLTO if needed.
    boolean bitcodeOutput =
        featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
            && CppFileTypes.LTO_SOURCE.matches(module.getFilename());
    Preconditions.checkState(!bitcodeOutput);

    builder.setVariables(
        setupSpecificCompileBuildVariables(
            commonToolchainVariables,
            ccCompilationContext,
            conlyopts,
            copts,
            cppConfiguration,
            cxxopts,
            fdoContext,
            auxiliaryFdoInputs,
            featureConfiguration,
            semantics,
            builder,
            sourceLabel,
            /* usePic= */ pic,
            /* needsFdoBuildVariables= */ ccRelativeName != null,
            fdoBuildVariables,
            ccCompilationContext.getCppModuleMap(),
            isCodeCoverageEnabled,
            gcnoFile,
            generateDwo,
            dwoFile,
            /* ltoIndexingFile= */ null,
            /* additionalBuildVariables= */ ImmutableMap.of()));

    builder.setGcnoFile(gcnoFile);
    builder.setDwoFile(dwoFile);

    semantics.finalizeCompileActionBuilder(configuration, featureConfiguration, builder);
    CppCompileAction compileAction = builder.buildOrThrowRuleError(ruleErrorConsumer);
    actionConstructionContext.registerAction(compileAction);
    Artifact objectFile = compileAction.getPrimaryOutput();
    if (pic) {
      result.addPicObjectFile(objectFile);
    } else {
      result.addObjectFile(objectFile);
    }
  }

  static void createParseHeaderAction(
      ActionConstructionContext actionConstructionContext,
      CcCompilationContext ccCompilationContext,
      CcToolchainProvider ccToolchain,
      BuildConfigurationValue configuration,
      ImmutableList<String> conlyopts,
      ImmutableList<String> copts,
      CppConfiguration cppConfiguration,
      ImmutableList<String> cxxopts,
      FdoContext fdoContext,
      NestedSet<Artifact> auxiliaryFdoInputs,
      FeatureConfiguration featureConfiguration,
      boolean generatePicAction,
      Label label,
      CcToolchainVariables commonToolchainVariables,
      ImmutableMap<String, String> fdoBuildVariables,
      RuleErrorConsumer ruleErrorConsumer,
      CppSemantics semantics,
      Label sourceLabel,
      String outputName,
      CcCompilationOutputs.Builder result,
      CppCompileActionBuilder builder)
      throws RuleErrorException, EvalException {
    String outputNameBase =
        CppHelper.getArtifactNameForCategory(
            ccToolchain, ArtifactCategory.GENERATED_HEADER, outputName);

    builder
        .setOutputs(
            actionConstructionContext,
            ruleErrorConsumer,
            label,
            ArtifactCategory.PROCESSED_HEADER,
            outputNameBase)
        // If we generate pic actions, we prefer the header actions to use the pic artifacts.
        .setPicMode(generatePicAction);
    builder.setVariables(
        setupSpecificCompileBuildVariables(
            commonToolchainVariables,
            ccCompilationContext,
            conlyopts,
            copts,
            cppConfiguration,
            cxxopts,
            fdoContext,
            auxiliaryFdoInputs,
            featureConfiguration,
            semantics,
            builder,
            sourceLabel,
            generatePicAction,
            /* needsFdoBuildVariables= */ false,
            fdoBuildVariables,
            ccCompilationContext.getCppModuleMap(),
            /* enableCoverage= */ false,
            /* gcnoFile= */ null,
            /* isUsingFission= */ false,
            /* dwoFile= */ null,
            /* ltoIndexingFile= */ null,
            /* additionalBuildVariables= */ ImmutableMap.of()));
    semantics.finalizeCompileActionBuilder(configuration, featureConfiguration, builder);
    CppCompileAction compileAction = builder.buildOrThrowRuleError(ruleErrorConsumer);
    actionConstructionContext.registerAction(compileAction);
    Artifact tokenFile = compileAction.getPrimaryOutput();
    result.addHeaderTokenFile(tokenFile);
  }

  static ImmutableList<Artifact> createModuleAction(
      ActionConstructionContext actionConstructionContext,
      CcCompilationContext ccCompilationContext,
      CcToolchainProvider ccToolchain,
      BuildConfigurationValue configuration,
      ImmutableList<String> conlyopts,
      ImmutableList<String> copts,
      CoptsFilter coptsFilter,
      CppConfiguration cppConfiguration,
      ImmutableList<String> cxxopts,
      ImmutableMap<String, String> executionInfo,
      FdoContext fdoContext,
      NestedSet<Artifact> auxiliaryFdoInputs,
      FeatureConfiguration featureConfiguration,
      boolean generateNoPicAction,
      boolean generatePicAction,
      Label label,
      CcToolchainVariables commonToolchainVariables,
      ImmutableMap<String, String> fdoBuildVariables,
      RuleErrorConsumer ruleErrorConsumer,
      CppSemantics semantics,
      CcCompilationOutputs.Builder result,
      CppModuleMap cppModuleMap,
      CppCompileActionBuilder builder)
      throws RuleErrorException, EvalException, InterruptedException {
    Artifact moduleMapArtifact = cppModuleMap.getArtifact();
    Label sourceLabel = Label.parseCanonicalUnchecked(cppModuleMap.getName());

    // A header module compile action is just like a normal compile action, but:
    // - the compiled source file is the module map
    // - it creates a header module (.pcm file).
    return createCompileSourceActionFromBuilder(
        actionConstructionContext,
        ccCompilationContext,
        ccToolchain,
        configuration,
        conlyopts,
        copts,
        cppConfiguration,
        cxxopts,
        fdoContext,
        auxiliaryFdoInputs,
        featureConfiguration,
        generateNoPicAction,
        generatePicAction,
        label,
        commonToolchainVariables,
        fdoBuildVariables,
        ruleErrorConsumer,
        semantics,
        sourceLabel,
        Path.of(sourceLabel.getName()).getFileName().toString(),
        result,
        moduleMapArtifact,
        builder,
        ArtifactCategory.CPP_MODULE,
        cppModuleMap,
        /* addObject= */ false,
        /* enableCoverage= */ false,
        /* generateDwo= */ false,
        /* bitcodeOutput= */ false);
  }

  @CanIgnoreReturnValue
  static ImmutableList<Artifact> createCompileSourceActionFromBuilder(
      ActionConstructionContext actionConstructionContext,
      CcCompilationContext ccCompilationContext,
      CcToolchainProvider ccToolchain,
      BuildConfigurationValue configuration,
      ImmutableList<String> conlyopts,
      ImmutableList<String> copts,
      CppConfiguration cppConfiguration,
      ImmutableList<String> cxxopts,
      FdoContext fdoContext,
      NestedSet<Artifact> auxiliaryFdoInputs,
      FeatureConfiguration featureConfiguration,
      boolean generateNoPicAction,
      boolean generatePicAction,
      Label label,
      CcToolchainVariables commonToolchainVariables,
      ImmutableMap<String, String> fdoBuildVariables,
      RuleErrorConsumer ruleErrorConsumer,
      CppSemantics semantics,
      Label sourceLabel,
      String outputName,
      CcCompilationOutputs.Builder result,
      Artifact sourceArtifact,
      CppCompileActionBuilder builder,
      ArtifactCategory outputCategory,
      CppModuleMap cppModuleMap,
      boolean addObject,
      boolean enableCoverage,
      boolean generateDwo,
      boolean bitcodeOutput)
      throws RuleErrorException, EvalException, InterruptedException {
    ImmutableList.Builder<Artifact> directOutputs = new ImmutableList.Builder<>();
    PathFragment ccRelativeName = sourceArtifact.getRootRelativePath();

    // Create PIC compile actions (same as no-PIC, but use -fPIC and
    // generate .pic.o, .pic.d, .pic.gcno instead of .o, .d, .gcno.)
    if (generatePicAction) {
      CppCompileActionBuilder picBuilder = new CppCompileActionBuilder(builder);
      picBuilder.setPicMode(true);
      Artifact picOutputFile =
          createPicOrNoPicCompileSourceAction(
              actionConstructionContext,
              ccCompilationContext,
              ccToolchain,
              configuration,
              conlyopts,
              copts,
              cppConfiguration,
              cxxopts,
              fdoContext,
              auxiliaryFdoInputs,
              featureConfiguration,
              label,
              commonToolchainVariables,
              fdoBuildVariables,
              ruleErrorConsumer,
              semantics,
              sourceLabel,
              outputName,
              result,
              sourceArtifact,
              picBuilder,
              outputCategory,
              cppModuleMap,
              addObject,
              enableCoverage,
              generateDwo,
              bitcodeOutput,
              ccRelativeName,
              /* usePic= */ true,
              /* additionalBuildVariables= */ ImmutableMap.of());
      directOutputs.add(picOutputFile);
      if (outputCategory == ArtifactCategory.CPP_MODULE) {
        result.addModuleFile(picOutputFile);
      }
    }

    if (generateNoPicAction) {
      Artifact noPicOutputFile =
          createPicOrNoPicCompileSourceAction(
              actionConstructionContext,
              ccCompilationContext,
              ccToolchain,
              configuration,
              conlyopts,
              copts,
              cppConfiguration,
              cxxopts,
              fdoContext,
              auxiliaryFdoInputs,
              featureConfiguration,
              label,
              commonToolchainVariables,
              fdoBuildVariables,
              ruleErrorConsumer,
              semantics,
              sourceLabel,
              outputName,
              result,
              sourceArtifact,
              builder,
              outputCategory,
              cppModuleMap,
              addObject,
              enableCoverage,
              generateDwo,
              bitcodeOutput,
              ccRelativeName,
              /* usePic= */ false,
              /* additionalBuildVariables= */ ImmutableMap.of());
      directOutputs.add(noPicOutputFile);
      if (outputCategory == ArtifactCategory.CPP_MODULE) {
        result.addModuleFile(noPicOutputFile);
      }
    }
    return directOutputs.build();
  }

  private static Artifact createPicOrNoPicCompileSourceAction(
      ActionConstructionContext actionConstructionContext,
      CcCompilationContext ccCompilationContext,
      CcToolchainProvider ccToolchain,
      BuildConfigurationValue configuration,
      ImmutableList<String> conlyopts,
      ImmutableList<String> copts,
      CppConfiguration cppConfiguration,
      ImmutableList<String> cxxopts,
      FdoContext fdoContext,
      NestedSet<Artifact> auxiliaryFdoInputs,
      FeatureConfiguration featureConfiguration,
      Label label,
      CcToolchainVariables commonToolchainVariables,
      ImmutableMap<String, String> fdoBuildVariables,
      RuleErrorConsumer ruleErrorConsumer,
      CppSemantics semantics,
      Label sourceLabel,
      String outputName,
      CcCompilationOutputs.Builder result,
      Artifact sourceArtifact,
      CppCompileActionBuilder builder,
      ArtifactCategory outputCategory,
      CppModuleMap cppModuleMap,
      boolean addObject,
      boolean enableCoverage,
      boolean generateDwo,
      boolean bitcodeOutput,
      PathFragment ccRelativeName,
      boolean usePic,
      ImmutableMap<String, String> additionalBuildVariables)
      throws RuleErrorException, EvalException, InterruptedException {
    builder.setOutputs(
        actionConstructionContext,
        ruleErrorConsumer,
        label,
        outputCategory,
        getOutputNameBaseWith(ccToolchain, outputName, usePic));
    String gcnoFileName =
        CppHelper.getArtifactNameForCategory(
            ccToolchain,
            ArtifactCategory.COVERAGE_DATA_FILE,
            getOutputNameBaseWith(ccToolchain, outputName, usePic));

    Artifact gcnoFile =
        enableCoverage && !cppConfiguration.useLLVMCoverageMapFormat()
            ? CppHelper.getCompileOutputArtifact(
                actionConstructionContext, label, gcnoFileName, configuration)
            : null;

    Artifact dwoFile =
        generateDwo && !bitcodeOutput
            ? getDwoFile(actionConstructionContext, configuration, builder.getOutputFile())
            : null;
    Artifact ltoIndexingFile =
        bitcodeOutput
            ? getLtoIndexingFile(
                actionConstructionContext,
                configuration,
                featureConfiguration,
                builder.getOutputFile())
            : null;

    builder.setVariables(
        setupSpecificCompileBuildVariables(
            commonToolchainVariables,
            ccCompilationContext,
            conlyopts,
            copts,
            cppConfiguration,
            cxxopts,
            fdoContext,
            auxiliaryFdoInputs,
            featureConfiguration,
            semantics,
            builder,
            sourceLabel,
            usePic,
            /* needsFdoBuildVariables= */ ccRelativeName != null && addObject,
            fdoBuildVariables,
            cppModuleMap,
            enableCoverage,
            gcnoFile,
            generateDwo,
            dwoFile,
            ltoIndexingFile,
            additionalBuildVariables));

    result.addTemps(
        createTempsActions(
            actionConstructionContext,
            ccCompilationContext,
            ccToolchain,
            configuration,
            conlyopts,
            copts,
            cppConfiguration,
            cxxopts,
            fdoContext,
            auxiliaryFdoInputs,
            featureConfiguration,
            label,
            commonToolchainVariables,
            fdoBuildVariables,
            ruleErrorConsumer,
            semantics,
            sourceArtifact,
            sourceLabel,
            outputName,
            builder,
            usePic,
            ccRelativeName));

    builder.setGcnoFile(gcnoFile);
    builder.setDwoFile(dwoFile);
    builder.setLtoIndexingFile(ltoIndexingFile);

    semantics.finalizeCompileActionBuilder(configuration, featureConfiguration, builder);
    CppCompileAction compileAction = builder.buildOrThrowRuleError(ruleErrorConsumer);
    actionConstructionContext.registerAction(compileAction);
    Artifact objectFile = compileAction.getPrimaryOutput();
    if (usePic) {
      if (addObject) {
        result.addPicObjectFile(objectFile);
      }
      if (dwoFile != null) {
        // Exec configuration targets don't produce .dwo files.
        result.addPicDwoFile(dwoFile);
      }
      if (gcnoFile != null) {
        result.addPicGcnoFile(gcnoFile);
      }
    } else {
      if (addObject) {
        result.addObjectFile(objectFile);
      }
      if (dwoFile != null) {
        // Exec configuration targets don't produce .dwo files.
        result.addDwoFile(dwoFile);
      }
      if (gcnoFile != null) {
        result.addGcnoFile(gcnoFile);
      }
    }
    if (addObject && bitcodeOutput) {
      result.addLtoBitcodeFile(
          objectFile,
          ltoIndexingFile,
          getCopts(
              conlyopts, copts, cppConfiguration, cxxopts, semantics, sourceArtifact, sourceLabel));
    }
    return objectFile;
  }

  /** Create the actions for "--save_temps". */
  private static ImmutableList<Artifact> createTempsActions(
      ActionConstructionContext actionConstructionContext,
      CcCompilationContext ccCompilationContext,
      CcToolchainProvider ccToolchain,
      BuildConfigurationValue configuration,
      ImmutableList<String> conlyopts,
      ImmutableList<String> copts,
      CppConfiguration cppConfiguration,
      ImmutableList<String> cxxopts,
      FdoContext fdoContext,
      NestedSet<Artifact> auxiliaryFdoInputs,
      FeatureConfiguration featureConfiguration,
      Label label,
      CcToolchainVariables commonToolchainVariables,
      ImmutableMap<String, String> fdoBuildVariables,
      RuleErrorConsumer ruleErrorConsumer,
      CppSemantics semantics,
      Artifact source,
      Label sourceLabel,
      String outputName,
      CppCompileActionBuilder builder,
      boolean usePic,
      PathFragment ccRelativeName)
      throws RuleErrorException, EvalException, InterruptedException {
    if (!cppConfiguration.getSaveTemps()) {
      return ImmutableList.of();
    }

    String path = source.getFilename();
    boolean isCFile = CppFileTypes.C_SOURCE.matches(path);
    boolean isCppFile = CppFileTypes.CPP_SOURCE.matches(path);
    boolean isObjcFile = CppFileTypes.OBJC_SOURCE.matches(path);
    boolean isObjcppFile = CppFileTypes.OBJCPP_SOURCE.matches(path);

    if (!isCFile && !isCppFile && !isObjcFile && !isObjcppFile) {
      return ImmutableList.of();
    }

    ArtifactCategory category =
        isCFile ? ArtifactCategory.PREPROCESSED_C_SOURCE : ArtifactCategory.PREPROCESSED_CPP_SOURCE;

    String outputArtifactNameBase = getOutputNameBaseWith(ccToolchain, outputName, usePic);

    CppCompileActionBuilder dBuilder = new CppCompileActionBuilder(builder);
    dBuilder.setOutputs(
        actionConstructionContext, ruleErrorConsumer, label, category, outputArtifactNameBase);
    dBuilder.setVariables(
        setupSpecificCompileBuildVariables(
            commonToolchainVariables,
            ccCompilationContext,
            conlyopts,
            copts,
            cppConfiguration,
            cxxopts,
            fdoContext,
            auxiliaryFdoInputs,
            featureConfiguration,
            semantics,
            dBuilder,
            sourceLabel,
            usePic,
            /* needsFdoBuildVariables= */ ccRelativeName != null,
            fdoBuildVariables,
            ccCompilationContext.getCppModuleMap(),
            /* enableCoverage= */ false,
            /* gcnoFile= */ null,
            /* isUsingFission= */ false,
            /* dwoFile= */ null,
            /* ltoIndexingFile= */ null,
            ImmutableMap.of(
                CompileBuildVariables.OUTPUT_PREPROCESS_FILE.getVariableName(),
                dBuilder.getRealOutputFilePath().getSafePathString())));
    semantics.finalizeCompileActionBuilder(configuration, featureConfiguration, dBuilder);
    CppCompileAction dAction = dBuilder.buildOrThrowRuleError(ruleErrorConsumer);
    actionConstructionContext.registerAction(dAction);

    CppCompileActionBuilder sdBuilder = new CppCompileActionBuilder(builder);
    sdBuilder.setOutputs(
        actionConstructionContext,
        ruleErrorConsumer,
        label,
        ArtifactCategory.GENERATED_ASSEMBLY,
        outputArtifactNameBase);
    sdBuilder.setVariables(
        setupSpecificCompileBuildVariables(
            commonToolchainVariables,
            ccCompilationContext,
            conlyopts,
            copts,
            cppConfiguration,
            cxxopts,
            fdoContext,
            auxiliaryFdoInputs,
            featureConfiguration,
            semantics,
            sdBuilder,
            sourceLabel,
            usePic,
            /* needsFdoBuildVariables= */ ccRelativeName != null,
            fdoBuildVariables,
            ccCompilationContext.getCppModuleMap(),
            /* enableCoverage= */ false,
            /* gcnoFile= */ null,
            /* isUsingFission= */ false,
            /* dwoFile= */ null,
            /* ltoIndexingFile= */ null,
            ImmutableMap.of(
                CompileBuildVariables.OUTPUT_ASSEMBLY_FILE.getVariableName(),
                sdBuilder.getRealOutputFilePath().getSafePathString())));
    semantics.finalizeCompileActionBuilder(configuration, featureConfiguration, sdBuilder);
    CppCompileAction sdAction = sdBuilder.buildOrThrowRuleError(ruleErrorConsumer);
    actionConstructionContext.registerAction(sdAction);

    return ImmutableList.of(dAction.getPrimaryOutput(), sdAction.getPrimaryOutput());
  }

  // Helper methods used when creating actions:

  private static String getOutputNameBaseWith(
      CcToolchainProvider ccToolchain, String base, boolean usePic) throws RuleErrorException {
    return usePic
        ? CppHelper.getArtifactNameForCategory(ccToolchain, ArtifactCategory.PIC_FILE, base)
        : base;
  }

  private static ImmutableList<String> collectPerFileCopts(
      CppConfiguration cppConfiguration, Artifact sourceFile, Label sourceLabel) {
    return cppConfiguration.getPerFileCopts().stream()
        .filter(
            perLabelOptions ->
                (sourceLabel != null && perLabelOptions.isIncluded(sourceLabel))
                    || perLabelOptions.isIncluded(sourceFile))
        .map(PerLabelOptions::getOptions)
        .flatMap(options -> options.stream())
        .collect(ImmutableList.toImmutableList());
  }

  private static Artifact getDwoFile(
      ActionConstructionContext actionConstructionContext,
      BuildConfigurationValue configuration,
      Artifact outputFile) {
    return actionConstructionContext.getRelatedArtifact(
        outputFile.getOutputDirRelativePath(configuration.isSiblingRepositoryLayout()), ".dwo");
  }

  @Nullable
  private static Artifact getLtoIndexingFile(
      ActionConstructionContext actionConstructionContext,
      BuildConfigurationValue configuration,
      FeatureConfiguration featureConfiguration,
      Artifact outputFile) {
    if (featureConfiguration.isEnabled(CppRuleClasses.NO_USE_LTO_INDEXING_BITCODE_FILE)) {
      return null;
    }
    String ext = Iterables.getOnlyElement(CppFileTypes.LTO_INDEXING_OBJECT_FILE.getExtensions());
    return actionConstructionContext.getRelatedArtifact(
        outputFile.getOutputDirRelativePath(configuration.isSiblingRepositoryLayout()), ext);
  }

  /**
   * Return flags that were specified on the Blaze command line. Take the filetype of sourceFilename
   * into account.
   */
  public static ImmutableList<String> getCoptsFromOptions(
      CppConfiguration config, CppSemantics semantics, String sourceFilename) {
    ImmutableList.Builder<String> flagsBuilder = ImmutableList.builder();

    flagsBuilder.addAll(config.getCopts());

    if (CppFileTypes.C_SOURCE.matches(sourceFilename)) {
      flagsBuilder.addAll(config.getConlyopts());
    }

    if (CppFileTypes.CPP_SOURCE.matches(sourceFilename)
        || CppFileTypes.CPP_HEADER.matches(sourceFilename)
        || CppFileTypes.CPP_MODULE_MAP.matches(sourceFilename)
        || CppFileTypes.OBJCPP_SOURCE.matches(sourceFilename)
        || CppFileTypes.CLIF_INPUT_PROTO.matches(sourceFilename)) {
      flagsBuilder.addAll(config.getCxxopts());
    }

    if (CppFileTypes.OBJC_SOURCE.matches(sourceFilename)
        || CppFileTypes.OBJCPP_SOURCE.matches(sourceFilename)
        || (CppFileTypes.CPP_HEADER.matches(sourceFilename)
            && semantics.language() == Language.OBJC)) {
      flagsBuilder.addAll(config.getObjcopts());
    }

    return flagsBuilder.build();
  }

  private static ImmutableList<String> getCopts(
      ImmutableList<String> conlyopts,
      ImmutableList<String> copts,
      CppConfiguration cppConfiguration,
      ImmutableList<String> cxxopts,
      CppSemantics semantics,
      Artifact sourceFile,
      Label sourceLabel) {
    ImmutableList.Builder<String> coptsList = ImmutableList.builder();
    String sourceFilename = sourceFile.getExecPathString();
    coptsList.addAll(getCoptsFromOptions(cppConfiguration, semantics, sourceFilename));
    coptsList.addAll(copts);

    if (CppFileTypes.C_SOURCE.matches(sourceFilename)) {
      coptsList.addAll(conlyopts);
    }

    if (CppFileTypes.CPP_SOURCE.matches(sourceFilename)
        || CppFileTypes.CPP_HEADER.matches(sourceFilename)
        || CppFileTypes.CPP_MODULE_MAP.matches(sourceFilename)
        || CppFileTypes.CLIF_INPUT_PROTO.matches(sourceFilename)
        || CppFileTypes.OBJCPP_SOURCE.matches(sourceFilename)) {
      coptsList.addAll(cxxopts);
    }

    if (sourceFile != null && sourceLabel != null) {
      coptsList.addAll(collectPerFileCopts(cppConfiguration, sourceFile, sourceLabel));
    }
    return coptsList.build();
  }
}
