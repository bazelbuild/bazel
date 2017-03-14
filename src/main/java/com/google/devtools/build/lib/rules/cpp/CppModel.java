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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcCompilationOutputs.Builder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.StringSequenceBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction.DotdFile;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.Picness;
import com.google.devtools.build.lib.rules.cpp.Link.Staticness;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * Representation of a C/C++ compilation. Its purpose is to share the code that creates compilation
 * actions between all classes that need to do so. It follows the builder pattern - load up the
 * necessary settings and then call {@link #createCcCompileActions}.
 *
 * <p>This class is not thread-safe, and it should only be used once for each set of source files,
 * i.e. calling {@link #createCcCompileActions} will throw an Exception if called twice.
 */
public final class CppModel {
  private final CppSemantics semantics;
  private final RuleContext ruleContext;
  private final BuildConfiguration configuration;
  private final CppConfiguration cppConfiguration;

  // compile model
  private CppCompilationContext context;
  private final Set<CppSource> sourceFiles = new LinkedHashSet<>();
  private final List<Artifact> mandatoryInputs = new ArrayList<>();
  private final List<String> copts = new ArrayList<>();
  @Nullable private Pattern nocopts;
  private boolean fake;
  private boolean maySaveTemps;
  private boolean onlySingleOutput;
  private CcCompilationOutputs compilationOutputs;

  // link model
  private final List<String> linkopts = new ArrayList<>();
  private LinkTargetType linkType = LinkTargetType.STATIC_LIBRARY;
  private boolean neverLink;
  private final List<Artifact> linkActionInputs = new ArrayList<>();
  private boolean allowInterfaceSharedObjects;
  private boolean createDynamicLibrary = true;
  private Artifact soImplArtifact;
  private FeatureConfiguration featureConfiguration;
  private List<VariablesExtension> variablesExtensions = new ArrayList<>();
  private final CcToolchainProvider ccToolchain;
  private final FdoSupportProvider fdoSupport;
  private String linkedArtifactNameSuffix = "";

  public CppModel(RuleContext ruleContext, CppSemantics semantics,
      CcToolchainProvider ccToolchain, FdoSupportProvider fdoSupport) {
    this(ruleContext, semantics, ccToolchain, fdoSupport, ruleContext.getConfiguration());
 }

  public CppModel(RuleContext ruleContext, CppSemantics semantics,
      CcToolchainProvider ccToolchain, FdoSupportProvider fdoSupport,
      BuildConfiguration configuration) {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);
    this.semantics = semantics;
    this.ccToolchain = Preconditions.checkNotNull(ccToolchain);
    this.fdoSupport = Preconditions.checkNotNull(fdoSupport);
    this.configuration = configuration;
    cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
  }

  private Artifact getDwoFile(Artifact outputFile) {
    return ruleContext.getRelatedArtifact(outputFile.getRootRelativePath(), ".dwo");
  }

  /**
   * If the cpp compilation is a fake, then it creates only a single compile action without PIC.
   * Defaults to false.
   */
  public CppModel setFake(boolean fake) {
    this.fake = fake;
    return this;
  }

  /**
   * If set, the CppModel only creates a single .o output that can be linked into a dynamic library,
   * i.e., it never generates both PIC and non-PIC outputs. Otherwise it creates outputs that can be
   * linked into both static binaries and dynamic libraries (if both require PIC or both require
   * non-PIC, then it still only creates a single output). Defaults to false.
   */
  public CppModel setOnlySingleOutput(boolean onlySingleOutput) {
    this.onlySingleOutput = onlySingleOutput;
    return this;
  }

  /**
   * Whether to create actions for temps. This defaults to false.
   */
  public CppModel setSaveTemps(boolean maySaveTemps) {
    this.maySaveTemps = maySaveTemps;
    return this;
  }

  /**
   * Sets the compilation context, i.e. include directories and allowed header files inclusions.
   */
  public CppModel setContext(CppCompilationContext context) {
    this.context = context;
    return this;
  }

  /**
   * Adds a single source file to be compiled. The given build variables will be added to those used
   * to compile this source file. Note that this should only be called for primary compilation
   * units, including module files or headers to be parsed or preprocessed.
   */
  public CppModel addCompilationUnitSources(
      Iterable<Artifact> sourceFiles, Label sourceLabel, Map<String, String> buildVariables,
      CppSource.Type type) {
    for (Artifact sourceFile : sourceFiles) {
      this.sourceFiles.add(CppSource.create(sourceFile, sourceLabel, buildVariables, type));
    }
    return this;
  }

  /**
   * Adds all the source files. Note that this should only be called for primary compilation units,
   * including module files or headers to be parsed or preprocessed.
   */
  public CppModel addCompilationUnitSources(Set<CppSource> sources) {
    this.sourceFiles.addAll(sources);
    return this;
  }

  /** Adds mandatory inputs. */
  public CppModel addMandatoryInputs(Collection<Artifact> artifacts) {
    this.mandatoryInputs.addAll(artifacts);
    return this;
  }

  /**
   * Adds the given copts.
   */
  public CppModel addCopts(Collection<String> copts) {
    this.copts.addAll(copts);
    return this;
  }

  /**
   * Sets the nocopts pattern. This is used to filter out flags from the system defined set of
   * flags. By default no filter is applied.
   */
  public CppModel setNoCopts(@Nullable Pattern nocopts) {
    this.nocopts = nocopts;
    return this;
  }

  /**
   * Adds the given linkopts to the optional dynamic library link command.
   */
  public CppModel addLinkopts(Collection<String> linkopts) {
    this.linkopts.addAll(linkopts);
    return this;
  }

  /**
   * Adds the given variablesExensions for templating the crosstool.
   *
   * <p>In general, we prefer the build variables (especially those that derive strictly from
   * the configuration) be learned by inspecting the CcToolchain, as passed to the rule in the
   * CcToolchainProvider.  However, for build variables that must be injected into the rule
   * implementation (ex. build variables learned from the BUILD file), should be added using the
   * VariablesExtension abstraction.  This allows the injection to construct non-trivial build
   * variables (lists, ect.).
   */
  public CppModel addVariablesExtension(Collection<VariablesExtension> variablesExtensions) {
    this.variablesExtensions.addAll(variablesExtensions);
    return this;
  }
  
  /**
   * Sets the link type used for the link actions. Note that only static links are supported at this
   * time.
   */
  public CppModel setLinkTargetType(LinkTargetType linkType) {
    this.linkType = linkType;
    return this;
  }

  public CppModel setNeverLink(boolean neverLink) {
    this.neverLink = neverLink;
    return this;
  }
  
  /**
   * Adds an artifact to the inputs of any link actions created by this CppModel.
   */
  public CppModel addLinkActionInputs(Collection<Artifact> inputs) {
    this.linkActionInputs.addAll(inputs);
    return this;
  }

  /**
   * Whether to allow interface dynamic libraries. Note that setting this to true only has an effect
   * if the configuration allows it. Defaults to false.
   */
  public CppModel setAllowInterfaceSharedObjects(boolean allowInterfaceSharedObjects) {
    // TODO(bazel-team): Set the default to true, and require explicit action to disable it.
    this.allowInterfaceSharedObjects = allowInterfaceSharedObjects;
    return this;
  }

  public CppModel setCreateDynamicLibrary(boolean createDynamicLibrary) {
    this.createDynamicLibrary = createDynamicLibrary;
    return this;
  }

  public CppModel setDynamicLibrary(Artifact soImplFilename) {
    this.soImplArtifact = soImplFilename;
    return this;
  }
  
  /**
   * Sets the feature configuration to be used for C/C++ actions. 
   */
  public CppModel setFeatureConfiguration(FeatureConfiguration featureConfiguration) {
    this.featureConfiguration = featureConfiguration;
    return this;
  }

  /*
   * Adds a suffix for paths of linked artifacts. Normally their paths are derived solely from rule
   * labels. In the case of multiple callers (e.g., aspects) acting on a single rule, they may
   * generate the same linked artifact and therefore lead to artifact conflicts. This method
   * provides a way to avoid this artifact conflict by allowing different callers acting on the same
   * rule to provide a suffix that will be used to scope their own linked artifacts.
   */
  public CppModel setLinkedArtifactNameSuffix(String suffix) {
    this.linkedArtifactNameSuffix = suffix;
    return this;
  }
  
  /**
   * @returns whether we want to provide header modules for the current target.
   */
  private boolean shouldProvideHeaderModules() {
    return featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULES)
        && !cppConfiguration.isLipoContextCollector();
  }

  /**
   * @return the non-pic header module artifact for the current target.
   */
  public Artifact getHeaderModule(Artifact moduleMapArtifact) {
    PathFragment objectDir = CppHelper.getObjDirectory(ruleContext.getLabel());
    PathFragment outputName = objectDir.getRelative(
        semantics.getEffectiveSourcePath(moduleMapArtifact));
    return ruleContext.getRelatedArtifact(outputName, ".pcm");
  }

  /**
   * @return the pic header module artifact for the current target.
   */
  public Artifact getPicHeaderModule(Artifact moduleMapArtifact) {
    PathFragment objectDir = CppHelper.getObjDirectory(ruleContext.getLabel());
    PathFragment outputName = objectDir.getRelative(
        semantics.getEffectiveSourcePath(moduleMapArtifact));
    return ruleContext.getRelatedArtifact(outputName, ".pic.pcm");
  }

  /**
   * @return whether this target needs to generate pic actions.
   */
  private boolean getGeneratePicActions() {
    return CppHelper.usePic(ruleContext, false);
  }

  /**
   * @return whether this target needs to generate non-pic actions.
   */
  private boolean getGenerateNoPicActions() {
    return
        // If we always need pic for everything, then don't bother to create a no-pic action.
        (!CppHelper.usePic(ruleContext, true) || !CppHelper.usePic(ruleContext, false))
        // onlySingleOutput guarantees that the code is only ever linked into a dynamic library - so
        // we don't need a no-pic action even if linking into a binary would require it.
        && !((onlySingleOutput && getGeneratePicActions()));
  }

  /**
   * @return whether this target needs to generate a pic header module.
   */
  public boolean getGeneratesPicHeaderModule() {
    return shouldProvideHeaderModules() && !fake && getGeneratePicActions();
  }

  /**
   * @return whether this target needs to generate a non-pic header module.
   */
  public boolean getGeneratesNoPicHeaderModule() {
    return shouldProvideHeaderModules() && !fake && getGenerateNoPicActions();
  }

  /**
   * Returns a {@code CppCompileActionBuilder} with the common fields for a C++ compile action being
   * initialized.
   */
  private CppCompileActionBuilder initializeCompileAction(
      Artifact sourceArtifact, Label sourceLabel) {
    CppCompileActionBuilder builder = createCompileActionBuilder(sourceArtifact, sourceLabel);
    if (nocopts != null) {
      builder.addNocopts(nocopts);
    }

    builder.setFeatureConfiguration(featureConfiguration);
    
    return builder;
  }

  /** Get the safe path strings for a list of paths to use in the build variables. */
  private ImmutableSet<String> getSafePathStrings(Collection<PathFragment> paths) {
    ImmutableSet.Builder<String> result = ImmutableSet.builder();
    for (PathFragment path : paths) {
      result.add(path.getSafePathString());
    }
    return result.build();
  }

  private void setupCompileBuildVariables(
      CppCompileActionBuilder builder,
      boolean usePic,
      PathFragment ccRelativeName,
      PathFragment autoFdoImportPath,
      Artifact gcnoFile,
      Artifact dwoFile,
      CppModuleMap cppModuleMap,
      Map<String, String> sourceSpecificBuildVariables) {
    CcToolchainFeatures.Variables.Builder buildVariables =
        new CcToolchainFeatures.Variables.Builder();
    
    // TODO(bazel-team): Pull out string constants for all build variables.

    CppCompilationContext builderContext = builder.getContext();
    Artifact sourceFile = builder.getSourceFile();
    Artifact outputFile = builder.getOutputFile();
    String realOutputFilePath;

    buildVariables.addStringVariable("source_file", sourceFile.getExecPathString());
    buildVariables.addStringVariable("output_file", outputFile.getExecPathString());

    if (builder.getTempOutputFile() != null) {
      realOutputFilePath = builder.getTempOutputFile().getPathString();
    } else {
      realOutputFilePath = builder.getOutputFile().getExecPathString();
    }

    if (FileType.contains(outputFile, CppFileTypes.ASSEMBLER, CppFileTypes.PIC_ASSEMBLER)) {
      buildVariables.addStringVariable("output_assembly_file", realOutputFilePath);
    } else if (FileType.contains(outputFile, CppFileTypes.PREPROCESSED_C,
        CppFileTypes.PREPROCESSED_CPP, CppFileTypes.PIC_PREPROCESSED_C,
        CppFileTypes.PIC_PREPROCESSED_CPP)) {
      buildVariables.addStringVariable("output_preprocess_file", realOutputFilePath);
    } else {
      buildVariables.addStringVariable("output_object_file", realOutputFilePath);
    }

    DotdFile dotdFile = CppFileTypes.mustProduceDotdFile(sourceFile)
        ? Preconditions.checkNotNull(builder.getDotdFile()) : null;
    // Set dependency_file to enable <object>.d file generation.
    if (dotdFile != null) {
      buildVariables.addStringVariable(
          "dependency_file", dotdFile.getSafeExecPath().getPathString());
    }

    if (featureConfiguration.isEnabled(CppRuleClasses.MODULE_MAPS) && cppModuleMap != null) {
      // If the feature is enabled and cppModuleMap is null, we are about to fail during analysis
      // in any case, but don't crash.
      buildVariables.addStringVariable("module_name", cppModuleMap.getName());
      buildVariables.addStringVariable(
          "module_map_file", cppModuleMap.getArtifact().getExecPathString());
      StringSequenceBuilder sequence = new StringSequenceBuilder();
      for (Artifact artifact : builderContext.getDirectModuleMaps()) {
        sequence.addValue(artifact.getExecPathString());
      }
      buildVariables.addCustomBuiltVariable("dependent_module_map_files", sequence);
    }
    if (featureConfiguration.isEnabled(CppRuleClasses.USE_HEADER_MODULES)) {
      // Module inputs will be set later when the action is executed.
      buildVariables.addStringSequenceVariable("module_files", ImmutableSet.<String>of());
    }
    if (featureConfiguration.isEnabled(CppRuleClasses.INCLUDE_PATHS)) {
      buildVariables.addStringSequenceVariable(
          "include_paths", getSafePathStrings(builderContext.getIncludeDirs()));
      buildVariables.addStringSequenceVariable(
          "quote_include_paths", getSafePathStrings(builderContext.getQuoteIncludeDirs()));
      buildVariables.addStringSequenceVariable(
          "system_include_paths", getSafePathStrings(builderContext.getSystemIncludeDirs()));
    }

    if (featureConfiguration.isEnabled(CppRuleClasses.PREPROCESSOR_DEFINES)) {
      String fdoBuildStamp = CppHelper.getFdoBuildStamp(ruleContext, fdoSupport.getFdoSupport());
      ImmutableList<String> defines;
      if (fdoBuildStamp != null) {
        // Stamp FDO builds with FDO subtype string
        defines = ImmutableList.<String>builder()
            .addAll(builderContext.getDefines())
            .add(CppConfiguration.FDO_STAMP_MACRO
                + "=\"" + CppHelper.getFdoBuildStamp(
                    ruleContext, fdoSupport.getFdoSupport()) + "\"")
            .build();
      } else {
        defines = builderContext.getDefines();
      }

      buildVariables.addStringSequenceVariable("preprocessor_defines", defines);
    }

    if (usePic) {
      if (!featureConfiguration.isEnabled(CppRuleClasses.PIC)) {
        ruleContext.ruleError("PIC compilation is requested but the toolchain does not support it");
      }
      buildVariables.addStringVariable("pic", "");
    }

    if (ccRelativeName != null) {
      fdoSupport.getFdoSupport().configureCompilation(
          builder, buildVariables, ruleContext, ccRelativeName, autoFdoImportPath, usePic,
          featureConfiguration, fdoSupport);
    }
    if (gcnoFile != null) {
      buildVariables.addStringVariable("gcov_gcno_file", gcnoFile.getExecPathString());
    }

    if (dwoFile != null) {
      buildVariables.addStringVariable("per_object_debug_info_file", dwoFile.getExecPathString());
    }

    buildVariables.addAllStringVariables(ccToolchain.getBuildVariables());

    buildVariables.addAllStringVariables(sourceSpecificBuildVariables);

    for (VariablesExtension extension : variablesExtensions) {
      extension.addVariables(buildVariables);
    }
    
    CcToolchainFeatures.Variables variables = buildVariables.build();
    builder.setVariables(variables);
  }
  
  /**
   * Constructs the C++ compiler actions. It generally creates one action for every specified source
   * file. It takes into account LIPO, fake-ness, coverage, and PIC, in addition to using the
   * settings specified on the current object. This method should only be called once.
   */
  public CcCompilationOutputs createCcCompileActions() {
    CcCompilationOutputs.Builder result = new CcCompilationOutputs.Builder();
    Preconditions.checkNotNull(context);
    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();

    if (shouldProvideHeaderModules()) {
      Collection<Artifact> modules = createModuleAction(result, context.getCppModuleMap());
      if (featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULE_CODEGEN)) {
        for (Artifact module : modules) {
          createModuleCodegenAction(result, module);
        }
      }
    } else if (context.getVerificationModuleMap() != null) {
      Collection<Artifact> modules = createModuleAction(result, context.getVerificationModuleMap());
      for (Artifact module : modules) {
        result.addHeaderTokenFile(module);
      }
    }

    for (CppSource source : sourceFiles) {
      Artifact sourceArtifact = source.getSource();
      Label sourceLabel = source.getLabel();
      String outputName = FileSystemUtils.removeExtension(
          semantics.getEffectiveSourcePath(sourceArtifact)).getPathString();
      CppCompileActionBuilder builder = initializeCompileAction(sourceArtifact, sourceLabel);

      builder.setSemantics(semantics);

      if (!sourceArtifact.isTreeArtifact()) {
        switch (source.getType()) {
          case HEADER:
            createHeaderAction(outputName, result, env, builder,
                CppFileTypes.mustProduceDotdFile(sourceArtifact));
            break;
          case CLIF_INPUT_PROTO:
            createClifMatchAction(outputName, result, env, builder);
            break;
          default:
            boolean bitcodeOutput =
                featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
                    && CppFileTypes.LTO_SOURCE.matches(sourceArtifact.getFilename());
            createSourceAction(
                outputName,
                result,
                env,
                sourceArtifact,
                builder,
                ArtifactCategory.OBJECT_FILE,
                context.getCppModuleMap(),
                /*addObject=*/ true,
                isCodeCoverageEnabled(),
                // The source action does not generate dwo when it has bitcode
                // output (since it isn't generating a native object with debug
                // info). In that case the LTOBackendAction will generate the dwo.
                /*generateDwo=*/ cppConfiguration.useFission() && !bitcodeOutput,
                CppFileTypes.mustProduceDotdFile(sourceArtifact),
                source.getBuildVariables());
            break;
        }
      } else {
        switch (source.getType()) {
          case HEADER:
            Artifact headerTokenFile =
                createCompileActionTemplate(
                    env,
                    source,
                    builder,
                    ImmutableList.of(
                        ArtifactCategory.GENERATED_HEADER, ArtifactCategory.PROCESSED_HEADER));
            result.addHeaderTokenFile(headerTokenFile);
            break;
          case SOURCE:
            Artifact objectFile =
                createCompileActionTemplate(
                    env, source, builder, ImmutableList.of(ArtifactCategory.OBJECT_FILE));
            result.addObjectFile(objectFile);
            break;
          default:
            throw new IllegalStateException(
                "Encountered invalid source types when creating CppCompileActionTemplates");
        }
      }
    }

    compilationOutputs = result.build();
    return compilationOutputs;
  }

  private void createHeaderAction(String outputName, Builder result, AnalysisEnvironment env,
      CppCompileActionBuilder builder, boolean generateDotd) {
    String outputNameBase = CppHelper.getArtifactNameForCategory(
        ruleContext, ccToolchain, ArtifactCategory.GENERATED_HEADER, outputName);

    builder
        .setOutputs(ruleContext, ArtifactCategory.PROCESSED_HEADER, outputNameBase, generateDotd)
        // If we generate pic actions, we prefer the header actions to use the pic artifacts.
        .setPicMode(this.getGeneratePicActions());
    setupCompileBuildVariables(
        builder,
        this.getGeneratePicActions(),
        /*ccRelativeName=*/ null,
        /*autoFdoImportPath=*/ null,
        /*gcnoFile=*/ null,
        /*dwoFile=*/ null,
        builder.getContext().getCppModuleMap(),
        ImmutableMap.<String, String>of());
    semantics.finalizeCompileActionBuilder(ruleContext, builder);
    CppCompileAction compileAction = builder.buildAndValidate(ruleContext);
    env.registerAction(compileAction);
    Artifact tokenFile = compileAction.getOutputFile();
    result.addHeaderTokenFile(tokenFile);
  }

  private void createModuleCodegenAction(CcCompilationOutputs.Builder result, Artifact module) {
    if (fake) {
      // We can't currently foresee a situation where we'd want nocompile tests for module codegen.
      // If we find one, support needs to be added here.
      return;
    }
    String outputName = semantics.getEffectiveSourcePath(module).getPathString();
    
    // TODO(djasper): Make this less hacky after refactoring how the PIC/noPIC actions are created.
    boolean pic = module.getFilename().contains(".pic.");

    // TODO(djasper): Investigate whether we need to use a label separate from that of the module
    // map. It is used for per-file-copts.
    CppCompileActionBuilder builder =
        initializeCompileAction(
            module, Label.parseAbsoluteUnchecked(context.getCppModuleMap().getName()));
    builder.setSemantics(semantics);
    builder.setPicMode(pic);
    builder.setOutputs(
        ruleContext,
        ArtifactCategory.OBJECT_FILE,
        outputName,
        CppFileTypes.mustProduceDotdFile(module));
    PathFragment ccRelativeName = semantics.getEffectiveSourcePath(module);

    String gcnoFileName =
        CppHelper.getArtifactNameForCategory(
            ruleContext, ccToolchain, ArtifactCategory.COVERAGE_DATA_FILE, outputName);
    // TODO(djasper): This is now duplicated. Refactor the various create..Action functions.
    Artifact gcnoFile =
        isCodeCoverageEnabled() && !cppConfiguration.isLipoOptimization()
            ? CppHelper.getCompileOutputArtifact(ruleContext, gcnoFileName, configuration)
            : null;

    boolean generateDwo = cppConfiguration.useFission();
    Artifact dwoFile = generateDwo ? getDwoFile(builder.getOutputFile()) : null;

    setupCompileBuildVariables(
        builder,
        /*usePic=*/ pic,
        ccRelativeName,
        module.getExecPath(),
        gcnoFile,
        dwoFile,
        builder.getContext().getCppModuleMap(),
        ImmutableMap.<String, String>of());

    builder.setGcnoFile(gcnoFile);
    builder.setDwoFile(dwoFile);

    semantics.finalizeCompileActionBuilder(ruleContext, builder);
    CppCompileAction compileAction = builder.build();
    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
    env.registerAction(compileAction);
    Artifact objectFile = compileAction.getOutputFile();
    if (pic) {
      result.addPicObjectFile(objectFile);
    } else {
      result.addObjectFile(objectFile);
    }
  }

  private Collection<Artifact> createModuleAction(
      CcCompilationOutputs.Builder result, CppModuleMap cppModuleMap) {
    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
    Label moduleMapLabel = Label.parseAbsoluteUnchecked(context.getCppModuleMap().getName());
    Artifact moduleMapArtifact = cppModuleMap.getArtifact();
    CppCompileActionBuilder builder = initializeCompileAction(moduleMapArtifact, moduleMapLabel);

    builder.setSemantics(semantics);

    // A header module compile action is just like a normal compile action, but:
    // - the compiled source file is the module map
    // - it creates a header module (.pcm file).
    return createSourceAction(
        FileSystemUtils.removeExtension(semantics.getEffectiveSourcePath(moduleMapArtifact))
            .getPathString(),
        result,
        env,
        moduleMapArtifact,
        builder,
        ArtifactCategory.CPP_MODULE,
        cppModuleMap,
        /*addObject=*/ false,
        /*enableCoverage=*/ false,
        /*generateDwo=*/ false,
        CppFileTypes.mustProduceDotdFile(moduleMapArtifact),
        ImmutableMap.<String, String>of());
  }

  private void createClifMatchAction(
      String outputName, Builder result, AnalysisEnvironment env, CppCompileActionBuilder builder) {
    builder
        .setOutputs(ruleContext, ArtifactCategory.CLIF_OUTPUT_PROTO, outputName, false)
        .setPicMode(false)
        // The additional headers in a clif action are both mandatory inputs and
        // need to be include-scanned.
        .addMandatoryInputs(mandatoryInputs)
        .addAdditionalIncludes(mandatoryInputs);
    setupCompileBuildVariables(
        builder,
        /* usePic=*/ false,
        /*ccRelativeName=*/ null,
        /*autoFdoImportPath=*/ null,
        /*gcnoFile=*/ null,
        /*dwoFile=*/ null,
        builder.getContext().getCppModuleMap(),
        /*sourceSpecificBuildVariables=*/ ImmutableMap.<String, String>of());
    semantics.finalizeCompileActionBuilder(ruleContext, builder);
    CppCompileAction compileAction = builder.buildAndValidate(ruleContext);
    env.registerAction(compileAction);
    Artifact tokenFile = compileAction.getOutputFile();
    result.addHeaderTokenFile(tokenFile);
  }

  private Collection<Artifact> createSourceAction(
      String outputName,
      CcCompilationOutputs.Builder result,
      AnalysisEnvironment env,
      Artifact sourceArtifact,
      CppCompileActionBuilder builder,
      ArtifactCategory outputCategory,
      CppModuleMap cppModuleMap,
      boolean addObject,
      boolean enableCoverage,
      boolean generateDwo,
      boolean generateDotd,
      Map<String, String> sourceSpecificBuildVariables) {
    ImmutableList.Builder<Artifact> directOutputs = new ImmutableList.Builder<>();
    PathFragment ccRelativeName = semantics.getEffectiveSourcePath(sourceArtifact);
    if (cppConfiguration.isLipoOptimization()) {
      // TODO(bazel-team): we shouldn't be needing this, merging context with the binary
      // is a superset of necessary information.
      LipoContextProvider lipoProvider =
          Preconditions.checkNotNull(CppHelper.getLipoContextProvider(ruleContext), outputName);
      builder.setContext(CppCompilationContext.mergeForLipo(lipoProvider.getLipoContext(),
          context));
    }
    boolean generatePicAction = getGeneratePicActions();
    boolean generateNoPicAction = getGenerateNoPicActions();
    Preconditions.checkState(generatePicAction || generateNoPicAction);
    if (fake) {
      boolean usePic = !generateNoPicAction;
      createFakeSourceAction(outputName, result, env, builder, outputCategory, addObject,
          ccRelativeName, sourceArtifact.getExecPath(), usePic, generateDotd);
    } else {
      // Create PIC compile actions (same as non-PIC, but use -fPIC and
      // generate .pic.o, .pic.d, .pic.gcno instead of .o, .d, .gcno.)
      if (generatePicAction) {
        String picOutputBase = CppHelper.getArtifactNameForCategory(ruleContext,
            ccToolchain, ArtifactCategory.PIC_FILE, outputName);
        CppCompileActionBuilder picBuilder = copyAsPicBuilder(
            builder, picOutputBase, outputCategory, generateDotd);
        String gcnoFileName = CppHelper.getArtifactNameForCategory(ruleContext,
            ccToolchain, ArtifactCategory.COVERAGE_DATA_FILE, picOutputBase);
        Artifact gcnoFile = enableCoverage
            ? CppHelper.getCompileOutputArtifact(ruleContext, gcnoFileName, configuration)
            : null;
        Artifact dwoFile = generateDwo ? getDwoFile(picBuilder.getOutputFile()) : null;

        setupCompileBuildVariables(
            picBuilder,
            /*usePic=*/ true,
            ccRelativeName,
            sourceArtifact.getExecPath(),
            gcnoFile,
            dwoFile,
            cppModuleMap,
            sourceSpecificBuildVariables);

        if (maySaveTemps) {
          result.addTemps(
              createTempsActions(sourceArtifact, outputName, picBuilder, /*usePic=*/true,
                /*generateDotd=*/ generateDotd, ccRelativeName));
        }

        picBuilder.setGcnoFile(gcnoFile);
        picBuilder.setDwoFile(dwoFile);

        semantics.finalizeCompileActionBuilder(ruleContext, picBuilder);
        CppCompileAction picAction = picBuilder.buildAndValidate(ruleContext);
        env.registerAction(picAction);
        directOutputs.add(picAction.getOutputFile());
        if (addObject) {
          result.addPicObjectFile(picAction.getOutputFile());

          if (featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
              && CppFileTypes.LTO_SOURCE.matches(sourceArtifact.getFilename())) {
            result.addLTOBitcodeFile(picAction.getOutputFile());
          }
        }
        if (dwoFile != null) {
          // Host targets don't produce .dwo files.
          result.addPicDwoFile(dwoFile);
        }
        if (cppConfiguration.isLipoContextCollector() && !generateNoPicAction) {
          result.addLipoScannable(picAction);
        }
      }

      if (generateNoPicAction) {
        Artifact noPicOutputFile = CppHelper.getCompileOutputArtifact(
            ruleContext,
            CppHelper.getArtifactNameForCategory(
                ruleContext, ccToolchain, outputCategory, outputName),
            configuration);
        builder.setOutputs(ruleContext, outputCategory, outputName, generateDotd);
        String gcnoFileName = CppHelper.getArtifactNameForCategory(ruleContext,
            ccToolchain, ArtifactCategory.COVERAGE_DATA_FILE, outputName);

        // Create non-PIC compile actions
        Artifact gcnoFile =
            !cppConfiguration.isLipoOptimization() && enableCoverage
                ? CppHelper.getCompileOutputArtifact(ruleContext, gcnoFileName, configuration)
                : null;

        Artifact noPicDwoFile = generateDwo ? getDwoFile(noPicOutputFile) : null;

        setupCompileBuildVariables(
            builder,
            /*usePic=*/ false,
            ccRelativeName,
            sourceArtifact.getExecPath(),
            gcnoFile,
            noPicDwoFile,
            cppModuleMap,
            sourceSpecificBuildVariables);

        if (maySaveTemps) {
          result.addTemps(
              createTempsActions(
                  sourceArtifact,
                  outputName,
                  builder,
                  /*usePic=*/ false,
                  /*generateDotd*/ generateDotd,
                  ccRelativeName));
        }

        builder.setGcnoFile(gcnoFile);
        builder.setDwoFile(noPicDwoFile);

        semantics.finalizeCompileActionBuilder(ruleContext, builder);
        CppCompileAction compileAction = builder.buildAndValidate(ruleContext);
        env.registerAction(compileAction);
        Artifact objectFile = compileAction.getOutputFile();
        directOutputs.add(objectFile);
        if (addObject) {
          result.addObjectFile(objectFile);
          if (featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
              && CppFileTypes.LTO_SOURCE.matches(sourceArtifact.getFilename())) {
            result.addLTOBitcodeFile(objectFile);
          }
        }
        if (noPicDwoFile != null) {
          // Host targets don't produce .dwo files.
          result.addDwoFile(noPicDwoFile);
        }
        if (cppConfiguration.isLipoContextCollector()) {
          result.addLipoScannable(compileAction);
        }
      }
    }
    return directOutputs.build();
  }

  private Artifact createCompileActionTemplate(AnalysisEnvironment env,
      CppSource source, CppCompileActionBuilder builder,
      Iterable<ArtifactCategory> outputCategories) {
    Artifact sourceArtifact = source.getSource();
    Artifact outputFiles = CppHelper.getCompileOutputTreeArtifact(ruleContext, sourceArtifact);
    // TODO(rduan): Dotd file output is not supported yet.
    builder.setOutputs(outputFiles, null);
    setupCompileBuildVariables(
        builder,
        /* usePic=*/ false,
        /*ccRelativeName=*/ null,
        /*autoFdoImportPath=*/ null,
        /*gcnoFile=*/ null,
        /*dwoFile=*/ null,
        builder.getContext().getCppModuleMap(),
        source.getBuildVariables());
    semantics.finalizeCompileActionBuilder(ruleContext, builder);
    CppCompileActionTemplate actionTemplate = new CppCompileActionTemplate(
        sourceArtifact,
        outputFiles,
        builder,
        cppConfiguration,
        outputCategories,
        ruleContext.getActionOwner());
    env.registerAction(actionTemplate);

    return outputFiles;
  }

  String getOutputNameBaseWith(String base, boolean usePic) {
    return usePic
        ? CppHelper.getArtifactNameForCategory(
            ruleContext, ccToolchain, ArtifactCategory.PIC_FILE, base)
        : base;
  }

  private void createFakeSourceAction(String outputName, CcCompilationOutputs.Builder result,
      AnalysisEnvironment env, CppCompileActionBuilder builder,
      ArtifactCategory outputCategory, boolean addObject, PathFragment ccRelativeName,
      PathFragment execPath, boolean usePic, boolean generateDotd) {
    String outputNameBase = getOutputNameBaseWith(outputName, usePic);
    String tempOutputName = ruleContext.getConfiguration().getBinFragment()
        .getRelative(CppHelper.getObjDirectory(ruleContext.getLabel()))
        .getRelative(
            CppHelper.getArtifactNameForCategory(ruleContext, ccToolchain, outputCategory,
                getOutputNameBaseWith(outputName + ".temp", usePic)))
        .getPathString();
    builder
        .setPicMode(usePic)
        .setOutputs(ruleContext, outputCategory, outputNameBase, generateDotd)
        .setTempOutputFile(new PathFragment(tempOutputName));

    setupCompileBuildVariables(
        builder,
        usePic,
        ccRelativeName,
        execPath,
        /*gcnoFile=*/ null,
        /*dwoFile=*/ null,
        builder.getContext().getCppModuleMap(),
        ImmutableMap.<String, String>of());
    semantics.finalizeCompileActionBuilder(ruleContext, builder);
    CppCompileAction action = builder.buildAndValidate(ruleContext);
    env.registerAction(action);
    if (addObject) {
      if (usePic) {
        result.addPicObjectFile(action.getOutputFile());
      } else {
        result.addObjectFile(action.getOutputFile());
      }
    }
  }

  /**
   * Returns the linked artifact resulting from a linking of the given type. Consults the feature
   * configuration to obtain an action_config that provides the artifact. If the feature
   * configuration provides no artifact, uses a default.
   *
   * <p>We cannot assume that the feature configuration contains an action_config for the link
   * action, because the linux link action depends on hardcoded values in
   * LinkCommandLine.getRawLinkArgv(), which are applied on the condition that an action_config is
   * not present.
   * TODO(b/30393154): Assert that the given link action has an action_config.
   *
   * @throws RuleErrorException
   */
  private Artifact getLinkedArtifact(LinkTargetType linkTargetType) throws RuleErrorException {
    Artifact result = null;
    Artifact linuxDefault = CppHelper.getLinuxLinkedArtifact(
        ruleContext, linkTargetType, linkedArtifactNameSuffix);

    try {
      String maybePicName = ruleContext.getLabel().getName() + linkedArtifactNameSuffix;
      if (linkTargetType.picness() == Picness.PIC) {
        maybePicName = CppHelper.getArtifactNameForCategory(
            ruleContext, ccToolchain, ArtifactCategory.PIC_FILE, maybePicName);
      }
      String linkedName = CppHelper.getArtifactNameForCategory(
          ruleContext, ccToolchain, linkTargetType.getLinkerOutput(), maybePicName);
      PathFragment artifactFragment = new PathFragment(ruleContext.getLabel().getName())
          .getParentDirectory().getRelative(linkedName);

      result = ruleContext.getPackageRelativeArtifact(
          artifactFragment, configuration.getBinDirectory(ruleContext.getRule().getRepository()));
    } catch (ExpansionException e) {
      ruleContext.throwWithRuleError(e.getMessage());
    }

    // If the linked artifact is not the linux default, then a FailAction is generated for the
    // linux default to satisfy the requirement of the implicit output.
    // TODO(b/30132703): Remove the implicit outputs of cc_library.
    if (!result.equals(linuxDefault)) {
      ruleContext.registerAction(
          new FailAction(
              ruleContext.getActionOwner(),
              ImmutableList.of(linuxDefault),
              String.format(
                  "the given toolchain supports creation of %s instead of %s",
                  linuxDefault.getExecPathString(), result.getExecPathString())));
    }

    return result;
  }

  /**
   * Constructs the C++ linker actions. It generally generates two actions, one for a static library
   * and one for a dynamic library. If PIC is required for shared libraries, but not for binaries,
   * it additionally creates a third action to generate a PIC static library.
   *
   * <p>For dynamic libraries, this method can additionally create an interface shared library that
   * can be used for linking, but doesn't contain any executable code. This increases the number of
   * cache hits for link actions. Call {@link #setAllowInterfaceSharedObjects(boolean)} to enable
   * this behavior.
   *
   * @throws RuleErrorException
   */
  public CcLinkingOutputs createCcLinkActions(
      CcCompilationOutputs ccOutputs, Iterable<Artifact> nonCodeLinkerInputs)
      throws RuleErrorException, InterruptedException {
    // For now only handle static links. Note that the dynamic library link below ignores linkType.
    // TODO(bazel-team): Either support non-static links or move this check to setLinkType().
    Preconditions.checkState(
        linkType.staticness() == Staticness.STATIC, "can only handle static links");

    CcLinkingOutputs.Builder result = new CcLinkingOutputs.Builder();
    if (cppConfiguration.isLipoContextCollector()) {
      // Don't try to create LIPO link actions in collector mode,
      // because it needs some data that's not available at this point.
      return result.build();
    }

    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
    boolean usePicForBinaries = CppHelper.usePic(ruleContext, true);
    boolean usePicForSharedLibs = CppHelper.usePic(ruleContext, false);

    // Create static library (.a). The linkType only reflects whether the library is alwayslink or
    // not. The PIC-ness is determined by whether we need to use PIC or not. There are three cases
    // for (usePicForSharedLibs usePicForBinaries):
    //
    // (1) (false false) -> no pic code
    // (2) (true false)  -> shared libraries as pic, but not binaries
    // (3) (true true)   -> both shared libraries and binaries as pic
    //
    // In case (3), we always need PIC, so only create one static library containing the PIC object
    // files. The name therefore does not match the content.
    //
    // Presumably, it is done this way because the .a file is an implicit output of every cc_library
    // rule, so we can't use ".pic.a" that in the always-PIC case.

    // If the crosstool is configured to select an output artifact, we use that selection.
    // Otherwise, we use linux defaults.
    Artifact linkedArtifact = getLinkedArtifact(linkType);
    PathFragment labelName = new PathFragment(ruleContext.getLabel().getName());
    String libraryIdentifier = ruleContext.getPackageDirectory().getRelative(
        labelName.replaceName("lib" + labelName.getBaseName())).getPathString();
    CppLinkAction maybePicAction =
        newLinkActionBuilder(linkedArtifact)
            .addObjectFiles(ccOutputs.getObjectFiles(usePicForBinaries))
            .addNonCodeInputs(nonCodeLinkerInputs)
            .addLTOBitcodeFiles(ccOutputs.getLtoBitcodeFiles())
            .setLinkType(linkType)
            .setLinkStaticness(LinkStaticness.FULLY_STATIC)
            .addActionInputs(linkActionInputs)
            .setLibraryIdentifier(libraryIdentifier)
            .addVariablesExtensions(variablesExtensions)
            .setFeatureConfiguration(featureConfiguration)
            .build();
    env.registerAction(maybePicAction);
    if (linkType != LinkTargetType.EXECUTABLE) {
      result.addStaticLibrary(maybePicAction.getOutputLibrary());
    }

    // Create a second static library (.pic.a). Only in case (2) do we need both PIC and non-PIC
    // static libraries. In that case, the first static library contains the non-PIC code, and this
    // one contains the PIC code, so the names match the content.
    if (!usePicForBinaries && usePicForSharedLibs) {
      LinkTargetType picLinkType = (linkType == LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY)
          ? LinkTargetType.ALWAYS_LINK_PIC_STATIC_LIBRARY
          : LinkTargetType.PIC_STATIC_LIBRARY;

      // If the crosstool is configured to select an output artifact, we use that selection.
      // Otherwise, we use linux defaults.
      Artifact picArtifact = getLinkedArtifact(picLinkType);
      CppLinkAction picAction =
          newLinkActionBuilder(picArtifact)
              .addObjectFiles(ccOutputs.getObjectFiles(true))
              .addLTOBitcodeFiles(ccOutputs.getLtoBitcodeFiles())
              .setLinkType(picLinkType)
              .setLinkStaticness(LinkStaticness.FULLY_STATIC)
              .addActionInputs(linkActionInputs)
              .setLibraryIdentifier(libraryIdentifier)
              .addVariablesExtensions(variablesExtensions)
              .setFeatureConfiguration(featureConfiguration)
              .build();
      env.registerAction(picAction);
      if (linkType != LinkTargetType.EXECUTABLE) {
        result.addPicStaticLibrary(picAction.getOutputLibrary());
      }
    }

    if (!createDynamicLibrary) {
      return result.build();
    }

    // Create dynamic library.
    Artifact soImpl;
    String mainLibraryIdentifier;
    if (soImplArtifact == null) {
      // If the crosstool is configured to select an output artifact, we use that selection.
      // Otherwise, we use linux defaults.
      soImpl = getLinkedArtifact(LinkTargetType.DYNAMIC_LIBRARY);
      mainLibraryIdentifier = libraryIdentifier;
    } else {
      // This branch is only used for vestigial Google-internal rules where the name of the output
      // file is explicitly specified in the BUILD file and as such, is platform-dependent. Thus,
      // we just hardcode some reasonable logic to compute the library identifier and hope that this
      // will eventually go away.
      soImpl = soImplArtifact;
      mainLibraryIdentifier = FileSystemUtils.removeExtension(
          soImpl.getRootRelativePath().getPathString());

    }

    List<String> sonameLinkopts = ImmutableList.of();
    Artifact soInterface = null;
    if (cppConfiguration.useInterfaceSharedObjects() && allowInterfaceSharedObjects) {
      soInterface =
          CppHelper.getLinuxLinkedArtifact(
              ruleContext, LinkTargetType.INTERFACE_DYNAMIC_LIBRARY, linkedArtifactNameSuffix);
      sonameLinkopts = ImmutableList.of("-Wl,-soname=" +
          SolibSymlinkAction.getDynamicLibrarySoname(soImpl.getRootRelativePath(), false));
    }
    
    // Should we also link in any libraries that this library depends on?
    // That is required on some systems...
    CppLinkActionBuilder linkActionBuilder =
        newLinkActionBuilder(soImpl)
            .setInterfaceOutput(soInterface)
            .addObjectFiles(ccOutputs.getObjectFiles(usePicForSharedLibs))
            .addNonCodeInputs(ccOutputs.getHeaderTokenFiles())
            .addLTOBitcodeFiles(ccOutputs.getLtoBitcodeFiles())
            .setLinkType(LinkTargetType.DYNAMIC_LIBRARY)
            .setLinkStaticness(LinkStaticness.DYNAMIC)
            .addActionInputs(linkActionInputs)
            .setLibraryIdentifier(mainLibraryIdentifier)
            .addLinkopts(linkopts)
            .addLinkopts(sonameLinkopts)
            .setRuntimeInputs(
                ArtifactCategory.DYNAMIC_LIBRARY,
                ccToolchain.getDynamicRuntimeLinkMiddleman(),
                ccToolchain.getDynamicRuntimeLinkInputs())
            .setFeatureConfiguration(featureConfiguration)
            .addVariablesExtensions(variablesExtensions);

    if (!ccOutputs.getLtoBitcodeFiles().isEmpty()
        && featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)) {
      linkActionBuilder.setLTOIndexing(true);
      linkActionBuilder.setUsePicForLTOBackendActions(usePicForSharedLibs);
      // If support is ever added for generating a dwp file for shared
      // library targets (e.g. when linkstatic=0), then this should change
      // to generate dwo files when cppConfiguration.useFission(),
      // and the dwp generating action for the shared library should
      // include all of the resulting dwo files.
      linkActionBuilder.setUseFissionForLTOBackendActions(false);
      CppLinkAction indexAction = linkActionBuilder.build();
      env.registerAction(indexAction);

      linkActionBuilder.setLTOIndexing(false);
    }

    CppLinkAction action = linkActionBuilder.build();
    env.registerAction(action);

    if (linkType == LinkTargetType.EXECUTABLE) {
      return result.build();
    }

    LibraryToLink dynamicLibrary = action.getOutputLibrary();
    LibraryToLink interfaceLibrary = action.getInterfaceOutputLibrary();
    if (interfaceLibrary == null) {
      interfaceLibrary = dynamicLibrary;
    }

    // If shared library has neverlink=1, then leave it untouched. Otherwise,
    // create a mangled symlink for it and from now on reference it through
    // mangled name only.
    if (neverLink) {
      result.addDynamicLibrary(interfaceLibrary);
      result.addExecutionDynamicLibrary(dynamicLibrary);
    } else {
      Artifact libraryLink = SolibSymlinkAction.getDynamicLibrarySymlink(
          ruleContext, interfaceLibrary.getArtifact(), false, false,
          ruleContext.getConfiguration());
      result.addDynamicLibrary(LinkerInputs.solibLibraryToLink(
          libraryLink, interfaceLibrary.getArtifact(), libraryIdentifier));
      Artifact implLibraryLink = SolibSymlinkAction.getDynamicLibrarySymlink(
          ruleContext, dynamicLibrary.getArtifact(), false, false,
          ruleContext.getConfiguration());
      result.addExecutionDynamicLibrary(LinkerInputs.solibLibraryToLink(
          implLibraryLink, dynamicLibrary.getArtifact(), libraryIdentifier));
    }
    return result.build();
  }

  private CppLinkActionBuilder newLinkActionBuilder(Artifact outputArtifact) {
    return new CppLinkActionBuilder(ruleContext, outputArtifact, ccToolchain, fdoSupport)
        .setCrosstoolInputs(ccToolchain.getLink())
        .addNonCodeInputs(context.getTransitiveCompilationPrerequisites());
  }

  /**
   * Creates a basic cpp compile action builder for source file. Configures options, crosstool
   * inputs, output and dotd file names, compilation context and copts.
   */
  private CppCompileActionBuilder createCompileActionBuilder(Artifact source, Label label) {
    CppCompileActionBuilder builder =
        new CppCompileActionBuilder(ruleContext, label, ccToolchain, configuration);
    builder.setSourceFile(source);
    builder.setContext(context).addCopts(copts);
    builder.addEnvironment(ccToolchain.getEnvironment());
    return builder;
  }

  /**
   * Creates cpp PIC compile action builder from the given builder by adding necessary copt and
   * changing output and dotd file names.
   */
  private CppCompileActionBuilder copyAsPicBuilder(CppCompileActionBuilder builder,
      String outputName, ArtifactCategory outputCategory,
      boolean generateDotd) {
    CppCompileActionBuilder picBuilder = new CppCompileActionBuilder(builder);
    picBuilder
        .setPicMode(true)
        .setOutputs(ruleContext, outputCategory, outputName, generateDotd);

    return picBuilder;
  }

  /**
   * Create the actions for "--save_temps".
   */
  private ImmutableList<Artifact> createTempsActions(Artifact source, String outputName,
      CppCompileActionBuilder builder, boolean usePic, boolean generateDotd,
      PathFragment ccRelativeName) {
    if (!cppConfiguration.getSaveTemps()) {
      return ImmutableList.of();
    }

    String path = source.getFilename();
    boolean isCFile = CppFileTypes.C_SOURCE.matches(path);
    boolean isCppFile = CppFileTypes.CPP_SOURCE.matches(path);

    if (!isCFile && !isCppFile) {
      return ImmutableList.of();
    }

    ArtifactCategory category = isCFile
        ? ArtifactCategory.PREPROCESSED_C_SOURCE : ArtifactCategory.PREPROCESSED_CPP_SOURCE;

    String outputArtifactNameBase = getOutputNameBaseWith(outputName, usePic);

    CppCompileActionBuilder dBuilder = new CppCompileActionBuilder(builder);
    dBuilder.setOutputs(ruleContext, category, outputArtifactNameBase, generateDotd);
    setupCompileBuildVariables(
        dBuilder,
        usePic,
        ccRelativeName,
        source.getExecPath(),
        null,
        null,
        builder.getContext().getCppModuleMap(),
        ImmutableMap.<String, String>of());
    semantics.finalizeCompileActionBuilder(ruleContext, dBuilder);
    CppCompileAction dAction = dBuilder.buildAndValidate(ruleContext);
    ruleContext.registerAction(dAction);

    CppCompileActionBuilder sdBuilder = new CppCompileActionBuilder(builder);
    sdBuilder.setOutputs(
        ruleContext, ArtifactCategory.GENERATED_ASSEMBLY, outputArtifactNameBase, generateDotd);
    setupCompileBuildVariables(
        sdBuilder,
        usePic,
        ccRelativeName,
        source.getExecPath(),
        null,
        null,
        builder.getContext().getCppModuleMap(),
        ImmutableMap.<String, String>of());
    semantics.finalizeCompileActionBuilder(ruleContext, sdBuilder);
    CppCompileAction sdAction = sdBuilder.buildAndValidate(ruleContext);
    ruleContext.registerAction(sdAction);

    return ImmutableList.of(
        dAction.getOutputFile(),
        sdAction.getOutputFile());
  }

  /**
   * Returns true iff code coverage is enabled for the given target.
   */
  private boolean isCodeCoverageEnabled() {
    if (configuration.isCodeCoverageEnabled()) {
      // If rule is matched by the instrumentation filter, enable instrumentation
      if (InstrumentedFilesCollector.shouldIncludeLocalSources(ruleContext)) {
        return true;
      }
      // At this point the rule itself is not matched by the instrumentation filter. However, we
      // might still want to instrument C++ rules if one of the targets listed in "deps" is
      // instrumented and, therefore, can supply header files that we would want to collect code
      // coverage for. For example, think about cc_test rule that tests functionality defined in a
      // header file that is supplied by the cc_library.
      //
      // Note that we only check direct prerequisites and not the transitive closure. This is done
      // for two reasons:
      // a) It is a good practice to declare libraries which you directly rely on. Including headers
      //    from a library hidden deep inside the transitive closure makes build dependencies less
      //    readable and can lead to unexpected breakage.
      // b) Traversing the transitive closure for each C++ compile action would require more complex
      //    implementation (with caching results of this method) to avoid O(N^2) slowdown.
      if (ruleContext.getRule().isAttrDefined("deps", BuildType.LABEL_LIST)) {
        for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
          if (dep.getProvider(CppCompilationContext.class) != null
              && InstrumentedFilesCollector.shouldIncludeLocalSources(configuration, dep)) {
            return true;
          }
        }
      }
    }
    return false;
  }
}
