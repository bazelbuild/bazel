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

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
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

/**
 * Representation of a C/C++ compilation. Its purpose is to share the code that creates compilation
 * actions between all classes that need to do so. It follows the builder pattern - load up the
 * necessary settings and then call {@link #createCcCompileActions}.
 *
 * <p>This class is not thread-safe, and it should only be used once for each set of source files,
 * i.e. calling {@link #createCcCompileActions} will throw an Exception if called twice.
 */
public final class CppModel {

  /** Name of the build variable for the path to the source file being compiled. */
  public static final String SOURCE_FILE_VARIABLE_NAME = "source_file";

  /** Name of the build variable for the path to the input file being processed. */
  public static final String INPUT_FILE_VARIABLE_NAME = "input_file";

  /** Name of the build variable for the path to the compilation output file. */
  public static final String OUTPUT_FILE_VARIABLE_NAME = "output_file";

  /**
   * Name of the build variable for the path to the compilation output file in case of assembly
   * source.
   */
  public static final String OUTPUT_ASSEMBLY_FILE_VARIABLE_NAME = "output_assembly_file";

  /**
   * Name of the build variable for the path to the compilation output file in case of preprocessed
   * source.
   */
  public static final String OUTPUT_PREPROCESS_FILE_VARIABLE_NAME = "output_preprocess_file";

  /** Name of the build variable for the path to the output file when output is an object file. */
  public static final String OUTPUT_OBJECT_FILE_VARIABLE_NAME = "output_object_file";

  /** Name of the build variable for the module file name. */
  public static final String MODULE_NAME_VARIABLE_NAME = "module_name";

  /** Name of the build variable for the module map file name. */
  public static final String MODULE_MAP_FILE_VARIABLE_NAME = "module_map_file";

  /** Name of the build variable for the dependent module map file name. */
  public static final String DEPENDENT_MODULE_MAP_FILES_VARIABLE_NAME =
      "dependent_module_map_files";

  /** Name of the build variable for the collection of module files. */
  public static final String MODULE_FILES_VARIABLE_NAME = "module_files";

  /**
   * Name of the build variable for includes that compiler needs to include into sources to be
   * compiled.
   */
  public static final String INCLUDES_VARIABLE_NAME = "includes";

  /**
   * Name of the build variable for the collection of include paths.
   *
   * @see CppCompilationContext#getIncludeDirs().
   */
  public static final String INCLUDE_PATHS_VARIABLE_NAME = "include_paths";

  /**
   * Name of the build variable for the collection of quote include paths.
   *
   * @see CppCompilationContext#getIncludeDirs().
   */
  public static final String QUOTE_INCLUDE_PATHS_VARIABLE_NAME = "quote_include_paths";

  /**
   * Name of the build variable for the collection of system include paths.
   *
   * @see CppCompilationContext#getIncludeDirs().
   */
  public static final String SYSTEM_INCLUDE_PATHS_VARIABLE_NAME = "system_include_paths";

  /** Name of the build variable for the dependency file path */
  public static final String DEPENDENCY_FILE_VARIABLE_NAME = "dependency_file";

  /** Name of the build variable for the collection of macros defined for preprocessor. */
  public static final String PREPROCESSOR_DEFINES_VARIABLE_NAME = "preprocessor_defines";

  /** Name of the build variable present when the output is compiled as position independent. */
  public static final String PIC_VARIABLE_NAME = "pic";

  /** Name of the build variable for the gcov coverage file path. */
  public static final String GCOV_GCNO_FILE_VARIABLE_NAME = "gcov_gcno_file";

  /** Name of the build variable for the per object debug info file. */
  public static final String PER_OBJECT_DEBUG_INFO_FILE_VARIABLE_NAME =
      "per_object_debug_info_file";

  /** Name of the build variable for the LTO indexing bitcode file. */
  public static final String LTO_INDEXING_BITCODE_FILE_VARIABLE_NAME = "lto_indexing_bitcode_file";

  /** Name of the build variable for stripopts for the strip action. */
  public static final String STRIPOPTS_VARIABLE_NAME = "stripopts";

  /**
   * Build variable for all flags coming from legacy crosstool fields, such as compiler_flag,
   * optional_compiler_flag, cxx_flag, optional_cxx_flag.
   */
  public static final String LEGACY_COMPILE_FLAGS_VARIABLE_NAME = "legacy_compile_flags";

  /**
   * Build variable for all flags coming from copt rule attribute, and from --copt, --cxxopt, or
   * --conlyopt options.
   */
  public static final String USER_COMPILE_FLAGS_VARIABLE_NAME = "user_compile_flags";

  /** Build variable for flags coming from unfiltered_cxx_flag CROSSTOOL fields. */
  public static final String UNFILTERED_COMPILE_FLAGS_VARIABLE_NAME = "unfiltered_compile_flags";

  /** Name of the build variable for the sysroot path variable name. */
  public static final String SYSROOT_VARIABLE_NAME = "sysroot";

  private final CppSemantics semantics;
  private final RuleContext ruleContext;
  private final BuildConfiguration configuration;
  private final CppConfiguration cppConfiguration;

  // compile model
  private CppCompilationContext context;
  private final Set<CppSource> sourceFiles = new LinkedHashSet<>();
  private final List<Artifact> mandatoryInputs = new ArrayList<>();
  private final ImmutableList<String> copts;
  private final Predicate<String> coptsFilter;
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
  private final ImmutableSet<String> features;

  public CppModel(
      RuleContext ruleContext,
      CppSemantics semantics,
      CcToolchainProvider ccToolchain,
      FdoSupportProvider fdoSupport,
      ImmutableList<String> copts) {
    this(
        ruleContext,
        semantics,
        ccToolchain,
        fdoSupport,
        ruleContext.getConfiguration(),
        copts,
        Predicates.alwaysTrue());
  }

  public CppModel(
      RuleContext ruleContext,
      CppSemantics semantics,
      CcToolchainProvider ccToolchain,
      FdoSupportProvider fdoSupport,
      ImmutableList<String> copts,
      Predicate<String> coptsFilter) {
    this(
        ruleContext,
        semantics,
        ccToolchain,
        fdoSupport,
        ruleContext.getConfiguration(),
        copts,
        coptsFilter);
 }

  public CppModel(
      RuleContext ruleContext,
      CppSemantics semantics,
      CcToolchainProvider ccToolchain,
      FdoSupportProvider fdoSupport,
      BuildConfiguration configuration,
      ImmutableList<String> copts,
      Predicate<String> coptsFilter) {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);
    this.semantics = semantics;
    this.ccToolchain = Preconditions.checkNotNull(ccToolchain);
    this.fdoSupport = Preconditions.checkNotNull(fdoSupport);
    this.configuration = configuration;
    this.copts = copts;
    this.coptsFilter = Preconditions.checkNotNull(coptsFilter);
    cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    features = ruleContext.getFeatures();
  }

  private Artifact getDwoFile(Artifact outputFile) {
    return ruleContext.getRelatedArtifact(outputFile.getRootRelativePath(), ".dwo");
  }

  private Artifact getLtoIndexingFile(Artifact outputFile) {
    String ext = Iterables.getOnlyElement(CppFileTypes.LTO_INDEXING_OBJECT_FILE.getExtensions());
    return ruleContext.getRelatedArtifact(outputFile.getRootRelativePath(), ext);
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

  /** Sets the feature configuration to be used for C/C++ actions. */
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
  private CppCompileActionBuilder initializeCompileAction(Artifact sourceArtifact) {
    CppCompileActionBuilder builder = createCompileActionBuilder(sourceArtifact);
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

  /**
   * Supplier that computes unfiltered_compile_flags lazily at the execution phase.
   *
   * <p>Dear friends of the lambda, this method exists to limit the scope of captured variables
   * only to arguments (to prevent accidental capture of enclosing instance which could regress
   * memory).
   */
  public static Supplier<ImmutableList<String>> getUnfilteredCompileFlagsSupplier(
      CcToolchainProvider ccToolchain, ImmutableSet<String> features) {
    return () -> ccToolchain.getUnfilteredCompilerOptions(features);
  }

  /**
   * Supplier that computes legacy_compile_flags lazily at the execution phase.
   *
   * <p>Dear friends of the lambda, this method exists to limit the scope of captured variables
   * only to arguments (to prevent accidental capture of enclosing instance which could regress
   * memory).
   */
  public static Supplier<ImmutableList<String>> getLegacyCompileFlagsSupplier(
      CppConfiguration cppConfiguration, String sourceFilename, ImmutableSet<String> features) {
    return () -> cppConfiguration.collectLegacyCompileFlags(sourceFilename, features);
  }

  private void setupCompileBuildVariables(
      CppCompileActionBuilder builder,
      Label sourceLabel,
      boolean usePic,
      PathFragment ccRelativeName,
      PathFragment autoFdoImportPath,
      Artifact gcnoFile,
      Artifact dwoFile,
      Artifact ltoIndexingFile,
      CppModuleMap cppModuleMap,
      Map<String, String> sourceSpecificBuildVariables) {
    CcToolchainFeatures.Variables.Builder buildVariables =
        new CcToolchainFeatures.Variables.Builder(ccToolchain.getBuildVariables());

    CppCompilationContext builderContext = builder.getContext();
    Artifact sourceFile = builder.getSourceFile();
    Artifact outputFile = builder.getOutputFile();
    String realOutputFilePath;

    buildVariables.addStringVariable(SOURCE_FILE_VARIABLE_NAME, sourceFile.getExecPathString());
    buildVariables.addStringVariable(OUTPUT_FILE_VARIABLE_NAME, outputFile.getExecPathString());
    buildVariables.addStringSequenceVariable(
        USER_COMPILE_FLAGS_VARIABLE_NAME,
        ImmutableList.<String>builder()
            .addAll(copts)
            .addAll(collectPerFileCopts(sourceFile, sourceLabel))
            .build());

    String sourceFilename = sourceFile.getExecPathString();
    buildVariables.addLazyStringSequenceVariable(
        LEGACY_COMPILE_FLAGS_VARIABLE_NAME,
        getLegacyCompileFlagsSupplier(cppConfiguration, sourceFilename, features));

    if (!CppFileTypes.OBJC_SOURCE.matches(sourceFilename)
        && !CppFileTypes.OBJCPP_SOURCE.matches(sourceFilename)) {
      buildVariables.addLazyStringSequenceVariable(
          UNFILTERED_COMPILE_FLAGS_VARIABLE_NAME,
          getUnfilteredCompileFlagsSupplier(ccToolchain, features));
    }

    if (builder.getTempOutputFile() != null) {
      realOutputFilePath = builder.getTempOutputFile().getPathString();
    } else {
      realOutputFilePath = builder.getOutputFile().getExecPathString();
    }

    if (FileType.contains(outputFile, CppFileTypes.ASSEMBLER, CppFileTypes.PIC_ASSEMBLER)) {
      buildVariables.addStringVariable(OUTPUT_ASSEMBLY_FILE_VARIABLE_NAME, realOutputFilePath);
    } else if (FileType.contains(outputFile, CppFileTypes.PREPROCESSED_C,
        CppFileTypes.PREPROCESSED_CPP, CppFileTypes.PIC_PREPROCESSED_C,
        CppFileTypes.PIC_PREPROCESSED_CPP)) {
      buildVariables.addStringVariable(OUTPUT_PREPROCESS_FILE_VARIABLE_NAME, realOutputFilePath);
    } else {
      buildVariables.addStringVariable(OUTPUT_OBJECT_FILE_VARIABLE_NAME, realOutputFilePath);
    }

    DotdFile dotdFile =
        isGenerateDotdFile(sourceFile) ? Preconditions.checkNotNull(builder.getDotdFile()) : null;
    // Set dependency_file to enable <object>.d file generation.
    if (dotdFile != null) {
      buildVariables.addStringVariable(
          DEPENDENCY_FILE_VARIABLE_NAME, dotdFile.getSafeExecPath().getPathString());
    }

    if (featureConfiguration.isEnabled(CppRuleClasses.MODULE_MAPS) && cppModuleMap != null) {
      // If the feature is enabled and cppModuleMap is null, we are about to fail during analysis
      // in any case, but don't crash.
      buildVariables.addStringVariable(MODULE_NAME_VARIABLE_NAME, cppModuleMap.getName());
      buildVariables.addStringVariable(
          MODULE_MAP_FILE_VARIABLE_NAME, cppModuleMap.getArtifact().getExecPathString());
      StringSequenceBuilder sequence = new StringSequenceBuilder();
      for (Artifact artifact : builderContext.getDirectModuleMaps()) {
        sequence.addValue(artifact.getExecPathString());
      }
      buildVariables.addCustomBuiltVariable(DEPENDENT_MODULE_MAP_FILES_VARIABLE_NAME, sequence);
    }
    if (featureConfiguration.isEnabled(CppRuleClasses.USE_HEADER_MODULES)) {
      // Module inputs will be set later when the action is executed.
      buildVariables.addStringSequenceVariable(MODULE_FILES_VARIABLE_NAME, ImmutableSet.of());
    }
    if (featureConfiguration.isEnabled(CppRuleClasses.INCLUDE_PATHS)) {
      buildVariables.addStringSequenceVariable(
          INCLUDE_PATHS_VARIABLE_NAME, getSafePathStrings(builderContext.getIncludeDirs()));
      buildVariables.addStringSequenceVariable(
          QUOTE_INCLUDE_PATHS_VARIABLE_NAME,
          getSafePathStrings(builderContext.getQuoteIncludeDirs()));
      buildVariables.addStringSequenceVariable(
          SYSTEM_INCLUDE_PATHS_VARIABLE_NAME,
          getSafePathStrings(builderContext.getSystemIncludeDirs()));
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

      buildVariables.addStringSequenceVariable(PREPROCESSOR_DEFINES_VARIABLE_NAME, defines);
    }

    if (usePic) {
      if (!featureConfiguration.isEnabled(CppRuleClasses.PIC)) {
        ruleContext.ruleError("PIC compilation is requested but the toolchain does not support it");
      }
      buildVariables.addStringVariable(PIC_VARIABLE_NAME, "");
    }

    if (ccRelativeName != null) {
      fdoSupport.getFdoSupport().configureCompilation(
          builder, buildVariables, ruleContext, ccRelativeName, autoFdoImportPath, usePic,
          featureConfiguration, fdoSupport);
    }
    if (gcnoFile != null) {
      buildVariables.addStringVariable(GCOV_GCNO_FILE_VARIABLE_NAME, gcnoFile.getExecPathString());
    }

    if (dwoFile != null) {
      buildVariables.addStringVariable(
          PER_OBJECT_DEBUG_INFO_FILE_VARIABLE_NAME, dwoFile.getExecPathString());
    }

    if (ltoIndexingFile != null) {
      buildVariables.addStringVariable(
          LTO_INDEXING_BITCODE_FILE_VARIABLE_NAME, ltoIndexingFile.getExecPathString());
    }

    buildVariables.addAllStringVariables(sourceSpecificBuildVariables);

    for (VariablesExtension extension : variablesExtensions) {
      extension.addVariables(buildVariables);
    }

    CcToolchainFeatures.Variables variables = buildVariables.build();
    builder.setVariables(variables);
  }

  private ImmutableList<String> collectPerFileCopts(Artifact sourceFile, Label sourceLabel) {
    return cppConfiguration
        .getPerFileCopts()
        .stream()
        .filter(
            perLabelOptions ->
                (sourceLabel != null && perLabelOptions.isIncluded(sourceLabel))
                    || perLabelOptions.isIncluded(sourceFile))
        .map(PerLabelOptions::getOptions)
        .flatMap(options -> options.stream())
        .collect(ImmutableList.toImmutableList());
  }

  /** Returns true if Dotd file should be generated. */
  private boolean isGenerateDotdFile(Artifact sourceArtifact) {
    return CppFileTypes.headerDiscoveryRequired(sourceArtifact)
        && !featureConfiguration.isEnabled(CppRuleClasses.PARSE_SHOWINCLUDES);
  }

  /**
   * Constructs the C++ compiler actions. It generally creates one action for every specified source
   * file. It takes into account LIPO, fake-ness, coverage, and PIC, in addition to using the
   * settings specified on the current object. This method should only be called once.
   */
  public CcCompilationOutputs createCcCompileActions() throws RuleErrorException {
    CcCompilationOutputs.Builder result = new CcCompilationOutputs.Builder();
    Preconditions.checkNotNull(context);
    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();

    if (shouldProvideHeaderModules()) {
      Label moduleMapLabel = Label.parseAbsoluteUnchecked(context.getCppModuleMap().getName());
      Collection<Artifact> modules = createModuleAction(result, context.getCppModuleMap());
      if (featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULE_CODEGEN)) {
        for (Artifact module : modules) {
          // TODO(djasper): Investigate whether we need to use a label separate from that of the
          // module map. It is used for per-file-copts.
          createModuleCodegenAction(result, moduleMapLabel, module);
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
      CppCompileActionBuilder builder = initializeCompileAction(sourceArtifact);

      builder.setSemantics(semantics);

      if (!sourceArtifact.isTreeArtifact()) {
        switch (source.getType()) {
          case HEADER:
            createHeaderAction(
                sourceLabel, outputName, result, env, builder, isGenerateDotdFile(sourceArtifact));
            break;
          case CLIF_INPUT_PROTO:
            createClifMatchAction(sourceLabel, outputName, result, env, builder);
            break;
          default:
            boolean bitcodeOutput =
                featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
                    && CppFileTypes.LTO_SOURCE.matches(sourceArtifact.getFilename());
            createSourceAction(
                sourceLabel,
                outputName,
                result,
                env,
                sourceArtifact,
                builder,
                ArtifactCategory.OBJECT_FILE,
                context.getCppModuleMap(),
                /* addObject= */ true,
                isCodeCoverageEnabled(),
                // The source action does not generate dwo when it has bitcode
                // output (since it isn't generating a native object with debug
                // info). In that case the LtoBackendAction will generate the dwo.
                /* generateDwo= */ cppConfiguration.useFission() && !bitcodeOutput,
                isGenerateDotdFile(sourceArtifact),
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

  private void createHeaderAction(
      Label sourceLabel,
      String outputName,
      Builder result,
      AnalysisEnvironment env,
      CppCompileActionBuilder builder,
      boolean generateDotd)
      throws RuleErrorException {
    String outputNameBase = CppHelper.getArtifactNameForCategory(
        ruleContext, ccToolchain, ArtifactCategory.GENERATED_HEADER, outputName);

    builder
        .setOutputs(ruleContext, ArtifactCategory.PROCESSED_HEADER, outputNameBase, generateDotd)
        // If we generate pic actions, we prefer the header actions to use the pic artifacts.
        .setPicMode(getGeneratePicActions());
    setupCompileBuildVariables(
        builder,
        sourceLabel,
        this.getGeneratePicActions(),
        /* ccRelativeName= */ null,
        /* autoFdoImportPath= */ null,
        /* gcnoFile= */ null,
        /* dwoFile= */ null,
        /* ltoIndexingFile= */ null,
        builder.getContext().getCppModuleMap(),
        /* sourceSpecificBuildVariables= */ ImmutableMap.of());
    semantics.finalizeCompileActionBuilder(
        ruleContext,
        builder,
        featureConfiguration.getFeatureSpecification(),
        coptsFilter,
        features);
    CppCompileAction compileAction = builder.buildOrThrowRuleError(ruleContext);
    env.registerAction(compileAction);
    Artifact tokenFile = compileAction.getOutputFile();
    result.addHeaderTokenFile(tokenFile);
  }

  private void createModuleCodegenAction(
      CcCompilationOutputs.Builder result, Label sourceLabel, Artifact module)
      throws RuleErrorException {
    if (fake) {
      // We can't currently foresee a situation where we'd want nocompile tests for module codegen.
      // If we find one, support needs to be added here.
      return;
    }
    String outputName = semantics.getEffectiveSourcePath(module).getPathString();

    // TODO(djasper): Make this less hacky after refactoring how the PIC/noPIC actions are created.
    boolean pic = module.getFilename().contains(".pic.");

    CppCompileActionBuilder builder = initializeCompileAction(module);
    builder.setSemantics(semantics);
    builder.setPicMode(pic);
    builder.setOutputs(
        ruleContext, ArtifactCategory.OBJECT_FILE, outputName, isGenerateDotdFile(module));
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
    // TODO(tejohnson): Add support for ThinLTO if needed.
    boolean bitcodeOutput =
        featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
            && CppFileTypes.LTO_SOURCE.matches(module.getFilename());
    Preconditions.checkState(!bitcodeOutput);

    setupCompileBuildVariables(
        builder,
        sourceLabel,
        /* usePic= */ pic,
        ccRelativeName,
        module.getExecPath(),
        gcnoFile,
        dwoFile,
        /* ltoIndexingFile= */ null,
        builder.getContext().getCppModuleMap(),
        /* sourceSpecificBuildVariables= */ ImmutableMap.of());

    builder.setGcnoFile(gcnoFile);
    builder.setDwoFile(dwoFile);

    semantics.finalizeCompileActionBuilder(
        ruleContext,
        builder,
        featureConfiguration.getFeatureSpecification(),
        coptsFilter,
        features);
    CppCompileAction compileAction = builder.buildOrThrowRuleError(ruleContext);
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
      CcCompilationOutputs.Builder result, CppModuleMap cppModuleMap) throws RuleErrorException {
    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
    Artifact moduleMapArtifact = cppModuleMap.getArtifact();
    CppCompileActionBuilder builder = initializeCompileAction(moduleMapArtifact);

    builder.setSemantics(semantics);

    // A header module compile action is just like a normal compile action, but:
    // - the compiled source file is the module map
    // - it creates a header module (.pcm file).
    return createSourceAction(
        Label.parseAbsoluteUnchecked(cppModuleMap.getName()),
        FileSystemUtils.removeExtension(semantics.getEffectiveSourcePath(moduleMapArtifact))
            .getPathString(),
        result,
        env,
        moduleMapArtifact,
        builder,
        ArtifactCategory.CPP_MODULE,
        cppModuleMap,
        /* addObject= */ false,
        /* enableCoverage= */ false,
        /* generateDwo= */ false,
        isGenerateDotdFile(moduleMapArtifact),
        ImmutableMap.of());
  }

  private void createClifMatchAction(
      Label sourceLabel,
      String outputName,
      Builder result,
      AnalysisEnvironment env,
      CppCompileActionBuilder builder)
      throws RuleErrorException {
    builder
        .setOutputs(
            ruleContext, ArtifactCategory.CLIF_OUTPUT_PROTO, outputName, /* generateDotd= */ true)
        .setPicMode(false)
        // The additional headers in a clif action are both mandatory inputs and
        // need to be include-scanned.
        .addMandatoryInputs(mandatoryInputs)
        .addAdditionalIncludes(mandatoryInputs);
    setupCompileBuildVariables(
        builder,
        sourceLabel,
        /* usePic= */ false,
        /* ccRelativeName= */ null,
        /* autoFdoImportPath= */ null,
        /* gcnoFile= */ null,
        /* dwoFile= */ null,
        /* ltoIndexingFile= */ null,
        builder.getContext().getCppModuleMap(),
        /* sourceSpecificBuildVariables= */ ImmutableMap.of());
    semantics.finalizeCompileActionBuilder(
        ruleContext,
        builder,
        featureConfiguration.getFeatureSpecification(),
        coptsFilter,
        features);
    CppCompileAction compileAction = builder.buildOrThrowRuleError(ruleContext);
    env.registerAction(compileAction);
    Artifact tokenFile = compileAction.getOutputFile();
    result.addHeaderTokenFile(tokenFile);
  }

  private Collection<Artifact> createSourceAction(
      Label sourceLabel,
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
      Map<String, String> sourceSpecificBuildVariables)
      throws RuleErrorException {
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
      createFakeSourceAction(
          sourceLabel,
          outputName,
          result,
          env,
          builder,
          outputCategory,
          addObject,
          ccRelativeName,
          sourceArtifact.getExecPath(),
          usePic,
          generateDotd);
    } else {
      boolean bitcodeOutput =
          featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
              && CppFileTypes.LTO_SOURCE.matches(sourceArtifact.getFilename());

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
        Artifact ltoIndexingFile =
            bitcodeOutput ? getLtoIndexingFile(picBuilder.getOutputFile()) : null;

        setupCompileBuildVariables(
            picBuilder,
            sourceLabel,
            /* usePic= */ true,
            ccRelativeName,
            sourceArtifact.getExecPath(),
            gcnoFile,
            dwoFile,
            ltoIndexingFile,
            cppModuleMap,
            sourceSpecificBuildVariables);

        if (maySaveTemps) {
          result.addTemps(
              createTempsActions(
                  sourceArtifact,
                  sourceLabel,
                  outputName,
                  picBuilder,
                  /* usePic= */ true,
                  /* generateDotd= */ generateDotd,
                  ccRelativeName));
        }

        picBuilder.setGcnoFile(gcnoFile);
        picBuilder.setDwoFile(dwoFile);
        picBuilder.setLtoIndexingFile(ltoIndexingFile);

        semantics.finalizeCompileActionBuilder(
            ruleContext,
            picBuilder,
            featureConfiguration.getFeatureSpecification(),
            coptsFilter,
            features);
        CppCompileAction picAction = picBuilder.buildOrThrowRuleError(ruleContext);
        env.registerAction(picAction);
        directOutputs.add(picAction.getOutputFile());
        if (addObject) {
          result.addPicObjectFile(picAction.getOutputFile());

          if (bitcodeOutput) {
            result.addLtoBitcodeFile(picAction.getOutputFile(), ltoIndexingFile);
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
        Artifact ltoIndexingFile =
            bitcodeOutput ? getLtoIndexingFile(builder.getOutputFile()) : null;

        setupCompileBuildVariables(
            builder,
            sourceLabel,
            /* usePic= */ false,
            ccRelativeName,
            sourceArtifact.getExecPath(),
            gcnoFile,
            noPicDwoFile,
            ltoIndexingFile,
            cppModuleMap,
            sourceSpecificBuildVariables);

        if (maySaveTemps) {
          result.addTemps(
              createTempsActions(
                  sourceArtifact,
                  sourceLabel,
                  outputName,
                  builder,
                  /* usePic= */ false,
                  generateDotd,
                  ccRelativeName));
        }

        builder.setGcnoFile(gcnoFile);
        builder.setDwoFile(noPicDwoFile);
        builder.setLtoIndexingFile(ltoIndexingFile);

        semantics.finalizeCompileActionBuilder(
            ruleContext,
            builder,
            featureConfiguration.getFeatureSpecification(),
            coptsFilter,
            features);
        CppCompileAction compileAction = builder.buildOrThrowRuleError(ruleContext);
        env.registerAction(compileAction);
        Artifact objectFile = compileAction.getOutputFile();
        directOutputs.add(objectFile);
        if (addObject) {
          result.addObjectFile(objectFile);
          if (bitcodeOutput) {
            result.addLtoBitcodeFile(objectFile, ltoIndexingFile);
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
    builder.setOutputs(outputFiles, /* dotdFile= */ null);
    setupCompileBuildVariables(
        builder,
        source.getLabel(),
        /* usePic= */ false,
        /* ccRelativeName= */ null,
        /* autoFdoImportPath= */ null,
        /* gcnoFile= */ null,
        /* dwoFile= */ null,
        /* ltoIndexingFile= */ null,
        builder.getContext().getCppModuleMap(),
        source.getBuildVariables());
    semantics.finalizeCompileActionBuilder(
        ruleContext,
        builder,
        featureConfiguration.getFeatureSpecification(),
        coptsFilter,
        features);
    // Make sure this builder doesn't reference ruleContext outside of analysis phase.
    CppCompileActionTemplate actionTemplate =
        new CppCompileActionTemplate(
            sourceArtifact,
            outputFiles,
            builder,
            ccToolchain,
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

  private void createFakeSourceAction(
      Label sourceLabel,
      String outputName,
      CcCompilationOutputs.Builder result,
      AnalysisEnvironment env,
      CppCompileActionBuilder builder,
      ArtifactCategory outputCategory,
      boolean addObject,
      PathFragment ccRelativeName,
      PathFragment execPath,
      boolean usePic,
      boolean generateDotd)
      throws RuleErrorException {
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
        .setTempOutputFile(PathFragment.create(tempOutputName));

    setupCompileBuildVariables(
        builder,
        sourceLabel,
        usePic,
        ccRelativeName,
        execPath,
        /* gcnoFile= */ null,
        /* dwoFile= */ null,
        /* ltoIndexingFile= */ null,
        builder.getContext().getCppModuleMap(),
        /* sourceSpecificBuildVariables= */ ImmutableMap.of());
    semantics.finalizeCompileActionBuilder(
        ruleContext,
        builder,
        featureConfiguration.getFeatureSpecification(),
        coptsFilter,
        features);
    CppCompileAction action = builder.buildOrThrowRuleError(ruleContext);
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
    Artifact linuxDefault =
        CppHelper.getLinuxLinkedArtifact(
            ruleContext, configuration, linkTargetType, linkedArtifactNameSuffix);

    try {
      String maybePicName = ruleContext.getLabel().getName() + linkedArtifactNameSuffix;
      if (linkTargetType.picness() == Picness.PIC) {
        maybePicName = CppHelper.getArtifactNameForCategory(
            ruleContext, ccToolchain, ArtifactCategory.PIC_FILE, maybePicName);
      }
      String linkedName = CppHelper.getArtifactNameForCategory(
          ruleContext, ccToolchain, linkTargetType.getLinkerOutput(), maybePicName);
      PathFragment artifactFragment = PathFragment.create(ruleContext.getLabel().getName())
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
    boolean usePicForBinaries = CppHelper.usePic(ruleContext, /* forBinary= */ true);
    boolean usePicForSharedLibs = CppHelper.usePic(ruleContext, /* forBinary= */ false);

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
    PathFragment labelName = PathFragment.create(ruleContext.getLabel().getName());
    String libraryIdentifier = ruleContext.getPackageDirectory().getRelative(
        labelName.replaceName("lib" + labelName.getBaseName())).getPathString();
    CppLinkAction maybePicAction =
        newLinkActionBuilder(linkedArtifact)
            .addObjectFiles(ccOutputs.getObjectFiles(usePicForBinaries))
            .addNonCodeInputs(nonCodeLinkerInputs)
            .addLtoBitcodeFiles(ccOutputs.getLtoBitcodeFiles())
            .setLinkType(linkType)
            .setLinkStaticness(LinkStaticness.FULLY_STATIC)
            .addActionInputs(linkActionInputs)
            .setLibraryIdentifier(libraryIdentifier)
            .addVariablesExtensions(variablesExtensions)
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
              .addObjectFiles(ccOutputs.getObjectFiles(/* usePic= */ true))
              .addLtoBitcodeFiles(ccOutputs.getLtoBitcodeFiles())
              .setLinkType(picLinkType)
              .setLinkStaticness(LinkStaticness.FULLY_STATIC)
              .addActionInputs(linkActionInputs)
              .setLibraryIdentifier(libraryIdentifier)
              .addVariablesExtensions(variablesExtensions)
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
              ruleContext,
              configuration,
              LinkTargetType.INTERFACE_DYNAMIC_LIBRARY,
              linkedArtifactNameSuffix);
      // TODO(b/28946988): Remove this hard-coded flag.
      if (!featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)) {
        sonameLinkopts =
            ImmutableList.of(
                "-Wl,-soname="
                    + SolibSymlinkAction.getDynamicLibrarySoname(
                        soImpl.getRootRelativePath(), /* preserveName= */ false));
      }
    }

    CppLinkActionBuilder dynamicLinkActionBuilder =
        newLinkActionBuilder(soImpl)
            .setInterfaceOutput(soInterface)
            .addObjectFiles(ccOutputs.getObjectFiles(usePicForSharedLibs))
            .addNonCodeInputs(ccOutputs.getHeaderTokenFiles())
            .addLtoBitcodeFiles(ccOutputs.getLtoBitcodeFiles())
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
            .addVariablesExtensions(variablesExtensions);

    if (CppHelper.shouldUseDefFile(featureConfiguration)) {
      Artifact defFile =
          CppHelper.createDefFileActions(
              ruleContext,
              ccToolchain.getDefParserTool(),
              ccOutputs.getObjectFiles(false),
              SolibSymlinkAction.getDynamicLibrarySoname(soImpl.getRootRelativePath(), true));
      dynamicLinkActionBuilder.setDefFile(defFile);
    }

    // On Windows, we cannot build a shared library with symbols unresolved, so here we dynamically
    // link to all it's dependencies.
    if (featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)) {
      CcLinkParams.Builder ccLinkParamsBuilder =
          CcLinkParams.builder(/* linkingStatically= */ false, /* linkShared= */ true);
      ccLinkParamsBuilder.addCcLibrary(ruleContext);
      dynamicLinkActionBuilder.addLinkParams(ccLinkParamsBuilder.build(), ruleContext);
    }

    if (!ccOutputs.getLtoBitcodeFiles().isEmpty()
        && featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)) {
      dynamicLinkActionBuilder.setLtoIndexing(true);
      dynamicLinkActionBuilder.setUsePicForLtoBackendActions(usePicForSharedLibs);
      CppLinkAction indexAction = dynamicLinkActionBuilder.build();
      env.registerAction(indexAction);

      dynamicLinkActionBuilder.setLtoIndexing(false);
    }

    CppLinkAction dynamicLinkAction = dynamicLinkActionBuilder.build();
    env.registerAction(dynamicLinkAction);

    if (linkType == LinkTargetType.EXECUTABLE) {
      return result.build();
    }

    LibraryToLink dynamicLibrary = dynamicLinkAction.getOutputLibrary();
    LibraryToLink interfaceLibrary = dynamicLinkAction.getInterfaceOutputLibrary();
    if (interfaceLibrary == null) {
      interfaceLibrary = dynamicLibrary;
    }

    // If shared library has neverlink=1, then leave it untouched. Otherwise,
    // create a mangled symlink for it and from now on reference it through
    // mangled name only.
    //
    // When COPY_DYNAMIC_LIBRARIES_TO_BINARY is enabled, we don't need to create the special
    // solibDir, instead we use the original interface library and dynamic library.
    if (neverLink
        || featureConfiguration.isEnabled(CppRuleClasses.COPY_DYNAMIC_LIBRARIES_TO_BINARY)) {
      result.addDynamicLibrary(interfaceLibrary);
      result.addExecutionDynamicLibrary(dynamicLibrary);
    } else {
      Artifact libraryLink =
          SolibSymlinkAction.getDynamicLibrarySymlink(
              ruleContext,
              ccToolchain.getSolibDirectory(),
              interfaceLibrary.getArtifact(),
              /* preserveName= */ false,
              /* prefixConsumer= */ false,
              ruleContext.getConfiguration());
      result.addDynamicLibrary(LinkerInputs.solibLibraryToLink(
          libraryLink, interfaceLibrary.getArtifact(), libraryIdentifier));
      Artifact implLibraryLink =
          SolibSymlinkAction.getDynamicLibrarySymlink(
              ruleContext,
              ccToolchain.getSolibDirectory(),
              dynamicLibrary.getArtifact(),
              /* preserveName= */ false,
              /* prefixConsumer= */ false,
              ruleContext.getConfiguration());
      result.addExecutionDynamicLibrary(LinkerInputs.solibLibraryToLink(
          implLibraryLink, dynamicLibrary.getArtifact(), libraryIdentifier));
    }
    return result.build();
  }

  private CppLinkActionBuilder newLinkActionBuilder(Artifact outputArtifact) {
    return new CppLinkActionBuilder(
            ruleContext, outputArtifact, ccToolchain, fdoSupport, featureConfiguration, semantics)
        .setCrosstoolInputs(ccToolchain.getLink())
        .addNonCodeInputs(context.getTransitiveCompilationPrerequisites());
  }

  /**
   * Creates a basic cpp compile action builder for source file. Configures options, crosstool
   * inputs, output and dotd file names, compilation context and copts.
   */
  private CppCompileActionBuilder createCompileActionBuilder(Artifact source) {
    CppCompileActionBuilder builder =
        new CppCompileActionBuilder(ruleContext, ccToolchain, configuration);
    builder.setSourceFile(source);
    builder.setContext(context);
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

  /** Create the actions for "--save_temps". */
  private ImmutableList<Artifact> createTempsActions(
      Artifact source,
      Label sourceLabel,
      String outputName,
      CppCompileActionBuilder builder,
      boolean usePic,
      boolean generateDotd,
      PathFragment ccRelativeName)
      throws RuleErrorException {
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
        sourceLabel,
        usePic,
        ccRelativeName,
        source.getExecPath(),
        /* gcnoFile= */ null,
        /* dwoFile= */ null,
        /* ltoIndexingFile= */ null,
        builder.getContext().getCppModuleMap(),
        /* sourceSpecificBuildVariables= */ ImmutableMap.of());
    semantics.finalizeCompileActionBuilder(
        ruleContext,
        dBuilder,
        featureConfiguration.getFeatureSpecification(),
        coptsFilter,
        features);
    CppCompileAction dAction = dBuilder.buildOrThrowRuleError(ruleContext);
    ruleContext.registerAction(dAction);

    CppCompileActionBuilder sdBuilder = new CppCompileActionBuilder(builder);
    sdBuilder.setOutputs(
        ruleContext, ArtifactCategory.GENERATED_ASSEMBLY, outputArtifactNameBase, generateDotd);
    setupCompileBuildVariables(
        sdBuilder,
        sourceLabel,
        usePic,
        ccRelativeName,
        source.getExecPath(),
        /* gcnoFile= */ null,
        /* dwoFile= */ null,
        /* ltoIndexingFile= */ null,
        builder.getContext().getCppModuleMap(),
        /* sourceSpecificBuildVariables= */ ImmutableMap.of());
    semantics.finalizeCompileActionBuilder(
        ruleContext,
        sdBuilder,
        featureConfiguration.getFeatureSpecification(),
        coptsFilter,
        features);
    CppCompileAction sdAction = sdBuilder.buildOrThrowRuleError(ruleContext);
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
