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

import static java.util.stream.Collectors.toCollection;

import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.LanguageDependentFragment;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMapBuilder;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CoptsFilter;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.StringSequenceBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction.DotdFile;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * A class to create C/C++ compile actions in a way that is consistent with cc_library. Rules that
 * generate source files and emulate cc_library on top of that should use this class instead of the
 * lower-level APIs in CppHelper and CppCompileActionBuilder.
 *
 * <p>Rules that want to use this class are required to have implicit dependencies on the toolchain,
 * the STL, the lipo context, and so on. Optionally, they can also have copts, and malloc
 * attributes, but note that these require explicit calls to the corresponding setter methods.
 */
public final class CcCompilationHelper {
  /** Similar to {@code OutputGroupInfo.HIDDEN_TOP_LEVEL}, but specific to header token files. */
  public static final String HIDDEN_HEADER_TOKENS =
      OutputGroupInfo.HIDDEN_OUTPUT_GROUP_PREFIX
          + "hidden_header_tokens"
          + OutputGroupInfo.INTERNAL_SUFFIX;

  private static final String PIC_CONFIGURATION_ERROR =
      "PIC compilation is requested but the toolchain does not support it";
  /** Name of the build variable for the path to the source file being compiled. */
  public static final String SOURCE_FILE_VARIABLE_NAME = "source_file";

  /** Name of the build variable for the path to the compilation output file. */
  public static final String OUTPUT_FILE_VARIABLE_NAME = "output_file";

  /**
   * Build variable for all flags coming from copt rule attribute, and from --copt, --cxxopt, or
   * --conlyopt options.
   */
  public static final String USER_COMPILE_FLAGS_VARIABLE_NAME = "user_compile_flags";

  /**
   * Build variable for all flags coming from legacy crosstool fields, such as compiler_flag,
   * optional_compiler_flag, cxx_flag, optional_cxx_flag.
   */
  public static final String LEGACY_COMPILE_FLAGS_VARIABLE_NAME = "legacy_compile_flags";

  /** Build variable for flags coming from unfiltered_cxx_flag CROSSTOOL fields. */
  public static final String UNFILTERED_COMPILE_FLAGS_VARIABLE_NAME = "unfiltered_compile_flags";

  /**
   * Name of the build variable for the path to the compilation output file in case of preprocessed
   * source.
   */
  public static final String OUTPUT_PREPROCESS_FILE_VARIABLE_NAME = "output_preprocess_file";

  /** Name of the build variable for the path to the output file when output is an object file. */
  public static final String OUTPUT_OBJECT_FILE_VARIABLE_NAME = "output_object_file";

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

  /** Name of the build variable for the collection of macros defined for preprocessor. */
  public static final String PREPROCESSOR_DEFINES_VARIABLE_NAME = "preprocessor_defines";

  /** Name of the build variable present when the output is compiled as position independent. */
  public static final String PIC_VARIABLE_NAME = "pic";

  /**
   * Name of the build variable for the path to the compilation output file in case of assembly
   * source.
   */
  private static final String OUTPUT_ASSEMBLY_FILE_VARIABLE_NAME = "output_assembly_file";

  /** Name of the build variable for the dependency file path */
  private static final String DEPENDENCY_FILE_VARIABLE_NAME = "dependency_file";

  /** Name of the build variable for the module file name. */
  private static final String MODULE_NAME_VARIABLE_NAME = "module_name";

  /** Name of the build variable for the module map file name. */
  private static final String MODULE_MAP_FILE_VARIABLE_NAME = "module_map_file";

  /** Name of the build variable for the dependent module map file name. */
  private static final String DEPENDENT_MODULE_MAP_FILES_VARIABLE_NAME =
      "dependent_module_map_files";

  /** Name of the build variable for the collection of module files. */
  private static final String MODULE_FILES_VARIABLE_NAME = "module_files";

  /** Name of the build variable for the gcov coverage file path. */
  private static final String GCOV_GCNO_FILE_VARIABLE_NAME = "gcov_gcno_file";

  /** Name of the build variable for the per object debug info file. */
  private static final String PER_OBJECT_DEBUG_INFO_FILE_VARIABLE_NAME =
      "per_object_debug_info_file";

  /** Name of the build variable for the LTO indexing bitcode file. */
  private static final String LTO_INDEXING_BITCODE_FILE_VARIABLE_NAME = "lto_indexing_bitcode_file";

  /**
   * A group of source file types and action names for builds controlled by CcCompilationHelper.
   * Determines what file types CcCompilationHelper considers sources and what action configs are
   * configured in the CROSSTOOL.
   */
  public enum SourceCategory {
    CC(
        FileTypeSet.of(
            CppFileTypes.CPP_SOURCE,
            CppFileTypes.CPP_HEADER,
            CppFileTypes.C_SOURCE,
            CppFileTypes.ASSEMBLER,
            CppFileTypes.ASSEMBLER_WITH_C_PREPROCESSOR,
            CppFileTypes.CLIF_INPUT_PROTO)),
    CC_AND_OBJC(
        FileTypeSet.of(
            CppFileTypes.CPP_SOURCE,
            CppFileTypes.CPP_HEADER,
            CppFileTypes.OBJC_SOURCE,
            CppFileTypes.OBJCPP_SOURCE,
            CppFileTypes.C_SOURCE,
            CppFileTypes.ASSEMBLER,
            CppFileTypes.ASSEMBLER_WITH_C_PREPROCESSOR));

    private final FileTypeSet sourceTypeSet;

    private SourceCategory(FileTypeSet sourceTypeSet) {
      this.sourceTypeSet = sourceTypeSet;
    }

    /** Returns the set of file types that are valid for this category. */
    public FileTypeSet getSourceTypes() {
      return sourceTypeSet;
    }
  }

  /** Function for extracting module maps from CppCompilationDependencies. */
  private static final Function<TransitiveInfoCollection, CppModuleMap> CPP_DEPS_TO_MODULES =
      dep -> {
        CppCompilationContext context = dep.getProvider(CppCompilationContext.class);
        return context == null ? null : context.getCppModuleMap();
      };

  /** Contains the providers as well as the compilation outputs, and the compilation context. */
  public static final class CompilationInfo {
    private final TransitiveInfoProviderMap providers;
    private final Map<String, NestedSet<Artifact>> outputGroups;
    private final CcCompilationOutputs compilationOutputs;
    private final CppCompilationContext context;

    private CompilationInfo(
        TransitiveInfoProviderMap providers,
        Map<String, NestedSet<Artifact>> outputGroups,
        CcCompilationOutputs compilationOutputs,
        CppCompilationContext context) {
      this.providers = providers;
      this.outputGroups = outputGroups;
      this.compilationOutputs = compilationOutputs;
      this.context = context;
    }

    public TransitiveInfoProviderMap getProviders() {
      return providers;
    }

    public Map<String, NestedSet<Artifact>> getOutputGroups() {
      return outputGroups;
    }

    public CcCompilationOutputs getCcCompilationOutputs() {
      return compilationOutputs;
    }

    public CppCompilationContext getCppCompilationContext() {
      return context;
    }
  }

  private final RuleContext ruleContext;
  private final CppSemantics semantics;
  private final BuildConfiguration configuration;
  private final CppConfiguration cppConfiguration;

  private final List<Artifact> publicHeaders = new ArrayList<>();
  private final List<Artifact> nonModuleMapHeaders = new ArrayList<>();
  private final List<Artifact> publicTextualHeaders = new ArrayList<>();
  private final List<Artifact> privateHeaders = new ArrayList<>();
  private final List<Artifact> additionalInputs = new ArrayList<>();
  private final List<Artifact> compilationMandatoryInputs = new ArrayList<>();
  private final List<Artifact> additionalIncludeScanningRoots = new ArrayList<>();
  private final List<PathFragment> additionalExportedHeaders = new ArrayList<>();
  private final List<CppModuleMap> additionalCppModuleMaps = new ArrayList<>();
  private final Set<CppSource> compilationUnitSources = new LinkedHashSet<>();
  private final List<Artifact> objectFiles = new ArrayList<>();
  private final List<Artifact> picObjectFiles = new ArrayList<>();
  private ImmutableList<String> copts = ImmutableList.of();
  private CoptsFilter coptsFilter = CoptsFilter.alwaysPasses();
  private final Set<String> defines = new LinkedHashSet<>();
  private final List<TransitiveInfoCollection> deps = new ArrayList<>();
  private final List<CppCompilationContext> depContexts = new ArrayList<>();
  private final List<PathFragment> looseIncludeDirs = new ArrayList<>();
  private final List<PathFragment> systemIncludeDirs = new ArrayList<>();
  private final List<PathFragment> includeDirs = new ArrayList<>();

  private HeadersCheckingMode headersCheckingMode = HeadersCheckingMode.LOOSE;
  private boolean fake;

  private boolean checkDepsGenerateCpp = true;
  private boolean emitCompileProviders;
  private final SourceCategory sourceCategory;
  private final List<VariablesExtension> variablesExtensions = new ArrayList<>();
  @Nullable private CppModuleMap cppModuleMap;
  private boolean propagateModuleMapToCompileAction = true;

  private final FeatureConfiguration featureConfiguration;
  private final CcToolchainProvider ccToolchain;
  private final FdoSupportProvider fdoSupport;
  private final ImmutableSet<String> features;
  private boolean useDeps = true;
  private boolean generateModuleMap = true;
  private String purpose = null;
  private boolean generateNoPic = true;

  // TODO(plf): Pull out of class.
  private CppCompilationContext cppCompilationContext;

  /**
   * Creates a CcCompilationHelper.
   *
   * @param ruleContext the RuleContext for the rule being built
   * @param semantics CppSemantics for the build
   * @param featureConfiguration activated features and action configs for the build
   * @param sourceCatagory the candidate source types for the build
   * @param ccToolchain the C++ toolchain provider for the build
   * @param fdoSupport the C++ FDO optimization support provider for the build
   */
  public CcCompilationHelper(
      RuleContext ruleContext,
      CppSemantics semantics,
      FeatureConfiguration featureConfiguration,
      SourceCategory sourceCatagory,
      CcToolchainProvider ccToolchain,
      FdoSupportProvider fdoSupport) {
    this(
        ruleContext,
        semantics,
        featureConfiguration,
        sourceCatagory,
        ccToolchain,
        fdoSupport,
        ruleContext.getConfiguration());
  }

  /**
   * Creates a CcCompilationHelper that outputs artifacts in a given configuration.
   *
   * @param ruleContext the RuleContext for the rule being built
   * @param semantics CppSemantics for the build
   * @param featureConfiguration activated features and action configs for the build
   * @param sourceCatagory the candidate source types for the build
   * @param ccToolchain the C++ toolchain provider for the build
   * @param fdoSupport the C++ FDO optimization support provider for the build
   * @param configuration the configuration that gives the directory of output artifacts
   */
  public CcCompilationHelper(
      RuleContext ruleContext,
      CppSemantics semantics,
      FeatureConfiguration featureConfiguration,
      SourceCategory sourceCatagory,
      CcToolchainProvider ccToolchain,
      FdoSupportProvider fdoSupport,
      BuildConfiguration configuration) {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);
    this.semantics = Preconditions.checkNotNull(semantics);
    this.featureConfiguration = Preconditions.checkNotNull(featureConfiguration);
    this.sourceCategory = Preconditions.checkNotNull(sourceCatagory);
    this.ccToolchain = Preconditions.checkNotNull(ccToolchain);
    this.fdoSupport = Preconditions.checkNotNull(fdoSupport);
    this.configuration = Preconditions.checkNotNull(configuration);
    this.cppConfiguration =
        Preconditions.checkNotNull(ruleContext.getFragment(CppConfiguration.class));
    this.features = ruleContext.getFeatures();
  }

  /**
   * Creates a CcCompilationHelper for cpp source files.
   *
   * @param ruleContext the RuleContext for the rule being built
   * @param semantics CppSemantics for the build
   * @param featureConfiguration activated features and action configs for the build
   * @param ccToolchain the C++ toolchain provider for the build
   * @param fdoSupport the C++ FDO optimization support provider for the build
   */
  public CcCompilationHelper(
      RuleContext ruleContext,
      CppSemantics semantics,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider ccToolchain,
      FdoSupportProvider fdoSupport) {
    this(ruleContext, semantics, featureConfiguration, SourceCategory.CC, ccToolchain, fdoSupport);
  }

  /** Sets fields that overlap for cc_library and cc_binary rules. */
  public CcCompilationHelper fromCommon(CcCommon common) {
    setCopts(common.getCopts());
    addDefines(common.getDefines());
    addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET));
    addLooseIncludeDirs(common.getLooseIncludeDirs());
    addSystemIncludeDirs(common.getSystemIncludeDirs());
    setCoptsFilter(common.getCoptsFilter());
    setHeadersCheckingMode(semantics.determineHeadersCheckingMode(ruleContext));
    return this;
  }

  /**
   * Adds {@code headers} as public header files. These files will be made visible to dependent
   * rules. They may be parsed/preprocessed or compiled into a header module depending on the
   * configuration.
   */
  public CcCompilationHelper addPublicHeaders(Collection<Artifact> headers) {
    for (Artifact header : headers) {
      addHeader(header, ruleContext.getLabel());
    }
    return this;
  }

  /**
   * Adds {@code headers} as public header files. These files will be made visible to dependent
   * rules. They may be parsed/preprocessed or compiled into a header module depending on the
   * configuration.
   */
  public CcCompilationHelper addPublicHeaders(Artifact... headers) {
    addPublicHeaders(Arrays.asList(headers));
    return this;
  }

  /**
   * Adds {@code headers} as public header files. These files will be made visible to dependent
   * rules. They may be parsed/preprocessed or compiled into a header module depending on the
   * configuration.
   */
  public CcCompilationHelper addPublicHeaders(Iterable<Pair<Artifact, Label>> headers) {
    for (Pair<Artifact, Label> header : headers) {
      addHeader(header.first, header.second);
    }
    return this;
  }

  /**
   * Add the corresponding files as public header files, i.e., these files will not be compiled, but
   * are made visible as includes to dependent rules in module maps.
   */
  public CcCompilationHelper addAdditionalExportedHeaders(
      Iterable<PathFragment> additionalExportedHeaders) {
    Iterables.addAll(this.additionalExportedHeaders, additionalExportedHeaders);
    return this;
  }

  /**
   * Add the corresponding files as public textual header files. These files will not be compiled
   * into a target's header module, but will be made visible as textual includes to dependent rules.
   */
  public CcCompilationHelper addPublicTextualHeaders(Iterable<Artifact> textualHeaders) {
    Iterables.addAll(this.publicTextualHeaders, textualHeaders);
    for (Artifact header : textualHeaders) {
      this.additionalExportedHeaders.add(header.getExecPath());
    }
    return this;
  }

  /**
   * Add the corresponding files as source files. These may also be header files, in which case they
   * will not be compiled, but also not made visible as includes to dependent rules. The given build
   * variables will be added to those used for compiling this source.
   */
  public CcCompilationHelper addSources(Collection<Artifact> sources) {
    for (Artifact source : sources) {
      addSource(source, ruleContext.getLabel());
    }
    return this;
  }

  /**
   * Add the corresponding files as source files. These may also be header files, in which case they
   * will not be compiled, but also not made visible as includes to dependent rules.
   */
  public CcCompilationHelper addSources(Iterable<Pair<Artifact, Label>> sources) {
    for (Pair<Artifact, Label> source : sources) {
      addSource(source.first, source.second);
    }
    return this;
  }

  /**
   * Add the corresponding files as source files. These may also be header files, in which case they
   * will not be compiled, but also not made visible as includes to dependent rules.
   */
  public CcCompilationHelper addSources(Artifact... sources) {
    return addSources(Arrays.asList(sources));
  }

  /** Add the corresponding files as non-header, non-source input files. */
  public CcCompilationHelper addAdditionalInputs(Collection<Artifact> inputs) {
    Iterables.addAll(additionalInputs, inputs);
    return this;
  }

  /**
   * Adds a header to {@code publicHeaders} and in case header processing is switched on for the
   * file type also to compilationUnitSources.
   */
  private void addHeader(Artifact header, Label label) {
    // We assume TreeArtifacts passed in are directories containing proper headers.
    boolean isHeader =
        CppFileTypes.CPP_HEADER.matches(header.getExecPath()) || header.isTreeArtifact();
    boolean isTextualInclude = CppFileTypes.CPP_TEXTUAL_INCLUDE.matches(header.getExecPath());
    publicHeaders.add(header);
    if (isTextualInclude || !isHeader || !shouldProcessHeaders()) {
      return;
    }
    compilationUnitSources.add(CppSource.create(header, label, CppSource.Type.HEADER));
  }

  /** Adds a header to {@code publicHeaders}, but not to this target's module map. */
  public CcCompilationHelper addNonModuleMapHeader(Artifact header) {
    Preconditions.checkNotNull(header);
    nonModuleMapHeaders.add(header);
    return this;
  }

  /**
   * Adds a source to {@code compilationUnitSources} if it is a compiled file type (including
   * parsed/preprocessed header) and to {@code privateHeaders} if it is a header.
   */
  private void addSource(Artifact source, Label label) {
    Preconditions.checkNotNull(featureConfiguration);
    boolean isHeader = CppFileTypes.CPP_HEADER.matches(source.getExecPath());
    boolean isTextualInclude = CppFileTypes.CPP_TEXTUAL_INCLUDE.matches(source.getExecPath());
    // We assume TreeArtifacts passed in are directories containing proper sources for compilation.
    boolean isCompiledSource =
        sourceCategory.getSourceTypes().matches(source.getExecPathString())
            || source.isTreeArtifact();
    if (isHeader || isTextualInclude) {
      privateHeaders.add(source);
    }
    if (isTextualInclude || !isCompiledSource || (isHeader && !shouldProcessHeaders())) {
      return;
    }
    boolean isClifInputProto = CppFileTypes.CLIF_INPUT_PROTO.matches(source.getExecPathString());
    CppSource.Type type;
    if (isHeader) {
      type = CppSource.Type.HEADER;
    } else if (isClifInputProto) {
      type = CppSource.Type.CLIF_INPUT_PROTO;
    } else {
      type = CppSource.Type.SOURCE;
    }
    compilationUnitSources.add(CppSource.create(source, label, type));
  }

  private boolean shouldProcessHeaders() {
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    // If parse_headers_verifies_modules is switched on, we verify that headers are
    // self-contained by building the module instead.
    return !cppConfiguration.getParseHeadersVerifiesModules()
        && (featureConfiguration.isEnabled(CppRuleClasses.PREPROCESS_HEADERS)
            || featureConfiguration.isEnabled(CppRuleClasses.PARSE_HEADERS));
  }

  /**
   * Returns the compilation unit sources. That includes all compiled source files as well as
   * headers that will be parsed or preprocessed. Each source file contains the label it arises from
   * in the build graph as well as {@code FeatureConfiguration} that should be used during its
   * compilation.
   */
  public ImmutableSet<CppSource> getCompilationUnitSources() {
    return ImmutableSet.copyOf(this.compilationUnitSources);
  }

  /**
   * Add the corresponding files as linker inputs for non-PIC links. If the corresponding files are
   * compiled with PIC, the final link may or may not fail. Note that the final link may not happen
   * here, if {@code --start_end_lib} is enabled, but instead at any binary that transitively
   * depends on the current rule.
   */
  public CcCompilationHelper addObjectFiles(Iterable<Artifact> objectFiles) {
    for (Artifact objectFile : objectFiles) {
      Preconditions.checkArgument(Link.OBJECT_FILETYPES.matches(objectFile.getFilename()));
    }
    Iterables.addAll(this.objectFiles, objectFiles);
    return this;
  }

  /**
   * Add the corresponding files as linker inputs for PIC links. If the corresponding files are not
   * compiled with PIC, the final link may or may not fail. Note that the final link may not happen
   * here, if {@code --start_end_lib} is enabled, but instead at any binary that transitively
   * depends on the current rule.
   */
  public CcCompilationHelper addPicObjectFiles(Iterable<Artifact> picObjectFiles) {
    for (Artifact objectFile : objectFiles) {
      Preconditions.checkArgument(Link.OBJECT_FILETYPES.matches(objectFile.getFilename()));
    }
    Iterables.addAll(this.picObjectFiles, picObjectFiles);
    return this;
  }

  public CcCompilationHelper setCopts(ImmutableList<String> copts) {
    this.copts = Preconditions.checkNotNull(copts);
    return this;
  }

  /** Sets a pattern that is used to filter copts; set to {@code null} for no filtering. */
  private void setCoptsFilter(CoptsFilter coptsFilter) {
    this.coptsFilter = Preconditions.checkNotNull(coptsFilter);
  }

  /** Adds the given defines to the compiler command line. */
  public CcCompilationHelper addDefines(Iterable<String> defines) {
    Iterables.addAll(this.defines, defines);
    return this;
  }

  /**
   * Adds the given targets as dependencies - this can include explicit dependencies on other rules
   * (like from a "deps" attribute) and also implicit dependencies on runtime libraries.
   */
  public CcCompilationHelper addDeps(Iterable<? extends TransitiveInfoCollection> deps) {
    for (TransitiveInfoCollection dep : deps) {
      this.deps.add(dep);
    }
    return this;
  }

  public CcCompilationHelper addDepContext(CppCompilationContext dep) {
    this.depContexts.add(Preconditions.checkNotNull(dep));
    return this;
  }

  /**
   * Adds the given precompiled files to this helper. Shared and static libraries are added as
   * compilation prerequisites, and object files are added as pic or non-pic object files
   * respectively.
   */
  public CcCompilationHelper addPrecompiledFiles(PrecompiledFiles precompiledFiles) {
    addObjectFiles(precompiledFiles.getObjectFiles(false));
    addPicObjectFiles(precompiledFiles.getObjectFiles(true));
    return this;
  }

  /**
   * Adds the given directories to the loose include directories that are only allowed to be
   * referenced when headers checking is {@link HeadersCheckingMode#LOOSE} or {@link
   * HeadersCheckingMode#WARN}.
   */
  private void addLooseIncludeDirs(Iterable<PathFragment> looseIncludeDirs) {
    Iterables.addAll(this.looseIncludeDirs, looseIncludeDirs);
  }

  /**
   * Adds the given directories to the system include directories (they are passed with {@code
   * "-isystem"} to the compiler); these are also passed to dependent rules.
   */
  public CcCompilationHelper addSystemIncludeDirs(Iterable<PathFragment> systemIncludeDirs) {
    Iterables.addAll(this.systemIncludeDirs, systemIncludeDirs);
    return this;
  }

  /**
   * Adds the given directories to the include directories (they are passed with {@code "-I"} to the
   * compiler); these are also passed to dependent rules.
   */
  public CcCompilationHelper addIncludeDirs(Iterable<PathFragment> includeDirs) {
    Iterables.addAll(this.includeDirs, includeDirs);
    return this;
  }

  /** Adds a variableExtension to template the crosstool. */
  public CcCompilationHelper addVariableExtension(VariablesExtension variableExtension) {
    Preconditions.checkNotNull(variableExtension);
    this.variablesExtensions.add(variableExtension);
    return this;
  }

  /** Sets a module map artifact for this build. */
  public CcCompilationHelper setCppModuleMap(CppModuleMap cppModuleMap) {
    Preconditions.checkNotNull(cppModuleMap);
    this.cppModuleMap = cppModuleMap;
    return this;
  }

  /** Signals that this target's module map should not be an input to c++ compile actions. */
  public CcCompilationHelper setPropagateModuleMapToCompileAction(boolean propagatesModuleMap) {
    this.propagateModuleMapToCompileAction = propagatesModuleMap;
    return this;
  }

  /** Sets the given headers checking mode. The default is {@link HeadersCheckingMode#LOOSE}. */
  public CcCompilationHelper setHeadersCheckingMode(HeadersCheckingMode headersCheckingMode) {
    this.headersCheckingMode = Preconditions.checkNotNull(headersCheckingMode);
    return this;
  }

  /**
   * Marks the resulting code as fake, i.e., the code will not actually be compiled or linked, but
   * instead, the compile command is written to a file and added to the runfiles. This is currently
   * used for non-compilation tests. Unfortunately, the design is problematic, so please don't add
   * any further uses.
   */
  public CcCompilationHelper setFake(boolean fake) {
    this.fake = fake;
    return this;
  }

  /**
   * Disables checking that the deps actually are C++ rules. By default, the {@link #compile} method
   * uses {@link LanguageDependentFragment.Checker#depSupportsLanguage} to check that all deps
   * provide C++ providers.
   */
  public CcCompilationHelper setCheckDepsGenerateCpp(boolean checkDepsGenerateCpp) {
    this.checkDepsGenerateCpp = checkDepsGenerateCpp;
    return this;
  }

  /**
   * Enables the output of the {@code files_to_compile} and {@code compilation_prerequisites} output
   * groups.
   */
  // TODO(bazel-team): We probably need to adjust this for the multi-language rules.
  public CcCompilationHelper enableCompileProviders() {
    this.emitCompileProviders = true;
    return this;
  }

  /**
   * Causes actions generated from this CcCompilationHelper not to use build semantics (includes,
   * headers, srcs) from dependencies.
   */
  public CcCompilationHelper doNotUseDeps() {
    this.useDeps = false;
    return this;
  }

  /** non-PIC actions won't be generated. */
  public CcCompilationHelper setGenerateNoPic(boolean generateNoPic) {
    this.generateNoPic = generateNoPic;
    return this;
  }

  /** Adds mandatory inputs for the compilation action. */
  public CcCompilationHelper addCompilationMandatoryInputs(
      Collection<Artifact> compilationMandatoryInputs) {
    this.compilationMandatoryInputs.addAll(compilationMandatoryInputs);
    return this;
  }

  /** Adds additional includes to be scanned. */
  // TODO(plf): This is only needed for CLIF. Investigate whether this is strictly necessary or
  // there is a way to avoid include scanning for CLIF rules.
  public CcCompilationHelper addAditionalIncludeScanningRoots(
      Collection<Artifact> additionalIncludeScanningRoots) {
    this.additionalIncludeScanningRoots.addAll(additionalIncludeScanningRoots);
    return this;
  }

  /**
   * Create the C++ compile actions, and the corresponding compilation related providers.
   *
   * @throws RuleErrorException
   */
  public CompilationInfo compile() throws RuleErrorException {
    if (checkDepsGenerateCpp) {
      for (LanguageDependentFragment dep :
          AnalysisUtils.getProviders(deps, LanguageDependentFragment.class)) {
        LanguageDependentFragment.Checker.depSupportsLanguage(
            ruleContext, dep, CppRuleClasses.LANGUAGE, "deps");
      }
    }

    cppCompilationContext = initializeCppCompilationContext();

    boolean compileHeaderModules = featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULES);
    Preconditions.checkState(
        !compileHeaderModules || cppCompilationContext.getCppModuleMap() != null,
        "All cc rules must support module maps.");

    // Create compile actions (both PIC and non-PIC).
    CcCompilationOutputs ccOutputs = createCcCompileActions();
    if (!objectFiles.isEmpty() || !picObjectFiles.isEmpty()) {
      // Merge the pre-compiled object files into the compiler outputs.
      ccOutputs =
          new CcCompilationOutputs.Builder()
              .merge(ccOutputs)
              .addLtoBitcodeFile(ccOutputs.getLtoBitcodeFiles())
              .addObjectFiles(objectFiles)
              .addPicObjectFiles(picObjectFiles)
              .build();
    }

    DwoArtifactsCollector dwoArtifacts =
        DwoArtifactsCollector.transitiveCollector(
            ccOutputs,
            deps,
            /*generateDwo=*/ false,
            /*ltoBackendArtifactsUsePic=*/ false,
            /*ltoBackendArtifacts=*/ ImmutableList.of());

    // Be very careful when adding new providers here - it can potentially affect a lot of rules.
    // We should consider merging most of these providers into a single provider.
    TransitiveInfoProviderMapBuilder providers =
        new TransitiveInfoProviderMapBuilder()
            .add(
                cppCompilationContext,
                new CppDebugFileProvider(
                    dwoArtifacts.getDwoArtifacts(), dwoArtifacts.getPicDwoArtifacts()),
                collectTransitiveLipoInfo(ccOutputs));

    Map<String, NestedSet<Artifact>> outputGroups = new TreeMap<>();
    outputGroups.put(OutputGroupInfo.TEMP_FILES, getTemps(ccOutputs));
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    if (emitCompileProviders) {
      boolean isLipoCollector = cppConfiguration.isLipoContextCollector();
      boolean processHeadersInDependencies = cppConfiguration.processHeadersInDependencies();
      boolean usePic = CppHelper.usePic(ruleContext, ccToolchain, false);
      outputGroups.put(
          OutputGroupInfo.FILES_TO_COMPILE,
          ccOutputs.getFilesToCompile(isLipoCollector, processHeadersInDependencies, usePic));
      outputGroups.put(
          OutputGroupInfo.COMPILATION_PREREQUISITES,
          CcCommon.collectCompilationPrerequisites(ruleContext, cppCompilationContext));
    }

    return new CompilationInfo(providers.build(), outputGroups, ccOutputs, cppCompilationContext);
  }

  @Immutable
  private static class PublicHeaders {
    private final ImmutableList<Artifact> headers;
    private final ImmutableList<Artifact> moduleMapHeaders;
    private final @Nullable PathFragment virtualIncludePath;

    private PublicHeaders(
        ImmutableList<Artifact> headers,
        ImmutableList<Artifact> moduleMapHeaders,
        PathFragment virtualIncludePath) {
      this.headers = headers;
      this.moduleMapHeaders = moduleMapHeaders;
      this.virtualIncludePath = virtualIncludePath;
    }

    private ImmutableList<Artifact> getHeaders() {
      return headers;
    }

    private ImmutableList<Artifact> getModuleMapHeaders() {
      return moduleMapHeaders;
    }

    @Nullable
    private PathFragment getVirtualIncludePath() {
      return virtualIncludePath;
    }
  }

  private PublicHeaders computePublicHeaders() {
    if (!ruleContext.attributes().has("strip_include_prefix", Type.STRING)
        || !ruleContext.attributes().has("include_prefix", Type.STRING)) {
      return new PublicHeaders(
          ImmutableList.copyOf(Iterables.concat(publicHeaders, nonModuleMapHeaders)),
          ImmutableList.copyOf(publicHeaders),
          null);
    }

    PathFragment prefix = null;
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("include_prefix")) {
      String prefixAttr = ruleContext.attributes().get("include_prefix", Type.STRING);
      prefix = PathFragment.create(prefixAttr);
      if (PathFragment.containsUplevelReferences(prefixAttr)) {
        ruleContext.attributeError("include_prefix", "should not contain uplevel references");
      }
      if (prefix.isAbsolute()) {
        ruleContext.attributeError("include_prefix", "should be a relative path");
      }
    }

    PathFragment stripPrefix;
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("strip_include_prefix")) {
      String stripPrefixAttr = ruleContext.attributes().get("strip_include_prefix", Type.STRING);
      if (PathFragment.containsUplevelReferences(stripPrefixAttr)) {
        ruleContext.attributeError("strip_include_prefix", "should not contain uplevel references");
      }
      stripPrefix = PathFragment.create(stripPrefixAttr);
      if (stripPrefix.isAbsolute()) {
        stripPrefix =
            ruleContext
                .getLabel()
                .getPackageIdentifier()
                .getRepository()
                .getSourceRoot()
                .getRelative(stripPrefix.toRelative());
      } else {
        stripPrefix = ruleContext.getPackageDirectory().getRelative(stripPrefix);
      }
    } else if (prefix != null) {
      stripPrefix = ruleContext.getPackageDirectory();
    } else {
      stripPrefix = null;
    }

    if (stripPrefix == null && prefix == null) {
      // Simple case, no magic needed
      return new PublicHeaders(
          ImmutableList.copyOf(Iterables.concat(publicHeaders, nonModuleMapHeaders)),
          ImmutableList.copyOf(publicHeaders),
          null);
    }

    if (ruleContext.hasErrors()) {
      return new PublicHeaders(ImmutableList.<Artifact>of(), ImmutableList.<Artifact>of(), null);
    }

    ImmutableList.Builder<Artifact> moduleHeadersBuilder = ImmutableList.builder();

    for (Artifact originalHeader : publicHeaders) {
      if (!originalHeader.getRootRelativePath().startsWith(stripPrefix)) {
        ruleContext.ruleError(
            String.format(
                "header '%s' is not under the specified strip prefix '%s'",
                originalHeader.getExecPathString(), stripPrefix.getPathString()));
        continue;
      }

      PathFragment includePath = originalHeader.getRootRelativePath().relativeTo(stripPrefix);
      if (prefix != null) {
        includePath = prefix.getRelative(includePath);
      }

      if (!originalHeader.getExecPath().equals(includePath)) {
        Artifact virtualHeader =
            ruleContext.getUniqueDirectoryArtifact(
                "_virtual_includes", includePath, ruleContext.getBinOrGenfilesDirectory());
        ruleContext.registerAction(
            new SymlinkAction(
                ruleContext.getActionOwner(),
                originalHeader,
                virtualHeader,
                "Symlinking virtual headers for " + ruleContext.getLabel()));
        moduleHeadersBuilder.add(virtualHeader);
      } else {
        moduleHeadersBuilder.add(originalHeader);
      }
    }

    ImmutableList<Artifact> moduleMapHeaders = moduleHeadersBuilder.build();
    ImmutableList<Artifact> virtualHeaders =
        ImmutableList.<Artifact>builder()
            .addAll(moduleMapHeaders)
            .addAll(nonModuleMapHeaders)
            .build();

    return new PublicHeaders(
        virtualHeaders,
        moduleMapHeaders,
        ruleContext
            .getBinOrGenfilesDirectory()
            .getExecPath()
            .getRelative(ruleContext.getUniqueDirectory("_virtual_includes")));
  }

  /**
   * Create context for cc compile action from generated inputs.
   *
   * <p>TODO(plf): Try to pull out CppCompilationContext building out of this class.
   */
  public CppCompilationContext initializeCppCompilationContext() {
    CppCompilationContext.Builder contextBuilder = new CppCompilationContext.Builder(ruleContext);

    // Setup the include path; local include directories come before those inherited from deps or
    // from the toolchain; in case of aliasing (same include file found on different entries),
    // prefer the local include rather than the inherited one.

    // Add in the roots for well-formed include names for source files and
    // generated files. It is important that the execRoot (EMPTY_FRAGMENT) comes
    // before the genfilesFragment to preferably pick up source files. Otherwise
    // we might pick up stale generated files.
    PathFragment repositoryPath =
        ruleContext.getLabel().getPackageIdentifier().getRepository().getPathUnderExecRoot();
    contextBuilder.addQuoteIncludeDir(repositoryPath);
    contextBuilder.addQuoteIncludeDir(
        ruleContext.getConfiguration().getGenfilesFragment().getRelative(repositoryPath));

    for (PathFragment systemIncludeDir : systemIncludeDirs) {
      contextBuilder.addSystemIncludeDir(systemIncludeDir);
    }
    for (PathFragment includeDir : includeDirs) {
      contextBuilder.addIncludeDir(includeDir);
    }

    PublicHeaders publicHeaders = computePublicHeaders();
    if (publicHeaders.getVirtualIncludePath() != null) {
      contextBuilder.addIncludeDir(publicHeaders.getVirtualIncludePath());
    }

    if (useDeps) {
      contextBuilder.mergeDependentContexts(
          AnalysisUtils.getProviders(deps, CppCompilationContext.class));
      contextBuilder.mergeDependentContexts(depContexts);
    }
    CppHelper.mergeToolchainDependentContext(ruleContext, ccToolchain, contextBuilder);

    // But defines come after those inherited from deps.
    contextBuilder.addDefines(defines);

    // There are no ordering constraints for declared include dirs/srcs, or the pregrepped headers.
    contextBuilder.addDeclaredIncludeSrcs(publicHeaders.getHeaders());
    contextBuilder.addDeclaredIncludeSrcs(publicTextualHeaders);
    contextBuilder.addDeclaredIncludeSrcs(privateHeaders);
    contextBuilder.addDeclaredIncludeSrcs(additionalInputs);
    contextBuilder.addNonCodeInputs(additionalInputs);
    contextBuilder.addModularHdrs(publicHeaders.getHeaders());
    contextBuilder.addModularHdrs(privateHeaders);
    contextBuilder.addTextualHdrs(publicTextualHeaders);
    contextBuilder.addPregreppedHeaders(
        CppHelper.createExtractInclusions(ruleContext, semantics, publicHeaders.getHeaders()));
    contextBuilder.addPregreppedHeaders(
        CppHelper.createExtractInclusions(ruleContext, semantics, publicTextualHeaders));
    contextBuilder.addPregreppedHeaders(
        CppHelper.createExtractInclusions(ruleContext, semantics, privateHeaders));

    // Add this package's dir to declaredIncludeDirs, & this rule's headers to declaredIncludeSrcs
    // Note: no include dir for STRICT mode.
    if (headersCheckingMode == HeadersCheckingMode.WARN) {
      contextBuilder.addDeclaredIncludeWarnDir(ruleContext.getLabel().getPackageFragment());
      for (PathFragment looseIncludeDir : looseIncludeDirs) {
        contextBuilder.addDeclaredIncludeWarnDir(looseIncludeDir);
      }
    } else if (headersCheckingMode == HeadersCheckingMode.LOOSE) {
      contextBuilder.addDeclaredIncludeDir(ruleContext.getLabel().getPackageFragment());
      for (PathFragment looseIncludeDir : looseIncludeDirs) {
        contextBuilder.addDeclaredIncludeDir(looseIncludeDir);
      }
    }

    if (featureConfiguration.isEnabled(CppRuleClasses.MODULE_MAPS)) {
      if (cppModuleMap == null) {
        cppModuleMap = CppHelper.createDefaultCppModuleMap(ruleContext, /*suffix=*/ "");
      }

      contextBuilder.setPropagateCppModuleMapAsActionInput(propagateModuleMapToCompileAction);
      contextBuilder.setCppModuleMap(cppModuleMap);
      // There are different modes for module compilation:
      // 1. We create the module map and compile the module so that libraries depending on us can
      //    use the resulting module artifacts in their compilation (compiled is true).
      // 2. We create the module map so that libraries depending on us will include the headers
      //    textually (compiled is false).
      boolean compiled =
          featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULES)
              || featureConfiguration.isEnabled(CppRuleClasses.COMPILE_ALL_MODULES);
      Iterable<CppModuleMap> dependentModuleMaps = collectModuleMaps();

      if (generateModuleMap) {
        Optional<Artifact> umbrellaHeader = cppModuleMap.getUmbrellaHeader();
        if (umbrellaHeader.isPresent()) {
          ruleContext.registerAction(
              createUmbrellaHeaderAction(umbrellaHeader.get(), publicHeaders));
        }

        ruleContext.registerAction(
            createModuleMapAction(cppModuleMap, publicHeaders, dependentModuleMaps, compiled));
      }
      if (getGeneratesPicHeaderModule()) {
        contextBuilder.setPicHeaderModule(getPicHeaderModule(cppModuleMap.getArtifact()));
      }
      if (getGeneratesNoPicHeaderModule()) {
        contextBuilder.setHeaderModule(getHeaderModule(cppModuleMap.getArtifact()));
      }
      if (!compiled
          && featureConfiguration.isEnabled(CppRuleClasses.PARSE_HEADERS)
          && featureConfiguration.isEnabled(CppRuleClasses.USE_HEADER_MODULES)
          && ruleContext.getFragment(CppConfiguration.class).getParseHeadersVerifiesModules()) {
        // Here, we are creating a compiled module to verify that headers are self-contained and
        // modules ready, but we don't use the corresponding module map or compiled file anywhere
        // else.
        CppModuleMap verificationMap =
            CppHelper.createDefaultCppModuleMap(ruleContext, /*suffix=*/ ".verify");
        ruleContext.registerAction(
            createModuleMapAction(
                verificationMap, publicHeaders, dependentModuleMaps, /*compiledModule=*/ true));
        contextBuilder.setVerificationModuleMap(verificationMap);
      }
    }
    contextBuilder.setPurpose(purpose);

    semantics.setupCompilationContext(ruleContext, contextBuilder);
    return contextBuilder.build();
  }

  /**
   * Collects all preprocessed header files (*.h.processed) from dependencies and the current rule.
   */
  public static NestedSet<Artifact> collectHeaderTokens(
      RuleContext ruleContext, CcCompilationOutputs ccCompilationOutputs) {
    NestedSetBuilder<Artifact> headerTokens = NestedSetBuilder.stableOrder();
    for (OutputGroupInfo dep :
        ruleContext.getPrerequisites("deps", Mode.TARGET, OutputGroupInfo.SKYLARK_CONSTRUCTOR)) {
      headerTokens.addTransitive(dep.getOutputGroup(CcCompilationHelper.HIDDEN_HEADER_TOKENS));
    }
    if (ruleContext.getFragment(CppConfiguration.class).processHeadersInDependencies()) {
      headerTokens.addAll(ccCompilationOutputs.getHeaderTokenFiles());
    }
    return headerTokens.build();
  }

  public void registerAdditionalModuleMap(CppModuleMap cppModuleMap) {
    this.additionalCppModuleMaps.add(Preconditions.checkNotNull(cppModuleMap));
  }

  /** Don't generate a module map for this target if a custom module map is provided. */
  public CcCompilationHelper doNotGenerateModuleMap() {
    generateModuleMap = false;
    return this;
  }

  /**
   * Sets the purpose for the context.
   *
   * @see CppCompilationContext.Builder#setPurpose
   * @param purpose must be a string which is suitable for use as a filename. A single rule may have
   *     many middlemen with distinct purposes.
   */
  public CcCompilationHelper setPurpose(@Nullable String purpose) {
    this.purpose = purpose;
    return this;
  }

  /**
   * Supplier that computes legacy_compile_flags lazily at the execution phase.
   *
   * <p>Dear friends of the lambda, this method exists to limit the scope of captured variables only
   * to arguments (to prevent accidental capture of enclosing instance which could regress memory).
   */
  public static Supplier<ImmutableList<String>> getLegacyCompileFlagsSupplier(
      CppConfiguration cppConfiguration,
      CcToolchainProvider toolchain,
      String sourceFilename,
      ImmutableSet<String> features) {
    return () -> {
      ImmutableList.Builder<String> legacyCompileFlags = ImmutableList.builder();
      legacyCompileFlags.addAll(
          CppHelper.getCompilerOptions(cppConfiguration, toolchain, features));
      if (CppFileTypes.C_SOURCE.matches(sourceFilename)) {
        legacyCompileFlags.addAll(cppConfiguration.getCOptions());
      }
      if (CppFileTypes.CPP_SOURCE.matches(sourceFilename)
          || CppFileTypes.CPP_HEADER.matches(sourceFilename)
          || CppFileTypes.CPP_MODULE_MAP.matches(sourceFilename)
          || CppFileTypes.CLIF_INPUT_PROTO.matches(sourceFilename)) {
        legacyCompileFlags.addAll(CppHelper.getCxxOptions(cppConfiguration, toolchain, features));
      }
      return legacyCompileFlags.build();
    };
  }

  /**
   * Supplier that computes unfiltered_compile_flags lazily at the execution phase.
   *
   * <p>Dear friends of the lambda, this method exists to limit the scope of captured variables only
   * to arguments (to prevent accidental capture of enclosing instance which could regress memory).
   */
  public static Supplier<ImmutableList<String>> getUnfilteredCompileFlagsSupplier(
      CcToolchainProvider ccToolchain, ImmutableSet<String> features) {
    return () -> ccToolchain.getUnfilteredCompilerOptions(features);
  }

  private UmbrellaHeaderAction createUmbrellaHeaderAction(
      Artifact umbrellaHeader, PublicHeaders publicHeaders) {
    return new UmbrellaHeaderAction(
        ruleContext.getActionOwner(),
        umbrellaHeader,
        featureConfiguration.isEnabled(CppRuleClasses.ONLY_DOTH_HEADERS_IN_MODULE_MAPS)
            ? Iterables.filter(publicHeaders.getModuleMapHeaders(), CppFileTypes.MODULE_MAP_HEADER)
            : publicHeaders.getModuleMapHeaders(),
        additionalExportedHeaders);
  }

  private CppModuleMapAction createModuleMapAction(
      CppModuleMap moduleMap,
      PublicHeaders publicHeaders,
      Iterable<CppModuleMap> dependentModuleMaps,
      boolean compiledModule) {
    return new CppModuleMapAction(
        ruleContext.getActionOwner(),
        moduleMap,
        featureConfiguration.isEnabled(CppRuleClasses.EXCLUDE_PRIVATE_HEADERS_IN_MODULE_MAPS)
            ? ImmutableList.<Artifact>of()
            : privateHeaders,
        featureConfiguration.isEnabled(CppRuleClasses.ONLY_DOTH_HEADERS_IN_MODULE_MAPS)
            ? Iterables.filter(publicHeaders.getModuleMapHeaders(), CppFileTypes.MODULE_MAP_HEADER)
            : publicHeaders.getModuleMapHeaders(),
        dependentModuleMaps,
        additionalExportedHeaders,
        compiledModule,
        featureConfiguration.isEnabled(CppRuleClasses.MODULE_MAP_HOME_CWD),
        featureConfiguration.isEnabled(CppRuleClasses.GENERATE_SUBMODULES),
        !featureConfiguration.isEnabled(CppRuleClasses.MODULE_MAP_WITHOUT_EXTERN_MODULE));
  }

  private Iterable<CppModuleMap> collectModuleMaps() {
    // Cpp module maps may be null for some rules. We filter the nulls out at the end.
    List<CppModuleMap> result =
        deps.stream().map(CPP_DEPS_TO_MODULES).collect(toCollection(ArrayList::new));
    if (ruleContext.getRule().getAttributeDefinition(":stl") != null) {
      CppCompilationContext stl =
          ruleContext.getPrerequisite(":stl", Mode.TARGET, CppCompilationContext.class);
      if (stl != null) {
        result.add(stl.getCppModuleMap());
      }
    }

    if (ccToolchain != null) {
      result.add(ccToolchain.getCppCompilationContext().getCppModuleMap());
    }
    for (CppModuleMap additionalCppModuleMap : additionalCppModuleMaps) {
      result.add(additionalCppModuleMap);
    }

    return Iterables.filter(result, Predicates.<CppModuleMap>notNull());
  }

  private TransitiveLipoInfoProvider collectTransitiveLipoInfo(CcCompilationOutputs outputs) {
    if (fdoSupport.getFdoSupport().getFdoRoot() == null) {
      return TransitiveLipoInfoProvider.EMPTY;
    }
    NestedSetBuilder<IncludeScannable> scannableBuilder = NestedSetBuilder.stableOrder();
    // TODO(bazel-team): Only fetch the STL prerequisite in one place.
    TransitiveInfoCollection stl = ruleContext.getPrerequisite(":stl", Mode.TARGET);
    if (stl != null) {
      TransitiveLipoInfoProvider provider = stl.getProvider(TransitiveLipoInfoProvider.class);
      if (provider != null) {
        scannableBuilder.addTransitive(provider.getTransitiveIncludeScannables());
      }
    }

    for (TransitiveLipoInfoProvider dep :
        AnalysisUtils.getProviders(deps, TransitiveLipoInfoProvider.class)) {
      scannableBuilder.addTransitive(dep.getTransitiveIncludeScannables());
    }

    for (IncludeScannable scannable : outputs.getLipoScannables()) {
      Preconditions.checkState(scannable.getIncludeScannerSources().size() == 1);
      scannableBuilder.add(scannable);
    }
    return new TransitiveLipoInfoProvider(scannableBuilder.build());
  }

  private NestedSet<Artifact> getTemps(CcCompilationOutputs compilationOutputs) {
    return ruleContext.getFragment(CppConfiguration.class).isLipoContextCollector()
        ? NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)
        : compilationOutputs.getTemps();
  }

  /** @return whether this target needs to generate a pic header module. */
  private boolean getGeneratesPicHeaderModule() {
    return shouldProvideHeaderModules() && !fake && getGeneratePicActions();
  }

  /** @return whether this target needs to generate a non-pic header module. */
  private boolean getGeneratesNoPicHeaderModule() {
    return shouldProvideHeaderModules() && !fake && getGenerateNoPicActions();
  }

  /** @return whether we want to provide header modules for the current target. */
  private boolean shouldProvideHeaderModules() {
    return featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULES)
        && !cppConfiguration.isLipoContextCollector();
  }

  /** @return whether this target needs to generate non-pic actions. */
  private boolean getGenerateNoPicActions() {
    if (!generateNoPic) {
      return false;
    }
    boolean picFeatureEnabled = featureConfiguration.isEnabled(CppRuleClasses.PIC);
    boolean usePicForBinaries = CppHelper.usePic(ruleContext, ccToolchain, true);
    boolean usePicForNonBinaries = CppHelper.usePic(ruleContext, ccToolchain, false);

    if (!usePicForNonBinaries) {
      // This means you have to be prepared to use non-pic output for dynamic libraries.
      return true;
    }

    // Either you're only making a dynamic library (onlySingleOutput) or pic should be used
    // in all cases.
    if (usePicForBinaries) {
      if (picFeatureEnabled) {
        return false;
      }
      ruleContext.ruleError(PIC_CONFIGURATION_ERROR);
    }

    return true;
  }

  /** @return whether this target needs to generate pic actions. */
  private boolean getGeneratePicActions() {
    return featureConfiguration.isEnabled(CppRuleClasses.PIC)
        && CppHelper.usePic(ruleContext, ccToolchain, false);
  }

  /** @return the non-pic header module artifact for the current target. */
  private Artifact getHeaderModule(Artifact moduleMapArtifact) {
    PathFragment objectDir = CppHelper.getObjDirectory(ruleContext.getLabel());
    PathFragment outputName = objectDir.getRelative(moduleMapArtifact.getRootRelativePath());
    return ruleContext.getRelatedArtifact(outputName, ".pcm");
  }

  /** @return the pic header module artifact for the current target. */
  private Artifact getPicHeaderModule(Artifact moduleMapArtifact) {
    PathFragment objectDir = CppHelper.getObjDirectory(ruleContext.getLabel());
    PathFragment outputName = objectDir.getRelative(moduleMapArtifact.getRootRelativePath());
    return ruleContext.getRelatedArtifact(outputName, ".pic.pcm");
  }

  /**
   * Constructs the C++ compiler actions. It generally creates one action for every specified source
   * file. It takes into account LIPO, fake-ness, coverage, and PIC, in addition to using the
   * settings specified on the current object. This method should only be called once.
   */
  private CcCompilationOutputs createCcCompileActions() throws RuleErrorException {
    CcCompilationOutputs.Builder result = new CcCompilationOutputs.Builder();
    Preconditions.checkNotNull(cppCompilationContext);
    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();

    if (shouldProvideHeaderModules()) {
      Label moduleMapLabel =
          Label.parseAbsoluteUnchecked(cppCompilationContext.getCppModuleMap().getName());
      Collection<Artifact> modules =
          createModuleAction(result, cppCompilationContext.getCppModuleMap());
      if (featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULE_CODEGEN)) {
        for (Artifact module : modules) {
          // TODO(djasper): Investigate whether we need to use a label separate from that of the
          // module map. It is used for per-file-copts.
          createModuleCodegenAction(result, moduleMapLabel, module);
        }
      }
    } else if (cppCompilationContext.getVerificationModuleMap() != null) {
      Collection<Artifact> modules =
          createModuleAction(result, cppCompilationContext.getVerificationModuleMap());
      for (Artifact module : modules) {
        result.addHeaderTokenFile(module);
      }
    }

    for (CppSource source : compilationUnitSources) {
      Artifact sourceArtifact = source.getSource();
      Label sourceLabel = source.getLabel();
      String outputName =
          FileSystemUtils.removeExtension(sourceArtifact.getRootRelativePath()).getPathString();
      CppCompileActionBuilder builder = initializeCompileAction(sourceArtifact);

      builder
          .setSemantics(semantics)
          .addMandatoryInputs(compilationMandatoryInputs)
          .addAdditionalIncludeScanningRoots(additionalIncludeScanningRoots);

      boolean bitcodeOutput =
          featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
              && CppFileTypes.LTO_SOURCE.matches(sourceArtifact.getFilename());

      if (!sourceArtifact.isTreeArtifact()) {
        switch (source.getType()) {
          case HEADER:
            createHeaderAction(
                sourceLabel, outputName, result, env, builder, isGenerateDotdFile(sourceArtifact));
            break;
          default:
            createSourceAction(
                sourceLabel,
                outputName,
                result,
                env,
                sourceArtifact,
                builder,
                // TODO(plf): Continue removing CLIF logic from C++. Follow up changes would include
                // refactoring CppSource.Type and ArtifactCategory to be classes instead of enums
                // that could be instantiated with arbitrary values.
                source.getType() == CppSource.Type.CLIF_INPUT_PROTO
                    ? ArtifactCategory.CLIF_OUTPUT_PROTO
                    : ArtifactCategory.OBJECT_FILE,
                cppCompilationContext.getCppModuleMap(),
                /* addObject= */ true,
                isCodeCoverageEnabled(),
                // The source action does not generate dwo when it has bitcode
                // output (since it isn't generating a native object with debug
                // info). In that case the LtoBackendAction will generate the dwo.
                CppHelper.shouldCreatePerObjectDebugInfo(
                        cppConfiguration, ccToolchain, featureConfiguration)
                    && !bitcodeOutput,
                isGenerateDotdFile(sourceArtifact));
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
                        ArtifactCategory.GENERATED_HEADER, ArtifactCategory.PROCESSED_HEADER),
                    false);
            result.addHeaderTokenFile(headerTokenFile);
            break;
          case SOURCE:
            Artifact objectFile =
                createCompileActionTemplate(
                    env, source, builder, ImmutableList.of(ArtifactCategory.OBJECT_FILE), false);
            result.addObjectFile(objectFile);

            if (getGeneratePicActions()) {
              Artifact picObjectFile =
                  createCompileActionTemplate(
                      env,
                      source,
                      builder,
                      ImmutableList.of(ArtifactCategory.PIC_OBJECT_FILE),
                      true);
              result.addPicObjectFile(picObjectFile);
            }
            break;
          default:
            throw new IllegalStateException(
                "Encountered invalid source types when creating CppCompileActionTemplates");
        }
      }
    }

    return result.build();
  }

  private Artifact createCompileActionTemplate(
      AnalysisEnvironment env,
      CppSource source,
      CppCompileActionBuilder builder,
      Iterable<ArtifactCategory> outputCategories,
      boolean usePic) {
    SpecialArtifact sourceArtifact = (SpecialArtifact) source.getSource();
    SpecialArtifact outputFiles =
        CppHelper.getCompileOutputTreeArtifact(ruleContext, sourceArtifact, usePic);
    // TODO(rduan): Dotd file output is not supported yet.
    builder.setOutputs(outputFiles, /* dotdFile= */ null);
    setupCompileBuildVariables(
        builder,
        source.getLabel(),
        usePic,
        /* ccRelativeName= */ null,
        /* autoFdoImportPath= */ null,
        /* gcnoFile= */ null,
        /* dwoFile= */ null,
        /* ltoIndexingFile= */ null,
        builder.getContext().getCppModuleMap());
    semantics.finalizeCompileActionBuilder(ruleContext, builder);
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

  private void setupCompileBuildVariables(
      CppCompileActionBuilder builder,
      Label sourceLabel,
      boolean usePic,
      PathFragment ccRelativeName,
      PathFragment autoFdoImportPath,
      Artifact gcnoFile,
      Artifact dwoFile,
      Artifact ltoIndexingFile,
      CppModuleMap cppModuleMap) {
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
        getLegacyCompileFlagsSupplier(cppConfiguration, ccToolchain, sourceFilename, features));

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
    } else if (FileType.contains(
        outputFile,
        CppFileTypes.PREPROCESSED_C,
        CppFileTypes.PREPROCESSED_CPP,
        CppFileTypes.PIC_PREPROCESSED_C,
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
        defines =
            ImmutableList.<String>builder()
                .addAll(builderContext.getDefines())
                .add(
                    CppConfiguration.FDO_STAMP_MACRO
                        + "=\""
                        + CppHelper.getFdoBuildStamp(ruleContext, fdoSupport.getFdoSupport())
                        + "\"")
                .build();
      } else {
        defines = builderContext.getDefines();
      }

      buildVariables.addStringSequenceVariable(PREPROCESSOR_DEFINES_VARIABLE_NAME, defines);
    }

    if (usePic) {
      if (!featureConfiguration.isEnabled(CppRuleClasses.PIC)) {
        ruleContext.ruleError(PIC_CONFIGURATION_ERROR);
      }
      buildVariables.addStringVariable(PIC_VARIABLE_NAME, "");
    }

    if (ccRelativeName != null) {
      fdoSupport
          .getFdoSupport()
          .configureCompilation(
              builder,
              buildVariables,
              ruleContext,
              ccRelativeName,
              autoFdoImportPath,
              usePic,
              featureConfiguration,
              fdoSupport);
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

    for (VariablesExtension extension : variablesExtensions) {
      extension.addVariables(buildVariables);
    }

    CcToolchainFeatures.Variables variables = buildVariables.build();
    builder.setVariables(variables);
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

  /**
   * Creates a basic cpp compile action builder for source file. Configures options, crosstool
   * inputs, output and dotd file names, compilation context and copts.
   */
  private CppCompileActionBuilder createCompileActionBuilder(Artifact source) {
    CppCompileActionBuilder builder =
        new CppCompileActionBuilder(ruleContext, ccToolchain, configuration);
    builder.setSourceFile(source);
    builder.setContext(cppCompilationContext);
    builder.addEnvironment(ccToolchain.getEnvironment());
    builder.setCoptsFilter(coptsFilter);
    return builder;
  }

  private void createModuleCodegenAction(
      CcCompilationOutputs.Builder result, Label sourceLabel, Artifact module)
      throws RuleErrorException {
    if (fake) {
      // We can't currently foresee a situation where we'd want nocompile tests for module codegen.
      // If we find one, support needs to be added here.
      return;
    }
    String outputName = module.getRootRelativePath().getPathString();

    // TODO(djasper): Make this less hacky after refactoring how the PIC/noPIC actions are created.
    boolean pic = module.getFilename().contains(".pic.");

    CppCompileActionBuilder builder = initializeCompileAction(module);
    builder.setSemantics(semantics);
    builder.setPicMode(pic);
    builder.setOutputs(
        ruleContext, ArtifactCategory.OBJECT_FILE, outputName, isGenerateDotdFile(module));
    PathFragment ccRelativeName = module.getRootRelativePath();

    String gcnoFileName =
        CppHelper.getArtifactNameForCategory(
            ruleContext, ccToolchain, ArtifactCategory.COVERAGE_DATA_FILE, outputName);
    // TODO(djasper): This is now duplicated. Refactor the various create..Action functions.
    Artifact gcnoFile =
        isCodeCoverageEnabled() && !CppHelper.isLipoOptimization(cppConfiguration, ccToolchain)
            ? CppHelper.getCompileOutputArtifact(ruleContext, gcnoFileName, configuration)
            : null;

    boolean generateDwo =
        CppHelper.shouldCreatePerObjectDebugInfo(
            cppConfiguration, ccToolchain, featureConfiguration);
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
        builder.getContext().getCppModuleMap());

    builder.setGcnoFile(gcnoFile);
    builder.setDwoFile(dwoFile);

    semantics.finalizeCompileActionBuilder(ruleContext, builder);
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

  /** Returns true if Dotd file should be generated. */
  private boolean isGenerateDotdFile(Artifact sourceArtifact) {
    return CppFileTypes.headerDiscoveryRequired(sourceArtifact)
        && !featureConfiguration.isEnabled(CppRuleClasses.PARSE_SHOWINCLUDES);
  }

  private void createHeaderAction(
      Label sourceLabel,
      String outputName,
      CcCompilationOutputs.Builder result,
      AnalysisEnvironment env,
      CppCompileActionBuilder builder,
      boolean generateDotd)
      throws RuleErrorException {
    String outputNameBase =
        CppHelper.getArtifactNameForCategory(
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
        builder.getContext().getCppModuleMap());
    semantics.finalizeCompileActionBuilder(ruleContext, builder);
    CppCompileAction compileAction = builder.buildOrThrowRuleError(ruleContext);
    env.registerAction(compileAction);
    Artifact tokenFile = compileAction.getOutputFile();
    result.addHeaderTokenFile(tokenFile);
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
        FileSystemUtils.removeExtension(moduleMapArtifact.getRootRelativePath()).getPathString(),
        result,
        env,
        moduleMapArtifact,
        builder,
        ArtifactCategory.CPP_MODULE,
        cppModuleMap,
        /* addObject= */ false,
        /* enableCoverage= */ false,
        /* generateDwo= */ false,
        isGenerateDotdFile(moduleMapArtifact));
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
      boolean generateDotd)
      throws RuleErrorException {
    ImmutableList.Builder<Artifact> directOutputs = new ImmutableList.Builder<>();
    PathFragment ccRelativeName = sourceArtifact.getRootRelativePath();
    if (CppHelper.isLipoOptimization(cppConfiguration, ccToolchain)) {
      // TODO(bazel-team): we shouldn't be needing this, merging context with the binary
      // is a superset of necessary information.
      LipoContextProvider lipoProvider =
          Preconditions.checkNotNull(CppHelper.getLipoContextProvider(ruleContext), outputName);
      builder.setContext(
          CppCompilationContext.mergeForLipo(lipoProvider.getLipoContext(), cppCompilationContext));
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
        String picOutputBase =
            CppHelper.getArtifactNameForCategory(
                ruleContext, ccToolchain, ArtifactCategory.PIC_FILE, outputName);
        CppCompileActionBuilder picBuilder =
            copyAsPicBuilder(builder, picOutputBase, outputCategory, generateDotd);
        String gcnoFileName =
            CppHelper.getArtifactNameForCategory(
                ruleContext, ccToolchain, ArtifactCategory.COVERAGE_DATA_FILE, picOutputBase);
        Artifact gcnoFile =
            enableCoverage
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
            cppModuleMap);

        result.addTemps(
            createTempsActions(
                sourceArtifact,
                sourceLabel,
                outputName,
                picBuilder,
                /* usePic= */ true,
                /* generateDotd= */ generateDotd,
                ccRelativeName));

        picBuilder.setGcnoFile(gcnoFile);
        picBuilder.setDwoFile(dwoFile);
        picBuilder.setLtoIndexingFile(ltoIndexingFile);

        semantics.finalizeCompileActionBuilder(ruleContext, picBuilder);
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
        Artifact noPicOutputFile =
            CppHelper.getCompileOutputArtifact(
                ruleContext,
                CppHelper.getArtifactNameForCategory(
                    ruleContext, ccToolchain, outputCategory, outputName),
                configuration);
        builder.setOutputs(ruleContext, outputCategory, outputName, generateDotd);
        String gcnoFileName =
            CppHelper.getArtifactNameForCategory(
                ruleContext, ccToolchain, ArtifactCategory.COVERAGE_DATA_FILE, outputName);

        // Create non-PIC compile actions
        Artifact gcnoFile =
            !CppHelper.isLipoOptimization(cppConfiguration, ccToolchain) && enableCoverage
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
            cppModuleMap);

        result.addTemps(
            createTempsActions(
                sourceArtifact,
                sourceLabel,
                outputName,
                builder,
                /* usePic= */ false,
                generateDotd,
                ccRelativeName));

        builder.setGcnoFile(gcnoFile);
        builder.setDwoFile(noPicDwoFile);
        builder.setLtoIndexingFile(ltoIndexingFile);

        semantics.finalizeCompileActionBuilder(ruleContext, builder);
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

  /**
   * Creates cpp PIC compile action builder from the given builder by adding necessary copt and
   * changing output and dotd file names.
   */
  private CppCompileActionBuilder copyAsPicBuilder(
      CppCompileActionBuilder builder,
      String outputName,
      ArtifactCategory outputCategory,
      boolean generateDotd)
      throws RuleErrorException {
    CppCompileActionBuilder picBuilder = new CppCompileActionBuilder(builder);
    picBuilder.setPicMode(true).setOutputs(ruleContext, outputCategory, outputName, generateDotd);

    return picBuilder;
  }

  String getOutputNameBaseWith(String base, boolean usePic) throws RuleErrorException {
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
    String tempOutputName =
        ruleContext
            .getConfiguration()
            .getBinFragment()
            .getRelative(CppHelper.getObjDirectory(ruleContext.getLabel()))
            .getRelative(
                CppHelper.getArtifactNameForCategory(
                    ruleContext,
                    ccToolchain,
                    outputCategory,
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
        builder.getContext().getCppModuleMap());
    semantics.finalizeCompileActionBuilder(ruleContext, builder);
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

  /** Returns true iff code coverage is enabled for the given target. */
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

  /** Get the safe path strings for a list of paths to use in the build variables. */
  private ImmutableSet<String> getSafePathStrings(Collection<PathFragment> paths) {
    ImmutableSet.Builder<String> result = ImmutableSet.builder();
    for (PathFragment path : paths) {
      result.add(path.getSafePathString());
    }
    return result.build();
  }

  private Artifact getDwoFile(Artifact outputFile) {
    return ruleContext.getRelatedArtifact(outputFile.getRootRelativePath(), ".dwo");
  }

  private Artifact getLtoIndexingFile(Artifact outputFile) {
    String ext = Iterables.getOnlyElement(CppFileTypes.LTO_INDEXING_OBJECT_FILE.getExtensions());
    return ruleContext.getRelatedArtifact(outputFile.getRootRelativePath(), ext);
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

    ArtifactCategory category =
        isCFile ? ArtifactCategory.PREPROCESSED_C_SOURCE : ArtifactCategory.PREPROCESSED_CPP_SOURCE;

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
        builder.getContext().getCppModuleMap());
    semantics.finalizeCompileActionBuilder(ruleContext, dBuilder);
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
        builder.getContext().getCppModuleMap());
    semantics.finalizeCompileActionBuilder(ruleContext, sdBuilder);
    CppCompileAction sdAction = sdBuilder.buildOrThrowRuleError(ruleContext);
    ruleContext.registerAction(sdAction);

    return ImmutableList.of(dAction.getOutputFile(), sdAction.getOutputFile());
  }
}
