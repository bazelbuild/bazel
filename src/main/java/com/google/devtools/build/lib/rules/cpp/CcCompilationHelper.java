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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
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
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;
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
        CcCompilationInfo ccCompilationInfo = dep.get(CcCompilationInfo.PROVIDER);
        CcCompilationContextInfo ccCompilationContextInfo = null;
        if (ccCompilationInfo != null) {
          ccCompilationContextInfo = ccCompilationInfo.getCcCompilationContextInfo();
        }
        return ccCompilationContextInfo == null ? null : ccCompilationContextInfo.getCppModuleMap();
      };

  /**
   * Contains the providers as well as the {@code CcCompilationOutputs} and the {@code
   * CcCompilationContextInfo}.
   */
  @SkylarkModule(
    name = "compilation_info",
    documented = false,
    category = SkylarkModuleCategory.BUILTIN,
    doc = "Helper class containing CC compilation providers."
  )
  // TODO(plf): Rename so that it's not confused with CcCompilationContextInfo and also consider
  // merging
  // this class with {@code CcCompilationOutputs}.
  public static final class CompilationInfo {
    private final TransitiveInfoProviderMap providers;
    private final Map<String, NestedSet<Artifact>> outputGroups;
    private final CcCompilationOutputs compilationOutputs;

    private CompilationInfo(
        TransitiveInfoProviderMap providers,
        Map<String, NestedSet<Artifact>> outputGroups,
        CcCompilationOutputs compilationOutputs) {
      this.providers = providers;
      this.outputGroups = outputGroups;
      this.compilationOutputs = compilationOutputs;
    }

    public TransitiveInfoProviderMap getProviders() {
      return providers;
    }

    public Map<String, NestedSet<Artifact>> getOutputGroups() {
      return outputGroups;
    }

    @SkylarkCallable(name = "cc_output_groups", documented = false)
    public Map<String, SkylarkNestedSet> getSkylarkOutputGroups() {
      Map<String, SkylarkNestedSet> skylarkOutputGroups = new TreeMap<>();
      for (Map.Entry<String, NestedSet<Artifact>> entry : outputGroups.entrySet()) {
        skylarkOutputGroups.put(
            entry.getKey(), SkylarkNestedSet.of(Artifact.class, entry.getValue()));
      }
      return skylarkOutputGroups;
    }

    @SkylarkCallable(name = "cc_compilation_outputs", documented = false)
    public CcCompilationOutputs getCcCompilationOutputs() {
      return compilationOutputs;
    }

    @SkylarkCallable(name = "cc_compilation_info", documented = false)
    public CcCompilationInfo getCcCompilationInfo() {
      return (CcCompilationInfo) providers.getProvider(CcCompilationInfo.PROVIDER.getKey());
    }

    public CcCompilationContextInfo getCcCompilationContextInfo() {
      return ((CcCompilationInfo) providers.getProvider(CcCompilationInfo.PROVIDER.getKey()))
          .getCcCompilationContextInfo();
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
  private final List<CcCompilationContextInfo> depCcCompilationContextInfos = new ArrayList<>();
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
  private boolean useDeps = true;
  private boolean generateModuleMap = true;
  private String purpose = null;
  private boolean generateNoPicAction;
  private boolean generatePicAction;

  // TODO(plf): Pull out of class.
  private CcCompilationContextInfo ccCompilationContextInfo;

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
    setGenerateNoPicAction(
        !CppHelper.usePicForDynamicLibraries(ruleContext, ccToolchain)
            || !CppHelper.usePicForBinaries(ruleContext, ccToolchain));
    setGeneratePicAction(
        CppHelper.usePicForDynamicLibraries(ruleContext, ccToolchain)
            || CppHelper.usePicForBinaries(ruleContext, ccToolchain));
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
  public CcCompilationHelper fromCommon(CcCommon common, ImmutableList<String> additionalCopts) {
    Preconditions.checkNotNull(additionalCopts);

    setCopts(Iterables.concat(common.getCopts(), additionalCopts));
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
   * Add the corresponding files as linker inputs for no-PIC links. If the corresponding files are
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

  public CcCompilationHelper setCopts(Iterable<String> copts) {
    this.copts = ImmutableList.copyOf(copts);
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

  public CcCompilationHelper addDepCcCompilationContextInfo(
      CcCompilationContextInfo ccCompilationContextInfo) {
    this.depCcCompilationContextInfos.add(Preconditions.checkNotNull(ccCompilationContextInfo));
    return this;
  }

  /**
   * Adds the given precompiled files to this helper. Shared and static libraries are added as
   * compilation prerequisites, and object files are added as pic or no-PIC object files
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

  /** Whether to generate no-PIC actions. */
  public CcCompilationHelper setGenerateNoPicAction(boolean generateNoPicAction) {
    this.generateNoPicAction = generateNoPicAction;
    return this;
  }

  /** Whether to generate PIC actions. */
  public CcCompilationHelper setGeneratePicAction(boolean generatePicAction) {
    this.generatePicAction = generatePicAction;
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

    ccCompilationContextInfo = initializeCcCompilationContextInfo();

    boolean compileHeaderModules = featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULES);
    Preconditions.checkState(
        !compileHeaderModules || ccCompilationContextInfo.getCppModuleMap() != null,
        "All cc rules must support module maps.");

    // Create compile actions (both PIC and no-PIC).
    CcCompilationOutputs ccOutputs = createCcCompileActions();
    if (!objectFiles.isEmpty() || !picObjectFiles.isEmpty()) {
      // Merge the pre-compiled object files into the compiler outputs.
      ccOutputs =
          new CcCompilationOutputs.Builder()
              .merge(ccOutputs)
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
                new CppDebugFileProvider(
                    dwoArtifacts.getDwoArtifacts(), dwoArtifacts.getPicDwoArtifacts()),
                collectTransitiveLipoInfo(ccOutputs));
    CcCompilationInfo.Builder ccCompilationInfoBuilder = CcCompilationInfo.Builder.create();
    ccCompilationInfoBuilder.setCcCompilationContextInfo(ccCompilationContextInfo);
    providers.put(ccCompilationInfoBuilder.build());

    Map<String, NestedSet<Artifact>> outputGroups = new TreeMap<>();
    outputGroups.put(OutputGroupInfo.TEMP_FILES, getTemps(ccOutputs));
    if (emitCompileProviders) {
      boolean isLipoCollector = cppConfiguration.isLipoContextCollector();
      boolean processHeadersInDependencies = cppConfiguration.processHeadersInDependencies();
      boolean usePic = CppHelper.usePicForDynamicLibraries(ruleContext, ccToolchain);
      outputGroups.put(
          OutputGroupInfo.FILES_TO_COMPILE,
          ccOutputs.getFilesToCompile(isLipoCollector, processHeadersInDependencies, usePic));
      outputGroups.put(
          OutputGroupInfo.COMPILATION_PREREQUISITES,
          CcCommon.collectCompilationPrerequisites(ruleContext, ccCompilationContextInfo));
    }

    return new CompilationInfo(providers.build(), outputGroups, ccOutputs);
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
   * Create {@code CcCompilationContextInfo} for cc compile action from generated inputs.
   *
   * <p>TODO(plf): Try to pull out CcCompilationContextInfo building out of this class.
   */
  public CcCompilationContextInfo initializeCcCompilationContextInfo() {
    CcCompilationContextInfo.Builder ccCompilationContextInfoBuilder =
        new CcCompilationContextInfo.Builder(ruleContext);

    // Setup the include path; local include directories come before those inherited from deps or
    // from the toolchain; in case of aliasing (same include file found on different entries),
    // prefer the local include rather than the inherited one.

    // Add in the roots for well-formed include names for source files and
    // generated files. It is important that the execRoot (EMPTY_FRAGMENT) comes
    // before the genfilesFragment to preferably pick up source files. Otherwise
    // we might pick up stale generated files.
    PathFragment repositoryPath =
        ruleContext.getLabel().getPackageIdentifier().getRepository().getPathUnderExecRoot();
    ccCompilationContextInfoBuilder.addQuoteIncludeDir(repositoryPath);
    ccCompilationContextInfoBuilder.addQuoteIncludeDir(
        ruleContext.getConfiguration().getGenfilesFragment().getRelative(repositoryPath));

    for (PathFragment systemIncludeDir : systemIncludeDirs) {
      ccCompilationContextInfoBuilder.addSystemIncludeDir(systemIncludeDir);
    }
    for (PathFragment includeDir : includeDirs) {
      ccCompilationContextInfoBuilder.addIncludeDir(includeDir);
    }

    PublicHeaders publicHeaders = computePublicHeaders();
    if (publicHeaders.getVirtualIncludePath() != null) {
      ccCompilationContextInfoBuilder.addIncludeDir(publicHeaders.getVirtualIncludePath());
    }

    if (useDeps) {
      ccCompilationContextInfoBuilder.mergeDependentCcCompilationContextInfos(
          CcCompilationInfo.getCcCompilationContextInfos(deps));
      ccCompilationContextInfoBuilder.mergeDependentCcCompilationContextInfos(
          depCcCompilationContextInfos);
    }
    CppHelper.mergeToolchainDependentCcCompilationContextInfo(
        ruleContext, ccToolchain, ccCompilationContextInfoBuilder);

    // But defines come after those inherited from deps.
    ccCompilationContextInfoBuilder.addDefines(defines);

    // There are no ordering constraints for declared include dirs/srcs, or the pregrepped headers.
    ccCompilationContextInfoBuilder.addDeclaredIncludeSrcs(publicHeaders.getHeaders());
    ccCompilationContextInfoBuilder.addDeclaredIncludeSrcs(publicTextualHeaders);
    ccCompilationContextInfoBuilder.addDeclaredIncludeSrcs(privateHeaders);
    ccCompilationContextInfoBuilder.addDeclaredIncludeSrcs(additionalInputs);
    ccCompilationContextInfoBuilder.addNonCodeInputs(additionalInputs);
    ccCompilationContextInfoBuilder.addModularHdrs(publicHeaders.getHeaders());
    ccCompilationContextInfoBuilder.addModularHdrs(privateHeaders);
    ccCompilationContextInfoBuilder.addTextualHdrs(publicTextualHeaders);
    ccCompilationContextInfoBuilder.addPregreppedHeaders(
        CppHelper.createExtractInclusions(ruleContext, semantics, publicHeaders.getHeaders()));
    ccCompilationContextInfoBuilder.addPregreppedHeaders(
        CppHelper.createExtractInclusions(ruleContext, semantics, publicTextualHeaders));
    ccCompilationContextInfoBuilder.addPregreppedHeaders(
        CppHelper.createExtractInclusions(ruleContext, semantics, privateHeaders));

    // Add this package's dir to declaredIncludeDirs, & this rule's headers to declaredIncludeSrcs
    // Note: no include dir for STRICT mode.
    if (headersCheckingMode == HeadersCheckingMode.WARN) {
      ccCompilationContextInfoBuilder.addDeclaredIncludeWarnDir(
          ruleContext.getLabel().getPackageFragment());
      for (PathFragment looseIncludeDir : looseIncludeDirs) {
        ccCompilationContextInfoBuilder.addDeclaredIncludeWarnDir(looseIncludeDir);
      }
    } else if (headersCheckingMode == HeadersCheckingMode.LOOSE) {
      ccCompilationContextInfoBuilder.addDeclaredIncludeDir(
          ruleContext.getLabel().getPackageFragment());
      for (PathFragment looseIncludeDir : looseIncludeDirs) {
        ccCompilationContextInfoBuilder.addDeclaredIncludeDir(looseIncludeDir);
      }
    }

    if (featureConfiguration.isEnabled(CppRuleClasses.MODULE_MAPS)) {
      if (cppModuleMap == null) {
        cppModuleMap = CppHelper.createDefaultCppModuleMap(ruleContext, /*suffix=*/ "");
      }

      ccCompilationContextInfoBuilder.setPropagateCppModuleMapAsActionInput(
          propagateModuleMapToCompileAction);
      ccCompilationContextInfoBuilder.setCppModuleMap(cppModuleMap);
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
        ccCompilationContextInfoBuilder.setPicHeaderModule(
            getPicHeaderModule(cppModuleMap.getArtifact()));
      }
      if (getGeneratesNoPicHeaderModule()) {
        ccCompilationContextInfoBuilder.setHeaderModule(
            getHeaderModule(cppModuleMap.getArtifact()));
      }
      if (!compiled
          && featureConfiguration.isEnabled(CppRuleClasses.PARSE_HEADERS)
          && featureConfiguration.isEnabled(CppRuleClasses.USE_HEADER_MODULES)
          && cppConfiguration.getParseHeadersVerifiesModules()) {
        // Here, we are creating a compiled module to verify that headers are self-contained and
        // modules ready, but we don't use the corresponding module map or compiled file anywhere
        // else.
        CppModuleMap verificationMap =
            CppHelper.createDefaultCppModuleMap(ruleContext, /*suffix=*/ ".verify");
        ruleContext.registerAction(
            createModuleMapAction(
                verificationMap, publicHeaders, dependentModuleMaps, /*compiledModule=*/ true));
        ccCompilationContextInfoBuilder.setVerificationModuleMap(verificationMap);
      }
    }
    ccCompilationContextInfoBuilder.setPurpose(purpose);

    semantics.setupCcCompilationContextInfo(ruleContext, ccCompilationContextInfoBuilder);
    return ccCompilationContextInfoBuilder.build();
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
   * Sets the purpose for the {@code CcCompilationContextInfo}.
   *
   * @see CcCompilationContextInfo.Builder#setPurpose
   * @param purpose must be a string which is suitable for use as a filename. A single rule may have
   *     many middlemen with distinct purposes.
   */
  public CcCompilationHelper setPurpose(@Nullable String purpose) {
    this.purpose = purpose;
    return this;
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
      CcCompilationInfo stl =
          ruleContext.getPrerequisite(":stl", Mode.TARGET, CcCompilationInfo.PROVIDER);
      if (stl != null) {
        result.add(stl.getCcCompilationContextInfo().getCppModuleMap());
      }
    }

    if (ccToolchain != null) {
      result.add(ccToolchain.getCcCompilationContextInfo().getCppModuleMap());
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
    return cppConfiguration.isLipoContextCollector()
        ? NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)
        : compilationOutputs.getTemps();
  }

  /** @return whether this target needs to generate a pic header module. */
  private boolean getGeneratesPicHeaderModule() {
    return shouldProvideHeaderModules() && !fake && generatePicAction;
  }

  /** @return whether this target needs to generate a no-PIC header module. */
  private boolean getGeneratesNoPicHeaderModule() {
    return shouldProvideHeaderModules() && !fake && generateNoPicAction;
  }

  /** @return whether we want to provide header modules for the current target. */
  private boolean shouldProvideHeaderModules() {
    return featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULES)
        && !cppConfiguration.isLipoContextCollector();
  }

  /** @return the no-PIC header module artifact for the current target. */
  private Artifact getHeaderModule(Artifact moduleMapArtifact) {
    PathFragment objectDir = CppHelper.getObjDirectory(ruleContext.getLabel());
    PathFragment outputName =
        objectDir.getRelative(
            cppConfiguration.shortenObjFilePath()
                ? moduleMapArtifact.getRootRelativePath().getBaseName()
                : moduleMapArtifact.getRootRelativePath().getPathString());
    return ruleContext.getRelatedArtifact(outputName, ".pcm");
  }

  /** @return the pic header module artifact for the current target. */
  private Artifact getPicHeaderModule(Artifact moduleMapArtifact) {
    PathFragment objectDir = CppHelper.getObjDirectory(ruleContext.getLabel());
    PathFragment outputName =
        objectDir.getRelative(
            cppConfiguration.shortenObjFilePath()
                ? moduleMapArtifact.getRootRelativePath().getBaseName()
                : moduleMapArtifact.getRootRelativePath().getPathString());
    return ruleContext.getRelatedArtifact(outputName, ".pic.pcm");
  }

  /**
   * Calculate the output names for object file paths from a set of source files.
   *
   * <p>The object file path is constructed in the following format:
   *    <bazel-bin>/<target_package_path>/_objs/<target_name>/<output_name>.<obj_extension>
   * When there's no two source files having the same basename:
   *   <output_name> = <prefixDir>/<source_file_base_name>
   * otherwise:
   *   <output_name> = <prefixDir>/N/<source_file_base_name>,
   *   N = the file’s order among the source files with the same basename, starts with 0
   *
   * <p>Examples: 1. Output names for ["lib1/foo.cc", "lib2/bar.cc"] are ["foo", "bar"]
   *              2. Output names for ["foo.cc", "bar.cc", "foo.cpp", "lib/foo.cc"]
   *                 are ["0/foo", "bar", "1/foo", "2/foo"]
   */
  private ImmutableMap<Artifact, String> calculateOutputNameMap(
      NestedSet<Artifact> sourceArtifacts, String prefixDir) {
    ImmutableMap.Builder<Artifact, String> builder = ImmutableMap.builder();

    HashMap<String, Integer> count = new LinkedHashMap<>();
    HashMap<String, Integer> number = new LinkedHashMap<>();
    for (Artifact source : sourceArtifacts) {
      String outputName =
          FileSystemUtils.removeExtension(source.getRootRelativePath()).getBaseName();
      count.put(outputName.toLowerCase(),
          count.getOrDefault(outputName.toLowerCase(), 0) + 1);
    }

    for (Artifact source : sourceArtifacts) {
      String outputName =
          FileSystemUtils.removeExtension(source.getRootRelativePath()).getBaseName();
      if (count.getOrDefault(outputName.toLowerCase(), 0) > 1) {
        int num = number.getOrDefault(outputName.toLowerCase(), 0);
        number.put(outputName.toLowerCase(), num + 1);
        outputName = num + "/" + outputName;
      }
      // If prefixDir is set, prepend it to the outputName
      if (prefixDir != null) {
        outputName = prefixDir + "/" + outputName;
      }
      builder.put(source, outputName);
    }

    return builder.build();
  }

  /**
   * Calculate outputNameMap for different source types separately. Returns a merged outputNameMap
   * for all artifacts.
   */
  private ImmutableMap<Artifact, String> calculateOutputNameMapByType(
      Set<CppSource> sources, String prefixDir) {
    ImmutableMap.Builder<Artifact, String> builder = ImmutableMap.builder();
    builder.putAll(
        calculateOutputNameMap(
            getSourceArtifactsByType(sources, CppSource.Type.SOURCE), prefixDir));
    builder.putAll(
        calculateOutputNameMap(
            getSourceArtifactsByType(sources, CppSource.Type.HEADER), prefixDir));
    // TODO(plf): Removing CLIF logic
    builder.putAll(
        calculateOutputNameMap(
            getSourceArtifactsByType(sources, CppSource.Type.CLIF_INPUT_PROTO), prefixDir));
    return builder.build();
  }

  private NestedSet<Artifact> getSourceArtifactsByType(
    Set<CppSource> sources, CppSource.Type type) {
    NestedSetBuilder<Artifact> result = NestedSetBuilder.stableOrder();
    result.addAll(sources
        .stream()
        .filter(source -> source.getType().equals(type))
        .map(CppSource::getSource)
        .collect(Collectors.toList()));
    return result.build();
  }

  /**
   * Constructs the C++ compiler actions. It generally creates one action for every specified source
   * file. It takes into account LIPO, fake-ness, coverage, and PIC, in addition to using the
   * settings specified on the current object. This method should only be called once.
   */
  private CcCompilationOutputs createCcCompileActions() throws RuleErrorException {
    CcCompilationOutputs.Builder result = new CcCompilationOutputs.Builder();
    Preconditions.checkNotNull(ccCompilationContextInfo);
    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();

    if (shouldProvideHeaderModules()) {
      Label moduleMapLabel =
          Label.parseAbsoluteUnchecked(ccCompilationContextInfo.getCppModuleMap().getName());
      Collection<Artifact> modules =
          createModuleAction(result, ccCompilationContextInfo.getCppModuleMap());
      if (featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULE_CODEGEN)) {
        for (Artifact module : modules) {
          // TODO(djasper): Investigate whether we need to use a label separate from that of the
          // module map. It is used for per-file-copts.
          createModuleCodegenAction(result, moduleMapLabel, module);
        }
      }
    } else if (ccCompilationContextInfo.getVerificationModuleMap() != null) {
      Collection<Artifact> modules =
          createModuleAction(result, ccCompilationContextInfo.getVerificationModuleMap());
      for (Artifact module : modules) {
        result.addHeaderTokenFile(module);
      }
    }

    ImmutableMap<Artifact, String> outputNameMap = null;
    if (cppConfiguration.shortenObjFilePath()) {
      String outputNamePrefixDir = null;
      // purpose is only used by objc rules, it ends with either "_non_objc_arc" or "_objc_arc".
      // Here we use it to distinguish arc and non-arc compilation.
      if (purpose != null) {
        outputNamePrefixDir = purpose.endsWith("_non_objc_arc") ? "non_arc" : "arc";
      }
      outputNameMap = calculateOutputNameMapByType(compilationUnitSources, outputNamePrefixDir);
    }

    for (CppSource source : compilationUnitSources) {
      Artifact sourceArtifact = source.getSource();
      Label sourceLabel = source.getLabel();
      CppCompileActionBuilder builder = initializeCompileAction(sourceArtifact);

      builder
          .setSemantics(semantics)
          .addMandatoryInputs(compilationMandatoryInputs)
          .addAdditionalIncludeScanningRoots(additionalIncludeScanningRoots);

      boolean bitcodeOutput =
          featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
              && CppFileTypes.LTO_SOURCE.matches(sourceArtifact.getFilename());

      String outputName = cppConfiguration.shortenObjFilePath()
          ? outputNameMap.get(sourceArtifact)
          : FileSystemUtils.removeExtension(sourceArtifact.getRootRelativePath()).getPathString();

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
                ccCompilationContextInfo.getCppModuleMap(),
                /* addObject= */ true,
                isCodeCoverageEnabled(),
                // The source action does not generate dwo when it has bitcode
                // output (since it isn't generating a native object with debug
                // info). In that case the LtoBackendAction will generate the dwo.
                ccToolchain.shouldCreatePerObjectDebugInfo(featureConfiguration) && !bitcodeOutput,
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
                    outputName,
                    builder,
                    ImmutableList.of(
                        ArtifactCategory.GENERATED_HEADER, ArtifactCategory.PROCESSED_HEADER),
                    false);
            result.addHeaderTokenFile(headerTokenFile);
            break;
          case SOURCE:
            Artifact objectFile =
                createCompileActionTemplate(
                    env,
                    source,
                    outputName,
                    builder,
                    ImmutableList.of(ArtifactCategory.OBJECT_FILE),
                    false);
            result.addObjectFile(objectFile);

            if (generatePicAction) {
              Artifact picObjectFile =
                  createCompileActionTemplate(
                      env,
                      source,
                      outputName,
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
      String outputName,
      CppCompileActionBuilder builder,
      Iterable<ArtifactCategory> outputCategories,
      boolean usePic) {
    SpecialArtifact sourceArtifact = (SpecialArtifact) source.getSource();
    SpecialArtifact outputFiles =
        CppHelper.getCompileOutputTreeArtifact(ruleContext, sourceArtifact, outputName, usePic);
    // TODO(rduan): Dotd file output is not supported yet.
    builder.setOutputs(outputFiles, /* dotdFile= */ null);
    builder.setVariables(
        setupCompileBuildVariables(
            builder,
            /* sourceLabel= */ null,
            /* outputName= */ null,
            usePic,
            /* ccRelativeName= */ null,
            /* autoFdoImportPath= */ null,
            ccCompilationContextInfo.getCppModuleMap(),
            /* gcnoFile= */ null,
            /* dwoFile= */ null,
            /* ltoIndexingFile= */ null,
            /* additionalBuildVariables= */ ImmutableMap.of()));
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

  /**
   * Return flags that were specified on the Blaze command line. Take the filetype of sourceFilename
   * into account.
   */
  public static ImmutableList<String> getCoptsFromOptions(
      CppConfiguration config, String sourceFilename) {
    ImmutableList.Builder<String> flagsBuilder = ImmutableList.builder();

    flagsBuilder.addAll(config.getCopts());

    if (CppFileTypes.C_SOURCE.matches(sourceFilename)) {
      flagsBuilder.addAll(config.getCOptions());
    }

    if (CppFileTypes.CPP_SOURCE.matches(sourceFilename)
        || CppFileTypes.CPP_HEADER.matches(sourceFilename)
        || CppFileTypes.CPP_MODULE_MAP.matches(sourceFilename)
        || CppFileTypes.CLIF_INPUT_PROTO.matches(sourceFilename)) {
      flagsBuilder.addAll(config.getCxxopts());
    }

    return flagsBuilder.build();
  }

  private Variables setupCompileBuildVariables(
      CppCompileActionBuilder builder,
      Label sourceLabel,
      String outputName,
      boolean usePic,
      PathFragment ccRelativeName,
      PathFragment autoFdoImportPath,
      CppModuleMap cppModuleMap,
      Artifact gcnoFile,
      Artifact dwoFile,
      Artifact ltoIndexingFile,
      ImmutableMap<String, String> additionalBuildVariables) {
    Artifact sourceFile = builder.getSourceFile();
    ImmutableList.Builder<String> userCompileFlags = ImmutableList.builder();
    userCompileFlags.addAll(getCoptsFromOptions(cppConfiguration, sourceFile.getExecPathString()));
    userCompileFlags.addAll(copts);
    if (sourceFile != null && sourceLabel != null) {
      userCompileFlags.addAll(collectPerFileCopts(sourceFile, sourceLabel));
    }
    String dotdFileExecPath = null;
    if (isGenerateDotdFile(builder.getSourceFile())) {
      Preconditions.checkNotNull(builder.getDotdFile());
      dotdFileExecPath = builder.getDotdFile().getSafeExecPath().getPathString();
    }
    ImmutableMap.Builder<String, String> allAdditionalBuildVariables = ImmutableMap.builder();
    allAdditionalBuildVariables.putAll(additionalBuildVariables);
    if (ccRelativeName != null) {
      allAdditionalBuildVariables.putAll(
          fdoSupport
              .getFdoSupport()
              .configureCompilation(
                  builder,
                  ruleContext,
                  ccRelativeName,
                  autoFdoImportPath,
                  PathFragment.create(outputName),
                  usePic,
                  featureConfiguration,
                  fdoSupport));
    }
    return CompileBuildVariables.setupVariablesOrReportRuleError(
        ruleContext,
        featureConfiguration,
        ccToolchain,
        sourceFile,
        builder.getOutputFile(),
        gcnoFile,
        dwoFile,
        ltoIndexingFile,
        ImmutableList.of(),
        userCompileFlags.build(),
        cppModuleMap,
        usePic,
        builder.getRealOutputFilePath(),
        CppHelper.getFdoBuildStamp(ruleContext, fdoSupport.getFdoSupport()),
        dotdFileExecPath,
        ImmutableList.copyOf(variablesExtensions),
        allAdditionalBuildVariables.build(),
        ccCompilationContextInfo.getDirectModuleMaps(),
        ccCompilationContextInfo.getIncludeDirs(),
        ccCompilationContextInfo.getQuoteIncludeDirs(),
        ccCompilationContextInfo.getSystemIncludeDirs(),
        ccCompilationContextInfo.getDefines());
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
   * inputs, output and dotd file names, {@code CcCompilationContextInfo} and copts.
   */
  private CppCompileActionBuilder createCompileActionBuilder(Artifact source) {
    CppCompileActionBuilder builder =
        new CppCompileActionBuilder(ruleContext, ccToolchain, configuration);
    builder.setSourceFile(source);
    builder.setCcCompilationContextInfo(ccCompilationContextInfo);
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
    String outputName =
        cppConfiguration.shortenObjFilePath()
            ? module.getRootRelativePath().getBaseName()
            : module.getRootRelativePath().getPathString();

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

    boolean generateDwo = ccToolchain.shouldCreatePerObjectDebugInfo(featureConfiguration);
    Artifact dwoFile = generateDwo ? getDwoFile(builder.getOutputFile()) : null;
    // TODO(tejohnson): Add support for ThinLTO if needed.
    boolean bitcodeOutput =
        featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
            && CppFileTypes.LTO_SOURCE.matches(module.getFilename());
    Preconditions.checkState(!bitcodeOutput);

    builder.setVariables(
        setupCompileBuildVariables(
            builder,
            sourceLabel,
            outputName,
            /* usePic= */ pic,
            ccRelativeName,
            module.getExecPath(),
            ccCompilationContextInfo.getCppModuleMap(),
            gcnoFile,
            dwoFile,
            /* ltoIndexingFile= */ null,
            /* additionalBuildVariables= */ ImmutableMap.of()));

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
        .setPicMode(generatePicAction);
    builder.setVariables(
        setupCompileBuildVariables(
            builder,
            sourceLabel,
            /* outputName= */ null,
            generatePicAction,
            /* ccRelativeName= */ null,
            /* autoFdoImportPath= */ null,
            ccCompilationContextInfo.getCppModuleMap(),
            /* gcnoFile= */ null,
            /* dwoFile= */ null,
            /* ltoIndexingFile= */ null,
            /* additionalBuildVariables= */ ImmutableMap.of()));
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
        // The header module(.pcm) is generated at most one file per target,
        // so it's safe to remove module map's package path from its output name.
        cppConfiguration.shortenObjFilePath()
            ? FileSystemUtils.removeExtension(moduleMapArtifact.getRootRelativePath())
                .getBaseName()
            : FileSystemUtils.removeExtension(moduleMapArtifact.getRootRelativePath())
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
      // TODO(bazel-team): we shouldn't be needing this, merging ccCompilationContextInfo with the
      // binary
      // is a superset of necessary information.
      LipoContextProvider lipoProvider =
          Preconditions.checkNotNull(CppHelper.getLipoContextProvider(ruleContext), outputName);
      builder.setCcCompilationContextInfo(
          CcCompilationContextInfo.mergeForLipo(
              lipoProvider.getLipoCcCompilationContextInfo(), ccCompilationContextInfo));
    }
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

      // Create PIC compile actions (same as no-PIC, but use -fPIC and
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

        picBuilder.setVariables(
            setupCompileBuildVariables(
                picBuilder,
                sourceLabel,
                outputName,
                /* usePic= */ true,
                ccRelativeName,
                sourceArtifact.getExecPath(),
                ccCompilationContextInfo.getCppModuleMap(),
                gcnoFile,
                dwoFile,
                ltoIndexingFile,
                /* additionalBuildVariables= */ ImmutableMap.of()));

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

        // Create no-PIC compile actions
        Artifact gcnoFile =
            !CppHelper.isLipoOptimization(cppConfiguration, ccToolchain) && enableCoverage
                ? CppHelper.getCompileOutputArtifact(ruleContext, gcnoFileName, configuration)
                : null;

        Artifact noPicDwoFile = generateDwo ? getDwoFile(noPicOutputFile) : null;
        Artifact ltoIndexingFile =
            bitcodeOutput ? getLtoIndexingFile(builder.getOutputFile()) : null;

        builder.setVariables(
            setupCompileBuildVariables(
                builder,
                sourceLabel,
                outputName,
                /* usePic= */ false,
                ccRelativeName,
                sourceArtifact.getExecPath(),
                cppModuleMap,
                gcnoFile,
                noPicDwoFile,
                ltoIndexingFile,
                /* additionalBuildVariables= */ ImmutableMap.of()));

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

    builder.setVariables(
        setupCompileBuildVariables(
            builder,
            sourceLabel,
            outputName,
            usePic,
            ccRelativeName,
            execPath,
            ccCompilationContextInfo.getCppModuleMap(),
            /* gcnoFile= */ null,
            /* dwoFile= */ null,
            /* ltoIndexingFile= */ null,
            /* additionalBuildVariables= */ ImmutableMap.of()));
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
          CcCompilationInfo ccCompilationInfo = dep.get(CcCompilationInfo.PROVIDER);
          if (ccCompilationInfo != null
              && ccCompilationInfo.getCcCompilationContextInfo() != null
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
    dBuilder.setVariables(
        setupCompileBuildVariables(
            dBuilder,
            sourceLabel,
            outputName,
            usePic,
            ccRelativeName,
            source.getExecPath(),
            ccCompilationContextInfo.getCppModuleMap(),
            /* gcnoFile= */ null,
            /* dwoFile= */ null,
            /* ltoIndexingFile= */ null,
            ImmutableMap.of(
                CompileBuildVariables.OUTPUT_PREPROCESS_FILE.getVariableName(),
                dBuilder.getRealOutputFilePath().getSafePathString())));
    semantics.finalizeCompileActionBuilder(ruleContext, dBuilder);
    CppCompileAction dAction = dBuilder.buildOrThrowRuleError(ruleContext);
    ruleContext.registerAction(dAction);

    CppCompileActionBuilder sdBuilder = new CppCompileActionBuilder(builder);
    sdBuilder.setOutputs(
        ruleContext, ArtifactCategory.GENERATED_ASSEMBLY, outputArtifactNameBase, generateDotd);
    sdBuilder.setVariables(
        setupCompileBuildVariables(
            sdBuilder,
            sourceLabel,
            outputName,
            usePic,
            ccRelativeName,
            source.getExecPath(),
            ccCompilationContextInfo.getCppModuleMap(),
            /* gcnoFile= */ null,
            /* dwoFile= */ null,
            /* ltoIndexingFile= */ null,
            ImmutableMap.of(
                CompileBuildVariables.OUTPUT_ASSEMBLY_FILE.getVariableName(),
                sdBuilder.getRealOutputFilePath().getSafePathString())));
    semantics.finalizeCompileActionBuilder(ruleContext, sdBuilder);
    CppCompileAction sdAction = sdBuilder.buildOrThrowRuleError(ruleContext);
    ruleContext.registerAction(sdAction);

    return ImmutableList.of(dAction.getOutputFile(), sdAction.getOutputFile());
  }
}
