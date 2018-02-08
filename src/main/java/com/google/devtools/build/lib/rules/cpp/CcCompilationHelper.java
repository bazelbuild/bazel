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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.LanguageDependentFragment;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMapBuilder;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CoptsFilter;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
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
 * lower-level APIs in CppHelper and CppModel.
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
  private boolean useDeps = true;
  private boolean generateModuleMap = true;
  private String purpose = null;
  private boolean generateNoPic = true;

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

    CppModel model = initializeCppModel();
    CppCompilationContext cppCompilationContext = initializeCppCompilationContext(model);
    model.setContext(cppCompilationContext);

    boolean compileHeaderModules = featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULES);
    Preconditions.checkState(
        !compileHeaderModules || cppCompilationContext.getCppModuleMap() != null,
        "All cc rules must support module maps.");

    // Create compile actions (both PIC and non-PIC).
    CcCompilationOutputs ccOutputs = model.createCcCompileActions();
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

  /** Creates the C/C++ compilation action creator. */
  private CppModel initializeCppModel() {
    return new CppModel(
            ruleContext, semantics, ccToolchain, fdoSupport, configuration, copts, coptsFilter)
        .addCompilationUnitSources(compilationUnitSources)
        .addCompilationMandatoryInputs(compilationMandatoryInputs)
        .addAdditionalIncludeScanningRoots(additionalIncludeScanningRoots)
        .setFake(fake)
        .setGenerateNoPic(generateNoPic)
        // Note: this doesn't actually save the temps, it just makes the CppModel use the
        // configurations --save_temps setting to decide whether to actually save the temps.
        .setSaveTemps(true)
        .setFeatureConfiguration(featureConfiguration)
        .addVariablesExtension(variablesExtensions);
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

  /** Creates context for cc compile action from generated inputs. */
  public CppCompilationContext initializeCppCompilationContext() {
    return initializeCppCompilationContext(initializeCppModel());
  }

  /**
   * Create context for cc compile action from generated inputs.
   *
   * <p>TODO(plf): Try to pull out CppCompilationContext building out of this class.
   */
  private CppCompilationContext initializeCppCompilationContext(CppModel model) {
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
      if (model.getGeneratesPicHeaderModule()) {
        contextBuilder.setPicHeaderModule(model.getPicHeaderModule(cppModuleMap.getArtifact()));
      }
      if (model.getGeneratesNoPicHeaderModule()) {
        contextBuilder.setHeaderModule(model.getHeaderModule(cppModuleMap.getArtifact()));
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

  static NestedSet<Artifact> collectHeaderTokens(
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
}
