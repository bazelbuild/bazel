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

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.LanguageDependentFragment;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.Staticness;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * A class to create C/C++ compile and link actions in a way that is consistent with cc_library.
 * Rules that generate source files and emulate cc_library on top of that should use this class
 * instead of the lower-level APIs in CppHelper and CppModel.
 *
 * <p>Rules that want to use this class are required to have implicit dependencies on the
 * toolchain, the STL, the lipo context, and so on. Optionally, they can also have copts,
 * and malloc attributes, but note that these require explicit calls to the corresponding setter
 * methods.
 */
public final class CcLibraryHelper {
  /**
   * Similar to {@code OutputGroupProvider.HIDDEN_TOP_LEVEL}, but specific to header token files.
   */
  public static final String HIDDEN_HEADER_TOKENS =
      OutputGroupProvider.HIDDEN_OUTPUT_GROUP_PREFIX
          + "hidden_header_tokens"
          + OutputGroupProvider.INTERNAL_SUFFIX;

  /**
   * A group of source file types and action names for builds controlled by CcLibraryHelper.
   * Determines what file types CcLibraryHelper considers sources and what action configs are
   * configured in the CROSSTOOL.
   */
  public static enum SourceCategory {
    CC(
        FileTypeSet.of(
            CppFileTypes.CPP_SOURCE,
            CppFileTypes.CPP_HEADER,
            CppFileTypes.C_SOURCE,
            CppFileTypes.ASSEMBLER,
            CppFileTypes.ASSEMBLER_WITH_C_PREPROCESSOR),
        ImmutableSet.<String>of(
            CppCompileAction.C_COMPILE,
            CppCompileAction.CPP_COMPILE,
            CppCompileAction.CPP_HEADER_PARSING,
            CppCompileAction.CPP_HEADER_PREPROCESSING,
            CppCompileAction.CPP_MODULE_COMPILE,
            CppCompileAction.ASSEMBLE,
            CppCompileAction.PREPROCESS_ASSEMBLE,
            CppCompileAction.CLIF_MATCH,
            Link.LinkTargetType.STATIC_LIBRARY.getActionName(),
            // We need to create pic-specific actions for link actions, as they will produce
            // differently named outputs.
            Link.LinkTargetType.PIC_STATIC_LIBRARY.getActionName(),
            Link.LinkTargetType.INTERFACE_DYNAMIC_LIBRARY.getActionName(),
            Link.LinkTargetType.DYNAMIC_LIBRARY.getActionName(),
            Link.LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY.getActionName(),
            Link.LinkTargetType.ALWAYS_LINK_PIC_STATIC_LIBRARY.getActionName(),
            Link.LinkTargetType.EXECUTABLE.getActionName())),
    CC_AND_OBJC(
        FileTypeSet.of(
            CppFileTypes.CPP_SOURCE,
            CppFileTypes.CPP_HEADER,
            CppFileTypes.OBJC_SOURCE,
            CppFileTypes.OBJCPP_SOURCE,
            CppFileTypes.C_SOURCE,
            CppFileTypes.ASSEMBLER,
            CppFileTypes.ASSEMBLER_WITH_C_PREPROCESSOR),
        ImmutableSet.<String>of(
            CppCompileAction.C_COMPILE,
            CppCompileAction.CPP_COMPILE,
            CppCompileAction.OBJC_COMPILE,
            CppCompileAction.OBJCPP_COMPILE,
            CppCompileAction.CPP_HEADER_PARSING,
            CppCompileAction.CPP_HEADER_PREPROCESSING,
            CppCompileAction.ASSEMBLE,
            CppCompileAction.PREPROCESS_ASSEMBLE,
            Link.LinkTargetType.STATIC_LIBRARY.getActionName(),
            // We need to create pic-specific actions for link actions, as they will produce
            // differently named outputs.
            Link.LinkTargetType.PIC_STATIC_LIBRARY.getActionName(),
            Link.LinkTargetType.INTERFACE_DYNAMIC_LIBRARY.getActionName(),
            Link.LinkTargetType.DYNAMIC_LIBRARY.getActionName(),
            Link.LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY.getActionName(),
            Link.LinkTargetType.ALWAYS_LINK_PIC_STATIC_LIBRARY.getActionName(),
            Link.LinkTargetType.EXECUTABLE.getActionName()));

    private final FileTypeSet sourceTypeSet;
    private final Set<String> actionConfigSet;

    private SourceCategory(FileTypeSet sourceTypeSet, Set<String> actionConfigSet) {
      this.sourceTypeSet = sourceTypeSet;
      this.actionConfigSet = actionConfigSet;
    }

    /**
     * Returns the set of file types that are valid for this category.
     */
    public FileTypeSet getSourceTypes() {
      return sourceTypeSet;
    }

    /** Returns the set of enabled actions for this category. */
    public Set<String> getActionConfigSet() {
      return actionConfigSet;
    }
  }

  /** Function for extracting module maps from CppCompilationDependencies. */
  public static final Function<TransitiveInfoCollection, CppModuleMap> CPP_DEPS_TO_MODULES =
    new Function<TransitiveInfoCollection, CppModuleMap>() {
      @Override
      @Nullable
      public CppModuleMap apply(TransitiveInfoCollection dep) {
        CppCompilationContext context = dep.getProvider(CppCompilationContext.class);
        return context == null ? null : context.getCppModuleMap();
      }
    };

  /**
   * Contains the providers as well as the compilation and linking outputs, and the compilation
   * context.
   */
  public static final class Info {
    private final TransitiveInfoProviderMap.Builder providers;
    private final ImmutableMap<String, NestedSet<Artifact>> outputGroups;
    private final CcCompilationOutputs compilationOutputs;
    private final CcLinkingOutputs linkingOutputs;
    private final CcLinkingOutputs linkingOutputsExcludingPrecompiledLibraries;
    private final CppCompilationContext context;

    private Info(
        TransitiveInfoProviderMap providers,
        Map<String, NestedSet<Artifact>> outputGroups,
        CcCompilationOutputs compilationOutputs,
        CcLinkingOutputs linkingOutputs,
        CcLinkingOutputs linkingOutputsExcludingPrecompiledLibraries,
        CppCompilationContext context) {
      this.providers = providers.toBuilder();
      this.outputGroups = ImmutableMap.copyOf(outputGroups);
      this.compilationOutputs = compilationOutputs;
      this.linkingOutputs = linkingOutputs;
      this.linkingOutputsExcludingPrecompiledLibraries =
          linkingOutputsExcludingPrecompiledLibraries;
      this.context = context;
    }

    public TransitiveInfoProviderMap getProviders() {
      return providers.build();
    }

    public ImmutableMap<String, NestedSet<Artifact>> getOutputGroups() {
      return outputGroups;
    }

    public CcCompilationOutputs getCcCompilationOutputs() {
      return compilationOutputs;
    }

    public CcLinkingOutputs getCcLinkingOutputs() {
      return linkingOutputs;
    }

    /**
     * Returns the linking outputs before adding the pre-compiled libraries. Avoid using this -
     * pre-compiled and locally compiled libraries should be treated identically. This method only
     * exists for backwards compatibility.
     */
    public CcLinkingOutputs getCcLinkingOutputsExcludingPrecompiledLibraries() {
      return linkingOutputsExcludingPrecompiledLibraries;
    }

    public CppCompilationContext getCppCompilationContext() {
      return context;
    }

    /**
     * Adds the static, pic-static, and dynamic (both compile-time and execution-time) libraries to
     * the given builder.
     */
    public void addLinkingOutputsTo(NestedSetBuilder<Artifact> filesBuilder) {
      filesBuilder
          .addAll(LinkerInputs.toLibraryArtifacts(linkingOutputs.getStaticLibraries()))
          .addAll(LinkerInputs.toLibraryArtifacts(linkingOutputs.getPicStaticLibraries()))
          .addAll(LinkerInputs.toNonSolibArtifacts(linkingOutputs.getDynamicLibraries()))
          .addAll(LinkerInputs.toNonSolibArtifacts(linkingOutputs.getExecutionDynamicLibraries()));
    }
  }

  private final RuleContext ruleContext;
  private final CppSemantics semantics;
  private final BuildConfiguration configuration;

  private final List<Artifact> publicHeaders = new ArrayList<>();
  private final List<Artifact> nonModuleMapHeaders = new ArrayList<>();
  private final List<Artifact> publicTextualHeaders = new ArrayList<>();
  private final List<Artifact> privateHeaders = new ArrayList<>();
  private final List<PathFragment> additionalExportedHeaders = new ArrayList<>();
  private final List<CppModuleMap> additionalCppModuleMaps = new ArrayList<>();
  private final Set<CppSource> compilationUnitSources = new LinkedHashSet<>();
  private final List<Artifact> objectFiles = new ArrayList<>();
  private final List<Artifact> picObjectFiles = new ArrayList<>();
  private final List<Artifact> nonCodeLinkerInputs = new ArrayList<>();
  private final List<String> copts = new ArrayList<>();
  private final List<String> linkopts = new ArrayList<>();
  @Nullable private Pattern nocopts;
  private final Set<String> defines = new LinkedHashSet<>();
  private final List<TransitiveInfoCollection> deps = new ArrayList<>();
  private final List<CppCompilationContext> depContexts = new ArrayList<>();
  private final NestedSetBuilder<Artifact> linkstamps = NestedSetBuilder.stableOrder();
  private final List<PathFragment> looseIncludeDirs = new ArrayList<>();
  private final List<PathFragment> systemIncludeDirs = new ArrayList<>();
  private final List<PathFragment> includeDirs = new ArrayList<>();
  private final List<Artifact> linkActionInputs = new ArrayList<>();

  @Nullable private Artifact dynamicLibrary;
  private LinkTargetType linkType = LinkTargetType.STATIC_LIBRARY;
  private HeadersCheckingMode headersCheckingMode = HeadersCheckingMode.LOOSE;
  private boolean neverlink;
  private boolean fake;

  private final List<LibraryToLink> staticLibraries = new ArrayList<>();
  private final List<LibraryToLink> picStaticLibraries = new ArrayList<>();
  private final List<LibraryToLink> dynamicLibraries = new ArrayList<>();

  private boolean emitLinkActions = true;
  private boolean emitLinkActionsIfEmpty;
  private boolean emitCcNativeLibrariesProvider;
  private boolean emitCcSpecificLinkParamsProvider;
  private boolean emitInterfaceSharedObjects;
  private boolean emitDynamicLibrary = true;
  private boolean checkDepsGenerateCpp = true;
  private boolean emitCompileProviders;
  private final SourceCategory sourceCategory;
  private List<VariablesExtension> variablesExtensions = new ArrayList<>();
  @Nullable private CppModuleMap cppModuleMap;
  private boolean propagateModuleMapToCompileAction = true;

  private final FeatureConfiguration featureConfiguration;
  private CcToolchainProvider ccToolchain;
  private final FdoSupportProvider fdoSupport;
  private String linkedArtifactNameSuffix = "";

  /**
   * Creates a CcLibraryHelper.
   *
   * @param ruleContext  the RuleContext for the rule being built
   * @param semantics  CppSemantics for the build
   * @param featureConfiguration  activated features and action configs for the build
   * @param sourceCatagory  the candidate source types for the build
   * @param ccToolchain the C++ toolchain provider for the build
   * @param fdoSupport the C++ FDO optimization support provider for the build
   */
  public CcLibraryHelper(
      RuleContext ruleContext,
      CppSemantics semantics,
      FeatureConfiguration featureConfiguration,
      SourceCategory sourceCatagory,
      CcToolchainProvider ccToolchain,
      FdoSupportProvider fdoSupport) {
    this(ruleContext, semantics, featureConfiguration, sourceCatagory, ccToolchain, fdoSupport,
        ruleContext.getConfiguration());
  }

  /**
   * Creates a CcLibraryHelper that outputs artifacts in a given configuration.
   *
   * @param ruleContext  the RuleContext for the rule being built
   * @param semantics  CppSemantics for the build
   * @param featureConfiguration  activated features and action configs for the build
   * @param sourceCatagory  the candidate source types for the build
   * @param ccToolchain the C++ toolchain provider for the build
   * @param fdoSupport the C++ FDO optimization support provider for the build
   * @param configuration the configuration that gives the directory of output artifacts
   */
  public CcLibraryHelper(
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
   * Creates a CcLibraryHelper for cpp source files.
   *
   * @param ruleContext the RuleContext for the rule being built
   * @param semantics CppSemantics for the build
   * @param featureConfiguration activated features and action configs for the build
   * @param ccToolchain the C++ toolchain provider for the build
   * @param fdoSupport the C++ FDO optimization support provider for the build
   */
  public CcLibraryHelper(
      RuleContext ruleContext, CppSemantics semantics, FeatureConfiguration featureConfiguration,
      CcToolchainProvider ccToolchain, FdoSupportProvider fdoSupport) {
    this(ruleContext, semantics, featureConfiguration, SourceCategory.CC, ccToolchain, fdoSupport);
  }

  /** Sets fields that overlap for cc_library and cc_binary rules. */
  public CcLibraryHelper fromCommon(CcCommon common) {
    this
        .addCopts(common.getCopts())
        .addDefines(common.getDefines())
        .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
        .addLooseIncludeDirs(common.getLooseIncludeDirs())
        .addNonCodeLinkerInputs(common.getLinkerScripts())
        .addSystemIncludeDirs(common.getSystemIncludeDirs())
        .setNoCopts(common.getNoCopts())
        .setHeadersCheckingMode(semantics.determineHeadersCheckingMode(ruleContext));
    return this;
  }

  /**
   * Adds {@code headers} as public header files. These files will be made visible to dependent
   * rules. They may be parsed/preprocessed or compiled into a header module depending on the
   * configuration.
   */
  public CcLibraryHelper addPublicHeaders(Collection<Artifact> headers) {
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
  public CcLibraryHelper addPublicHeaders(Artifact... headers) {
    addPublicHeaders(Arrays.asList(headers));
    return this;
  }

  /**
   * Adds {@code headers} as public header files. These files will be made visible to dependent
   * rules. They may be parsed/preprocessed or compiled into a header module depending on the
   * configuration.
   */
  public CcLibraryHelper addPublicHeaders(Iterable<Pair<Artifact, Label>> headers) {
    for (Pair<Artifact, Label> header : headers) {
      addHeader(header.first, header.second);
    }
    return this;
  }

  /**
   * Add the corresponding files as public header files, i.e., these files will not be compiled, but
   * are made visible as includes to dependent rules in module maps.
   */
  public CcLibraryHelper addAdditionalExportedHeaders(
      Iterable<PathFragment> additionalExportedHeaders) {
    Iterables.addAll(this.additionalExportedHeaders, additionalExportedHeaders);
    return this;
  }

  /**
   * Add the corresponding files as public textual header files. These files will not be compiled
   * into a target's header module, but will be made visible as textual includes to dependent rules.
   */
  public CcLibraryHelper addPublicTextualHeaders(Iterable<Artifact> textualHeaders) {
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
  public CcLibraryHelper addSources(
      Collection<Artifact> sources, Map<String, String> buildVariables) {
    Preconditions.checkNotNull(buildVariables);
    for (Artifact source : sources) {
      addSource(source, ruleContext.getLabel(), buildVariables);
    }
    return this;
  }

  /**
   * Add the corresponding files as source files. These may also be header files, in which case they
   * will not be compiled, but also not made visible as includes to dependent rules. The given
   * sources will be built without extra, source-specific build variables.
   */
  public CcLibraryHelper addSources(Collection<Artifact> sources) {
    addSources(sources, ImmutableMap.<String, String>of());
    return this;
  }

  /**
   * Add the corresponding files as source files. These may also be header files, in which case they
   * will not be compiled, but also not made visible as includes to dependent rules. The given
   * sources will be built without extra, source-specific build variables.
   */
  public CcLibraryHelper addSources(Iterable<Pair<Artifact, Label>> sources) {
    for (Pair<Artifact, Label> source : sources) {
      addSource(source.first, source.second, ImmutableMap.<String, String>of());
    }
    return this;
  }

  /**
   * Add the corresponding files as source files. These may also be header files, in which case they
   * will not be compiled, but also not made visible as includes to dependent rules. The given
   * sources will be built without extra, source-specific build variables.
   */
  public CcLibraryHelper addSources(Artifact... sources) {
    return addSources(Arrays.asList(sources));
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
    compilationUnitSources.add(
        CppSource.create(header, label, ImmutableMap.<String, String>of(), CppSource.Type.HEADER));
  }

  /** Adds a header to {@code publicHeaders}, but not to this target's module map. */
  public CcLibraryHelper addNonModuleMapHeader(Artifact header) {
    Preconditions.checkNotNull(header);
    nonModuleMapHeaders.add(header);
    return this;
  }

  /**
   * Adds a source to {@code compilationUnitSources} if it is a compiled file type (including
   * parsed/preprocessed header) and to {@code privateHeaders} if it is a header. The given build
   * variables will be added to those used for compiling this source.
   */
  private void addSource(Artifact source, Label label, Map<String, String> buildVariables) {
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
    compilationUnitSources.add(CppSource.create(source, label, buildVariables, type));
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
  public CcLibraryHelper addObjectFiles(Iterable<Artifact> objectFiles) {
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
  public CcLibraryHelper addPicObjectFiles(Iterable<Artifact> picObjectFiles) {
    for (Artifact objectFile : objectFiles) {
      Preconditions.checkArgument(Link.OBJECT_FILETYPES.matches(objectFile.getFilename()));
    }
    Iterables.addAll(this.picObjectFiles, picObjectFiles);
    return this;
  }

  /**
   * Adds the corresponding non-code files as linker inputs.
   */
  public CcLibraryHelper addNonCodeLinkerInputs(Iterable<Artifact> nonCodeLinkerInputs) {
    for (Artifact nonCodeLinkerInput : nonCodeLinkerInputs) {
      String basename = nonCodeLinkerInput.getFilename();
      Preconditions.checkArgument(!Link.OBJECT_FILETYPES.matches(basename));
      Preconditions.checkArgument(!Link.ARCHIVE_LIBRARY_FILETYPES.matches(basename));
      Preconditions.checkArgument(!Link.SHARED_LIBRARY_FILETYPES.matches(basename));
      this.nonCodeLinkerInputs.add(nonCodeLinkerInput);
    }

    return this;
  }

  /**
   * Add the corresponding files as static libraries into the linker outputs (i.e., after the linker
   * action) - this makes them available for linking to binary rules that depend on this rule.
   */
  public CcLibraryHelper addStaticLibraries(Iterable<LibraryToLink> libraries) {
    Iterables.addAll(staticLibraries, libraries);
    return this;
  }

  /**
   * Add the corresponding files as static libraries into the linker outputs (i.e., after the linker
   * action) - this makes them available for linking to binary rules that depend on this rule.
   */
  public CcLibraryHelper addPicStaticLibraries(Iterable<LibraryToLink> libraries) {
    Iterables.addAll(picStaticLibraries, libraries);
    return this;
  }

  /**
   * Add the corresponding files as dynamic libraries into the linker outputs (i.e., after the
   * linker action) - this makes them available for linking to binary rules that depend on this
   * rule.
   */
  public CcLibraryHelper addDynamicLibraries(Iterable<LibraryToLink> libraries) {
    Iterables.addAll(dynamicLibraries, libraries);
    return this;
  }

  /**
   * Adds the copts to the compile command line.
   */
  public CcLibraryHelper addCopts(Iterable<String> copts) {
    Iterables.addAll(this.copts, copts);
    return this;
  }

  /**
   * Sets a pattern that is used to filter copts; set to {@code null} for no filtering.
   */
  public CcLibraryHelper setNoCopts(@Nullable Pattern nocopts) {
    this.nocopts = nocopts;
    return this;
  }

  /**
   * Adds the given options as linker options to the link command.
   */
  public CcLibraryHelper addLinkopts(Iterable<String> linkopts) {
    Iterables.addAll(this.linkopts, linkopts);
    return this;
  }

  /**
   * Adds the given defines to the compiler command line.
   */
  public CcLibraryHelper addDefines(Iterable<String> defines) {
    Iterables.addAll(this.defines, defines);
    return this;
  }

  /**
   * Adds the given targets as dependencies - this can include explicit dependencies on other
   * rules (like from a "deps" attribute) and also implicit dependencies on runtime libraries.
   */
  public CcLibraryHelper addDeps(Iterable<? extends TransitiveInfoCollection> deps) {
    for (TransitiveInfoCollection dep : deps) {
      this.deps.add(dep);
    }
    return this;
  }

  public CcLibraryHelper addDepContext(CppCompilationContext dep) {
    this.depContexts.add(Preconditions.checkNotNull(dep));
    return this;
  }

  /**
   * Adds the given linkstamps. Note that linkstamps are usually not compiled at the library level,
   * but only in the dependent binary rules.
   */
  public CcLibraryHelper addLinkstamps(Iterable<? extends TransitiveInfoCollection> linkstamps) {
    for (TransitiveInfoCollection linkstamp : linkstamps) {
      this.linkstamps.addTransitive(linkstamp.getProvider(FileProvider.class).getFilesToBuild());
    }
    return this;
  }

  /**
   * Adds the given precompiled files to this helper. Shared and static libraries are added as
   * compilation prerequisites, and object files are added as pic or non-pic object files
   * respectively.
   */
  public CcLibraryHelper addPrecompiledFiles(PrecompiledFiles precompiledFiles) {
    addObjectFiles(precompiledFiles.getObjectFiles(false));
    addPicObjectFiles(precompiledFiles.getObjectFiles(true));
    return this;
  }

  /**
   * Adds the given directories to the loose include directories that are only allowed to be
   * referenced when headers checking is {@link HeadersCheckingMode#LOOSE} or {@link
   * HeadersCheckingMode#WARN}.
   */
  public CcLibraryHelper addLooseIncludeDirs(Iterable<PathFragment> looseIncludeDirs) {
    Iterables.addAll(this.looseIncludeDirs, looseIncludeDirs);
    return this;
  }

  /**
   * Adds the given directories to the system include directories (they are passed with {@code
   * "-isystem"} to the compiler); these are also passed to dependent rules.
   */
  public CcLibraryHelper addSystemIncludeDirs(Iterable<PathFragment> systemIncludeDirs) {
    Iterables.addAll(this.systemIncludeDirs, systemIncludeDirs);
    return this;
  }

  /**
   * Adds the given directories to the include directories (they are passed with {@code "-I"} to
   * the compiler); these are also passed to dependent rules.
   */
  public CcLibraryHelper addIncludeDirs(Iterable<PathFragment> includeDirs) {
    Iterables.addAll(this.includeDirs, includeDirs);
    return this;
  }

  /** Adds the given artifact to the input of any generated link actions. */
  public CcLibraryHelper addLinkActionInput(Artifact input) {
    Preconditions.checkNotNull(input);
    this.linkActionInputs.add(input);
    return this;
  }

  /**
   * Adds a variableExtension to template the crosstool.
   */
  public CcLibraryHelper addVariableExtension(VariablesExtension variableExtension) {
    Preconditions.checkNotNull(variableExtension);
    this.variablesExtensions.add(variableExtension);
    return this;
  }

  /**
   * Sets a module map artifact for this build.
   */
  public CcLibraryHelper setCppModuleMap(CppModuleMap cppModuleMap) {
    Preconditions.checkNotNull(cppModuleMap);
    this.cppModuleMap = cppModuleMap;
    return this;
  }

  /**
   * Signals that this target's module map should not be an input to c++ compile actions.
   */
  public CcLibraryHelper setPropagateModuleMapToCompileAction(boolean propagatesModuleMap) {
    this.propagateModuleMapToCompileAction = propagatesModuleMap;
    return this;
  }

  /**
   * Overrides the path for the generated dynamic library - this should only be called if the
   * dynamic library is an implicit or explicit output of the rule, i.e., if it is accessible by
   * name from other rules in the same package. Set to {@code null} to use the default computation.
   */
  public CcLibraryHelper setDynamicLibrary(@Nullable Artifact dynamicLibrary) {
    this.dynamicLibrary = dynamicLibrary;
    return this;
  }

  /**
   * Marks the output of this rule as alwayslink, i.e., the corresponding symbols will be retained
   * by the linker even if they are not otherwise used. This is useful for libraries that register
   * themselves somewhere during initialization.
   *
   * <p>This only sets the link type (see {@link #setLinkType}), either to a static library or to
   * an alwayslink static library (blaze uses a different file extension to signal alwayslink to
   * downstream code).
   */
  public CcLibraryHelper setAlwayslink(boolean alwayslink) {
    linkType = alwayslink
        ? LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY
        : LinkTargetType.STATIC_LIBRARY;
    return this;
  }

  /**
   * Directly set the link type. This can be used instead of {@link #setAlwayslink}. Setting
   * anything other than a static link causes this class to skip the link action creation.
   */
  public CcLibraryHelper setLinkType(LinkTargetType linkType) {
    this.linkType = Preconditions.checkNotNull(linkType);
    return this;
  }

  /**
   * Marks the resulting code as neverlink, i.e., the code will not be linked into dependent
   * libraries or binaries - the header files are still available.
   */
  public CcLibraryHelper setNeverLink(boolean neverlink) {
    this.neverlink = neverlink;
    return this;
  }

  /**
   * Sets the given headers checking mode. The default is {@link HeadersCheckingMode#LOOSE}.
   */
  public CcLibraryHelper setHeadersCheckingMode(HeadersCheckingMode headersCheckingMode) {
    this.headersCheckingMode = Preconditions.checkNotNull(headersCheckingMode);
    return this;
  }

  /**
   * Marks the resulting code as fake, i.e., the code will not actually be compiled or linked, but
   * instead, the compile command is written to a file and added to the runfiles. This is currently
   * used for non-compilation tests. Unfortunately, the design is problematic, so please don't add
   * any further uses.
   */
  public CcLibraryHelper setFake(boolean fake) {
    this.fake = fake;
    return this;
  }

  /*
   * Adds a suffix for paths of linked artifacts. Normally their paths are derived solely from rule
   * labels. In the case of multiple callers (e.g., aspects) acting on a single rule, they may
   * generate the same linked artifact and therefore lead to artifact conflicts. This method
   * provides a way to avoid this artifact conflict by allowing different callers acting on the same
   * rule to provide a suffix that will be used to scope their own linked artifacts.
   */
  public CcLibraryHelper setLinkedArtifactNameSuffix(String suffix) {
    this.linkedArtifactNameSuffix = Preconditions.checkNotNull(suffix);
    return this;
  }

  /**
   * This adds the {@link CcNativeLibraryProvider} to the providers created by this class.
   */
  public CcLibraryHelper enableCcNativeLibrariesProvider() {
    this.emitCcNativeLibrariesProvider = true;
    return this;
  }

  /**
   * This adds the {@link CcSpecificLinkParamsProvider} to the providers created by this class.
   * Otherwise the result will contain an instance of {@link CcLinkParamsProvider}.
   */
  public CcLibraryHelper enableCcSpecificLinkParamsProvider() {
    this.emitCcSpecificLinkParamsProvider = true;
    return this;
  }

  public CcLibraryHelper setEmitLinkActions(boolean emitLinkActions) {
    this.emitLinkActions = emitLinkActions;
    return this;
  }

  /**
   * Enables or disables generation of link actions if there are no object files. Some rules declare
   * a <code>.a</code> or <code>.so</code> implicit output, which requires that these files are
   * created even if there are no object files, so be careful when calling this.
   *
   * <p>This is disabled by default.
   */
  public CcLibraryHelper setGenerateLinkActionsIfEmpty(boolean emitLinkActionsIfEmpty) {
    this.emitLinkActionsIfEmpty = emitLinkActionsIfEmpty;
    return this;
  }

  /**
   * Enables the optional generation of interface dynamic libraries - this is only used when the
   * linker generates a dynamic library, and only if the crosstool supports it. The default is not
   * to generate interface dynamic libraries.
   */
  public CcLibraryHelper enableInterfaceSharedObjects() {
    this.emitInterfaceSharedObjects = true;
    return this;
  }

  /**
   * This enables or disables the generation of a dynamic library link action. The default is to
   * generate a dynamic library. Note that the selection between dynamic or static linking is
   * performed at the binary rule level.
   */
  public CcLibraryHelper setCreateDynamicLibrary(boolean emitDynamicLibrary) {
    this.emitDynamicLibrary = emitDynamicLibrary;
    return this;
  }

  /**
   * Disables checking that the deps actually are C++ rules. By default, the {@link #build} method
   * uses {@link LanguageDependentFragment.Checker#depSupportsLanguage} to check that all deps
   * provide C++ providers.
   */
  public CcLibraryHelper setCheckDepsGenerateCpp(boolean checkDepsGenerateCpp) {
    this.checkDepsGenerateCpp = checkDepsGenerateCpp;
    return this;
  }

  /**
   * Enables the output of the {@code files_to_compile} and {@code compilation_prerequisites}
   * output groups.
   */
  // TODO(bazel-team): We probably need to adjust this for the multi-language rules.
  public CcLibraryHelper enableCompileProviders() {
    this.emitCompileProviders = true;
    return this;
  }

  /**
   * Create the C++ compile and link actions, and the corresponding C++-related providers.
   *
   * @throws RuleErrorException
   */
  public Info build() throws RuleErrorException, InterruptedException {
    // Fail early if there is no lipo context collector on the rule - otherwise we end up failing
    // in lipo optimization.
    Preconditions.checkState(
        // 'cc_inc_library' rules do not compile, and thus are not affected by LIPO.
        ruleContext.getRule().getRuleClass().equals("cc_inc_library")
            || ruleContext.isAttrDefined(":lipo_context_collector", BuildType.LABEL));

    if (checkDepsGenerateCpp) {
      for (LanguageDependentFragment dep :
          AnalysisUtils.getProviders(deps, LanguageDependentFragment.class)) {
        LanguageDependentFragment.Checker.depSupportsLanguage(
            ruleContext, dep, CppRuleClasses.LANGUAGE);
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
              .addLTOBitcodeFile(ccOutputs.getLtoBitcodeFiles())
              .addObjectFiles(objectFiles)
              .addPicObjectFiles(picObjectFiles)
              .build();
    }

    // Create link actions (only if there are object files or if explicitly requested).
    CcLinkingOutputs ccLinkingOutputs = CcLinkingOutputs.EMPTY;
    if (emitLinkActions && (emitLinkActionsIfEmpty || !ccOutputs.isEmpty())) {
      // On some systems, the linker gives an error message if there are no input files. Even with
      // the check above, this can still happen if there is a .nopic.o or .o files in srcs, but no
      // other files. To fix that, we'd have to check for each link action individually.
      //
      // An additional pre-existing issue is that the header check tokens are dropped if we don't
      // generate any link actions, effectively disabling header checking in some cases.
      if (linkType.staticness() == Staticness.STATIC) {
        // TODO(bazel-team): This can't create the link action for a cc_binary yet.
        ccLinkingOutputs = model.createCcLinkActions(ccOutputs, nonCodeLinkerInputs);
      }
    }
    CcLinkingOutputs originalLinkingOutputs = ccLinkingOutputs;
    if (!(
        staticLibraries.isEmpty() && picStaticLibraries.isEmpty() && dynamicLibraries.isEmpty())) {

      CcLinkingOutputs.Builder newOutputsBuilder = new CcLinkingOutputs.Builder();
      if (!ccOutputs.isEmpty()) {
        // Add the linked outputs of this rule iff we had anything to put in them, but then
        // make sure we're not colliding with some library added from the inputs.
        newOutputsBuilder.merge(originalLinkingOutputs);
        ImmutableSetMultimap<String, LibraryToLink> precompiledLibraryMap =
            CcLinkingOutputs.getLibrariesByIdentifier(
                Iterables.concat(staticLibraries, picStaticLibraries, dynamicLibraries));
        ImmutableSetMultimap<String, LibraryToLink> linkedLibraryMap =
            originalLinkingOutputs.getLibrariesByIdentifier();
        for (String matchingIdentifier :
            Sets.intersection(precompiledLibraryMap.keySet(), linkedLibraryMap.keySet())) {
          Iterable<Artifact> matchingInputLibs =
              LinkerInputs.toNonSolibArtifacts(precompiledLibraryMap.get(matchingIdentifier));
          Iterable<Artifact> matchingOutputLibs =
              LinkerInputs.toNonSolibArtifacts(linkedLibraryMap.get(matchingIdentifier));
          ruleContext.ruleError(
              "Can't put "
                  + Joiner.on(", ")
                      .join(Iterables.transform(matchingInputLibs, FileType.TO_FILENAME))
                  + " into the srcs of a "
                  + ruleContext.getRuleClassNameForLogging()
                  + " with the same name ("
                  + ruleContext.getRule().getName()
                  + ") which also contains other code or objects to link; it shares a name with "
                  + Joiner.on(", ")
                      .join(Iterables.transform(matchingOutputLibs, FileType.TO_FILENAME))
                  + " (output compiled and linked from the non-library sources of this rule), "
                  + "which could cause confusion");
        }
      }

      // Merge the pre-compiled libraries (static & dynamic) into the linker outputs.
      ccLinkingOutputs =
          newOutputsBuilder
              .addStaticLibraries(staticLibraries)
              .addPicStaticLibraries(picStaticLibraries)
              .addDynamicLibraries(dynamicLibraries)
              .addExecutionDynamicLibraries(dynamicLibraries)
              .build();
    }

    DwoArtifactsCollector dwoArtifacts =
        DwoArtifactsCollector.transitiveCollector(
            ruleContext,
            ccOutputs,
            deps, /*generateDwo=*/
            false, /*ltoBackendArtifactsUsePic=*/
            false, /*ltoBackendArtifacts=*/
            ImmutableList.<LTOBackendArtifacts>of());
    Runfiles cppStaticRunfiles = collectCppRunfiles(ccLinkingOutputs, true);
    Runfiles cppSharedRunfiles = collectCppRunfiles(ccLinkingOutputs, false);

    // By very careful when adding new providers here - it can potentially affect a lot of rules.
    // We should consider merging most of these providers into a single provider.
    TransitiveInfoProviderMap.Builder providers =
        TransitiveInfoProviderMap.builder()
            .add(
                new CppRunfilesProvider(cppStaticRunfiles, cppSharedRunfiles),
                cppCompilationContext,
                new CppDebugFileProvider(
                    dwoArtifacts.getDwoArtifacts(), dwoArtifacts.getPicDwoArtifacts()),
                collectTransitiveLipoInfo(ccOutputs));
    Map<String, NestedSet<Artifact>> outputGroups = new TreeMap<>();

    if (shouldAddLinkerOutputArtifacts(ruleContext, ccOutputs)) {
      addLinkerOutputArtifacts(outputGroups, ccOutputs);
    }

    outputGroups.put(OutputGroupProvider.TEMP_FILES, getTemps(ccOutputs));
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    if (emitCompileProviders) {
      boolean isLipoCollector = cppConfiguration.isLipoContextCollector();
      boolean processHeadersInDependencies = cppConfiguration.processHeadersInDependencies();
      boolean usePic = CppHelper.usePic(ruleContext, false);
      outputGroups.put(
          OutputGroupProvider.FILES_TO_COMPILE,
          ccOutputs.getFilesToCompile(isLipoCollector, processHeadersInDependencies, usePic));
      outputGroups.put(OutputGroupProvider.COMPILATION_PREREQUISITES,
          CcCommon.collectCompilationPrerequisites(ruleContext, cppCompilationContext));
    }

    // TODO(bazel-team): Maybe we can infer these from other data at the places where they are
    // used.
    if (emitCcNativeLibrariesProvider) {
      providers.add(new CcNativeLibraryProvider(collectNativeCcLibraries(ccLinkingOutputs)));
    }
    providers.put(
        CcExecutionDynamicLibrariesProvider.class,
        collectExecutionDynamicLibraryArtifacts(ccLinkingOutputs.getExecutionDynamicLibraries()));

    boolean forcePic = cppConfiguration.forcePic();
    if (emitCcSpecificLinkParamsProvider) {
      providers.add(
          new CcSpecificLinkParamsProvider(
              createCcLinkParamsStore(ccLinkingOutputs, cppCompilationContext, forcePic)));
    } else {
      providers.add(
          new CcLinkParamsProvider(
              createCcLinkParamsStore(ccLinkingOutputs, cppCompilationContext, forcePic)));
    }
    return new Info(
        providers.build(),
        outputGroups,
        ccOutputs,
        ccLinkingOutputs,
        originalLinkingOutputs,
        cppCompilationContext);
  }

  /**
   * Returns true if the appropriate attributes for linker output artifacts are defined, and either
   * the compile action produces object files or the build is configured to produce an archive or
   * dynamic library even in the absence of object files.
   */
  private boolean shouldAddLinkerOutputArtifacts(
      RuleContext ruleContext, CcCompilationOutputs ccOutputs) {
    return (ruleContext.attributes().has("alwayslink", Type.BOOLEAN)
        && ruleContext.attributes().has("linkstatic", Type.BOOLEAN)
        && (emitLinkActionsIfEmpty || !ccOutputs.isEmpty()));
  }

  /**
   * Adds linker output artifacts to the given map, to be registered on the configured target as
   * output groups.
   */
  private void addLinkerOutputArtifacts(Map<String, NestedSet<Artifact>> outputGroups,
      CcCompilationOutputs ccOutputs) {
    NestedSetBuilder<Artifact> archiveFile = new NestedSetBuilder<>(Order.STABLE_ORDER);
    NestedSetBuilder<Artifact> dynamicLibrary = new NestedSetBuilder<>(Order.STABLE_ORDER);

    if (ruleContext.attributes().get("alwayslink", Type.BOOLEAN)) {
      archiveFile.add(
          CppHelper.getLinuxLinkedArtifact(
              ruleContext,
              Link.LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY,
              linkedArtifactNameSuffix));
    } else {
      archiveFile.add(
          CppHelper.getLinuxLinkedArtifact(
              ruleContext, Link.LinkTargetType.STATIC_LIBRARY, linkedArtifactNameSuffix));
    }

    if (!ruleContext.attributes().get("linkstatic", Type.BOOLEAN)
        && !ccOutputs.isEmpty()) {
      dynamicLibrary.add(
          CppHelper.getLinuxLinkedArtifact(
              ruleContext, Link.LinkTargetType.DYNAMIC_LIBRARY, linkedArtifactNameSuffix));
    }

    outputGroups.put("archive", archiveFile.build());
    outputGroups.put("dynamic_library", dynamicLibrary.build());
  }

  /**
   * Creates the C/C++ compilation action creator.
   */
  private CppModel initializeCppModel() {
    return new CppModel(ruleContext, semantics, ccToolchain, fdoSupport, configuration)
        .addCompilationUnitSources(compilationUnitSources)
        .addCopts(copts)
        .setLinkTargetType(linkType)
        .setNeverLink(neverlink)
        .addLinkActionInputs(linkActionInputs)
        .setFake(fake)
        .setAllowInterfaceSharedObjects(emitInterfaceSharedObjects)
        .setCreateDynamicLibrary(emitDynamicLibrary)
        // Note: this doesn't actually save the temps, it just makes the CppModel use the
        // configurations --save_temps setting to decide whether to actually save the temps.
        .setSaveTemps(true)
        .setNoCopts(nocopts)
        .setDynamicLibrary(dynamicLibrary)
        .addLinkopts(linkopts)
        .setFeatureConfiguration(featureConfiguration)
        .addVariablesExtension(variablesExtensions)
        .setLinkedArtifactNameSuffix(linkedArtifactNameSuffix);
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

    public ImmutableList<Artifact> getHeaders() {
      return headers;
    }

    public ImmutableList<Artifact> getModuleMapHeaders() {
      return moduleMapHeaders;
    }

    @Nullable
    public PathFragment getVirtualIncludePath() {
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

    PathFragment prefix =
        ruleContext.attributes().isAttributeValueExplicitlySpecified("include_prefix")
            ? new PathFragment(ruleContext.attributes().get("include_prefix", Type.STRING))
            : null;

    PathFragment stripPrefix;
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("strip_include_prefix")) {
      stripPrefix = new PathFragment(
          ruleContext.attributes().get("strip_include_prefix", Type.STRING));
      if (stripPrefix.isAbsolute()) {
        stripPrefix = ruleContext.getLabel().getPackageIdentifier().getRepository().getSourceRoot()
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

    if (stripPrefix.containsUplevelReferences()) {
      ruleContext.attributeError("strip_include_prefix",
          "should not contain uplevel references");
    }

    if (prefix != null && prefix.containsUplevelReferences()) {
      ruleContext.attributeError("include_prefix", "should not contain uplevel references");
    }

    if (prefix != null && prefix.isAbsolute()) {
      ruleContext.attributeError("include_prefix", "should be a relative path");
    }

    if (ruleContext.hasErrors()) {
      return new PublicHeaders(ImmutableList.<Artifact>of(), ImmutableList.<Artifact>of(), null);
    }

    ImmutableList.Builder<Artifact> moduleHeadersBuilder = ImmutableList.builder();

    for (Artifact originalHeader : publicHeaders) {
      if (!originalHeader.getRootRelativePath().startsWith(stripPrefix)) {
        ruleContext.ruleError(String.format(
            "header '%s' is not under the specified strip prefix '%s'",
            originalHeader.getExecPathString(), stripPrefix.getPathString()));
        continue;
      }

      PathFragment includePath = originalHeader.getRootRelativePath().relativeTo(stripPrefix);
      if (prefix != null) {
        includePath = prefix.getRelative(includePath);
      }

      Artifact virtualHeader = ruleContext.getUniqueDirectoryArtifact("_virtual_includes",
          includePath, ruleContext.getBinOrGenfilesDirectory());
      ruleContext.registerAction(new SymlinkAction(
          ruleContext.getActionOwner(), originalHeader, virtualHeader,
          "Symlinking virtual headers for " + ruleContext.getLabel()));
      moduleHeadersBuilder.add(virtualHeader);
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

  /** Create context for cc compile action from generated inputs. */
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

    contextBuilder.mergeDependentContexts(
        AnalysisUtils.getProviders(deps, CppCompilationContext.class));
    contextBuilder.mergeDependentContexts(depContexts);
    CppHelper.mergeToolchainDependentContext(ruleContext, ccToolchain, contextBuilder);

    // But defines come after those inherited from deps.
    contextBuilder.addDefines(defines);

    // There are no ordering constraints for declared include dirs/srcs, or the pregrepped headers.
    contextBuilder.addDeclaredIncludeSrcs(publicHeaders.getHeaders());
    contextBuilder.addDeclaredIncludeSrcs(publicTextualHeaders);
    contextBuilder.addDeclaredIncludeSrcs(privateHeaders);
    contextBuilder.addModularHdrs(publicHeaders.getHeaders());
    contextBuilder.addModularHdrs(privateHeaders);
    contextBuilder.addTextualHdrs(publicTextualHeaders);
    contextBuilder.addPregreppedHeaderMap(
        CppHelper.createExtractInclusions(ruleContext, semantics, publicHeaders.getHeaders()));
    contextBuilder.addPregreppedHeaderMap(
        CppHelper.createExtractInclusions(ruleContext, semantics, publicTextualHeaders));
    contextBuilder.addPregreppedHeaderMap(
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
      ruleContext.registerAction(
          createModuleMapAction(cppModuleMap, publicHeaders, dependentModuleMaps, compiled));
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

    semantics.setupCompilationContext(ruleContext, contextBuilder);
    return contextBuilder.build();
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

  /**
   * Creates context for cc compile action from generated inputs.
   */
  public CppCompilationContext initializeCppCompilationContext() {
    return initializeCppCompilationContext(initializeCppModel());
  }

  private Iterable<CppModuleMap> collectModuleMaps() {
    // Cpp module maps may be null for some rules. We filter the nulls out at the end.
    List<CppModuleMap> result = new ArrayList<>();
    Iterables.addAll(result, Iterables.transform(deps, CPP_DEPS_TO_MODULES));
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
    for (OutputGroupProvider dep :
        ruleContext.getPrerequisites("deps", Mode.TARGET, OutputGroupProvider.class)) {
      headerTokens.addTransitive(dep.getOutputGroup(CcLibraryHelper.HIDDEN_HEADER_TOKENS));
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

  private Runfiles collectCppRunfiles(
      CcLinkingOutputs ccLinkingOutputs, boolean linkingStatically) {
    Runfiles.Builder builder = new Runfiles.Builder(
        ruleContext.getWorkspaceName(), ruleContext.getConfiguration().legacyExternalRunfiles());
    builder.addTargets(deps, RunfilesProvider.DEFAULT_RUNFILES);
    builder.addTargets(deps, CppRunfilesProvider.runfilesFunction(linkingStatically));
    // Add the shared libraries to the runfiles.
    builder.addArtifacts(ccLinkingOutputs.getLibrariesForRunfiles(linkingStatically));
    return builder.build();
  }

  private CcLinkParamsStore createCcLinkParamsStore(
      final CcLinkingOutputs ccLinkingOutputs, final CppCompilationContext cppCompilationContext,
      final boolean forcePic) {
    return new CcLinkParamsStore() {
      @Override
      protected void collect(
          CcLinkParams.Builder builder, boolean linkingStatically, boolean linkShared) {
        builder.addLinkstamps(linkstamps.build(), cppCompilationContext);
        builder.addTransitiveTargets(
            deps,
            CcLinkParamsProvider.TO_LINK_PARAMS,
            CcSpecificLinkParamsProvider.TO_LINK_PARAMS);
        if (!neverlink) {
          builder.addLibraries(
              ccLinkingOutputs.getPreferredLibraries(
                  linkingStatically, /*preferPic=*/ linkShared || forcePic));
          builder.addLinkOpts(linkopts);
        }
      }
    };
  }

  private NestedSet<LinkerInput> collectNativeCcLibraries(CcLinkingOutputs ccLinkingOutputs) {
    NestedSetBuilder<LinkerInput> result = NestedSetBuilder.linkOrder();
    result.addAll(ccLinkingOutputs.getDynamicLibraries());
    for (CcNativeLibraryProvider dep :
        AnalysisUtils.getProviders(deps, CcNativeLibraryProvider.class)) {
      result.addTransitive(dep.getTransitiveCcNativeLibraries());
    }

    return result.build();
  }

  private CcExecutionDynamicLibrariesProvider collectExecutionDynamicLibraryArtifacts(
      List<LibraryToLink> executionDynamicLibraries) {
    Iterable<Artifact> artifacts = LinkerInputs.toLibraryArtifacts(executionDynamicLibraries);
    if (!Iterables.isEmpty(artifacts)) {
      return new CcExecutionDynamicLibrariesProvider(
          NestedSetBuilder.wrap(Order.STABLE_ORDER, artifacts));
    }

    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    for (CcExecutionDynamicLibrariesProvider dep :
        AnalysisUtils.getProviders(deps, CcExecutionDynamicLibrariesProvider.class)) {
      builder.addTransitive(dep.getExecutionDynamicLibraryArtifacts());
    }
    return builder.isEmpty()
        ? CcExecutionDynamicLibrariesProvider.EMPTY
        : new CcExecutionDynamicLibrariesProvider(builder.build());
  }

  private NestedSet<Artifact> getTemps(CcCompilationOutputs compilationOutputs) {
    return ruleContext.getFragment(CppConfiguration.class).isLipoContextCollector()
        ? NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)
        : compilationOutputs.getTemps();
  }

  public void registerAdditionalModuleMap(CppModuleMap cppModuleMap) {
    this.additionalCppModuleMaps.add(Preconditions.checkNotNull(cppModuleMap));
  }
}
