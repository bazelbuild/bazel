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
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
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
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
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
  static final FileTypeSet SOURCE_TYPES =
      FileTypeSet.of(
          CppFileTypes.CPP_SOURCE,
          CppFileTypes.CPP_HEADER,
          CppFileTypes.C_SOURCE,
          CppFileTypes.ASSEMBLER,
          CppFileTypes.ASSEMBLER_WITH_C_PREPROCESSOR);
  
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
    private final ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>
        providers;
    private final ImmutableMap<String, NestedSet<Artifact>> outputGroups;
    private final CcCompilationOutputs compilationOutputs;
    private final CcLinkingOutputs linkingOutputs;
    private final CcLinkingOutputs linkingOutputsExcludingPrecompiledLibraries;
    private final CppCompilationContext context;

    private Info(Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers,
        Map<String, NestedSet<Artifact>> outputGroups,
        CcCompilationOutputs compilationOutputs, CcLinkingOutputs linkingOutputs,
        CcLinkingOutputs linkingOutputsExcludingPrecompiledLibraries,
        CppCompilationContext context) {
      this.providers = ImmutableMap.copyOf(providers);
      this.outputGroups = ImmutableMap.copyOf(outputGroups);
      this.compilationOutputs = compilationOutputs;
      this.linkingOutputs = linkingOutputs;
      this.linkingOutputsExcludingPrecompiledLibraries =
          linkingOutputsExcludingPrecompiledLibraries;
      this.context = context;
    }

    public Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> getProviders() {
      return providers;
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
      filesBuilder.addAll(LinkerInputs.toLibraryArtifacts(linkingOutputs.getStaticLibraries()));
      filesBuilder.addAll(LinkerInputs.toLibraryArtifacts(linkingOutputs.getPicStaticLibraries()));
      filesBuilder.addAll(LinkerInputs.toNonSolibArtifacts(linkingOutputs.getDynamicLibraries()));
      filesBuilder.addAll(
          LinkerInputs.toNonSolibArtifacts(linkingOutputs.getExecutionDynamicLibraries()));
    }
  }

  private final RuleContext ruleContext;
  private final BuildConfiguration configuration;
  private final CppSemantics semantics;

  private final List<Artifact> publicHeaders = new ArrayList<>();
  private final List<Artifact> publicTextualHeaders = new ArrayList<>();
  private final List<Artifact> privateHeaders = new ArrayList<>();
  private final List<PathFragment> additionalExportedHeaders = new ArrayList<>();
  private final List<Pair<Artifact, Label>> compilationUnitSources = new ArrayList<>();
  private final List<Artifact> objectFiles = new ArrayList<>();
  private final List<Artifact> picObjectFiles = new ArrayList<>();
  private final List<String> copts = new ArrayList<>();
  @Nullable private Pattern nocopts;
  private final List<String> linkopts = new ArrayList<>();
  private final Set<String> defines = new LinkedHashSet<>();
  private final List<TransitiveInfoCollection> deps = new ArrayList<>();
  private final List<Artifact> linkstamps = new ArrayList<>();
  private final List<Artifact> prerequisites = new ArrayList<>();
  private final List<PathFragment> looseIncludeDirs = new ArrayList<>();
  private final List<PathFragment> systemIncludeDirs = new ArrayList<>();
  private final List<PathFragment> includeDirs = new ArrayList<>();
  @Nullable private Artifact dynamicLibrary;
  private LinkTargetType linkType = LinkTargetType.STATIC_LIBRARY;
  private HeadersCheckingMode headersCheckingMode = HeadersCheckingMode.LOOSE;
  private boolean neverlink;
  private boolean fake;

  private final List<LibraryToLink> staticLibraries = new ArrayList<>();
  private final List<LibraryToLink> picStaticLibraries = new ArrayList<>();
  private final List<LibraryToLink> dynamicLibraries = new ArrayList<>();

  private boolean emitLinkActionsIfEmpty;
  private boolean emitCcNativeLibrariesProvider;
  private boolean emitCcSpecificLinkParamsProvider;
  private boolean emitInterfaceSharedObjects;
  private boolean emitDynamicLibrary = true;
  private boolean checkDepsGenerateCpp = true;
  private boolean emitCompileProviders;

  private final FeatureConfiguration featureConfiguration;

  public CcLibraryHelper(RuleContext ruleContext, CppSemantics semantics,
      FeatureConfiguration featureConfiguration) {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);
    this.configuration = ruleContext.getConfiguration();
    this.semantics = Preconditions.checkNotNull(semantics);
    this.featureConfiguration = Preconditions.checkNotNull(featureConfiguration);
  }

  /**
   * Sets fields that overlap for cc_library and cc_binary rules.
   */
  public CcLibraryHelper fromCommon(CcCommon common) {
    this
        .addCopts(common.getCopts())
        .addDefines(common.getDefines())
        .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
        .addIncludeDirs(common.getIncludeDirs())
        .addLooseIncludeDirs(common.getLooseIncludeDirs())
        .addPicIndependentObjectFiles(common.getLinkerScripts())
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
   * Add the corresponding files as source files. These may also be header files, in which case
   * they will not be compiled, but also not made visible as includes to dependent rules.
   */
  public CcLibraryHelper addSources(Collection<Artifact> sources) {
    for (Artifact source : sources) {
      addSource(source, ruleContext.getLabel());
    }
    return this;
  }

  /**
   * Add the corresponding files as source files. These may also be header files, in which case
   * they will not be compiled, but also not made visible as includes to dependent rules.
   */
  public CcLibraryHelper addSources(Iterable<Pair<Artifact, Label>> sources) {
    for (Pair<Artifact, Label> source : sources) {
      addSource(source.first, source.second);
    }
    return this;
  }

  /**
   * Add the corresponding files as source files. These may also be header files, in which case
   * they will not be compiled, but also not made visible as includes to dependent rules.
   */
  public CcLibraryHelper addSources(Artifact... sources) {
    return addSources(Arrays.asList(sources));
  }
  
  /**
   * Adds a header to {@code publicHeaders} and in case header processing is switched on for the
   * file type also to compilationUnitSources.
   */
  private void addHeader(Artifact header, Label label) {
    boolean isHeader = CppFileTypes.CPP_HEADER.matches(header.getExecPath());
    boolean isTextualInclude = CppFileTypes.CPP_TEXTUAL_INCLUDE.matches(header.getExecPath());
    publicHeaders.add(header);
    if (isTextualInclude || !isHeader || !shouldProcessHeaders()) {
      return;
    }
    compilationUnitSources.add(Pair.of(header, label));
  }

  /**
   * Adds a source to {@code compilationUnitSources} if it is a compiled file type (including
   * parsed/preprocessed header) and to {@code privateHeaders} if it is a header.
   */
  private void addSource(Artifact source, Label label) {
    boolean isHeader = CppFileTypes.CPP_HEADER.matches(source.getExecPath());
    boolean isTextualInclude = CppFileTypes.CPP_TEXTUAL_INCLUDE.matches(source.getExecPath());
    boolean isCompiledSource = SOURCE_TYPES.matches(source.getExecPathString());
    if (isHeader || isTextualInclude) {
      privateHeaders.add(source);
    }
    if (isTextualInclude || !isCompiledSource || (isHeader && !shouldProcessHeaders())) {
      return;
    }
    compilationUnitSources.add(Pair.of(source, label));
  }

  private boolean shouldProcessHeaders() {
    return featureConfiguration.isEnabled(CppRuleClasses.PREPROCESS_HEADERS)
        || featureConfiguration.isEnabled(CppRuleClasses.PARSE_HEADERS);
  }

  /**
   * Returns the compilation unit sources. That includes all compiled source files as well
   * as headers that will be parsed or preprocessed.
   */
  public ImmutableList<Pair<Artifact, Label>> getCompilationUnitSources() {
    return ImmutableList.copyOf(this.compilationUnitSources);
  }

  /**
   * Add the corresponding files as linker inputs for non-PIC links. If the corresponding files are
   * compiled with PIC, the final link may or may not fail. Note that the final link may not happen
   * here, if {@code --start_end_lib} is enabled, but instead at any binary that transitively
   * depends on the current rule.
   */
  public CcLibraryHelper addObjectFiles(Iterable<Artifact> objectFiles) {
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
    Iterables.addAll(this.picObjectFiles, picObjectFiles);
    return this;
  }

  /**
   * Add the corresponding files as linker inputs for both PIC and non-PIC links.
   */
  public CcLibraryHelper addPicIndependentObjectFiles(Iterable<Artifact> objectFiles) {
    addPicObjectFiles(objectFiles);
    return addObjectFiles(objectFiles);
  }

  /**
   * Add the corresponding files as linker inputs for both PIC and non-PIC links.
   */
  public CcLibraryHelper addPicIndependentObjectFiles(Artifact... objectFiles) {
    return addPicIndependentObjectFiles(Arrays.asList(objectFiles));
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
      Preconditions.checkArgument(dep.getConfiguration() == null
          || configuration.equalsOrIsSupersetOf(dep.getConfiguration()),
          "dep " + dep.getLabel() + " has a different config than " + ruleContext.getLabel());
      this.deps.add(dep);
    }
    return this;
  }

  /**
   * Adds the given linkstamps. Note that linkstamps are usually not compiled at the library level,
   * but only in the dependent binary rules.
   */
  public CcLibraryHelper addLinkstamps(Iterable<? extends TransitiveInfoCollection> linkstamps) {
    for (TransitiveInfoCollection linkstamp : linkstamps) {
      Iterables.addAll(this.linkstamps,
          linkstamp.getProvider(FileProvider.class).getFilesToBuild());
    }
    return this;
  }

  /**
   * Adds the given prerequisites as prerequisites for the generated compile actions. This ensures
   * that the corresponding files exist - otherwise the action fails. Note that these dependencies
   * add edges to the action graph, and can therefore increase the length of the critical path,
   * i.e., make the build slower.
   */
  public CcLibraryHelper addCompilationPrerequisites(Iterable<Artifact> prerequisites) {
    Iterables.addAll(this.prerequisites, prerequisites);
    return this;
  }

  /**
   * Adds the given precompiled files to this helper. Shared and static libraries are added as
   * compilation prerequisites, and object files are added as pic or non-pic object files
   * respectively.
   */
  public CcLibraryHelper addPrecompiledFiles(PrecompiledFiles precompiledFiles) {
    addCompilationPrerequisites(precompiledFiles.getSharedLibraries());
    addCompilationPrerequisites(precompiledFiles.getStaticLibraries());
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
   * Adds the given directories to the quote include directories (they are passed with {@code
   * "-iquote"} to the compiler); these are also passed to dependent rules.
   */
  public CcLibraryHelper addIncludeDirs(Iterable<PathFragment> includeDirs) {
    Iterables.addAll(this.includeDirs, includeDirs);
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
   */
  public Info build() {
    // Fail early if there is no lipo context collector on the rule - otherwise we end up failing
    // in lipo optimization.
    Preconditions.checkState(
        // 'cc_inc_library' rules do not compile, and thus are not affected by LIPO.
        ruleContext.getRule().getRuleClass().equals("cc_inc_library")
        || ruleContext.getRule().isAttrDefined(":lipo_context_collector", BuildType.LABEL));

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
    if (emitLinkActionsIfEmpty || !ccOutputs.isEmpty()) {
      // On some systems, the linker gives an error message if there are no input files. Even with
      // the check above, this can still happen if there is a .nopic.o or .o files in srcs, but no
      // other files. To fix that, we'd have to check for each link action individually.
      //
      // An additional pre-existing issue is that the header check tokens are dropped if we don't
      // generate any link actions, effectively disabling header checking in some cases.
      if (linkType.isStaticLibraryLink()) {
        // TODO(bazel-team): This can't create the link action for a cc_binary yet.
        ccLinkingOutputs = model.createCcLinkActions(ccOutputs);
      }
    }
    CcLinkingOutputs originalLinkingOutputs = ccLinkingOutputs;
    if (!(
        staticLibraries.isEmpty() && picStaticLibraries.isEmpty() && dynamicLibraries.isEmpty())) {
      // Merge the pre-compiled libraries (static & dynamic) into the linker outputs.
      ccLinkingOutputs = new CcLinkingOutputs.Builder()
          .merge(ccLinkingOutputs)
          .addStaticLibraries(staticLibraries)
          .addPicStaticLibraries(picStaticLibraries)
          .addDynamicLibraries(dynamicLibraries)
          .addExecutionDynamicLibraries(dynamicLibraries)
          .build();
    }

    DwoArtifactsCollector dwoArtifacts = DwoArtifactsCollector.transitiveCollector(ccOutputs, deps);
    Runfiles cppStaticRunfiles = collectCppRunfiles(ccLinkingOutputs, true);
    Runfiles cppSharedRunfiles = collectCppRunfiles(ccLinkingOutputs, false);

    // By very careful when adding new providers here - it can potentially affect a lot of rules.
    // We should consider merging most of these providers into a single provider.
    Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers =
        new LinkedHashMap<>();
    providers.put(CppRunfilesProvider.class,
        new CppRunfilesProvider(cppStaticRunfiles, cppSharedRunfiles));
    providers.put(CppCompilationContext.class, cppCompilationContext);
    providers.put(CppDebugFileProvider.class, new CppDebugFileProvider(
        dwoArtifacts.getDwoArtifacts(), dwoArtifacts.getPicDwoArtifacts()));
    providers.put(TransitiveLipoInfoProvider.class, collectTransitiveLipoInfo(ccOutputs));
    Map<String, NestedSet<Artifact>> outputGroups = new TreeMap<>();
    outputGroups.put(OutputGroupProvider.TEMP_FILES, getTemps(ccOutputs));
    if (emitCompileProviders) {
      boolean isLipoCollector =
          ruleContext.getFragment(CppConfiguration.class).isLipoContextCollector();
      boolean processHeadersInDependencies =
          ruleContext.getFragment(CppConfiguration.class).processHeadersInDependencies();
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
      providers.put(CcNativeLibraryProvider.class,
          new CcNativeLibraryProvider(collectNativeCcLibraries(ccLinkingOutputs)));
    }
    providers.put(CcExecutionDynamicLibrariesProvider.class,
        collectExecutionDynamicLibraryArtifacts(ccLinkingOutputs.getExecutionDynamicLibraries()));

    boolean forcePic = ruleContext.getFragment(CppConfiguration.class).forcePic();
    if (emitCcSpecificLinkParamsProvider) {
      providers.put(CcSpecificLinkParamsProvider.class, new CcSpecificLinkParamsProvider(
          createCcLinkParamsStore(ccLinkingOutputs, cppCompilationContext, forcePic)));
    } else {
      providers.put(CcLinkParamsProvider.class, new CcLinkParamsProvider(
          createCcLinkParamsStore(ccLinkingOutputs, cppCompilationContext, forcePic)));
    }
    return new Info(providers, outputGroups, ccOutputs, ccLinkingOutputs, originalLinkingOutputs,
        cppCompilationContext);
  }

  /**
   * Creates the C/C++ compilation action creator.
   */
  private CppModel initializeCppModel() {
    return new CppModel(ruleContext, semantics)
        .addCompilationUnitSources(compilationUnitSources)
        .addCopts(copts)
        .setLinkTargetType(linkType)
        .setNeverLink(neverlink)
        .setFake(fake)
        .setAllowInterfaceSharedObjects(emitInterfaceSharedObjects)
        .setCreateDynamicLibrary(emitDynamicLibrary)
        // Note: this doesn't actually save the temps, it just makes the CppModel use the
        // configurations --save_temps setting to decide whether to actually save the temps.
        .setSaveTemps(true)
        .setNoCopts(nocopts)
        .setDynamicLibrary(dynamicLibrary)
        .addLinkopts(linkopts)
        .setFeatureConfiguration(featureConfiguration);
  }

  /**
   * Create context for cc compile action from generated inputs.
   */
  private CppCompilationContext initializeCppCompilationContext(CppModel model) {
    CppCompilationContext.Builder contextBuilder =
        new CppCompilationContext.Builder(ruleContext);

    // Setup the include path; local include directories come before those inherited from deps or
    // from the toolchain; in case of aliasing (same include file found on different entries),
    // prefer the local include rather than the inherited one.

    // Add in the roots for well-formed include names for source files and
    // generated files. It is important that the execRoot (EMPTY_FRAGMENT) comes
    // before the genfilesFragment to preferably pick up source files. Otherwise
    // we might pick up stale generated files.
    PathFragment repositoryPath =
        ruleContext.getLabel().getPackageIdentifier().getRepository().getPathFragment();
    contextBuilder.addQuoteIncludeDir(repositoryPath);
    contextBuilder.addQuoteIncludeDir(
        ruleContext.getConfiguration().getGenfilesFragment().getRelative(repositoryPath));

    for (PathFragment systemIncludeDir : systemIncludeDirs) {
      contextBuilder.addSystemIncludeDir(systemIncludeDir);
    }
    for (PathFragment includeDir : includeDirs) {
      contextBuilder.addIncludeDir(includeDir);
    }

    contextBuilder.mergeDependentContexts(
        AnalysisUtils.getProviders(deps, CppCompilationContext.class));
    CppHelper.mergeToolchainDependentContext(ruleContext, contextBuilder);

    // But defines come after those inherited from deps.
    contextBuilder.addDefines(defines);

    // There are no ordering constraints for declared include dirs/srcs, or the pregrepped headers.
    contextBuilder.addDeclaredIncludeSrcs(publicHeaders);
    contextBuilder.addDeclaredIncludeSrcs(publicTextualHeaders);
    contextBuilder.addDeclaredIncludeSrcs(privateHeaders);
    contextBuilder.addPregreppedHeaderMap(
        CppHelper.createExtractInclusions(ruleContext, semantics, publicHeaders));
    contextBuilder.addPregreppedHeaderMap(
        CppHelper.createExtractInclusions(ruleContext, semantics, publicTextualHeaders));
    contextBuilder.addPregreppedHeaderMap(
        CppHelper.createExtractInclusions(ruleContext, semantics, privateHeaders));
    contextBuilder.addCompilationPrerequisites(prerequisites);

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
      CppModuleMap cppModuleMap = CppHelper.addCppModuleMapToContext(ruleContext, contextBuilder);
      // TODO(bazel-team): addCppModuleMapToContext second-guesses whether module maps should
      // actually be enabled, so we need to double-check here. Who would write code like this?
      if (cppModuleMap != null) {
        CppModuleMapAction action =
            new CppModuleMapAction(
                ruleContext.getActionOwner(),
                cppModuleMap,
                privateHeaders,
                publicHeaders,
                collectModuleMaps(),
                additionalExportedHeaders,
                featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULES),
                featureConfiguration.isEnabled(CppRuleClasses.MODULE_MAP_HOME_CWD),
                featureConfiguration.isEnabled(CppRuleClasses.GENERATE_SUBMODULES),
                !featureConfiguration.isEnabled(CppRuleClasses.MODULE_MAP_WITHOUT_EXTERN_MODULE));
        ruleContext.registerAction(action);
      }
      if (model.getGeneratesPicHeaderModule()) {
        contextBuilder.setPicHeaderModule(model.getPicHeaderModule(cppModuleMap.getArtifact()));
      }
      if (model.getGeneratesNoPicHeaderModule()) {
        contextBuilder.setHeaderModule(model.getHeaderModule(cppModuleMap.getArtifact()));
      }
      if (featureConfiguration.isEnabled(CppRuleClasses.USE_HEADER_MODULES)
          && featureConfiguration.isEnabled(CppRuleClasses.TRANSITIVE_MODULE_MAPS)) {
        contextBuilder.setProvideTransitiveModuleMaps(true);
      }
    }

    semantics.setupCompilationContext(ruleContext, contextBuilder);
    return contextBuilder.build();
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

    CcToolchainProvider toolchain = CppHelper.getToolchain(ruleContext);
    if (toolchain != null) {
      result.add(toolchain.getCppCompilationContext().getCppModuleMap());
    }

    return Iterables.filter(result, Predicates.<CppModuleMap>notNull());
  }

  private TransitiveLipoInfoProvider collectTransitiveLipoInfo(CcCompilationOutputs outputs) {
    if (ruleContext.getFragment(CppConfiguration.class).getFdoSupport().getFdoRoot() == null) {
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
    Runfiles.Builder builder = new Runfiles.Builder(ruleContext.getWorkspaceName());
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
      protected void collect(CcLinkParams.Builder builder, boolean linkingStatically,
          boolean linkShared) {
        builder.addLinkstamps(linkstamps, cppCompilationContext);
        builder.addTransitiveTargets(deps,
            CcLinkParamsProvider.TO_LINK_PARAMS, CcSpecificLinkParamsProvider.TO_LINK_PARAMS);
        if (!neverlink) {
          builder.addLibraries(ccLinkingOutputs.getPreferredLibraries(linkingStatically,
              /*preferPic=*/linkShared || forcePic));
          builder.addLinkOpts(linkopts);
        }
      }
    };
  }

  private NestedSet<LinkerInput> collectNativeCcLibraries(CcLinkingOutputs ccLinkingOutputs) {
    NestedSetBuilder<LinkerInput> result = NestedSetBuilder.linkOrder();
    result.addAll(ccLinkingOutputs.getDynamicLibraries());
    for (CcNativeLibraryProvider dep : AnalysisUtils.getProviders(
        deps, CcNativeLibraryProvider.class)) {
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
}
