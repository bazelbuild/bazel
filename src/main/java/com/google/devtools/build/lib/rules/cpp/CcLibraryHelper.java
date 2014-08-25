// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.AnalysisUtils;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.lib.view.TempsProvider;
import com.google.devtools.build.lib.view.TransitiveInfoCollection;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A class to create C/C++ compile and link actions in a way that is consistent with cc_library.
 * Rules that generate source files and emulate cc_library on top of that should use this class
 * instead of the lower-level APIs in CppHelper and CppModel.
 *
 * <p>Rules that want to use this class are required to have implicit dependencies on the
 * toolchain, the STL and so on. Optionally, they can also have copts, plugins, and malloc
 * attributes, but note that these require explicit calls to the corresponding setter methods.
 */
public final class CcLibraryHelper {
  /**
   * Contains the providers as well as the compilation and linking outputs, and the compilation
   * context.
   */
  public static final class Info {
    private final Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers;
    private final CcCompilationOutputs compilationOutputs;
    private final CcLinkingOutputs linkingOutputs;
    private final CppCompilationContext context;

    private Info(Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers,
        CcCompilationOutputs compilationOutputs, CcLinkingOutputs linkingOutputs,
        CppCompilationContext context) {
      this.providers = Collections.unmodifiableMap(providers);
      this.compilationOutputs = compilationOutputs;
      this.linkingOutputs = linkingOutputs;
      this.context = context;
    }

    public Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> getProviders() {
      return providers;
    }

    public CcCompilationOutputs getCcCompilationOutputs() {
      return compilationOutputs;
    }

    public CcLinkingOutputs getCcLinkingOutputs() {
      return linkingOutputs;
    }

    public CppCompilationContext getCppCompilationContext() {
      return context;
    }

    /**
     * Adds the static, pic-static, and dynamic (both compile-time and execution-time) libraries to
     * the given builder.
     */
    public void addLinkingOutputsTo(NestedSetBuilder<Artifact> filesBuilder) {
      filesBuilder.addAll(Link.toLibraryArtifacts(linkingOutputs.getStaticLibraries()));
      filesBuilder.addAll(Link.toLibraryArtifacts(linkingOutputs.getPicStaticLibraries()));
      filesBuilder.addAll(Link.toNonSolibArtifacts(linkingOutputs.getDynamicLibraries()));
      filesBuilder.addAll(Link.toNonSolibArtifacts(linkingOutputs.getExecutionDynamicLibraries()));
    }
  }

  private final RuleContext ruleContext;
  private final CppSemantics semantics;

  private final List<Artifact> headers = new ArrayList<>();
  private final List<Artifact> sources = new ArrayList<>();
  private final List<String> copts = new ArrayList<>();
  private final List<TransitiveInfoCollection> deps = new ArrayList<>();
  private final List<CcPluginInfoProvider> plugins = new ArrayList<>();
  private LinkTargetType linkType = LinkTargetType.STATIC_LIBRARY;
  private HeadersCheckingMode headersCheckingMode = HeadersCheckingMode.LOOSE;

  private boolean emitCppModuleMaps = true;
  private boolean emitCompileActionsIfEmpty = true;
  private boolean emitCcNativeLibrariesProvider;
  private boolean emitCcSpecificLinkParamsProvider;

  public CcLibraryHelper(RuleContext ruleContext, CppSemantics semantics) {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);
    this.semantics = Preconditions.checkNotNull(semantics);
  }

  /**
   * Add the corresponding files as header files, i.e., these files will not be compiled, but are
   * made visible as includes to dependent rules.
   */
  public CcLibraryHelper addHeaders(Collection<Artifact> headers) {
    this.headers.addAll(headers);
    return this;
  }

  /**
   * Add the corresponding files as header files, i.e., these files will not be compiled, but are
   * made visible as includes to dependent rules.
   */
  public CcLibraryHelper addHeaders(Artifact... headers) {
    return addHeaders(Arrays.asList(headers));
  }

  /**
   * Add the corresponding files as source files. These may also be header files, in which case
   * they will not be compiled, but also not made visible as includes to dependent rules.
   */
  // TODO(bazel-team): This is inconsistent with the documentation on CppModel.
  public CcLibraryHelper addSources(Collection<Artifact> sources) {
    this.sources.addAll(sources);
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
   * Adds the copts to the compile command line.
   */
  public CcLibraryHelper addCopts(Iterable<String> copts) {
    Iterables.addAll(this.copts, copts);
    return this;
  }

  /**
   * Adds the given targets as dependencies - this can include explicit dependencies on other
   * rules (like from a "deps" attribute) and also implicit dependencies on runtime libraries.
   */
  public CcLibraryHelper addDeps(Iterable<? extends TransitiveInfoCollection> deps) {
    Iterables.addAll(this.deps, deps);
    return this;
  }

  /**
   * Adds the given targets as dependencies - this can include explicit dependencies on other
   * rules (like from a "deps" attribute) and also implicit dependencies on runtime libraries.
   */
  public CcLibraryHelper addDeps(TransitiveInfoCollection... deps) {
    return addDeps(Arrays.asList(deps));
  }

  /**
   * Adds the given targets as C++ compiler plugins; non-plugin targets are silently ignored.
   */
  public CcLibraryHelper addPlugins(Iterable<? extends TransitiveInfoCollection> plugins) {
    Iterables.addAll(this.plugins, AnalysisUtils.getProviders(plugins, CcPluginInfoProvider.class));
    return this;
  }

  /**
   * Marks the output of this rule as alwayslink, i.e., the corresponding symbols will be retained
   * by the linker even if they are not otherwise used. This is useful for libraries that register
   * themselves somewhere during initialization.
   */
  public CcLibraryHelper setAlwayslink(boolean alwayslink) {
    linkType = alwayslink
        ? LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY
        : LinkTargetType.STATIC_LIBRARY;
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
   * This adds the {@link CcNativeLibraryProvider} to the providers created by this class.
   */
  public CcLibraryHelper enableCcNativeLibrariesProvider() {
    this.emitCcNativeLibrariesProvider = true;
    return this;
  }

  /**
   * This adds the {@link CcSpecificLinkParamsProvider} to the providers created by this class.
   */
  public CcLibraryHelper enableCcSpecificLinkParamsProvider() {
    this.emitCcSpecificLinkParamsProvider = true;
    return this;
  }

  /**
   * This disables C++ module map generation for the current rule. Don't call this unless you know
   * what you are doing.
   */
  public CcLibraryHelper disableCppModuleMapGeneration() {
    this.emitCppModuleMaps = false;
    return this;
  }

  /**
   * Disables generation of compile actions if there are no sources. Some rules declare a .a or .so
   * implicit output, which requires that these files are created even if there are no source files,
   * so be careful when calling this.
   */
  public CcLibraryHelper disableCompileActionsIfEmpty() {
    this.emitCompileActionsIfEmpty = false;
    return this;
  }

  /**
   * Create the C++ compile and link actions, and the corresponding C++-related providers.
   */
  public Info build() {
    CppCompilationContext cppCompilationContext = initializeCppCompilationContext();
    CcLinkingOutputs ccLinkingOutputs = CcLinkingOutputs.EMPTY;
    CcCompilationOutputs ccOutputs = new CcCompilationOutputs.Builder().build();
    if (emitCompileActionsIfEmpty || !sources.isEmpty()) {
      CppModel model = new CppModel(ruleContext, semantics)
          .addSources(sources, ruleContext.getLabel())
          .addCopts(copts)
          .addPlugins(plugins)
          .setContext(cppCompilationContext)
          .setLinkTargetType(linkType)
          // Note: this doesn't actually save the temps, it just makes the CppModel use the
          // configurations --save_temps setting to decide whether to actually save the temps.
          .setSaveTemps(true);
      ccOutputs = model.createCcCompileActions();
      ccLinkingOutputs = model.createCcLinkActions(ccOutputs);
    }

    DwoArtifactsCollector dwoArtifacts = DwoArtifactsCollector.transitiveCollector(
        ccOutputs,
        ImmutableList.<TransitiveInfoCollection>builder()
            .addAll(deps)
            .build());

    boolean forcePic = ruleContext.getFragment(CppConfiguration.class).forcePic();
    Runfiles cppStaticRunfiles = collectCppRunfiles(ccLinkingOutputs, true, forcePic);
    Runfiles cppSharedRunfiles = collectCppRunfiles(ccLinkingOutputs, false, forcePic);

    // By very careful when adding new providers here - it can potentially affect a lot of rules.
    // We should consider merging most of these providers into a single provider.
    Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers =
        new LinkedHashMap<>();
    providers.put(CppRunfilesProvider.class,
        new CppRunfilesProvider(cppStaticRunfiles, cppSharedRunfiles));
    providers.put(CppCompilationContext.class, cppCompilationContext);
    providers.put(CppDebugFileProvider.class, new CppDebugFileProvider(
        dwoArtifacts.getDwoArtifacts(), dwoArtifacts.getPicDwoArtifacts()));
    providers.put(FdoProfilingInfoProvider.class, collectTransitiveLipoInfo());
    providers.put(TempsProvider.class, getTemps(ccOutputs));

    // TODO(bazel-team): Maybe we can infer these from other data at the places where they are
    // used.
    if (emitCcNativeLibrariesProvider) {
      providers.put(CcNativeLibraryProvider.class,
          new CcNativeLibraryProvider(collectNativeCcLibraries(ccLinkingOutputs)));
    }
    providers.put(CcExecutionDynamicLibrariesProvider.class,
        collectExecutionDynamicLibraryArtifacts(ccLinkingOutputs.getExecutionDynamicLibraries()));

    if (emitCcSpecificLinkParamsProvider) {
      providers.put(CcSpecificLinkParamsProvider.class, new CcSpecificLinkParamsProvider(
          createCcLinkParamsStore(ccLinkingOutputs, forcePic)));
    }
    return new Info(providers, ccOutputs, ccLinkingOutputs, cppCompilationContext);
  }

  /**
   * Create context for cc compile action from generated inputs.
   */
  private CppCompilationContext initializeCppCompilationContext() {
    CppCompilationContext.Builder contextBuilder =
        new CppCompilationContext.Builder(ruleContext);
    contextBuilder.mergeDependentContexts(
        AnalysisUtils.getProviders(deps, CppCompilationContext.class));
    CppHelper.mergeToolchainDependentContext(ruleContext, contextBuilder);
    if (emitCppModuleMaps) {
      CppModuleMap cppModuleMap = CppHelper.addCppModuleMapToContext(ruleContext, contextBuilder);
      // TODO(bazel-team): This probably needs to also propagate the dependencies' module maps.
      CppModuleMapAction action = new CppModuleMapAction(ruleContext.getActionOwner(),
          cppModuleMap, ImmutableList.<Artifact>of(),
          ImmutableList.<Artifact>copyOf(headers), ImmutableList.<CppModuleMap>of());
      ruleContext.getAnalysisEnvironment().registerAction(action);
    }

    // This is the default include path.
    // Add in the roots for well-formed include names for source files and
    // generated files. It is important that the execRoot (EMPTY_FRAGMENT) comes
    // before the genfilesFragment to preferably pick up source files. Otherwise
    // we might pick up stale generated files.
    contextBuilder.addQuoteIncludeDir(PathFragment.EMPTY_FRAGMENT);
    contextBuilder.addQuoteIncludeDir(ruleContext.getConfiguration().getGenfilesFragment());

    // Add this package's dir to declaredIncludeDirs, & this rule's headers to declaredIncludeSrcs
    // Note: no include dir for STRICT mode.
    if (headersCheckingMode == HeadersCheckingMode.WARN) {
      contextBuilder.addDeclaredIncludeWarnDir(ruleContext.getLabel().getPackageFragment());
    } else if (headersCheckingMode == HeadersCheckingMode.LOOSE) {
      contextBuilder.addDeclaredIncludeDir(ruleContext.getLabel().getPackageFragment());
    }

    contextBuilder.addDeclaredIncludeSrcs(headers);
    contextBuilder.addPregreppedHeaderMap(
        CppHelper.createExtractInclusions(ruleContext, headers));
    return contextBuilder.build();
  }

  private FdoProfilingInfoProvider collectTransitiveLipoInfo() {
    if (ruleContext.getFragment(CppConfiguration.class).getFdoSupport().getFdoRoot() == null) {
      return FdoProfilingInfoProvider.EMPTY;
    }
    NestedSetBuilder<Label> builder = NestedSetBuilder.stableOrder();
    // TODO(bazel-team): Only fetch the STL prerequisite in one place.
    TransitiveInfoCollection stl = ruleContext.getPrerequisite(":stl", Mode.TARGET);
    if (stl != null) {
      FdoProfilingInfoProvider provider = stl.getProvider(FdoProfilingInfoProvider.class);
      if (provider != null) {
        builder.addTransitive(provider.getTransitiveLipoLabels());
      }
    }

    for (FdoProfilingInfoProvider dep :
        AnalysisUtils.getProviders(deps, FdoProfilingInfoProvider.class)) {
      builder.addTransitive(dep.getTransitiveLipoLabels());
    }

    builder.add(ruleContext.getLabel());
    return new FdoProfilingInfoProvider(builder.build());
  }

  private Runfiles collectCppRunfiles(
      CcLinkingOutputs ccLinkingOutputs, boolean linkingStatically, boolean forcePic) {
    Runfiles.Builder builder = new Runfiles.Builder();
    builder.addTargets(deps, RunfilesProvider.DEFAULT_RUNFILES);
    builder.addTargets(deps, CppRunfilesProvider.runfilesFunction(linkingStatically));
    // Add the shared libraries to the runfiles.
    builder.addArtifacts(ccLinkingOutputs.getLibrariesForRunfiles(linkingStatically, forcePic));
    return builder.build();
  }

  private CcLinkParamsStore createCcLinkParamsStore(
      final CcLinkingOutputs ccLinkingOutputs, final boolean forcePic) {
    return new CcLinkParamsStore() {
      @Override
      protected void collect(CcLinkParams.Builder builder, boolean linkingStatically,
          boolean linkShared) {
        builder.addTransitiveLangTargets(deps, CcSpecificLinkParamsProvider.TO_LINK_PARAMS);
        builder.addTransitiveTargets(deps);
        builder.addLibraries(ccLinkingOutputs.getPreferredLibraries(linkingStatically,
            /*preferPic=*/linkShared || forcePic));
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
    Iterable<Artifact> artifacts = Link.toLibraryArtifacts(executionDynamicLibraries);
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

  private TempsProvider getTemps(CcCompilationOutputs compilationOutputs) {
    return ruleContext.getFragment(CppConfiguration.class).isLipoContextCollector()
        ? new TempsProvider(ImmutableList.<Artifact>of())
        : new TempsProvider(compilationOutputs.getTemps());
  }
}
