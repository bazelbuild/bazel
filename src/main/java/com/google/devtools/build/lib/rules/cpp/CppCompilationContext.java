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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Immutable store of information needed for C++ compilation that is aggregated
 * across dependencies.
 */
@Immutable
public final class CppCompilationContext implements TransitiveInfoProvider {
  /** An empty compilation context. */
  public static final CppCompilationContext EMPTY = new Builder(null).build();

  private final CommandLineContext commandLineContext;
  
  private final NestedSet<PathFragment> declaredIncludeDirs;
  private final NestedSet<PathFragment> declaredIncludeWarnDirs;
  private final NestedSet<Artifact> declaredIncludeSrcs;
  
  /**
   * Module maps from direct dependencies.
   */
  private final NestedSet<Artifact> directModuleMaps;

  private final NestedSet<Pair<Artifact, Artifact>> pregreppedHdrs;
  
  private final ModuleInfo moduleInfo;
  private final ModuleInfo picModuleInfo;

  private final CppModuleMap cppModuleMap;
  
  // Derived from depsContexts.
  private final ImmutableSet<Artifact> compilationPrerequisites;

  private CppCompilationContext(
      CommandLineContext commandLineContext,
      ImmutableSet<Artifact> compilationPrerequisites,
      NestedSet<PathFragment> declaredIncludeDirs,
      NestedSet<PathFragment> declaredIncludeWarnDirs,
      NestedSet<Artifact> declaredIncludeSrcs,
      NestedSet<Pair<Artifact, Artifact>> pregreppedHdrs,
      ModuleInfo moduleInfo,
      ModuleInfo picModuleInfo,
      NestedSet<Artifact> directModuleMaps,
      CppModuleMap cppModuleMap) {
    Preconditions.checkNotNull(commandLineContext);
    this.commandLineContext = commandLineContext;
    this.declaredIncludeDirs = declaredIncludeDirs;
    this.declaredIncludeWarnDirs = declaredIncludeWarnDirs;
    this.declaredIncludeSrcs = declaredIncludeSrcs;
    this.directModuleMaps = directModuleMaps;
    this.pregreppedHdrs = pregreppedHdrs;
    this.moduleInfo = moduleInfo;
    this.picModuleInfo = picModuleInfo;
    this.cppModuleMap = cppModuleMap;
    this.compilationPrerequisites = compilationPrerequisites;
  }

  /**
   * Returns the transitive compilation prerequisites consolidated into middlemen
   * prerequisites, or an empty set if there are no prerequisites.
   *
   * <p>Transitive compilation prerequisites are the prerequisites that will be needed by all
   * reverse dependencies; note that these do specifically not include any compilation prerequisites
   * that are only needed by the rule itself (for example, compiled source files from the
   * {@code srcs} attribute).
   *
   * <p>To reduce the number of edges in the action graph, we express the dependency on compilation
   * prerequisites as a transitive dependency via a middleman.
   * After they have been accumulated (using
   * {@link Builder#addCompilationPrerequisites(Iterable)},
   * {@link Builder#mergeDependentContext(CppCompilationContext)}, and
   * {@link Builder#mergeDependentContexts(Iterable)}, they are consolidated
   * into a single middleman Artifact when {@link Builder#build()} is called.
   *
   * <p>The returned set can be empty if there are no prerequisites. Usually it
   * contains a single middleman, but if LIPO is used there can be two.
   */
  public ImmutableSet<Artifact> getTransitiveCompilationPrerequisites() {
    return compilationPrerequisites;
  }

  /**
   * Returns the immutable list of include directories to be added with "-I"
   * (possibly empty but never null). This includes the include dirs from the
   * transitive deps closure of the target. This list does not contain
   * duplicates. All fragments are either absolute or relative to the exec root
   * (see {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}).
   */
  public ImmutableList<PathFragment> getIncludeDirs() {
    return commandLineContext.includeDirs;
  }

  /**
   * Returns the immutable list of include directories to be added with
   * "-iquote" (possibly empty but never null). This includes the include dirs
   * from the transitive deps closure of the target. This list does not contain
   * duplicates. All fragments are either absolute or relative to the exec root
   * (see {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}).
   */
  public ImmutableList<PathFragment> getQuoteIncludeDirs() {
    return commandLineContext.quoteIncludeDirs;
  }

  /**
   * Returns the immutable list of include directories to be added with
   * "-isystem" (possibly empty but never null). This includes the include dirs
   * from the transitive deps closure of the target. This list does not contain
   * duplicates. All fragments are either absolute or relative to the exec root
   * (see {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}).
   */
  public ImmutableList<PathFragment> getSystemIncludeDirs() {
    return commandLineContext.systemIncludeDirs;
  }

  /**
   * Returns the immutable set of declared include directories, relative to a
   * "-I" or "-iquote" directory" (possibly empty but never null). The returned
   * collection may contain duplicate elements.
   *
   * <p>Note: The iteration order of this list is preserved as ide_build_info
   * writes these directories and sources out and the ordering will help when
   * used by consumers.
   */
  public NestedSet<PathFragment> getDeclaredIncludeDirs() {
    return declaredIncludeDirs;
  }

  /**
   * Returns the immutable set of include directories, relative to a "-I" or
   * "-iquote" directory", from which inclusion will produce a warning (possibly
   * empty but never null). The returned collection may contain duplicate
   * elements.
   *
   * <p>Note: The iteration order of this list is preserved as ide_build_info
   * writes these directories and sources out and the ordering will help when
   * used by consumers.
   */
  public NestedSet<PathFragment> getDeclaredIncludeWarnDirs() {
    return declaredIncludeWarnDirs;
  }

  /**
   * Returns the immutable set of headers that have been declared in the
   * {@code src} or {@code headers attribute} (possibly empty but never null).
   * The returned collection may contain duplicate elements.
   *
   * <p>Note: The iteration order of this list is preserved as ide_build_info
   * writes these directories and sources out and the ordering will help when
   * used by consumers.
   */
  public NestedSet<Artifact> getDeclaredIncludeSrcs() {
    return declaredIncludeSrcs;
  }

  /**
   * Returns the immutable pairs of (header file, pregrepped header file).  The value artifacts
   * (pregrepped header file) are generated by {@link ExtractInclusionAction}.
   */
  NestedSet<Pair<Artifact, Artifact>> getPregreppedHeaders() {
    return pregreppedHdrs;
  }

  public NestedSet<Artifact> getTransitiveModules(boolean usePic) {
    return usePic ? picModuleInfo.transitiveModules : moduleInfo.transitiveModules;
  }

  public Collection<TransitiveModuleHeaders> getUsedModules(
      boolean usePic, Set<Artifact> usedHeaders) {
    return usePic
        ? picModuleInfo.getUsedModules(usedHeaders)
        : moduleInfo.getUsedModules(usedHeaders);
  }

  /**
   * Returns the immutable set of additional transitive inputs needed for
   * compilation, like C++ module map artifacts.
   */
  public NestedSet<Artifact> getAdditionalInputs() {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    builder.addTransitive(directModuleMaps);
    if (cppModuleMap != null) {
      builder.add(cppModuleMap.getArtifact());
    }
    return builder.build();
  }

  /**
   * @return modules maps from direct dependencies.
   */
  public NestedSet<Artifact> getDirectModuleMaps() {
    return directModuleMaps;
  }

  /**
   * @return all declared headers of the current module if the current target
   * is compiled as a module.
   */
  protected Set<Artifact> getHeaderModuleSrcs() {
    return new ImmutableSet.Builder<Artifact>()
        .addAll(moduleInfo.modularHeaders)
        .addAll(moduleInfo.textualHeaders)
        .build();
  }

  /**
   * Returns the set of defines needed to compile this target (possibly empty
   * but never null). This includes definitions from the transitive deps closure
   * for the target. The order of the returned collection is deterministic.
   */
  public ImmutableList<String> getDefines() {
    return commandLineContext.defines;
  }

  /**
   * Returns a context that is based on a given context but returns empty sets
   * for {@link #getDeclaredIncludeDirs()} and {@link #getDeclaredIncludeWarnDirs()}.
   */
  public static CppCompilationContext disallowUndeclaredHeaders(CppCompilationContext context) {
    return new CppCompilationContext(
        context.commandLineContext,
        context.compilationPrerequisites,
        NestedSetBuilder.<PathFragment>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<PathFragment>emptySet(Order.STABLE_ORDER),
        context.declaredIncludeSrcs,
        context.pregreppedHdrs,
        context.moduleInfo,
        context.picModuleInfo,
        context.directModuleMaps,
        context.cppModuleMap);
  }

  /**
   * Returns the context for a LIPO compile action. This uses the include dirs
   * and defines of the library, but the declared inclusion dirs/srcs from both
   * the library and the owner binary.
   *
   * <p>TODO(bazel-team): this might make every LIPO target have an unnecessary large set of
   * inclusion dirs/srcs. The correct behavior would be to merge only the contexts
   * of actual referred targets (as listed in .imports file).
   *
   * <p>Undeclared inclusion checking ({@link #getDeclaredIncludeDirs()},
   * {@link #getDeclaredIncludeWarnDirs()}, and
   * {@link #getDeclaredIncludeSrcs()}) needs to use the union of the contexts
   * of the involved source files.
   *
   * <p>For include and define command line flags ({@link #getIncludeDirs()}
   * {@link #getQuoteIncludeDirs()}, {@link #getSystemIncludeDirs()}, and
   * {@link #getDefines()}) LIPO compilations use the same values as non-LIPO
   * compilation.
   *
   * <p>Include scanning is not handled by this method. See
   * {@code IncludeScannable#getAuxiliaryScannables()} instead.
   *
   * @param ownerContext the compilation context of the owner binary
   * @param libContext the compilation context of the library
   */
  public static CppCompilationContext mergeForLipo(CppCompilationContext ownerContext,
      CppCompilationContext libContext) {
    ImmutableSet.Builder<Artifact> prerequisites = ImmutableSet.builder();
    prerequisites.addAll(ownerContext.compilationPrerequisites);
    prerequisites.addAll(libContext.compilationPrerequisites);
    ModuleInfo.Builder moduleInfo = new ModuleInfo.Builder();
    moduleInfo.merge(ownerContext.moduleInfo);
    moduleInfo.merge(libContext.moduleInfo);
    ModuleInfo.Builder picModuleInfo = new ModuleInfo.Builder();
    picModuleInfo.merge(ownerContext.picModuleInfo);
    picModuleInfo.merge(libContext.picModuleInfo);
    return new CppCompilationContext(
        libContext.commandLineContext,
        prerequisites.build(),
        mergeSets(ownerContext.declaredIncludeDirs, libContext.declaredIncludeDirs),
        mergeSets(ownerContext.declaredIncludeWarnDirs, libContext.declaredIncludeWarnDirs),
        mergeSets(ownerContext.declaredIncludeSrcs, libContext.declaredIncludeSrcs),
        mergeSets(ownerContext.pregreppedHdrs, libContext.pregreppedHdrs),
        moduleInfo.build(),
        picModuleInfo.build(),
        mergeSets(ownerContext.directModuleMaps, libContext.directModuleMaps),
        libContext.cppModuleMap);
  }
  
  /**
   * Return a nested set containing all elements from {@code s1} and {@code s2}.
   */
  private static <T> NestedSet<T> mergeSets(NestedSet<T> s1, NestedSet<T> s2) {
    NestedSetBuilder<T> builder = NestedSetBuilder.stableOrder();
    builder.addTransitive(s1);
    builder.addTransitive(s2);
    return builder.build();
  }

  /** @return the C++ module map of the owner. */
  public CppModuleMap getCppModuleMap() {
    return cppModuleMap;
  }

  /**
   * The parts of the compilation context that influence the command line of
   * compilation actions.
   */
  @Immutable
  private static class CommandLineContext {
    private final ImmutableList<PathFragment> includeDirs;
    private final ImmutableList<PathFragment> quoteIncludeDirs;
    private final ImmutableList<PathFragment> systemIncludeDirs;
    private final ImmutableList<String> defines;

    CommandLineContext(ImmutableList<PathFragment> includeDirs,
        ImmutableList<PathFragment> quoteIncludeDirs,
        ImmutableList<PathFragment> systemIncludeDirs,
        ImmutableList<String> defines) {
      this.includeDirs = includeDirs;
      this.quoteIncludeDirs = quoteIncludeDirs;
      this.systemIncludeDirs = systemIncludeDirs;
      this.defines = defines;
    }
  }

  /**
   * Builder class for {@link CppCompilationContext}.
   */
  public static class Builder {
    private String purpose;
    private final Set<Artifact> compilationPrerequisites = new LinkedHashSet<>();
    private final Set<PathFragment> includeDirs = new LinkedHashSet<>();
    private final Set<PathFragment> quoteIncludeDirs = new LinkedHashSet<>();
    private final Set<PathFragment> systemIncludeDirs = new LinkedHashSet<>();
    private final NestedSetBuilder<PathFragment> declaredIncludeDirs =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<PathFragment> declaredIncludeWarnDirs =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> declaredIncludeSrcs =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Pair<Artifact, Artifact>> pregreppedHdrs =
        NestedSetBuilder.stableOrder();
    private final ModuleInfo.Builder moduleInfo = new ModuleInfo.Builder();
    private final ModuleInfo.Builder picModuleInfo = new ModuleInfo.Builder();
    private final NestedSetBuilder<Artifact> directModuleMaps = NestedSetBuilder.stableOrder();
    private final Set<String> defines = new LinkedHashSet<>();
    private CppModuleMap cppModuleMap;

    /** The rule that owns the context */
    private final RuleContext ruleContext;

    /**
     * Creates a new builder for a {@link CppCompilationContext} instance.
     */
    public Builder(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    /**
     * Overrides the purpose of this context. This is useful if a Target
     * needs more than one CppCompilationContext. (The purpose is used to
     * construct the name of the prerequisites middleman for the context, and
     * all artifacts for a given Target must have distinct names.)
     *
     * @param purpose must be a string which is suitable for use as a filename.
     * A single rule may have many middlemen with distinct purposes.
     *
     * @see MiddlemanFactory#createErrorPropagatingMiddleman
     */
    public Builder setPurpose(String purpose) {
      this.purpose = purpose;
      return this;
    }

    public String getPurpose() {
      return purpose;
    }

    /**
     * Merges the context of a dependency into this one by adding the contents
     * of all of its attributes.
     */
    public Builder mergeDependentContext(CppCompilationContext otherContext) {
      Preconditions.checkNotNull(otherContext);
      compilationPrerequisites.addAll(otherContext.getTransitiveCompilationPrerequisites());
      includeDirs.addAll(otherContext.getIncludeDirs());
      quoteIncludeDirs.addAll(otherContext.getQuoteIncludeDirs());
      systemIncludeDirs.addAll(otherContext.getSystemIncludeDirs());
      declaredIncludeDirs.addTransitive(otherContext.getDeclaredIncludeDirs());
      declaredIncludeWarnDirs.addTransitive(otherContext.getDeclaredIncludeWarnDirs());
      declaredIncludeSrcs.addTransitive(otherContext.getDeclaredIncludeSrcs());
      pregreppedHdrs.addTransitive(otherContext.getPregreppedHeaders());
      moduleInfo.addTransitive(otherContext.moduleInfo);
      picModuleInfo.addTransitive(otherContext.picModuleInfo);

      // All module maps of direct dependencies are inputs to the current compile independently of
      // the build type.
      if (otherContext.getCppModuleMap() != null) {
        directModuleMaps.add(otherContext.getCppModuleMap().getArtifact());
      }

      defines.addAll(otherContext.getDefines());
      return this;
    }

    /**
     * Merges the context of some targets into this one by adding the contents
     * of all of their attributes. Targets that do not implement
     * {@link CppCompilationContext} are ignored.
     */
    public Builder mergeDependentContexts(Iterable<CppCompilationContext> targets) {
      for (CppCompilationContext target : targets) {
        mergeDependentContext(target);
      }
      return this;
    }

    /**
     * Adds multiple compilation prerequisites.
     *
     * <p>There are two kinds of "compilation prerequisites": declared header files and pregrepped
     * headers.
     */
    public Builder addCompilationPrerequisites(Iterable<Artifact> prerequisites) {
      // LIPO collector must not add compilation prerequisites in order to avoid
      // the creation of a middleman action.
      for (Artifact prerequisite : prerequisites) {
        String basename = prerequisite.getFilename();
        Preconditions.checkArgument(!Link.OBJECT_FILETYPES.matches(basename));
        Preconditions.checkArgument(!Link.ARCHIVE_LIBRARY_FILETYPES.matches(basename));
        Preconditions.checkArgument(!Link.SHARED_LIBRARY_FILETYPES.matches(basename));
      }
      Iterables.addAll(compilationPrerequisites, prerequisites);
      return this;
    }

    /**
     * Add a single include directory to be added with "-I". It can be either
     * relative to the exec root (see
     * {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}) or
     * absolute. Before it is stored, the include directory is normalized.
     */
    public Builder addIncludeDir(PathFragment includeDir) {
      includeDirs.add(includeDir.normalize());
      return this;
    }

    /**
     * Add multiple include directories to be added with "-I". These can be
     * either relative to the exec root (see {@link
     * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}) or absolute. The
     * entries are normalized before they are stored.
     */
    public Builder addIncludeDirs(Iterable<PathFragment> includeDirs) {
      for (PathFragment includeDir : includeDirs) {
        addIncludeDir(includeDir);
      }
      return this;
    }

    /**
     * Add a single include directory to be added with "-iquote". It can be
     * either relative to the exec root (see {@link
     * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}) or absolute. Before it
     * is stored, the include directory is normalized.
     */
    public Builder addQuoteIncludeDir(PathFragment quoteIncludeDir) {
      quoteIncludeDirs.add(quoteIncludeDir.normalize());
      return this;
    }

    /**
     * Add a single include directory to be added with "-isystem". It can be
     * either relative to the exec root (see {@link
     * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}) or absolute. Before it
     * is stored, the include directory is normalized.
     */
    public Builder addSystemIncludeDir(PathFragment systemIncludeDir) {
      systemIncludeDirs.add(systemIncludeDir.normalize());
      return this;
    }

    /**
     * Add a single declared include dir, relative to a "-I" or "-iquote"
     * directory".
     */
    public Builder addDeclaredIncludeDir(PathFragment dir) {
      declaredIncludeDirs.add(dir);
      return this;
    }

    /**
     * Add a single declared include directory, relative to a "-I" or "-iquote"
     * directory", from which inclusion will produce a warning.
     */
    public Builder addDeclaredIncludeWarnDir(PathFragment dir) {
      declaredIncludeWarnDirs.add(dir);
      return this;
    }

    /**
     * Adds a header that has been declared in the {@code src} or {@code headers attribute}. The
     * header will also be added to the compilation prerequisites.
     */
    public Builder addDeclaredIncludeSrc(Artifact header) {
      declaredIncludeSrcs.add(header);
      compilationPrerequisites.add(header);
      return this;
    }

    /**
     * Adds multiple headers that have been declared in the {@code src} or {@code headers
     * attribute}. The headers will also be added to the compilation prerequisites.
     */
    public Builder addDeclaredIncludeSrcs(Collection<Artifact> declaredIncludeSrcs) {
      this.declaredIncludeSrcs.addAll(declaredIncludeSrcs);
      return addCompilationPrerequisites(declaredIncludeSrcs);
    }

    public Builder addModularHdrs(Collection<Artifact> headers) {
      this.moduleInfo.addHeaders(headers);
      this.picModuleInfo.addHeaders(headers);
      return this;
    }

    public Builder addTextualHdrs(Collection<Artifact> headers) {
      this.moduleInfo.addTextualHeaders(headers);
      this.picModuleInfo.addTextualHeaders(headers);
      return this;
    }

    /**
     * Add a map of generated source or header Artifact to an output Artifact after grepping
     * the file for include statements.
     */
    public Builder addPregreppedHeaderMap(Map<Artifact, Artifact> pregrepped) {
      addCompilationPrerequisites(pregrepped.values());
      for (Map.Entry<Artifact, Artifact> entry : pregrepped.entrySet()) {
        this.pregreppedHdrs.add(Pair.of(entry.getKey(), entry.getValue()));
      }
      return this;
    }

    /**
     * Adds a single define.
     */
    public Builder addDefine(String define) {
      defines.add(define);
      return this;
    }

    /**
     * Adds multiple defines.
     */
    public Builder addDefines(Iterable<String> defines) {
      Iterables.addAll(this.defines, defines);
      return this;
    }

    /**
     * Sets the C++ module map.
     */
    public Builder setCppModuleMap(CppModuleMap cppModuleMap) {
      this.cppModuleMap = cppModuleMap;
      return this;
    }

    /**
     * Sets the C++ header module in non-pic mode.
     * @param headerModule The .pcm file generated for this library.
     */
    public Builder setHeaderModule(Artifact headerModule) {
      this.moduleInfo.setHeaderModule(headerModule);
      return this;
    }

    /**
     * Sets the C++ header module in pic mode.
     * @param picHeaderModule The .pic.pcm file generated for this library.
     */
    public Builder setPicHeaderModule(Artifact picHeaderModule) {
      this.picModuleInfo.setHeaderModule(picHeaderModule);
      return this;
    }

    /**
     * Builds the {@link CppCompilationContext}.
     */
    public CppCompilationContext build() {
      return build(
          ruleContext == null ? null : ruleContext.getActionOwner(),
          ruleContext == null ? null : ruleContext.getAnalysisEnvironment().getMiddlemanFactory());
    }

    @VisibleForTesting  // productionVisibility = Visibility.PRIVATE
    public CppCompilationContext build(ActionOwner owner, MiddlemanFactory middlemanFactory) {
      // We don't create middlemen in LIPO collector subtree, because some target CT
      // will do that instead.
      Artifact prerequisiteStampFile = (ruleContext != null
          && ruleContext.getFragment(CppConfiguration.class).isLipoContextCollector())
          ? getMiddlemanArtifact(middlemanFactory)
          : createMiddleman(owner, middlemanFactory);

      return new CppCompilationContext(
          new CommandLineContext(
              ImmutableList.copyOf(includeDirs),
              ImmutableList.copyOf(quoteIncludeDirs),
              ImmutableList.copyOf(systemIncludeDirs),
              ImmutableList.copyOf(defines)),
          prerequisiteStampFile == null
              ? ImmutableSet.<Artifact>of()
              : ImmutableSet.of(prerequisiteStampFile),
          declaredIncludeDirs.build(),
          declaredIncludeWarnDirs.build(),
          declaredIncludeSrcs.build(),
          pregreppedHdrs.build(),
          moduleInfo.build(),
          picModuleInfo.build(),
          directModuleMaps.build(),
          cppModuleMap);
    }

    /**
     * Creates a middleman for the compilation prerequisites.
     *
     * @return the middleman or null if there are no prerequisites
     */
    private Artifact createMiddleman(ActionOwner owner,
        MiddlemanFactory middlemanFactory) {
      if (compilationPrerequisites.isEmpty()) {
        return null;
      }

      // Compilation prerequisites gathered in the compilationPrerequisites
      // must be generated prior to executing C++ compilation step that depends
      // on them (since these prerequisites include all potential header files, etc
      // that could be referenced during compilation). So there is a definite need
      // to ensure scheduling edge dependency. However, those prerequisites should
      // have no effect on the decision whether C++ compilation should happen in
      // the first place - only CppCompileAction outputs (*.o and *.d files) and
      // all files referenced by the *.d file should be used to make that decision.
      // If this action was never executed, then *.d file would be missing, forcing
      // compilation to occur. If *.d file is present and has not changed then the
      // only reason that would force us to re-compile would be change in one of
      // the files referenced by the *.d file, since no other files participated
      // in the compilation. We also need to propagate errors through this
      // dependency link. So we use an error propagating middleman.
      // Such middleman will be ignored by the dependency checker yet will still
      // represent an edge in the action dependency graph - forcing proper execution
      // order and error propagation.
      String name =
          cppModuleMap != null ? cppModuleMap.getName() : ruleContext.getLabel().toString();
      return middlemanFactory.createErrorPropagatingMiddleman(
          owner, name, purpose,
          ImmutableList.copyOf(compilationPrerequisites),
          ruleContext.getConfiguration().getMiddlemanDirectory(
              ruleContext.getRule().getRepository()));
    }

    /**
     * Returns the same set of artifacts as createMiddleman() would, but without
     * actually creating middlemen.
     */
    private Artifact getMiddlemanArtifact(MiddlemanFactory middlemanFactory) {
      if (compilationPrerequisites.isEmpty()) {
        return null;
      }

      return middlemanFactory.getErrorPropagatingMiddlemanArtifact(
          ruleContext.getLabel().toString(),
          purpose,
          ruleContext.getConfiguration().getMiddlemanDirectory(
              ruleContext.getRule().getRepository()));
    }
  }
  
  /**
   * Gathers data about the direct and transitive .pcm files belonging to this context. Can be to
   * either gather data on PIC or on no-PIC .pcm files.
   */
  @Immutable
  public static final class ModuleInfo {
    /**
     * The module built for this context. If null, then no module is being compiled for this
     * context.
     */
    private final Artifact headerModule;

    /** All header files that are compiled into this module. */
    private final ImmutableSet<Artifact> modularHeaders;

    /** All header files that are contained in this module. */
    private final ImmutableSet<Artifact> textualHeaders;

    /**
     * All transitive modules that this context depends on, excluding headerModule.
     */
    private final NestedSet<Artifact> transitiveModules;

    /**
     * All information about mapping transitive headers to transitive modules.
     */
    public final NestedSet<TransitiveModuleHeaders> transitiveModuleHeaders;

    public ModuleInfo(
        Artifact headerModule,
        ImmutableSet<Artifact> modularHeaders,
        ImmutableSet<Artifact> textualHeaders,
        NestedSet<Artifact> transitiveModules,
        NestedSet<TransitiveModuleHeaders> transitiveModuleHeaders) {
      this.headerModule = headerModule;
      this.modularHeaders = modularHeaders;
      this.textualHeaders = textualHeaders;
      this.transitiveModules = transitiveModules;
      this.transitiveModuleHeaders = transitiveModuleHeaders;
    }

    public Collection<TransitiveModuleHeaders> getUsedModules(Set<Artifact> usedHeaders) {
      List<TransitiveModuleHeaders> result = new ArrayList<>();
      for (TransitiveModuleHeaders transitiveModule : transitiveModuleHeaders) {
        if (transitiveModule.module.equals(headerModule)) {
          // Do not add the module of the current rule for both:
          // 1. the module compile itself
          // 2. compiles of other translation units of the same rule.
          continue;
        }
        boolean providesUsedHeader = false;
        for (Artifact header : transitiveModule.headers) {
          if (usedHeaders.contains(header)) {
            providesUsedHeader = true;
            break;
          }
        }
        if (providesUsedHeader) {
          result.add(transitiveModule);
        }
      }
      return result;
    }

    /**
     * Builder class for {@link ModuleInfo}.
     */
    public static class Builder {
      private Artifact headerModule = null;
      private final Set<Artifact> modularHeaders = new LinkedHashSet<>();
      private final Set<Artifact> textualHeaders = new LinkedHashSet<>();
      private final NestedSetBuilder<Artifact> transitiveModules = NestedSetBuilder.stableOrder();
      private final NestedSetBuilder<TransitiveModuleHeaders> transitiveModuleHeaders =
          NestedSetBuilder.stableOrder();

      public Builder setHeaderModule(Artifact headerModule) {
        this.headerModule = headerModule;
        return this;
      }

      public Builder addHeaders(Collection<Artifact> headers) {
        this.modularHeaders.addAll(headers);
        return this;
      }

      public Builder addTextualHeaders(Collection<Artifact> headers) {
        this.textualHeaders.addAll(headers);
        return this;
      }

      /**
       * Merges a {@link ModuleInfo} into this one. In contrast to addTransitive, this doesn't add
       * the dependent module to transitiveModules, but just merges the transitive sets. The main
       * usage is to merge multiple {@link ModuleInfo} instances for Lipo.
       */
      public Builder merge(ModuleInfo other) {
        if (headerModule == null) {
          headerModule = other.headerModule;
        }
        modularHeaders.addAll(other.modularHeaders);
        textualHeaders.addAll(other.textualHeaders);
        transitiveModules.addTransitive(other.transitiveModules);
        transitiveModuleHeaders.addTransitive(other.transitiveModuleHeaders);
        return this;
      }

      /**
       * Adds the {@link ModuleInfo} of a dependency and builds up the transitive data structures.
       */
      public Builder addTransitive(ModuleInfo moduleInfo) {
        if (moduleInfo.headerModule != null) {
          transitiveModules.add(moduleInfo.headerModule);
        }
        transitiveModules.addTransitive(moduleInfo.transitiveModules);
        transitiveModuleHeaders.addTransitive(moduleInfo.transitiveModuleHeaders);
        return this;
      }

      public ModuleInfo build() {
        ImmutableSet<Artifact> modularHeaders = ImmutableSet.copyOf(this.modularHeaders);
        NestedSet<Artifact> transitiveModules = this.transitiveModules.build();
        if (headerModule != null) {
          transitiveModuleHeaders.add(
              new TransitiveModuleHeaders(headerModule, modularHeaders, transitiveModules));
        }
        return new ModuleInfo(
            headerModule,
            modularHeaders,
            ImmutableSet.copyOf(this.textualHeaders),
            transitiveModules,
            transitiveModuleHeaders.build());
      }
    }
  }

  /**
   * Collects data for a specific module in a special format that makes pruning easy.
   */
  @Immutable
  public static final class TransitiveModuleHeaders {
    /**
     * The module that we are calculating information for.
     */
    private final Artifact module;

    /**
     * The headers compiled into this module.
     */
    private final ImmutableSet<Artifact> headers;

    /**
     * This nested set contains 'module' as well as all targets it transitively depends on.
     * If any of the 'headers' is used, all of these modules a required for the compilation.
     */
    private final NestedSet<Artifact> transitiveModules;

    public TransitiveModuleHeaders(
        Artifact module,
        ImmutableSet<Artifact> headers,
        NestedSet<Artifact> transitiveModules) {
      this.module = module;
      this.headers = headers;
      this.transitiveModules = transitiveModules;
    }

    public Artifact getModule() {
      return module;
    }

    public Collection<Artifact> getTransitiveModules() {
      return transitiveModules.toCollection();
    }
  }
}
