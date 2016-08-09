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
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
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

  /**
   * All declared headers of the current module, if compiled as a header module.
   */
  private final NestedSet<Artifact> headerModuleSrcs;

  /**
   * All header modules in the transitive closure of {@code topLevelHeaderModules}.
   */
  private final NestedSet<Artifact> impliedHeaderModules;
  private final NestedSet<Artifact> picImpliedHeaderModules;

  private final NestedSet<Pair<Artifact, Artifact>> pregreppedHdrs;

  /**
   * All header modules in our transitive closure that are not in the transitive closure of
   * another header module in our transitive closure.
   */
  private final NestedSet<Artifact> topLevelHeaderModules;
  private final NestedSet<Artifact> picTopLevelHeaderModules;

  /**
   * The module maps from all targets the current target depends on transitively.
   */
  private final NestedSet<Artifact> transitiveModuleMaps;

  private final CppModuleMap cppModuleMap;
  private final Artifact headerModule;
  private final Artifact picHeaderModule;
  
  // True if this context is for a compilation that needs transitive module maps.
  private final boolean provideTransitiveModuleMaps;
  
  // True if this context is for a compilation that should use header modules from dependencies.
  private final boolean useHeaderModules;

  // Derived from depsContexts; no need to consider it for equals/hashCode.
  private final ImmutableSet<Artifact> compilationPrerequisites;

  private CppCompilationContext(
      CommandLineContext commandLineContext,
      ImmutableSet<Artifact> compilationPrerequisites,
      NestedSet<PathFragment> declaredIncludeDirs,
      NestedSet<PathFragment> declaredIncludeWarnDirs,
      NestedSet<Artifact> declaredIncludeSrcs,
      NestedSet<Pair<Artifact, Artifact>> pregreppedHdrs,
      NestedSet<Artifact> headerModuleSrcs,
      NestedSet<Artifact> topLevelHeaderModules,
      NestedSet<Artifact> picTopLevelHeaderModules,
      NestedSet<Artifact> impliedHeaderModules,
      NestedSet<Artifact> picImpliedHeaderModules,
      NestedSet<Artifact> transitiveModuleMaps,
      NestedSet<Artifact> directModuleMaps,
      CppModuleMap cppModuleMap,
      Artifact headerModule,
      Artifact picHeaderModule,
      boolean provideTransitiveModuleMaps,
      boolean useHeaderModules) {
    Preconditions.checkNotNull(commandLineContext);
    this.commandLineContext = commandLineContext;
    this.declaredIncludeDirs = declaredIncludeDirs;
    this.declaredIncludeWarnDirs = declaredIncludeWarnDirs;
    this.declaredIncludeSrcs = declaredIncludeSrcs;
    this.directModuleMaps = directModuleMaps;
    this.headerModuleSrcs = headerModuleSrcs;
    this.impliedHeaderModules = impliedHeaderModules;
    this.picImpliedHeaderModules = picImpliedHeaderModules;
    this.pregreppedHdrs = pregreppedHdrs;
    this.topLevelHeaderModules = topLevelHeaderModules;
    this.picTopLevelHeaderModules = picTopLevelHeaderModules;
    this.transitiveModuleMaps = transitiveModuleMaps;
    this.cppModuleMap = cppModuleMap;
    this.headerModule = headerModule;
    this.picHeaderModule = picHeaderModule;
    this.provideTransitiveModuleMaps = provideTransitiveModuleMaps;
    this.useHeaderModules = useHeaderModules;
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

  /**
   * Returns the immutable set of additional transitive inputs needed for
   * compilation, like C++ module map artifacts.
   */
  public NestedSet<Artifact> getAdditionalInputs(boolean usePic) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    if (useHeaderModules) {
      if (usePic) {
        builder.addTransitive(picTopLevelHeaderModules);
        builder.addTransitive(picImpliedHeaderModules);
      } else {
        builder.addTransitive(topLevelHeaderModules);
        builder.addTransitive(impliedHeaderModules);
      }
    }
    builder.addTransitive(directModuleMaps);
    if (provideTransitiveModuleMaps) {
      builder.addTransitive(transitiveModuleMaps);
    }
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
   * @return modules maps in the transitive closure that are not from direct dependencies.
   */
  private NestedSet<Artifact> getTransitiveModuleMaps() {
    return transitiveModuleMaps;
  }

  /**
   * @return all declared headers of the current module if the current target
   * is compiled as a module.
   */
  protected NestedSet<Artifact> getHeaderModuleSrcs() {
    return headerModuleSrcs;
  }

  /**
   * @return all header modules in our transitive closure that are not in the transitive closure
   * of another header module in our transitive closure.
   */
  protected NestedSet<Artifact> getTopLevelHeaderModules(boolean usePic) {
    return usePic ? picTopLevelHeaderModules : topLevelHeaderModules;
  }

  /**
   * @return all header modules in the transitive closure of {@code getTopLevelHeaderModules()}.
   */
  protected NestedSet<Artifact> getImpliedHeaderModules(boolean usePic) {
    return usePic ? picImpliedHeaderModules : impliedHeaderModules;
  }

  /**
   * Returns the set of defines needed to compile this target (possibly empty
   * but never null). This includes definitions from the transitive deps closure
   * for the target. The order of the returned collection is deterministic.
   */
  public ImmutableList<String> getDefines() {
    return commandLineContext.defines;
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }
    if (!(obj instanceof CppCompilationContext)) {
      return false;
    }
    CppCompilationContext other = (CppCompilationContext) obj;
    return Objects.equals(headerModule, other.headerModule)
        && Objects.equals(cppModuleMap, other.cppModuleMap)
        && Objects.equals(picHeaderModule, other.picHeaderModule)
        && commandLineContext.equals(other.commandLineContext)
        && Objects.equals(declaredIncludeDirs, other.declaredIncludeDirs)
        && Objects.equals(declaredIncludeWarnDirs, other.declaredIncludeWarnDirs)
        && Objects.equals(declaredIncludeSrcs, other.declaredIncludeSrcs)
        && Objects.equals(directModuleMaps, other.directModuleMaps)
        && Objects.equals(headerModuleSrcs, other.headerModuleSrcs)
        && Objects.equals(impliedHeaderModules, other.impliedHeaderModules)
        && Objects.equals(picImpliedHeaderModules, other.picImpliedHeaderModules)
        && Objects.equals(topLevelHeaderModules, other.topLevelHeaderModules)
        && Objects.equals(picTopLevelHeaderModules, other.picTopLevelHeaderModules)
        && Objects.equals(transitiveModuleMaps, other.transitiveModuleMaps)
        && Objects.equals(compilationPrerequisites, other.compilationPrerequisites)
        && (provideTransitiveModuleMaps == other.provideTransitiveModuleMaps)
        && (useHeaderModules == other.useHeaderModules);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        headerModule,
        picHeaderModule,
        commandLineContext,
        declaredIncludeDirs,
        declaredIncludeWarnDirs,
        declaredIncludeSrcs,
        directModuleMaps,
        headerModuleSrcs,
        impliedHeaderModules,
        picImpliedHeaderModules,
        transitiveModuleMaps,
        topLevelHeaderModules,
        picTopLevelHeaderModules,
        compilationPrerequisites,
        cppModuleMap,
        provideTransitiveModuleMaps,
        useHeaderModules);
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
        context.headerModuleSrcs,
        context.topLevelHeaderModules,
        context.picTopLevelHeaderModules,
        context.impliedHeaderModules,
        context.picImpliedHeaderModules,
        context.transitiveModuleMaps,
        context.directModuleMaps,
        context.cppModuleMap,
        context.headerModule,
        context.picHeaderModule,
        context.provideTransitiveModuleMaps,
        context.useHeaderModules);
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
    return new CppCompilationContext(
        libContext.commandLineContext,
        prerequisites.build(),
        mergeSets(ownerContext.declaredIncludeDirs, libContext.declaredIncludeDirs),
        mergeSets(ownerContext.declaredIncludeWarnDirs, libContext.declaredIncludeWarnDirs),
        mergeSets(ownerContext.declaredIncludeSrcs, libContext.declaredIncludeSrcs),
        mergeSets(ownerContext.pregreppedHdrs, libContext.pregreppedHdrs),
        mergeSets(ownerContext.headerModuleSrcs, libContext.headerModuleSrcs),
        mergeSets(ownerContext.topLevelHeaderModules, libContext.topLevelHeaderModules),
        mergeSets(ownerContext.picTopLevelHeaderModules, libContext.picTopLevelHeaderModules),
        mergeSets(ownerContext.impliedHeaderModules, libContext.impliedHeaderModules),
        mergeSets(ownerContext.picImpliedHeaderModules, libContext.picImpliedHeaderModules),
        mergeSets(ownerContext.transitiveModuleMaps, libContext.transitiveModuleMaps),
        mergeSets(ownerContext.directModuleMaps, libContext.directModuleMaps),
        libContext.cppModuleMap,
        libContext.headerModule,
        libContext.picHeaderModule,
        libContext.provideTransitiveModuleMaps,
        libContext.useHeaderModules);
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

  /**
   * @return the C++ module map of the owner.
   */
  public CppModuleMap getCppModuleMap() {
    return cppModuleMap;
  }

  /**
   * @return the non-pic C++ header module of the owner.
   */
  private Artifact getHeaderModule() {
    return headerModule;
  }

  /**
   * @return the pic C++ header module of the owner.
   */
  private Artifact getPicHeaderModule() {
    return picHeaderModule;
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

    @Override
    public boolean equals(Object obj) {
      if (obj == this) {
        return true;
      }
      if (!(obj instanceof CommandLineContext)) {
        return false;
      }
      CommandLineContext other = (CommandLineContext) obj;
      return Objects.equals(includeDirs, other.includeDirs)
          && Objects.equals(quoteIncludeDirs, other.quoteIncludeDirs)
          && Objects.equals(systemIncludeDirs, other.systemIncludeDirs)
          && Objects.equals(defines, other.defines);
    }

    @Override
    public int hashCode() {
      return Objects.hash(includeDirs, quoteIncludeDirs, systemIncludeDirs, defines);
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
    private final NestedSetBuilder<Artifact> headerModuleSrcs =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> topLevelHeaderModules =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> picTopLevelHeaderModules =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> impliedHeaderModules =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> picImpliedHeaderModules =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> transitiveModuleMaps =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> directModuleMaps =
        NestedSetBuilder.stableOrder();
    private final Set<String> defines = new LinkedHashSet<>();
    private CppModuleMap cppModuleMap;
    private Artifact headerModule;
    private Artifact picHeaderModule;
    private boolean provideTransitiveModuleMaps = false;
    private boolean useHeaderModules = false;

    /** The rule that owns the context */
    private final RuleContext ruleContext;

    /**
     * Creates a new builder for a {@link CppCompilationContext} instance.
     */
    public Builder(RuleContext ruleContext) {
      this(ruleContext, /*forInterface=*/ false);
    }

    /**
     * Creates a new builder for a {@link CppCompilationContext} instance.
     *
     * @param forInterface if true, this context is designated for the compilation of an interface.
     */
    public Builder(RuleContext ruleContext, boolean forInterface) {
      this.ruleContext = ruleContext;
      this.purpose = forInterface ? "cpp_interface_prerequisites" : "cpp_compilation_prerequisites";
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

      NestedSet<Artifact> othersTransitiveModuleMaps = otherContext.getTransitiveModuleMaps();
      NestedSet<Artifact> othersDirectModuleMaps = otherContext.getDirectModuleMaps();
      NestedSet<Artifact> othersTopLevelHeaderModules =
          otherContext.getTopLevelHeaderModules(/*usePic=*/ false);
      NestedSet<Artifact> othersPicTopLevelHeaderModules =
          otherContext.getTopLevelHeaderModules(/*usePic=*/ true);

      // Forward transitive information.
      // The other target's transitive module maps do not include its direct module maps, so we
      // add both.
      transitiveModuleMaps.addTransitive(othersTransitiveModuleMaps);
      transitiveModuleMaps.addTransitive(othersDirectModuleMaps);
      impliedHeaderModules.addTransitive(otherContext.getImpliedHeaderModules(/*usePic=*/ false));
      picImpliedHeaderModules.addTransitive(otherContext.getImpliedHeaderModules(/*usePic=*/ true));
      topLevelHeaderModules.addTransitive(othersTopLevelHeaderModules);
      picTopLevelHeaderModules.addTransitive(othersPicTopLevelHeaderModules);

      // All module maps of direct dependencies are inputs to the current compile independently of
      // the build type.
      if (otherContext.getCppModuleMap() != null) {
        directModuleMaps.add(otherContext.getCppModuleMap().getArtifact());
      }
      if (otherContext.getHeaderModule() != null || otherContext.getPicHeaderModule() != null) {
        // If the other context is for a target that compiles a header module, that context's
        // header module becomes our top-level header module, and its top-level header modules
        // become our implied header modules.
        impliedHeaderModules.addTransitive(othersTopLevelHeaderModules);
        picImpliedHeaderModules.addTransitive(othersPicTopLevelHeaderModules);
        if (otherContext.getHeaderModule() != null) {
          topLevelHeaderModules.add(otherContext.getHeaderModule());
        }
        if (otherContext.getPicHeaderModule() != null) {
          picTopLevelHeaderModules.add(otherContext.getPicHeaderModule());
        }
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
     */
    public Builder addCompilationPrerequisites(Iterable<Artifact> prerequisites) {
      // LIPO collector must not add compilation prerequisites in order to avoid
      // the creation of a middleman action.
      for (Artifact prerequisite : prerequisites) {
        Preconditions.checkArgument(!Link.OBJECT_FILETYPES.matches(prerequisite.getFilename()));
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
      headerModuleSrcs.add(header);
      return this;
    }

    /**
     * Adds multiple headers that have been declared in the {@code src} or {@code headers
     * attribute}. The headers will also be added to the compilation prerequisites.
     */
    public Builder addDeclaredIncludeSrcs(Iterable<Artifact> declaredIncludeSrcs) {
      this.declaredIncludeSrcs.addAll(declaredIncludeSrcs);
      this.headerModuleSrcs.addAll(declaredIncludeSrcs);
      return addCompilationPrerequisites(declaredIncludeSrcs);
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
     */
    public Builder setHeaderModule(Artifact headerModule) {
      this.headerModule = headerModule;
      return this;
    }

    /**
     * Sets the C++ header module in pic mode.
     */
    public Builder setPicHeaderModule(Artifact picHeaderModule) {
      this.picHeaderModule = picHeaderModule;
      return this;
    }
    
    /**
     * Sets that the context will be used by a compilation that needs transitive module maps.
     */
    public Builder setProvideTransitiveModuleMaps(boolean provideTransitiveModuleMaps) {
      this.provideTransitiveModuleMaps = provideTransitiveModuleMaps;
      return this;
    }
    
    /**
     * Sets that the context will be used by a compilation that uses header modules provided by
     * its dependencies.
     */
    public Builder setUseHeaderModules(boolean useHeaderModules) {
      this.useHeaderModules = useHeaderModules;
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
      // During merging we might have put header modules into topLevelHeaderModules that were
      // also in the transitive closure of a different header module; we need to filter those out.
      NestedSet<Artifact> impliedHeaderModules = this.impliedHeaderModules.build();
      NestedSet<Artifact> topLevelHeaderModules =
          filterTopLevelHeaderModules(this.topLevelHeaderModules.build(), impliedHeaderModules);
      NestedSet<Artifact> picImpliedHeaderModules = this.picImpliedHeaderModules.build();
      NestedSet<Artifact> picTopLevelHeaderModules =
          filterTopLevelHeaderModules(
              this.picTopLevelHeaderModules.build(), picImpliedHeaderModules);
      
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
          headerModuleSrcs.build(),
          topLevelHeaderModules,
          picTopLevelHeaderModules,
          impliedHeaderModules,
          picImpliedHeaderModules,
          transitiveModuleMaps.build(),
          directModuleMaps.build(),
          cppModuleMap,
          headerModule,
          picHeaderModule,
          provideTransitiveModuleMaps,
          useHeaderModules);
    }
    
    /**
     * Filter out artifacts from {@code topLevelHeaderModuels} that are also in
     * {@code impliedHeaderModules}.
     */
    private static NestedSet<Artifact> filterTopLevelHeaderModules(
        NestedSet<Artifact> topLevelHeaderModules, NestedSet<Artifact> impliedHeaderModules) {
      NestedSetBuilder<Artifact> filtered = NestedSetBuilder.stableOrder();
      Set<Artifact> impliedHeaderModulesSet = impliedHeaderModules.toSet();
      for (Artifact artifact : topLevelHeaderModules) {
        if (!impliedHeaderModulesSet.contains(artifact)) {
          filtered.add(artifact);
        }
      }
      return filtered.build();
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
      return middlemanFactory.createErrorPropagatingMiddleman(
          owner, ruleContext.getLabel().toString(), purpose,
          ImmutableList.copyOf(compilationPrerequisites),
          ruleContext.getConfiguration().getMiddlemanDirectory());
    }

    /**
     * Returns the same set of artifacts as createMiddleman() would, but without
     * actually creating middlemen.
     */
    private Artifact getMiddlemanArtifact(MiddlemanFactory middlemanFactory) {
      if (compilationPrerequisites.isEmpty()) {
        return null;
      }

      return middlemanFactory.getErrorPropagatingMiddlemanArtifact(ruleContext.getLabel()
          .toString(), purpose, ruleContext.getConfiguration().getMiddlemanDirectory());
    }
  }
}
