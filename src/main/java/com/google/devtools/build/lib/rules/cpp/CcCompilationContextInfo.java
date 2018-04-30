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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CppHelper.PregreppedHeader;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Immutable store of information needed for C++ compilation that is aggregated across dependencies.
 */
@Immutable
@AutoCodec
@SkylarkModule(
  name = "cc_compilation_info",
  documented = false,
  category = SkylarkModuleCategory.PROVIDER,
  doc =
      "Immutable store of information needed for C++ compilation that is aggregated across "
          + "dependencies."
)
// TODO(b/77669139): Rename to CcCompilationContext.
public final class CcCompilationContextInfo {
  /** An empty {@code CcCompilationContextInfo}. */
  public static final CcCompilationContextInfo EMPTY = new Builder(null).build();

  private final CommandLineCcCompilationContextInfo commandLineCcCompilationContextInfo;

  private final NestedSet<PathFragment> declaredIncludeDirs;
  private final NestedSet<PathFragment> declaredIncludeWarnDirs;
  private final NestedSet<Artifact> declaredIncludeSrcs;

  /**
   * Module maps from direct dependencies.
   */
  private final NestedSet<Artifact> directModuleMaps;

  /** Non-code mandatory compilation inputs. */
  private final NestedSet<Artifact> nonCodeInputs;

  private final NestedSet<PregreppedHeader> pregreppedHdrs;

  private final ModuleInfo moduleInfo;
  private final ModuleInfo picModuleInfo;

  private final CppModuleMap cppModuleMap;
  private final CppModuleMap verificationModuleMap;

  private final boolean propagateModuleMapAsActionInput;

  // Derived from depsContexts.
  private final ImmutableSet<Artifact> compilationPrerequisites;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  CcCompilationContextInfo(
      CommandLineCcCompilationContextInfo commandLineCcCompilationContextInfo,
      ImmutableSet<Artifact> compilationPrerequisites,
      NestedSet<PathFragment> declaredIncludeDirs,
      NestedSet<PathFragment> declaredIncludeWarnDirs,
      NestedSet<Artifact> declaredIncludeSrcs,
      NestedSet<PregreppedHeader> pregreppedHdrs,
      NestedSet<Artifact> nonCodeInputs,
      ModuleInfo moduleInfo,
      ModuleInfo picModuleInfo,
      NestedSet<Artifact> directModuleMaps,
      CppModuleMap cppModuleMap,
      @Nullable CppModuleMap verificationModuleMap,
      boolean propagateModuleMapAsActionInput) {
    Preconditions.checkNotNull(commandLineCcCompilationContextInfo);
    this.commandLineCcCompilationContextInfo = commandLineCcCompilationContextInfo;
    this.declaredIncludeDirs = declaredIncludeDirs;
    this.declaredIncludeWarnDirs = declaredIncludeWarnDirs;
    this.declaredIncludeSrcs = declaredIncludeSrcs;
    this.directModuleMaps = directModuleMaps;
    this.pregreppedHdrs = pregreppedHdrs;
    this.moduleInfo = moduleInfo;
    this.picModuleInfo = picModuleInfo;
    this.cppModuleMap = cppModuleMap;
    this.nonCodeInputs = nonCodeInputs;
    this.verificationModuleMap = verificationModuleMap;
    this.compilationPrerequisites = compilationPrerequisites;
    this.propagateModuleMapAsActionInput = propagateModuleMapAsActionInput;
  }

  /**
   * Returns the transitive compilation prerequisites consolidated into middlemen prerequisites, or
   * an empty set if there are no prerequisites.
   *
   * <p>Transitive compilation prerequisites are the prerequisites that will be needed by all
   * reverse dependencies; note that these do specifically not include any compilation prerequisites
   * that are only needed by the rule itself (for example, compiled source files from the {@code
   * srcs} attribute).
   *
   * <p>To reduce the number of edges in the action graph, we express the dependency on compilation
   * prerequisites as a transitive dependency via a middleman. After they have been accumulated
   * (using {@link Builder#addCompilationPrerequisites(Iterable)}, {@link
   * Builder#mergeDependentCcCompilationContextInfo(CcCompilationContextInfo)}, and {@link
   * Builder#mergeDependentCcCompilationContextInfos(Iterable)}, they are consolidated into a single
   * middleman Artifact when {@link Builder#build()} is called.
   *
   * <p>The returned set can be empty if there are no prerequisites. Usually it contains a single
   * middleman, but if LIPO is used there can be two.
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
    return commandLineCcCompilationContextInfo.includeDirs;
  }

  /**
   * Returns the immutable list of include directories to be added with
   * "-iquote" (possibly empty but never null). This includes the include dirs
   * from the transitive deps closure of the target. This list does not contain
   * duplicates. All fragments are either absolute or relative to the exec root
   * (see {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}).
   */
  public ImmutableList<PathFragment> getQuoteIncludeDirs() {
    return commandLineCcCompilationContextInfo.quoteIncludeDirs;
  }

  /**
   * Returns the immutable list of include directories to be added with
   * "-isystem" (possibly empty but never null). This includes the include dirs
   * from the transitive deps closure of the target. This list does not contain
   * duplicates. All fragments are either absolute or relative to the exec root
   * (see {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}).
   */
  public ImmutableList<PathFragment> getSystemIncludeDirs() {
    return commandLineCcCompilationContextInfo.systemIncludeDirs;
  }

  /**
   * Returns the immutable set of declared include directories, relative to a "-I" or "-iquote"
   * directory" (possibly empty but never null).
   */
  public NestedSet<PathFragment> getDeclaredIncludeDirs() {
    return declaredIncludeDirs;
  }

  /**
   * Returns the immutable set of include directories, relative to a "-I" or "-iquote" directory",
   * from which inclusion will produce a warning (possibly empty but never null).
   */
  public NestedSet<PathFragment> getDeclaredIncludeWarnDirs() {
    return declaredIncludeWarnDirs;
  }

  /**
   * Returns the immutable set of headers that have been declared in the {@code srcs} or {@code
   * hdrs} attribute (possibly empty but never null).
   */
  public NestedSet<Artifact> getDeclaredIncludeSrcs() {
    return declaredIncludeSrcs;
  }

  /** Returns headers given as textual_hdrs in this target. */
  public ImmutableSet<Artifact> getTextualHdrs() {
    return moduleInfo.textualHeaders;
  }

  /**
   * Returns the immutable pairs of (header file, pregrepped header file). The value artifacts
   * (pregrepped header file) are generated by {@link ExtractInclusionAction}.
   */
  NestedSet<PregreppedHeader> getPregreppedHeaders() {
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
    builder.addTransitive(nonCodeInputs);
    if (cppModuleMap != null && propagateModuleMapAsActionInput) {
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
    return commandLineCcCompilationContextInfo.defines;
  }

  /**
   * Returns a {@code CcCompilationContextInfo} that is based on a given {@code
   * CcCompilationContextInfo} but returns empty sets for {@link #getDeclaredIncludeDirs()} and
   * {@link #getDeclaredIncludeWarnDirs()}.
   */
  public static CcCompilationContextInfo disallowUndeclaredHeaders(
      CcCompilationContextInfo ccCompilationContextInfo) {
    return new CcCompilationContextInfo(
        ccCompilationContextInfo.commandLineCcCompilationContextInfo,
        ccCompilationContextInfo.compilationPrerequisites,
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        ccCompilationContextInfo.declaredIncludeSrcs,
        ccCompilationContextInfo.pregreppedHdrs,
        ccCompilationContextInfo.nonCodeInputs,
        ccCompilationContextInfo.moduleInfo,
        ccCompilationContextInfo.picModuleInfo,
        ccCompilationContextInfo.directModuleMaps,
        ccCompilationContextInfo.cppModuleMap,
        ccCompilationContextInfo.verificationModuleMap,
        ccCompilationContextInfo.propagateModuleMapAsActionInput);
  }

  /**
   * Returns the context for a LIPO compile action. This uses the include dirs and defines of the
   * library, but the declared inclusion dirs/srcs from both the library and the owner binary.
   *
   * <p>TODO(bazel-team): this might make every LIPO target have an unnecessary large set of
   * inclusion dirs/srcs. The correct behavior would be to merge only the contexts of actual
   * referred targets (as listed in .imports file).
   *
   * <p>Undeclared inclusion checking ({@link #getDeclaredIncludeDirs()}, {@link
   * #getDeclaredIncludeWarnDirs()}, and {@link #getDeclaredIncludeSrcs()}) needs to use the union
   * of the contexts of the involved source files.
   *
   * <p>For include and define command line flags ({@link #getIncludeDirs()} {@link
   * #getQuoteIncludeDirs()}, {@link #getSystemIncludeDirs()}, and {@link #getDefines()}) LIPO
   * compilations use the same values as non-LIPO compilation.
   *
   * <p>Include scanning is not handled by this method. See {@code
   * IncludeScannable#getAuxiliaryScannables()} instead.
   *
   * @param ownerCcCompilationContextInfo the {@code CcCompilationContextInfo} of the owner binary
   * @param libCcCompilationContextInfo the {@code CcCompilationContextInfo} of the library
   */
  public static CcCompilationContextInfo mergeForLipo(
      CcCompilationContextInfo ownerCcCompilationContextInfo,
      CcCompilationContextInfo libCcCompilationContextInfo) {
    ImmutableSet.Builder<Artifact> prerequisites = ImmutableSet.builder();
    prerequisites.addAll(ownerCcCompilationContextInfo.compilationPrerequisites);
    prerequisites.addAll(libCcCompilationContextInfo.compilationPrerequisites);
    ModuleInfo.Builder moduleInfo = new ModuleInfo.Builder();
    moduleInfo.merge(ownerCcCompilationContextInfo.moduleInfo);
    moduleInfo.merge(libCcCompilationContextInfo.moduleInfo);
    ModuleInfo.Builder picModuleInfo = new ModuleInfo.Builder();
    picModuleInfo.merge(ownerCcCompilationContextInfo.picModuleInfo);
    picModuleInfo.merge(libCcCompilationContextInfo.picModuleInfo);
    return new CcCompilationContextInfo(
        libCcCompilationContextInfo.commandLineCcCompilationContextInfo,
        prerequisites.build(),
        mergeSets(
            ownerCcCompilationContextInfo.declaredIncludeDirs,
            libCcCompilationContextInfo.declaredIncludeDirs),
        mergeSets(
            ownerCcCompilationContextInfo.declaredIncludeWarnDirs,
            libCcCompilationContextInfo.declaredIncludeWarnDirs),
        mergeSets(
            ownerCcCompilationContextInfo.declaredIncludeSrcs,
            libCcCompilationContextInfo.declaredIncludeSrcs),
        mergeSets(
            ownerCcCompilationContextInfo.pregreppedHdrs,
            libCcCompilationContextInfo.pregreppedHdrs),
        mergeSets(
            ownerCcCompilationContextInfo.nonCodeInputs, libCcCompilationContextInfo.nonCodeInputs),
        moduleInfo.build(),
        picModuleInfo.build(),
        mergeSets(
            ownerCcCompilationContextInfo.directModuleMaps,
            libCcCompilationContextInfo.directModuleMaps),
        libCcCompilationContextInfo.cppModuleMap,
        libCcCompilationContextInfo.verificationModuleMap,
        libCcCompilationContextInfo.propagateModuleMapAsActionInput);
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

  /** @return the C++ module map of the owner. */
  public CppModuleMap getVerificationModuleMap() {
    return verificationModuleMap;
  }

  /**
   * The parts of the {@code CcCompilationContextInfo} that influence the command line of
   * compilation actions.
   */
  @Immutable
  @AutoCodec
  @VisibleForSerialization
  static class CommandLineCcCompilationContextInfo {
    private final ImmutableList<PathFragment> includeDirs;
    private final ImmutableList<PathFragment> quoteIncludeDirs;
    private final ImmutableList<PathFragment> systemIncludeDirs;
    private final ImmutableList<String> defines;

    CommandLineCcCompilationContextInfo(
        ImmutableList<PathFragment> includeDirs,
        ImmutableList<PathFragment> quoteIncludeDirs,
        ImmutableList<PathFragment> systemIncludeDirs,
        ImmutableList<String> defines) {
      this.includeDirs = includeDirs;
      this.quoteIncludeDirs = quoteIncludeDirs;
      this.systemIncludeDirs = systemIncludeDirs;
      this.defines = defines;
    }
  }

  /** Builder class for {@link CcCompilationContextInfo}. */
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
    private final NestedSetBuilder<PregreppedHeader> pregreppedHdrs =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> nonCodeInputs = NestedSetBuilder.stableOrder();
    private final ModuleInfo.Builder moduleInfo = new ModuleInfo.Builder();
    private final ModuleInfo.Builder picModuleInfo = new ModuleInfo.Builder();
    private final NestedSetBuilder<Artifact> directModuleMaps = NestedSetBuilder.stableOrder();
    private final Set<String> defines = new LinkedHashSet<>();
    private CppModuleMap cppModuleMap;
    private CppModuleMap verificationModuleMap;
    private boolean propagateModuleMapAsActionInput = true;

    /** The rule that owns the context */
    private final RuleContext ruleContext;

    /** Creates a new builder for a {@link CcCompilationContextInfo} instance. */
    public Builder(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    /**
     * Overrides the purpose of this context. This is useful if a Target needs more than one
     * CcCompilationContextInfo. (The purpose is used to construct the name of the prerequisites
     * middleman for the context, and all artifacts for a given Target must have distinct names.)
     *
     * @param purpose must be a string which is suitable for use as a filename. A single rule may
     *     have many middlemen with distinct purposes.
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
     * Merges the {@link CcCompilationContextInfo} of a dependency into this one by adding the
     * contents of all of its attributes.
     */
    public Builder mergeDependentCcCompilationContextInfo(
        CcCompilationContextInfo otherCcCompilationContextInfo) {
      Preconditions.checkNotNull(otherCcCompilationContextInfo);
      compilationPrerequisites.addAll(
          otherCcCompilationContextInfo.getTransitiveCompilationPrerequisites());
      includeDirs.addAll(otherCcCompilationContextInfo.getIncludeDirs());
      quoteIncludeDirs.addAll(otherCcCompilationContextInfo.getQuoteIncludeDirs());
      systemIncludeDirs.addAll(otherCcCompilationContextInfo.getSystemIncludeDirs());
      declaredIncludeDirs.addTransitive(otherCcCompilationContextInfo.getDeclaredIncludeDirs());
      declaredIncludeWarnDirs.addTransitive(
          otherCcCompilationContextInfo.getDeclaredIncludeWarnDirs());
      declaredIncludeSrcs.addTransitive(otherCcCompilationContextInfo.getDeclaredIncludeSrcs());
      pregreppedHdrs.addTransitive(otherCcCompilationContextInfo.getPregreppedHeaders());
      moduleInfo.addTransitive(otherCcCompilationContextInfo.moduleInfo);
      picModuleInfo.addTransitive(otherCcCompilationContextInfo.picModuleInfo);
      nonCodeInputs.addTransitive(otherCcCompilationContextInfo.nonCodeInputs);

      // All module maps of direct dependencies are inputs to the current compile independently of
      // the build type.
      if (otherCcCompilationContextInfo.getCppModuleMap() != null) {
        directModuleMaps.add(otherCcCompilationContextInfo.getCppModuleMap().getArtifact());
      }

      defines.addAll(otherCcCompilationContextInfo.getDefines());
      return this;
    }

    /**
     * Merges the {@code CcCompilationContextInfo}s of some targets into this one by adding the
     * contents of all of their attributes. Targets that do not implement {@link
     * CcCompilationContextInfo} are ignored.
     */
    public Builder mergeDependentCcCompilationContextInfos(
        Iterable<CcCompilationContextInfo> targets) {
      for (CcCompilationContextInfo target : targets) {
        mergeDependentCcCompilationContextInfo(target);
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
      includeDirs.add(includeDir);
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
      quoteIncludeDirs.add(quoteIncludeDir);
      return this;
    }

    /**
     * Add a single include directory to be added with "-isystem". It can be
     * either relative to the exec root (see {@link
     * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}) or absolute. Before it
     * is stored, the include directory is normalized.
     */
    public Builder addSystemIncludeDir(PathFragment systemIncludeDir) {
      systemIncludeDirs.add(systemIncludeDir);
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
     *
     * <p>Filters out fileset directory artifacts, which are not valid inputs.
     */
    public Builder addDeclaredIncludeSrc(Artifact header) {
      if (!header.isFileset()) {
        declaredIncludeSrcs.add(header);
        compilationPrerequisites.add(header);
      }
      return this;
    }

    /**
     * Adds multiple headers that have been declared in the {@code src} or {@code headers
     * attribute}. The headers will also be added to the compilation prerequisites.
     *
     * <p>Filters out fileset directory artifacts, which are not valid inputs.
     */
    public Builder addDeclaredIncludeSrcs(Collection<Artifact> declaredIncludeSrcs) {
      for (Artifact source : declaredIncludeSrcs) {
        addDeclaredIncludeSrc(source);
      }
      return this;
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
     * Add a map of generated source or header Artifact to an output Artifact after grepping the
     * file for include statements.
     */
    public Builder addPregreppedHeaders(List<PregreppedHeader> pregrepped) {
      addCompilationPrerequisites(
          pregrepped
              .stream()
              .map(pregreppedHeader -> pregreppedHeader.greppedHeader())
              .collect(Collectors.toList()));
      this.pregreppedHdrs.addAll(pregrepped);
      return this;
    }

    /** Add a set of required non-code compilation input. */
    public Builder addNonCodeInputs(Iterable<Artifact> inputs) {
      nonCodeInputs.addAll(inputs);
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

    /** Sets the C++ module map. */
    public Builder setCppModuleMap(CppModuleMap cppModuleMap) {
      this.cppModuleMap = cppModuleMap;
      return this;
    }

    /** Sets the C++ module map used to verify that headers are modules compatible. */
    public Builder setVerificationModuleMap(CppModuleMap verificationModuleMap) {
      this.verificationModuleMap = verificationModuleMap;
      return this;
    }

    /**
     * Causes the module map to be passed as an action input to dependant compilations.
     */
    public Builder setPropagateCppModuleMapAsActionInput(boolean propagateModuleMap) {
      this.propagateModuleMapAsActionInput = propagateModuleMap;
      return this;
    }

    /**
     * Sets the C++ header module in non-pic mode.
     *
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

    /** Builds the {@link CcCompilationContextInfo}. */
    public CcCompilationContextInfo build() {
      return build(
          ruleContext == null ? null : ruleContext.getActionOwner(),
          ruleContext == null ? null : ruleContext.getAnalysisEnvironment().getMiddlemanFactory());
    }

    @VisibleForTesting // productionVisibility = Visibility.PRIVATE
    public CcCompilationContextInfo build(ActionOwner owner, MiddlemanFactory middlemanFactory) {
      Preconditions.checkState(
          Objects.equals(moduleInfo.textualHeaders, picModuleInfo.textualHeaders),
          "Module and PIC module's textual headers are expected to be identical");
      // We don't create middlemen in LIPO collector subtree, because some target CT
      // will do that instead.
      Artifact prerequisiteStampFile = (ruleContext != null
          && ruleContext.getFragment(CppConfiguration.class).isLipoContextCollector())
          ? getMiddlemanArtifact(middlemanFactory)
          : createMiddleman(owner, middlemanFactory);

      return new CcCompilationContextInfo(
          new CommandLineCcCompilationContextInfo(
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
          nonCodeInputs.build(),
          moduleInfo.build(),
          picModuleInfo.build(),
          directModuleMaps.build(),
          cppModuleMap,
          verificationModuleMap,
          propagateModuleMapAsActionInput);
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
  @AutoCodec
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

  /** Collects data for a specific module in a special format that makes pruning easy. */
  @Immutable
  @AutoCodec
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
