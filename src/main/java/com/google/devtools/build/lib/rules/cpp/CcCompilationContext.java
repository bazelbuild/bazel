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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScanningHeaderData;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcCompilationContextApi;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Immutable store of information needed for C++ compilation that is aggregated across dependencies.
 */
@Immutable
@AutoCodec
// TODO(b/77669139): Rename to CcCompilationContext.
public final class CcCompilationContext implements CcCompilationContextApi {
  /** An empty {@code CcCompilationContext}. */
  public static final CcCompilationContext EMPTY = new Builder(null).build();

  private final CommandLineCcCompilationContext commandLineCcCompilationContext;

  private final NestedSet<PathFragment> declaredIncludeDirs;
  private final NestedSet<Artifact> declaredIncludeSrcs;

  /** Module maps from direct dependencies. */
  private final ImmutableList<Artifact> directModuleMaps;

  /** Non-code mandatory compilation inputs. */
  private final NestedSet<Artifact> nonCodeInputs;

  private final HeaderInfo headerInfo;
  private final NestedSet<HeaderInfo> transitiveHeaderInfos;
  private final NestedSet<Artifact> transitiveModules;
  private final NestedSet<Artifact> transitivePicModules;

  private final CppModuleMap cppModuleMap;
  private final CppModuleMap verificationModuleMap;

  private final boolean propagateModuleMapAsActionInput;

  // Derived from depsContexts.
  private final ImmutableSet<Artifact> compilationPrerequisites;

  private final CppConfiguration.HeadersCheckingMode headersCheckingMode;

  // Maps the Bazel generated paths of virtual include headers back to their original path relative
  // to the workspace directory.
  // For example it can map
  // "bazel-out/k8-fastbuild/bin/include/common/_virtual_includes/strategy/strategy.h"
  // back to the path of the header in the workspace directory "include/common/strategy.h".
  // This is needed only when code coverage collection is enabled, to report the actual source file
  // name in the coverage output file.
  private final ImmutableMap<String, String> virtualToOriginalHeaders;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  CcCompilationContext(
      CommandLineCcCompilationContext commandLineCcCompilationContext,
      ImmutableSet<Artifact> compilationPrerequisites,
      NestedSet<PathFragment> declaredIncludeDirs,
      NestedSet<Artifact> declaredIncludeSrcs,
      NestedSet<Artifact> nonCodeInputs,
      HeaderInfo headerInfo,
      NestedSet<HeaderInfo> transitiveHeaderInfos,
      NestedSet<Artifact> transitiveModules,
      NestedSet<Artifact> transitivePicModules,
      ImmutableList<Artifact> directModuleMaps,
      CppModuleMap cppModuleMap,
      @Nullable CppModuleMap verificationModuleMap,
      boolean propagateModuleMapAsActionInput,
      CppConfiguration.HeadersCheckingMode headersCheckingMode,
      ImmutableMap<String, String> virtualToOriginalHeaders) {
    Preconditions.checkNotNull(commandLineCcCompilationContext);
    this.commandLineCcCompilationContext = commandLineCcCompilationContext;
    this.declaredIncludeDirs = declaredIncludeDirs;
    this.declaredIncludeSrcs = declaredIncludeSrcs;
    this.directModuleMaps = directModuleMaps;
    this.headerInfo = headerInfo;
    this.transitiveHeaderInfos = transitiveHeaderInfos;
    this.transitiveModules = transitiveModules;
    this.transitivePicModules = transitivePicModules;
    this.cppModuleMap = cppModuleMap;
    this.nonCodeInputs = nonCodeInputs;
    this.verificationModuleMap = verificationModuleMap;
    this.compilationPrerequisites = compilationPrerequisites;
    this.propagateModuleMapAsActionInput = propagateModuleMapAsActionInput;
    this.headersCheckingMode = headersCheckingMode;
    this.virtualToOriginalHeaders = virtualToOriginalHeaders;
  }

  @Override
  public SkylarkNestedSet getSkylarkDefines() {
    return SkylarkNestedSet.of(
        String.class, NestedSetBuilder.wrap(Order.STABLE_ORDER, getDefines()));
  }

  @Override
  public SkylarkNestedSet getSkylarkHeaders() {
    return SkylarkNestedSet.of(Artifact.class, getDeclaredIncludeSrcs());
  }

  @Override
  public SkylarkNestedSet getSkylarkDeclaredIncludeDirs() {
    return SkylarkNestedSet.of(
        String.class,
        NestedSetBuilder.wrap(
            Order.STABLE_ORDER,
            getSystemIncludeDirs().stream()
                .map(PathFragment::getPathString)
                .collect(ImmutableList.toImmutableList())));
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
   * ({@link Builder#mergeDependentCcCompilationContext(CcCompilationContext)}, and {@link
   * Builder#mergeDependentCcCompilationContexts(Iterable)}, they are consolidated into a single
   * middleman Artifact when {@link Builder#build()} is called.
   *
   * <p>The returned set can be empty if there are no prerequisites. Usually, it contains a single
   * middleman.
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
    return commandLineCcCompilationContext.includeDirs;
  }

  /**
   * Returns the immutable list of include directories to be added with
   * "-iquote" (possibly empty but never null). This includes the include dirs
   * from the transitive deps closure of the target. This list does not contain
   * duplicates. All fragments are either absolute or relative to the exec root
   * (see {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}).
   */
  public ImmutableList<PathFragment> getQuoteIncludeDirs() {
    return commandLineCcCompilationContext.quoteIncludeDirs;
  }

  /**
   * Returns the immutable list of include directories to be added with
   * "-isystem" (possibly empty but never null). This includes the include dirs
   * from the transitive deps closure of the target. This list does not contain
   * duplicates. All fragments are either absolute or relative to the exec root
   * (see {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}).
   */
  public ImmutableList<PathFragment> getSystemIncludeDirs() {
    return commandLineCcCompilationContext.systemIncludeDirs;
  }

  /**
   * Returns the immutable set of declared include directories, relative to a "-I" or "-iquote"
   * directory" (possibly empty but never null).
   */
  public NestedSet<PathFragment> getDeclaredIncludeDirs() {
    return declaredIncludeDirs;
  }

  /**
   * Returns the immutable set of headers that have been declared in the {@code srcs} or {@code
   * hdrs} attribute (possibly empty but never null).
   */
  public NestedSet<Artifact> getDeclaredIncludeSrcs() {
    return declaredIncludeSrcs;
  }

  /** Returns headers given as textual_hdrs in this target. */
  public Iterable<Artifact> getTextualHdrs() {
    return headerInfo.textualHeaders;
  }

  public IncludeScanningHeaderData createIncludeScanningHeaderData(
      boolean usePic, boolean createModularHeaders) {
    // We'd prefer for these types to use ImmutableSet/ImmutableMap. However, constructing these is
    // substantially more costly in a way that shows up in profiles.
    Map<PathFragment, Artifact> pathToLegalOutputArtifact = new HashMap<>();
    Set<Artifact> modularHeaders = new HashSet<>();
    for (HeaderInfo transitiveHeaderInfo : transitiveHeaderInfos) {
      boolean isModule = createModularHeaders && transitiveHeaderInfo.getModule(usePic) != null;
      for (Artifact a : transitiveHeaderInfo.modularHeaders) {
        if (!a.isSourceArtifact()) {
          pathToLegalOutputArtifact.put(a.getExecPath(), a);
        }
        if (isModule) {
          modularHeaders.add(a);
        }
      }
      for (Artifact a : transitiveHeaderInfo.textualHeaders) {
        if (!a.isSourceArtifact()) {
          pathToLegalOutputArtifact.put(a.getExecPath(), a);
        }
      }
    }
    removeArtifactsFromSet(modularHeaders, headerInfo.modularHeaders);
    removeArtifactsFromSet(modularHeaders, headerInfo.textualHeaders);
    return new IncludeScanningHeaderData(
        Collections.unmodifiableMap(pathToLegalOutputArtifact),
        Collections.unmodifiableSet(modularHeaders));
  }

  /** Simple container for a collection of headers and corresponding modules. */
  public static class HeadersAndModules {
    public final Collection<Artifact> headers;
    public final Collection<Artifact> modules;

    HeadersAndModules(int expectedHeaderCount) {
      headers = new HashSet<>(expectedHeaderCount);
      modules = new LinkedHashSet<>();
    }
  }

  /**
   * Returns a list of all headers from {@code includes} that are properly declared as well as all
   * the modules that they are in.
   */
  public HeadersAndModules computeDeclaredHeadersAndUsedModules(
      boolean usePic, Set<Artifact> includes) {
    HeadersAndModules result = new HeadersAndModules(includes.size());
    for (HeaderInfo transitiveHeaderInfo : transitiveHeaderInfos) {
      Artifact module = transitiveHeaderInfo.getModule(usePic);
      for (Artifact header : transitiveHeaderInfo.modularHeaders) {
        if (includes.contains(header)) {
          if (module != null) {
            result.modules.add(module);
          }
          result.headers.add(header);
        }
      }
      for (Artifact header : transitiveHeaderInfo.textualHeaders) {
        if (includes.contains(header)) {
          result.headers.add(header);
        }
      }
    }
    // Do not add the module of the current rule for both:
    // 1. the module compile itself
    // 2. compiles of other translation units of the same rule.
    result.modules.remove(headerInfo.getModule(usePic));
    return result;
  }

  private void removeArtifactsFromSet(Set<Artifact> set, Iterable<Artifact> artifacts) {
    // Do not use Iterables.removeAll() or Set.removeAll() here as with the given container sizes,
    // that needlessly deteriorates to a quadratic algorithm.
    for (Artifact artifact : artifacts) {
      set.remove(artifact);
    }
  }

  public NestedSet<Artifact> getTransitiveModules(boolean usePic) {
    return usePic ? transitivePicModules : transitiveModules;
  }

  /**
   * Returns the immutable set of additional transitive inputs needed for
   * compilation, like C++ module map artifacts.
   */
  public NestedSet<Artifact> getAdditionalInputs() {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    builder.addAll(directModuleMaps);
    builder.addTransitive(nonCodeInputs);
    if (cppModuleMap != null && propagateModuleMapAsActionInput) {
      builder.add(cppModuleMap.getArtifact());
    }
    return builder.build();
  }

  /** @return modules maps from direct dependencies. */
  public Iterable<Artifact> getDirectModuleMaps() {
    return directModuleMaps;
  }

  /**
   * @return all declared headers of the current module if the current target
   * is compiled as a module.
   */
  protected Set<Artifact> getHeaderModuleSrcs() {
    return new ImmutableSet.Builder<Artifact>()
        .addAll(headerInfo.modularHeaders)
        .addAll(headerInfo.textualHeaders)
        .build();
  }

  /**
   * Returns the set of defines needed to compile this target (possibly empty
   * but never null). This includes definitions from the transitive deps closure
   * for the target. The order of the returned collection is deterministic.
   */
  public ImmutableList<String> getDefines() {
    return commandLineCcCompilationContext.defines;
  }

  /**
   * Returns a {@code CcCompilationContext} that is based on a given {@code CcCompilationContext}
   * but returns empty sets for {@link #getDeclaredIncludeDirs()}.
   */
  public static CcCompilationContext disallowUndeclaredHeaders(
      CcCompilationContext ccCompilationContext) {
    return new CcCompilationContext(
        ccCompilationContext.commandLineCcCompilationContext,
        ccCompilationContext.compilationPrerequisites,
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        ccCompilationContext.declaredIncludeSrcs,
        ccCompilationContext.nonCodeInputs,
        ccCompilationContext.headerInfo,
        ccCompilationContext.transitiveHeaderInfos,
        ccCompilationContext.transitiveModules,
        ccCompilationContext.transitivePicModules,
        ccCompilationContext.directModuleMaps,
        ccCompilationContext.cppModuleMap,
        ccCompilationContext.verificationModuleMap,
        ccCompilationContext.propagateModuleMapAsActionInput,
        ccCompilationContext.headersCheckingMode,
        ccCompilationContext.virtualToOriginalHeaders);
  }

  /** @return the C++ module map of the owner. */
  public CppModuleMap getCppModuleMap() {
    return cppModuleMap;
  }

  /** @return the C++ module map of the owner. */
  public CppModuleMap getVerificationModuleMap() {
    return verificationModuleMap;
  }

  public CppConfiguration.HeadersCheckingMode getHeadersCheckingMode() {
    return headersCheckingMode;
  }

  public static ImmutableList<CcCompilationContext> getCcCompilationContexts(
      Iterable<? extends TransitiveInfoCollection> deps) {
    ImmutableList.Builder<CcCompilationContext> ccCompilationContextsBuilder =
        ImmutableList.builder();
    for (CcInfo ccInfo : AnalysisUtils.getProviders(deps, CcInfo.PROVIDER)) {
      ccCompilationContextsBuilder.add(ccInfo.getCcCompilationContext());
    }
    return ccCompilationContextsBuilder.build();
  }

  public static CcCompilationContext merge(Collection<CcCompilationContext> ccCompilationContexts) {
    CcCompilationContext.Builder builder =
        new CcCompilationContext.Builder(/* ruleContext= */ null);
    builder.mergeDependentCcCompilationContexts(ccCompilationContexts);
    return builder.build();
  }

  public ImmutableMap<String, String> getVirtualToOriginalHeaders() {
    return virtualToOriginalHeaders;
  }

  /**
   * The parts of the {@code CcCompilationContext} that influence the command line of compilation
   * actions.
   */
  @Immutable
  @AutoCodec
  @VisibleForSerialization
  static class CommandLineCcCompilationContext {
    private final ImmutableList<PathFragment> includeDirs;
    private final ImmutableList<PathFragment> quoteIncludeDirs;
    private final ImmutableList<PathFragment> systemIncludeDirs;
    private final ImmutableList<String> defines;

    CommandLineCcCompilationContext(
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

  /** Builder class for {@link CcCompilationContext}. */
  public static class Builder {
    private String purpose;
    private final Set<Artifact> compilationPrerequisites = new LinkedHashSet<>();
    private final Set<PathFragment> includeDirs = new LinkedHashSet<>();
    private final Set<PathFragment> quoteIncludeDirs = new LinkedHashSet<>();
    private final Set<PathFragment> systemIncludeDirs = new LinkedHashSet<>();
    private final NestedSetBuilder<PathFragment> declaredIncludeDirs =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> declaredIncludeSrcs =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> nonCodeInputs = NestedSetBuilder.stableOrder();
    private final HeaderInfo.Builder headerInfoBuilder = new HeaderInfo.Builder();
    private final NestedSetBuilder<HeaderInfo> transitiveHeaderInfo =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> transitiveModules = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> transitivePicModules = NestedSetBuilder.stableOrder();
    private final Set<Artifact> directModuleMaps = new LinkedHashSet<>();
    private final Set<String> defines = new LinkedHashSet<>();
    private CppModuleMap cppModuleMap;
    private CppModuleMap verificationModuleMap;
    private boolean propagateModuleMapAsActionInput = true;
    private CppConfiguration.HeadersCheckingMode headersCheckingMode =
        CppConfiguration.HeadersCheckingMode.STRICT;
    private Map<String, String> virtualToOriginalHeaders = new HashMap<>();

    /** The rule that owns the context */
    private final RuleContext ruleContext;

    /** Creates a new builder for a {@link CcCompilationContext} instance. */
    public Builder(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    /**
     * Overrides the purpose of this context. This is useful if a Target needs more than one
     * CcCompilationContext. (The purpose is used to construct the name of the prerequisites
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
     * Merges the {@link CcCompilationContext} of a dependency into this one by adding the contents
     * of all of its attributes.
     */
    public Builder mergeDependentCcCompilationContext(
        CcCompilationContext otherCcCompilationContext) {
      Preconditions.checkNotNull(otherCcCompilationContext);
      compilationPrerequisites.addAll(
          otherCcCompilationContext.getTransitiveCompilationPrerequisites());
      includeDirs.addAll(otherCcCompilationContext.getIncludeDirs());
      quoteIncludeDirs.addAll(otherCcCompilationContext.getQuoteIncludeDirs());
      systemIncludeDirs.addAll(otherCcCompilationContext.getSystemIncludeDirs());
      declaredIncludeDirs.addTransitive(otherCcCompilationContext.getDeclaredIncludeDirs());
      declaredIncludeSrcs.addTransitive(otherCcCompilationContext.getDeclaredIncludeSrcs());
      transitiveHeaderInfo.addTransitive(otherCcCompilationContext.transitiveHeaderInfos);
      transitiveModules.addTransitive(otherCcCompilationContext.transitiveModules);
      if (otherCcCompilationContext.headerInfo.headerModule != null) {
        transitiveModules.add(otherCcCompilationContext.headerInfo.headerModule);
      }
      transitivePicModules.addTransitive(otherCcCompilationContext.transitivePicModules);
      if (otherCcCompilationContext.headerInfo.picHeaderModule != null) {
        transitivePicModules.add(otherCcCompilationContext.headerInfo.picHeaderModule);
      }

      nonCodeInputs.addTransitive(otherCcCompilationContext.nonCodeInputs);

      // All module maps of direct dependencies are inputs to the current compile independently of
      // the build type.
      if (otherCcCompilationContext.getCppModuleMap() != null) {
        directModuleMaps.add(otherCcCompilationContext.getCppModuleMap().getArtifact());
      }

      defines.addAll(otherCcCompilationContext.getDefines());
      virtualToOriginalHeaders.putAll(otherCcCompilationContext.getVirtualToOriginalHeaders());
      return this;
    }

    /**
     * Merges the {@code CcCompilationContext}s of some targets into this one by adding the contents
     * of all of their attributes. Targets that do not implement {@link CcCompilationContext} are
     * ignored.
     */
    public Builder mergeDependentCcCompilationContexts(Iterable<CcCompilationContext> targets) {
      for (CcCompilationContext target : targets) {
        mergeDependentCcCompilationContext(target);
      }
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
     * Add a single include directory to be added with "-isystem". It can be either relative to the
     * exec root (see {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot})
     * or absolute. Before it is stored, the include directory is normalized.
     */
    public Builder addSystemIncludeDirs(Iterable<PathFragment> systemIncludeDirs) {
      Iterables.addAll(this.systemIncludeDirs, systemIncludeDirs);
      return this;
    }

    /** Add a single declared include dir, relative to a "-I" or "-iquote" directory". */
    public Builder addDeclaredIncludeDir(PathFragment dir) {
      declaredIncludeDirs.add(dir);
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
    public Builder addDeclaredIncludeSrcs(Iterable<Artifact> declaredIncludeSrcs) {
      for (Artifact source : declaredIncludeSrcs) {
        addDeclaredIncludeSrc(source);
      }
      return this;
    }

    public Builder addModularHdrs(Collection<Artifact> headers) {
      this.headerInfoBuilder.addHeaders(headers);
      return this;
    }

    public Builder addTextualHdrs(Collection<Artifact> headers) {
      this.headerInfoBuilder.addTextualHeaders(headers);
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
      this.headerInfoBuilder.setHeaderModule(headerModule);
      return this;
    }

    /**
     * Sets the C++ header module in pic mode.
     * @param picHeaderModule The .pic.pcm file generated for this library.
     */
    public Builder setPicHeaderModule(Artifact picHeaderModule) {
      this.headerInfoBuilder.setPicHeaderModule(picHeaderModule);
      return this;
    }

    public Builder setHeadersCheckingMode(
        CppConfiguration.HeadersCheckingMode headersCheckingMode) {
      this.headersCheckingMode = headersCheckingMode;
      return this;
    }

    public Builder addVirtualToOriginalHeaders(Map<String, String> virtualToOriginalHeaders) {
      this.virtualToOriginalHeaders.putAll(virtualToOriginalHeaders);
      return this;
    }

    /** Builds the {@link CcCompilationContext}. */
    public CcCompilationContext build() {
      return build(
          ruleContext == null ? null : ruleContext.getActionOwner(),
          ruleContext == null ? null : ruleContext.getAnalysisEnvironment().getMiddlemanFactory());
    }

    @VisibleForTesting // productionVisibility = Visibility.PRIVATE
    public CcCompilationContext build(ActionOwner owner, MiddlemanFactory middlemanFactory) {
      Artifact prerequisiteStampFile = createMiddleman(owner, middlemanFactory);
      HeaderInfo headerInfo = headerInfoBuilder.build();
      transitiveHeaderInfo.add(headerInfo);

      return new CcCompilationContext(
          new CommandLineCcCompilationContext(
              ImmutableList.copyOf(includeDirs),
              ImmutableList.copyOf(quoteIncludeDirs),
              ImmutableList.copyOf(systemIncludeDirs),
              ImmutableList.copyOf(defines)),
          // TODO(b/110873917): We don't have the middle man compilation prerequisite, therefore, we
          // use the compilation prerequisites as they were passed to the builder, i.e. we use every
          // header instead of a middle man.
          prerequisiteStampFile == null
              ? ImmutableSet.copyOf(compilationPrerequisites)
              : ImmutableSet.of(prerequisiteStampFile),
          declaredIncludeDirs.build(),
          declaredIncludeSrcs.build(),
          nonCodeInputs.build(),
          headerInfo,
          transitiveHeaderInfo.build(),
          transitiveModules.build(),
          transitivePicModules.build(),
          ImmutableList.copyOf(directModuleMaps),
          cppModuleMap,
          verificationModuleMap,
          propagateModuleMapAsActionInput,
          headersCheckingMode,
          ImmutableMap.copyOf(virtualToOriginalHeaders));
    }

    /**
     * Creates a middleman for the compilation prerequisites.
     *
     * @return the middleman or null if there are no prerequisites
     */
    private Artifact createMiddleman(ActionOwner owner,
        MiddlemanFactory middlemanFactory) {
      if (middlemanFactory == null || compilationPrerequisites.isEmpty()) {
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
  }

  /**
   * Gathers data about the PIC and no-PIC .pcm files belonging to this context and the associated
   * information about the headers, e.g. modular vs. textual headers and pre-grepped header files.
   */
  @Immutable
  @AutoCodec
  public static final class HeaderInfo {
    /**
     * The modules built for this context. If null, then no module is being compiled for this
     * context.
     */
    private final Artifact headerModule;
    private final Artifact picHeaderModule;

    /** All header files that are compiled into this module. */
    private final ImmutableList<Artifact> modularHeaders;

    /** All header files that are contained in this module. */
    private final ImmutableList<Artifact> textualHeaders;

    public HeaderInfo(
        Artifact headerModule,
        Artifact picHeaderModule,
        ImmutableList<Artifact> modularHeaders,
        ImmutableList<Artifact> textualHeaders) {
      this.headerModule = headerModule;
      this.picHeaderModule = picHeaderModule;
      this.modularHeaders = modularHeaders;
      this.textualHeaders = textualHeaders;
    }

    public Artifact getModule(boolean pic) {
      return pic ? picHeaderModule : headerModule;
    }

    /**
     * Builder class for {@link HeaderInfo}.
     */
    public static class Builder {
      private Artifact headerModule = null;
      private Artifact picHeaderModule = null;
      private final Set<Artifact> modularHeaders = new HashSet<>();
      private final Set<Artifact> textualHeaders = new HashSet<>();

      public Builder setHeaderModule(Artifact headerModule) {
        this.headerModule = headerModule;
        return this;
      }

      public Builder setPicHeaderModule(Artifact headerModule) {
        this.picHeaderModule = headerModule;
        return this;
      }

      public Builder addHeaders(Collection<Artifact> headers) {
        // TODO(djasper): CPP_TEXTUAL_INCLUDEs are currently special cased here and in
        // CppModuleMapAction. These should be moved to a place earlier in the Action construction.
        for (Artifact header : headers) {
          if (header.isFileType(CppFileTypes.CPP_TEXTUAL_INCLUDE)) {
            this.textualHeaders.add(header);
          } else {
            this.modularHeaders.add(header);
          }
        }
        return this;
      }

      public Builder addTextualHeaders(Collection<Artifact> headers) {
        this.textualHeaders.addAll(headers);
        return this;
      }

      public HeaderInfo build() {
        return new HeaderInfo(
            headerModule,
            picHeaderModule,
            ImmutableList.copyOf(modularHeaders),
            ImmutableList.copyOf(textualHeaders));
      }
    }
  }
}
