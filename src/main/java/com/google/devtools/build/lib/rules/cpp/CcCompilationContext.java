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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.compacthashmap.CompactHashMap;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScanningHeaderData;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcCompilationContextApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;

/**
 * Immutable store of information needed for C++ compilation that is aggregated across dependencies.
 */
@Immutable
public final class CcCompilationContext implements CcCompilationContextApi<Artifact> {
  /** An empty {@code CcCompilationContext}. */
  public static final CcCompilationContext EMPTY =
      builder(/* actionConstructionContext= */ null, /* configuration= */ null, /* label= */ null)
          .build();

  private final CommandLineCcCompilationContext commandLineCcCompilationContext;

  private final NestedSet<PathFragment> looseHdrsDirs;
  private final NestedSet<Artifact> declaredIncludeSrcs;

  /** Module maps from direct dependencies. */
  private final ImmutableList<Artifact> directModuleMaps;

  /** Module maps from dependencies that will be re-exported by this compilation context. */
  private final ImmutableList<CppModuleMap> exportingModuleMaps;

  /** Non-code mandatory compilation inputs. */
  private final NestedSet<Artifact> nonCodeInputs;

  private final HeaderInfo headerInfo;
  private final NestedSet<Artifact> transitiveModules;
  private final NestedSet<Artifact> transitivePicModules;

  private final CppModuleMap cppModuleMap;

  private final boolean propagateModuleMapAsActionInput;

  // Derived from depsContexts.
  private final NestedSet<Artifact> compilationPrerequisites;

  private final CppConfiguration.HeadersCheckingMode headersCheckingMode;

  // Each pair maps the Bazel generated paths of virtual include headers back to their original path
  // relative to the workspace directory.
  // For example it can map
  // "bazel-out/k8-fastbuild/bin/include/common/_virtual_includes/strategy/strategy.h"
  // back to the path of the header in the workspace directory "include/common/strategy.h".
  // This is needed only when code coverage collection is enabled, to report the actual source file
  // name in the coverage output file.
  private final NestedSet<Tuple> virtualToOriginalHeaders;
  /**
   * Caches the actual number of transitive headers reachable through transitiveHeaderInfos. We need
   * to create maps to store these and so caching this count can substantially help with memory
   * allocations.
   */
  private int transitiveHeaderCount;
  /** Aftifacts generated by the header validation actions. */
  private final NestedSet<Artifact> headerTokens;

  private CcCompilationContext(
      CommandLineCcCompilationContext commandLineCcCompilationContext,
      NestedSet<Artifact> compilationPrerequisites,
      NestedSet<PathFragment> looseHdrsDirs,
      NestedSet<Artifact> declaredIncludeSrcs,
      NestedSet<Artifact> nonCodeInputs,
      HeaderInfo headerInfo,
      NestedSet<Artifact> transitiveModules,
      NestedSet<Artifact> transitivePicModules,
      ImmutableList<Artifact> directModuleMaps,
      ImmutableList<CppModuleMap> exportingModuleMaps,
      CppModuleMap cppModuleMap,
      boolean propagateModuleMapAsActionInput,
      CppConfiguration.HeadersCheckingMode headersCheckingMode,
      NestedSet<Tuple> virtualToOriginalHeaders,
      NestedSet<Artifact> headerTokens) {
    Preconditions.checkNotNull(commandLineCcCompilationContext);
    this.commandLineCcCompilationContext = commandLineCcCompilationContext;
    this.looseHdrsDirs = looseHdrsDirs;
    this.declaredIncludeSrcs = declaredIncludeSrcs;
    this.directModuleMaps = directModuleMaps;
    this.exportingModuleMaps = exportingModuleMaps;
    this.headerInfo = headerInfo;
    this.transitiveModules = transitiveModules;
    this.transitivePicModules = transitivePicModules;
    this.cppModuleMap = cppModuleMap;
    this.nonCodeInputs = nonCodeInputs;
    this.compilationPrerequisites = compilationPrerequisites;
    this.propagateModuleMapAsActionInput = propagateModuleMapAsActionInput;
    this.headersCheckingMode = headersCheckingMode;
    this.virtualToOriginalHeaders = virtualToOriginalHeaders;
    this.transitiveHeaderCount = -1;
    this.headerTokens = headerTokens;
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  @Override
  public Depset getStarlarkDefines() {
    return Depset.of(String.class, NestedSetBuilder.wrap(Order.STABLE_ORDER, getDefines()));
  }

  @Override
  public Depset getStarlarkNonTransitiveDefines() {
    return Depset.of(
        String.class, NestedSetBuilder.wrap(Order.STABLE_ORDER, getNonTransitiveDefines()));
  }

  @Override
  public Depset getStarlarkHeaders() {
    return Depset.of(Artifact.class, getDeclaredIncludeSrcs());
  }

  @Override
  public StarlarkList<Artifact> getStarlarkDirectModularHeaders() {
    return StarlarkList.immutableCopyOf(
        ImmutableList.<Artifact>builder()
            .addAll(getDirectPublicHdrs())
            .addAll(getDirectPrivateHdrs())
            .addAll(headerInfo.separateModuleHeaders)
            .build());
  }

  @Override
  public StarlarkList<Artifact> getStarlarkDirectPublicHeaders() {
    return StarlarkList.immutableCopyOf(getDirectPublicHdrs());
  }

  @Override
  public StarlarkList<Artifact> getStarlarkDirectPrivateHeaders() {
    return StarlarkList.immutableCopyOf(getDirectPrivateHdrs());
  }

  @Override
  public StarlarkList<Artifact> getStarlarkDirectTextualHeaders() {
    return StarlarkList.immutableCopyOf(getTextualHdrs());
  }

  @Override
  public Depset getStarlarkSystemIncludeDirs() {
    return Depset.of(
        String.class,
        NestedSetBuilder.wrap(
            Order.STABLE_ORDER,
            getSystemIncludeDirs().stream()
                .map(PathFragment::getSafePathString)
                .collect(ImmutableList.toImmutableList())));
  }

  @Override
  public Depset getStarlarkFrameworkIncludeDirs() {
    return Depset.of(
        String.class,
        NestedSetBuilder.wrap(
            Order.STABLE_ORDER,
            getFrameworkIncludeDirs().stream()
                .map(PathFragment::getSafePathString)
                .collect(ImmutableList.toImmutableList())));
  }

  @Override
  public Depset getStarlarkIncludeDirs() {
    return Depset.of(
        String.class,
        NestedSetBuilder.wrap(
            Order.STABLE_ORDER,
            getIncludeDirs().stream()
                .map(PathFragment::getSafePathString)
                .collect(ImmutableList.toImmutableList())));
  }

  @Override
  public Depset getStarlarkExternalIncludeDirs() {
    return Depset.of(
        String.class,
        NestedSetBuilder.wrap(
            Order.STABLE_ORDER,
            getExternalIncludeDirs().stream()
                .map(PathFragment::getSafePathString)
                .collect(ImmutableList.toImmutableList())));
  }

  @Override
  public Depset getStarlarkQuoteIncludeDirs() {
    return Depset.of(
        String.class,
        NestedSetBuilder.wrap(
            Order.STABLE_ORDER,
            getQuoteIncludeDirs().stream()
                .map(PathFragment::getSafePathString)
                .collect(ImmutableList.toImmutableList())));
  }

  @Override
  public Depset getStarlarkTransitiveCompilationPrerequisites(StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Artifact.class, getTransitiveCompilationPrerequisites());
  }

  @Override
  public Depset getStarlarkValidationArtifacts() {
    return Depset.of(Artifact.class, getHeaderTokens());
  }

  @Override
  public Depset getStarlarkVirtualToOriginalHeaders(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Tuple.class, getVirtualToOriginalHeaders());
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
   * ({@link Builder#addDependentCcCompilationContext(CcCompilationContext)}, and {@link
   * Builder#addDependentCcCompilationContexts(Iterable)}, they are consolidated into a single
   * middleman Artifact when {@link Builder#build()} is called.
   *
   * <p>The returned set can be empty if there are no prerequisites. Usually, it contains a single
   * middleman.
   */
  public NestedSet<Artifact> getTransitiveCompilationPrerequisites() {
    return compilationPrerequisites;
  }

  /**
   * Returns the immutable list of include directories to be added with "-I" (possibly empty but
   * never null). This includes the include dirs from the transitive deps closure of the target.
   * This list does not contain duplicates. All fragments are either absolute or relative to the
   * exec root (see {@link
   * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot(String)}).
   */
  public ImmutableList<PathFragment> getIncludeDirs() {
    return commandLineCcCompilationContext.includeDirs;
  }

  /**
   * Returns the immutable list of include directories to be added with "-iquote" (possibly empty
   * but never null). This includes the include dirs from the transitive deps closure of the target.
   * This list does not contain duplicates. All fragments are either absolute or relative to the
   * exec root (see {@link
   * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot(String)}).
   */
  public ImmutableList<PathFragment> getQuoteIncludeDirs() {
    return commandLineCcCompilationContext.quoteIncludeDirs;
  }

  /**
   * Returns the immutable list of include directories to be added with "-isystem" (possibly empty
   * but never null). This includes the include dirs from the transitive deps closure of the target.
   * This list does not contain duplicates. All fragments are either absolute or relative to the
   * exec root (see {@link
   * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot(String)}).
   */
  public ImmutableList<PathFragment> getSystemIncludeDirs() {
    return commandLineCcCompilationContext.systemIncludeDirs;
  }

  /**
   * Returns the immutable list of include directories to be added with "-F" (possibly empty but
   * never null). This includes the include dirs from the transitive deps closure of the target.
   * This list does not contain duplicates. All fragments are either absolute or relative to the
   * exec root (see {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}).
   */
  public ImmutableList<PathFragment> getFrameworkIncludeDirs() {
    return commandLineCcCompilationContext.frameworkIncludeDirs;
  }

  /**
   * Returns the immutable list of external include directories (possibly empty but never null).
   * This includes the include dirs from the transitive deps closure of the target. This list does
   * not contain duplicates. All fragments are either absolute or relative to the exec root (see
   * {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot(String)}).
   */
  public ImmutableList<PathFragment> getExternalIncludeDirs() {
    return commandLineCcCompilationContext.externalIncludeDirs;
  }

  /**
   * Returns the immutable set of declared include directories, relative to a "-I" or "-iquote"
   * directory" (possibly empty but never null).
   */
  public NestedSet<PathFragment> getLooseHdrsDirs() {
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  /**
   * Returns the immutable set of headers that have been declared in the {@code srcs} or {@code
   * hdrs} attribute (possibly empty but never null).
   */
  public NestedSet<Artifact> getDeclaredIncludeSrcs() {
    return declaredIncludeSrcs;
  }

  /** Returns headers given as textual_hdrs in this target. */
  public ImmutableList<Artifact> getTextualHdrs() {
    return headerInfo.textualHeaders;
  }

  /** Returns public headers (given as {@code hdrs}) in this target. */
  public ImmutableList<Artifact> getDirectPublicHdrs() {
    return headerInfo.modularPublicHeaders;
  }

  /** Returns private headers (given as {@code srcs}) in this target. */
  public ImmutableList<Artifact> getDirectPrivateHdrs() {
    return headerInfo.modularPrivateHeaders;
  }

  public NestedSet<Artifact> getHeaderTokens() {
    return headerTokens;
  }

  /** Helper class for creating include scanning header data. */
  public static class IncludeScanningHeaderDataHelper {
    private IncludeScanningHeaderDataHelper() {}

    public static void handleArtifact(
        Artifact artifact,
        Map<PathFragment, Artifact> pathToLegalArtifact,
        ArrayList<Artifact> treeArtifacts) {
      if (artifact.isTreeArtifact()) {
        treeArtifacts.add(artifact);
        return;
      }
      pathToLegalArtifact.put(artifact.getExecPath(), artifact);
    }

    /**
     * Enter the TreeArtifactValues in each TreeArtifact into pathToLegalArtifact. Returns true on
     * success.
     *
     * <p>If a TreeArtifact's value is missing, returns false, and leave pathToLegalArtifact
     * unmodified.
     */
    public static boolean handleTreeArtifacts(
        Environment env,
        Map<PathFragment, Artifact> pathToLegalArtifact,
        ArrayList<Artifact> treeArtifacts)
        throws InterruptedException {
      if (treeArtifacts.isEmpty()) {
        return true;
      }
      SkyframeLookupResult result = env.getValuesAndExceptions(treeArtifacts);
      if (env.valuesMissing()) {
        return false;
      }
      for (Artifact treeArtifact : treeArtifacts) {
        SkyValue value = result.get(treeArtifact);
        if (value == null) {
          BugReport.sendBugReport(
              new IllegalStateException(
                  "Some value from " + treeArtifacts + " was missing, this should never happen"));
          return false;
        }
        checkState(
            value instanceof TreeArtifactValue, "SkyValue %s is not TreeArtifactValue", value);
        TreeArtifactValue treeArtifactValue = (TreeArtifactValue) value;
        for (TreeFileArtifact treeFileArtifact : treeArtifactValue.getChildren()) {
          pathToLegalArtifact.put(treeFileArtifact.getExecPath(), treeFileArtifact);
        }
      }
      return true;
    }
  }

  /**
   * Passes a list of headers to the include scanning helper for handling, and optionally adds them
   * to a set that tracks modular headers.
   *
   * <p>This is factored out into its own method not only to reduce code duplication below, but also
   * to improve JIT optimization for this performance-sensitive region.
   */
  private static void handleHeadersForIncludeScanning(
      ImmutableList<Artifact> headers,
      Map<PathFragment, Artifact> pathToLegalArtifact,
      ArrayList<Artifact> treeArtifacts,
      boolean isModule,
      Set<Artifact> modularHeaders) {
    // Not using range-based for loops here and below as the additional overhead of the
    // ImmutableList iterators has shown up in profiles.
    for (int i = 0; i < headers.size(); i++) {
      Artifact a = headers.get(i);
      IncludeScanningHeaderDataHelper.handleArtifact(a, pathToLegalArtifact, treeArtifacts);
      if (isModule) {
        modularHeaders.add(a);
      }
    }
  }

  /**
   * This method returns null when a required SkyValue is missing and a Skyframe restart is
   * required.
   */
  @Nullable
  public IncludeScanningHeaderData.Builder createIncludeScanningHeaderData(
      Environment env, boolean usePic, boolean createModularHeaders) throws InterruptedException {
    Collection<HeaderInfo> transitiveHeaderInfos = headerInfo.getTransitiveCollection();
    ArrayList<Artifact> treeArtifacts = new ArrayList<>();
    // We'd prefer for these types to use ImmutableSet/ImmutableMap. However, constructing these is
    // substantially more costly in a way that shows up in profiles.
    Map<PathFragment, Artifact> pathToLegalArtifact =
        CompactHashMap.createWithExpectedSize(
            transitiveHeaderCount == -1 ? transitiveHeaderInfos.size() : transitiveHeaderCount);
    Set<Artifact> modularHeaders =
        CompactHashSet.createWithExpectedSize(transitiveHeaderInfos.size());
    // Not using range-based for loops here and below as the additional overhead of the
    // ImmutableList iterators has shown up in profiles.
    for (HeaderInfo transitiveHeaderInfo : transitiveHeaderInfos) {
      boolean isModule = createModularHeaders && transitiveHeaderInfo.getModule(usePic) != null;
      handleHeadersForIncludeScanning(
          transitiveHeaderInfo.modularHeaders,
          pathToLegalArtifact,
          treeArtifacts,
          isModule,
          modularHeaders);
      handleHeadersForIncludeScanning(
          transitiveHeaderInfo.separateModuleHeaders,
          pathToLegalArtifact,
          treeArtifacts,
          isModule,
          modularHeaders);
      handleHeadersForIncludeScanning(
          transitiveHeaderInfo.textualHeaders,
          pathToLegalArtifact,
          treeArtifacts,
          /* isModule= */ false,
          null);
    }
    if (!IncludeScanningHeaderDataHelper.handleTreeArtifacts(
        env, pathToLegalArtifact, treeArtifacts)) {
      return null;
    }
    if (transitiveHeaderCount == -1) {
      transitiveHeaderCount = pathToLegalArtifact.size();
    }
    removeArtifactsFromSet(modularHeaders, headerInfo.modularHeaders);
    removeArtifactsFromSet(modularHeaders, headerInfo.textualHeaders);
    removeArtifactsFromSet(modularHeaders, headerInfo.separateModuleHeaders);
    return new IncludeScanningHeaderData.Builder(pathToLegalArtifact, modularHeaders);
  }

  /**
   * Returns a list of all headers from {@code includes} that are properly declared as well as all
   * the modules that they are in.
   */
  public Set<DerivedArtifact> computeUsedModules(
      boolean usePic, Set<Artifact> includes, boolean separate) {
    CompactHashSet<DerivedArtifact> modules = CompactHashSet.create();
    for (HeaderInfo transitiveHeaderInfo : headerInfo.getTransitiveCollection()) {
      DerivedArtifact module = transitiveHeaderInfo.getModule(usePic);
      if (module == null) {
        // If we don't have a main module, there is also not going to be a separate module. This is
        // verified when constructing HeaderInfo instances.
        continue;
      }
      // Not using range-based for loops here as often there is exactly one element in this list
      // and the amount of garbage created by SingletonImmutableList.iterator() is significant.
      for (int i = 0; i < transitiveHeaderInfo.modularHeaders.size(); i++) {
        Artifact header = transitiveHeaderInfo.modularHeaders.get(i);
        if (includes.contains(header)) {
          modules.add(module);
          break;
        }
      }
      for (int i = 0; i < transitiveHeaderInfo.separateModuleHeaders.size(); i++) {
        Artifact header = transitiveHeaderInfo.separateModuleHeaders.get(i);
        if (includes.contains(header)) {
          modules.add(transitiveHeaderInfo.getSeparateModule(usePic));
          break;
        }
      }
    }
    // Do not add the module of the current rule for both:
    // 1. the module compile itself
    // 2. compiles of other translation units of the same rule.
    modules.remove(separate ? headerInfo.getSeparateModule(usePic) : headerInfo.getModule(usePic));
    return modules;
  }

  private static void removeArtifactsFromSet(Set<Artifact> set, ImmutableList<Artifact> artifacts) {
    // Not using iterators here as the resulting overhead is significant in profiles. Do not use
    // Iterables.removeAll() or Set.removeAll() here as with the given container sizes, that
    // needlessly deteriorates to a quadratic algorithm.
    for (int i = 0; i < artifacts.size(); i++) {
      set.remove(artifacts.get(i));
    }
  }

  public NestedSet<Artifact> getTransitiveModules(boolean usePic) {
    return usePic ? transitivePicModules : transitiveModules;
  }

  @Override
  public Depset getStarlarkTransitiveModules(boolean usePic, StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Artifact.class, getTransitiveModules(usePic));
  }

  /**
   * Returns the immutable set of additional transitive inputs needed for compilation, like C++
   * module map artifacts.
   */
  public NestedSet<Artifact> getAdditionalInputs() {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    addAdditionalInputs(builder);
    return builder.build();
  }

  @Override
  public Depset getStarlarkAdditionalInputs(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Artifact.class, getAdditionalInputs());
  }

  /** Adds additional transitive inputs needed for compilation to builder. */
  void addAdditionalInputs(NestedSetBuilder<Artifact> builder) {
    builder.addAll(directModuleMaps);
    builder.addTransitive(nonCodeInputs);
    if (cppModuleMap != null && propagateModuleMapAsActionInput) {
      builder.add(cppModuleMap.getArtifact());
    }
  }

  /** @return modules maps from direct dependencies. */
  public Iterable<Artifact> getDirectModuleMaps() {
    return directModuleMaps;
  }

  DerivedArtifact getHeaderModule(boolean usePic) {
    return headerInfo.getModule(usePic);
  }

  DerivedArtifact getSeparateHeaderModule(boolean usePic) {
    return headerInfo.getSeparateModule(usePic);
  }

  /**
   * @return all declared headers of the current module if the current target is compiled as a
   *     module.
   */
  ImmutableList<Artifact> getHeaderModuleSrcs(boolean separateModule) {
    if (separateModule) {
      return headerInfo.separateModuleHeaders;
    }
    return new ImmutableSet.Builder<Artifact>()
        .addAll(headerInfo.modularHeaders)
        .addAll(headerInfo.textualHeaders)
        .addAll(headerInfo.separateModuleHeaders)
        .build()
        .asList();
  }

  /**
   * Returns the set of defines needed to compile this target. This includes definitions from the
   * transitive deps closure for the target. The order of the returned collection is deterministic.
   */
  public ImmutableList<String> getDefines() {
    return commandLineCcCompilationContext.defines;
  }

  /**
   * Returns the set of defines needed to compile this target. This doesn't include definitions from
   * the transitive deps closure for the target.
   */
  ImmutableList<String> getNonTransitiveDefines() {
    return commandLineCcCompilationContext.localDefines;
  }

  /**
   * Returns a {@code CcCompilationContext} that is based on a given {@code CcCompilationContext},
   * with {@code extraHeaderTokens} added to the header tokens.
   */
  public static CcCompilationContext createWithExtraHeaderTokens(
      CcCompilationContext ccCompilationContext, Iterable<Artifact> extraHeaderTokens) {
    NestedSetBuilder<Artifact> headerTokens = NestedSetBuilder.stableOrder();
    headerTokens.addTransitive(ccCompilationContext.getHeaderTokens());
    headerTokens.addAll(extraHeaderTokens);
    return new CcCompilationContext(
        ccCompilationContext.commandLineCcCompilationContext,
        ccCompilationContext.compilationPrerequisites,
        ccCompilationContext.looseHdrsDirs,
        ccCompilationContext.declaredIncludeSrcs,
        ccCompilationContext.nonCodeInputs,
        ccCompilationContext.headerInfo,
        ccCompilationContext.transitiveModules,
        ccCompilationContext.transitivePicModules,
        ccCompilationContext.directModuleMaps,
        ccCompilationContext.exportingModuleMaps,
        ccCompilationContext.cppModuleMap,
        ccCompilationContext.propagateModuleMapAsActionInput,
        ccCompilationContext.headersCheckingMode,
        ccCompilationContext.virtualToOriginalHeaders,
        headerTokens.build());
  }

  /** @return the C++ module map of the owner. */
  public CppModuleMap getCppModuleMap() {
    return cppModuleMap;
  }

  /** Returns the list of dependencies' C++ module maps re-exported by this compilation context. */
  public ImmutableList<CppModuleMap> getExportingModuleMaps() {
    return exportingModuleMaps;
  }

  public NestedSet<Tuple> getVirtualToOriginalHeaders() {
    return virtualToOriginalHeaders;
  }

  /**
   * The parts of the {@code CcCompilationContext} that influence the command line of compilation
   * actions.
   */
  @Immutable
  private static final class CommandLineCcCompilationContext {
    private final ImmutableList<PathFragment> includeDirs;
    private final ImmutableList<PathFragment> quoteIncludeDirs;
    private final ImmutableList<PathFragment> systemIncludeDirs;
    private final ImmutableList<PathFragment> frameworkIncludeDirs;
    private final ImmutableList<PathFragment> externalIncludeDirs;
    private final ImmutableList<String> defines;
    private final ImmutableList<String> localDefines;

    CommandLineCcCompilationContext(
        ImmutableList<PathFragment> includeDirs,
        ImmutableList<PathFragment> quoteIncludeDirs,
        ImmutableList<PathFragment> systemIncludeDirs,
        ImmutableList<PathFragment> frameworkIncludeDirs,
        ImmutableList<PathFragment> externalIncludeDirs,
        ImmutableList<String> defines,
        ImmutableList<String> localDefines) {
      this.includeDirs = includeDirs;
      this.quoteIncludeDirs = quoteIncludeDirs;
      this.systemIncludeDirs = systemIncludeDirs;
      this.frameworkIncludeDirs = frameworkIncludeDirs;
      this.externalIncludeDirs = externalIncludeDirs;
      this.defines = defines;
      this.localDefines = localDefines;
    }
  }

  /** Creates a new builder for a {@link CcCompilationContext} instance. */
  public static Builder builder(
      ActionConstructionContext actionConstructionContext,
      BuildConfigurationValue configuration,
      Label label) {
    return new Builder(actionConstructionContext, configuration, label);
  }

  /** Builder class for {@link CcCompilationContext}. */
  public static class Builder {
    private String purpose;
    private final NestedSetBuilder<Artifact> compilationPrerequisites =
        NestedSetBuilder.stableOrder();
    private final TransitiveSetHelper<PathFragment> includeDirs = new TransitiveSetHelper<>();
    private final TransitiveSetHelper<PathFragment> quoteIncludeDirs = new TransitiveSetHelper<>();
    private final TransitiveSetHelper<PathFragment> systemIncludeDirs = new TransitiveSetHelper<>();
    private final TransitiveSetHelper<PathFragment> frameworkIncludeDirs =
        new TransitiveSetHelper<>();
    private final TransitiveSetHelper<PathFragment> externalIncludeDirs =
        new TransitiveSetHelper<>();
    private final NestedSetBuilder<PathFragment> looseHdrsDirs = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> declaredIncludeSrcs = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> nonCodeInputs = NestedSetBuilder.stableOrder();
    private final HeaderInfo.Builder headerInfoBuilder = new HeaderInfo.Builder();
    private final Set<String> defines = new LinkedHashSet<>();
    private final ImmutableList.Builder<CcCompilationContext> deps = ImmutableList.builder();
    private final ImmutableList.Builder<CcCompilationContext> exportedDeps =
        ImmutableList.builder();
    private final Set<String> localDefines = new LinkedHashSet<>();
    private CppModuleMap cppModuleMap;
    private boolean propagateModuleMapAsActionInput = true;
    private CppConfiguration.HeadersCheckingMode headersCheckingMode =
        CppConfiguration.HeadersCheckingMode.STRICT;
    private final NestedSetBuilder<Tuple> virtualToOriginalHeaders = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> headerTokens = NestedSetBuilder.stableOrder();

    /** The rule that owns the context */
    private final ActionConstructionContext actionConstructionContext;

    private final BuildConfigurationValue configuration;
    private final Label label;

    /** Creates a new builder for a {@link CcCompilationContext} instance. */
    private Builder(
        ActionConstructionContext actionConstructionContext,
        BuildConfigurationValue configuration,
        Label label) {
      this.actionConstructionContext = actionConstructionContext;
      this.configuration = configuration;
      this.label = label;
    }

    /**
     * Overrides the purpose of this context. This is useful if a Target needs more than one
     * CcCompilationContext. (The purpose is used to construct the name of the prerequisites
     * middleman for the context, and all artifacts for a given Target must have distinct names.)
     *
     * @param purpose must be a string which is suitable for use as a filename. A single rule may
     *     have many middlemen with distinct purposes.
     * @see MiddlemanFactory#createSchedulingDependencyMiddleman
     */
    @CanIgnoreReturnValue
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
    @CanIgnoreReturnValue
    public Builder addDependentCcCompilationContext(
        CcCompilationContext otherCcCompilationContext) {
      deps.add(otherCcCompilationContext);
      return this;
    }

    private void mergeDependentCcCompilationContext(
        CcCompilationContext otherCcCompilationContext,
        TransitiveSetHelper<String> allDefines,
        NestedSetBuilder<Artifact> transitiveModules,
        NestedSetBuilder<Artifact> transitivePicModules,
        Set<Artifact> directModuleMaps) {
      Preconditions.checkNotNull(otherCcCompilationContext);
      compilationPrerequisites.addTransitive(
          otherCcCompilationContext.getTransitiveCompilationPrerequisites());
      includeDirs.addTransitive(otherCcCompilationContext.getIncludeDirs());
      quoteIncludeDirs.addTransitive(otherCcCompilationContext.getQuoteIncludeDirs());
      systemIncludeDirs.addTransitive(otherCcCompilationContext.getSystemIncludeDirs());
      frameworkIncludeDirs.addTransitive(otherCcCompilationContext.getFrameworkIncludeDirs());
      externalIncludeDirs.addTransitive(otherCcCompilationContext.getExternalIncludeDirs());
      looseHdrsDirs.addTransitive(otherCcCompilationContext.getLooseHdrsDirs());
      declaredIncludeSrcs.addTransitive(otherCcCompilationContext.getDeclaredIncludeSrcs());
      headerInfoBuilder.addDep(otherCcCompilationContext.headerInfo);

      transitiveModules.addTransitive(otherCcCompilationContext.transitiveModules);
      addIfNotNull(transitiveModules, otherCcCompilationContext.headerInfo.headerModule);
      addIfNotNull(transitiveModules, otherCcCompilationContext.headerInfo.separateModule);

      transitivePicModules.addTransitive(otherCcCompilationContext.transitivePicModules);
      addIfNotNull(transitivePicModules, otherCcCompilationContext.headerInfo.picHeaderModule);
      addIfNotNull(transitivePicModules, otherCcCompilationContext.headerInfo.separatePicModule);

      nonCodeInputs.addTransitive(otherCcCompilationContext.nonCodeInputs);

      // All module maps of direct dependencies are inputs to the current compile independently of
      // the build type.
      if (otherCcCompilationContext.getCppModuleMap() != null) {
        directModuleMaps.add(otherCcCompilationContext.getCppModuleMap().getArtifact());
      }
      // Likewise, module maps re-exported from dependencies are inputs to the current compile.
      for (CppModuleMap moduleMap : otherCcCompilationContext.exportingModuleMaps) {
        directModuleMaps.add(moduleMap.getArtifact());
      }

      allDefines.addTransitive(otherCcCompilationContext.getDefines());
      virtualToOriginalHeaders.addTransitive(
          otherCcCompilationContext.getVirtualToOriginalHeaders());

      headerTokens.addTransitive(otherCcCompilationContext.getHeaderTokens());
    }

    private static void addIfNotNull(
        NestedSetBuilder<Artifact> builder, @Nullable Artifact artifact) {
      if (artifact != null) {
        builder.add(artifact);
      }
    }

    /**
     * Merges the given {@code CcCompilationContext}s into this one by adding the contents of their
     * attributes.
     */
    @CanIgnoreReturnValue
    public Builder addDependentCcCompilationContexts(
        Iterable<CcCompilationContext> ccCompilationContexts) {
      deps.addAll(ccCompilationContexts);
      return this;
    }

    /**
     * Merges the given {@code CcCompilationContext}s into this one by adding the contents of their
     * attributes, and re-exporting the direct headers and module maps of {@code
     * exportedCcCompilationContexts} through this one.
     */
    @CanIgnoreReturnValue
    public Builder addDependentCcCompilationContexts(
        Iterable<CcCompilationContext> exportedCcCompilationContexts,
        Iterable<CcCompilationContext> ccCompilationContexts) {
      deps.addAll(ccCompilationContexts);
      exportedDeps.addAll(exportedCcCompilationContexts);
      return this;
    }

    private void mergeDependentCcCompilationContexts(
        Iterable<CcCompilationContext> exportedCcCompilationContexts,
        Iterable<CcCompilationContext> ccCompilationContexts,
        TransitiveSetHelper<String> allDefines,
        NestedSetBuilder<Artifact> transitiveModules,
        NestedSetBuilder<Artifact> transitivePicModules,
        Set<Artifact> directModuleMaps,
        Set<CppModuleMap> exportingModuleMaps) {
      for (CcCompilationContext ccCompilationContext :
          Iterables.concat(exportedCcCompilationContexts, ccCompilationContexts)) {
        mergeDependentCcCompilationContext(
            ccCompilationContext,
            allDefines,
            transitiveModules,
            transitivePicModules,
            directModuleMaps);
      }

      for (CcCompilationContext ccCompilationContext : exportedCcCompilationContexts) {
        // For each of the exported contexts, re-export its own module map and all of the module
        // maps that it exports.
        CppModuleMap moduleMap = ccCompilationContext.getCppModuleMap();
        if (moduleMap != null) {
          exportingModuleMaps.add(moduleMap);
        }
        exportingModuleMaps.addAll(ccCompilationContext.exportingModuleMaps);

        // Merge the modular and textual headers from the compilation context so that they are also
        // re-exported.
        headerInfoBuilder.mergeHeaderInfo(ccCompilationContext.headerInfo);
      }
    }

    /**
     * Add a single include directory to be added with "-I". It can be either relative to the exec
     * root (see {@link
     * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot(String)}) or absolute.
     * Before it is stored, the include directory is normalized.
     */
    @CanIgnoreReturnValue
    public Builder addIncludeDir(PathFragment includeDir) {
      includeDirs.add(includeDir);
      return this;
    }

    /** See {@link #addIncludeDir(PathFragment)} */
    @CanIgnoreReturnValue
    public Builder addIncludeDirs(Iterable<PathFragment> includeDirs) {
      this.includeDirs.addAll(includeDirs);
      return this;
    }

    /**
     * Add a single include directory to be added with "-iquote". It can be either relative to the
     * exec root (see {@link
     * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot(String)}) or absolute.
     * Before it is stored, the include directory is normalized.
     */
    @CanIgnoreReturnValue
    public Builder addQuoteIncludeDir(PathFragment quoteIncludeDir) {
      quoteIncludeDirs.add(quoteIncludeDir);
      return this;
    }

    /** See {@link #addQuoteIncludeDir(PathFragment)} */
    @CanIgnoreReturnValue
    public Builder addQuoteIncludeDirs(Iterable<PathFragment> quoteIncludeDirs) {
      this.quoteIncludeDirs.addAll(quoteIncludeDirs);
      return this;
    }

    /**
     * Add include directories to be added with "-isystem". It can be either relative to the exec
     * root (see {@link
     * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot(String)}) or absolute.
     * Before it is stored, the include directory is normalized.
     */
    @CanIgnoreReturnValue
    public Builder addSystemIncludeDirs(Iterable<PathFragment> systemIncludeDirs) {
      this.systemIncludeDirs.addAll(systemIncludeDirs);
      return this;
    }

    /** Add framewrok include directories to be added with "-F". */
    @CanIgnoreReturnValue
    public Builder addFrameworkIncludeDirs(Iterable<PathFragment> frameworkIncludeDirs) {
      this.frameworkIncludeDirs.addAll(frameworkIncludeDirs);
      return this;
    }

    /**
     * Mark specified include directory as external, coming from an external workspace. It can be
     * added with "-isystem" (GCC) or --system-header-prefix (Clang) to suppress warnings coming
     * from external files.
     */
    @CanIgnoreReturnValue
    public Builder addExternalIncludeDir(PathFragment externalIncludeDir) {
      this.externalIncludeDirs.add(externalIncludeDir);
      return this;
    }

    /**
     * Mark specified include directories as external, coming from an external workspace. These can
     * be added with "-isystem" (GCC) or --system-header-prefix (Clang) to suppress warnings coming
     * from external files.
     */
    @CanIgnoreReturnValue
    public Builder addExternalIncludeDirs(Iterable<PathFragment> externalIncludeDirs) {
      this.externalIncludeDirs.addAll(externalIncludeDirs);
      return this;
    }

    /** Add a single declared include dir, relative to a "-I" or "-iquote" directory". */
    @CanIgnoreReturnValue
    public Builder addLooseHdrsDir(PathFragment dir) {
      looseHdrsDirs.add(dir);
      return this;
    }

    /**
     * Adds a header that has been declared in the {@code src} or {@code headers attribute}. The
     * header will also be added to the compilation prerequisites.
     *
     * <p>Filters out fileset directory artifacts, which are not valid inputs.
     */
    @CanIgnoreReturnValue
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
    @CanIgnoreReturnValue
    public Builder addDeclaredIncludeSrcs(Iterable<Artifact> declaredIncludeSrcs) {
      for (Artifact source : declaredIncludeSrcs) {
        addDeclaredIncludeSrc(source);
      }
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addModularPublicHdrs(Collection<Artifact> headers) {
      this.headerInfoBuilder.addPublicHeaders(headers);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addModularPrivateHdrs(Collection<Artifact> headers) {
      this.headerInfoBuilder.addPrivateHeaders(headers);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addTextualHdrs(Collection<Artifact> headers) {
      this.headerInfoBuilder.addTextualHeaders(headers);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setSeparateModuleHdrs(
        Collection<Artifact> headers,
        DerivedArtifact separateModule,
        DerivedArtifact separatePicModule) {
      this.headerInfoBuilder.setSeparateModuleHdrs(headers, separateModule, separatePicModule);
      return this;
    }

    /** Add a set of required non-code compilation input. */
    @CanIgnoreReturnValue
    public Builder addNonCodeInputs(Iterable<Artifact> inputs) {
      nonCodeInputs.addAll(inputs);
      return this;
    }

    /** Adds a single define. */
    @CanIgnoreReturnValue
    public Builder addDefine(String define) {
      defines.add(define);
      return this;
    }

    /** Adds multiple defines. */
    @CanIgnoreReturnValue
    public Builder addDefines(Iterable<String> defines) {
      Iterables.addAll(this.defines, defines);
      return this;
    }

    /** Adds multiple non-transitive defines. */
    @CanIgnoreReturnValue
    public Builder addNonTransitiveDefines(Iterable<String> defines) {
      Iterables.addAll(this.localDefines, defines);
      return this;
    }

    /** Sets the C++ module map. */
    @CanIgnoreReturnValue
    public Builder setCppModuleMap(CppModuleMap cppModuleMap) {
      this.cppModuleMap = cppModuleMap;
      return this;
    }

    /** Causes the module map to be passed as an action input to dependant compilations. */
    @CanIgnoreReturnValue
    public Builder setPropagateCppModuleMapAsActionInput(boolean propagateModuleMap) {
      this.propagateModuleMapAsActionInput = propagateModuleMap;
      return this;
    }

    /**
     * Sets the C++ header module in non-pic mode.
     *
     * @param headerModule The .pcm file generated for this library.
     */
    @CanIgnoreReturnValue
    Builder setHeaderModule(DerivedArtifact headerModule) {
      this.headerInfoBuilder.setHeaderModule(headerModule);
      return this;
    }

    /**
     * Sets the C++ header module in pic mode.
     *
     * @param picHeaderModule The .pic.pcm file generated for this library.
     */
    @CanIgnoreReturnValue
    Builder setPicHeaderModule(DerivedArtifact picHeaderModule) {
      this.headerInfoBuilder.setPicHeaderModule(picHeaderModule);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setHeadersCheckingMode(
        CppConfiguration.HeadersCheckingMode headersCheckingMode) {
      this.headersCheckingMode = headersCheckingMode;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addVirtualToOriginalHeaders(NestedSet<Tuple> virtualToOriginalHeaders) {
      this.virtualToOriginalHeaders.addTransitive(virtualToOriginalHeaders);
      return this;
    }

    /** Builds the {@link CcCompilationContext}. */
    public CcCompilationContext build() {
      return build(
          actionConstructionContext == null ? null : actionConstructionContext.getActionOwner(),
          actionConstructionContext == null
              ? null
              : actionConstructionContext.getAnalysisEnvironment().getMiddlemanFactory());
    }

    @VisibleForTesting // productionVisibility = Visibility.PRIVATE
    public CcCompilationContext build(ActionOwner owner, MiddlemanFactory middlemanFactory) {
      TransitiveSetHelper<String> allDefines = new TransitiveSetHelper<>();
      NestedSetBuilder<Artifact> transitiveModules = NestedSetBuilder.stableOrder();
      NestedSetBuilder<Artifact> transitivePicModules = NestedSetBuilder.stableOrder();
      Set<Artifact> directModuleMaps = new LinkedHashSet<>();
      Set<CppModuleMap> exportingModuleMaps = new LinkedHashSet<>();
      mergeDependentCcCompilationContexts(
          exportedDeps.build(),
          deps.build(),
          allDefines,
          transitiveModules,
          transitivePicModules,
          directModuleMaps,
          exportingModuleMaps);

      allDefines.addAll(defines);

      HeaderInfo headerInfo = headerInfoBuilder.build();
      NestedSet<Artifact> constructedPrereq = createMiddleman(owner, middlemanFactory);

      return new CcCompilationContext(
          new CommandLineCcCompilationContext(
              includeDirs.getMergedResult(),
              quoteIncludeDirs.getMergedResult(),
              systemIncludeDirs.getMergedResult(),
              frameworkIncludeDirs.getMergedResult(),
              externalIncludeDirs.getMergedResult(),
              allDefines.getMergedResult(),
              ImmutableList.copyOf(localDefines)),
          constructedPrereq,
          looseHdrsDirs.build(),
          declaredIncludeSrcs.build(),
          nonCodeInputs.build(),
          headerInfo,
          transitiveModules.build(),
          transitivePicModules.build(),
          ImmutableList.copyOf(directModuleMaps),
          ImmutableList.copyOf(exportingModuleMaps),
          cppModuleMap,
          propagateModuleMapAsActionInput,
          headersCheckingMode,
          virtualToOriginalHeaders.build(),
          headerTokens.build());
    }

    /** Creates a middleman for the compilation prerequisites. */
    private NestedSet<Artifact> createMiddleman(
        ActionOwner owner, MiddlemanFactory middlemanFactory) {
      if (middlemanFactory == null) {
        // TODO(b/110873917): We don't have a middleman factory, therefore, we use the compilation
        // prerequisites as they were passed to the builder, i.e. we use every header instead of a
        // middle man.
        return compilationPrerequisites.build();
      }

      if (compilationPrerequisites.isEmpty()) {
        return compilationPrerequisites.build();
      }

      if (!configuration.getFragment(CppConfiguration.class).useSchedulingMiddlemen()) {
        return compilationPrerequisites.build();
      }

      // Compilation prerequisites gathered in the compilationPrerequisites must be generated prior
      // to executing a C++ compilation step that depends on them (since these prerequisites include
      // all potential header files, etc that could be referenced during compilation). So there is a
      // definite need to ensure a scheduling dependency. However, those prerequisites should have
      // no effect on the decision whether C++ compilation should happen in the first place - only
      // the CppCompileAction outputs (.o and .d files) and all files referenced by the .d file
      // should be used to make that decision.
      //
      // If this action was never executed, then the .d file is missing, forcing compilation. If the
      // .d file is present and has not changed then the only reason that would force us to
      // re-compile would be change in one of the files referenced by the .d file, since no other
      // files participated in the compilation.
      //
      // We also need to propagate errors through this dependency link. Therefore, we use a
      // scheduling dependency middleman. Such a middleman will be ignored by the dependency checker
      // yet still represents an edge in the action dependency graph - forcing proper execution
      // order and error propagation.
      String name = cppModuleMap != null ? cppModuleMap.getName() : label.toString();
      Artifact prerequisiteStampFile =
          middlemanFactory.createSchedulingDependencyMiddleman(
              owner,
              name,
              purpose,
              compilationPrerequisites.build(),
              configuration.getMiddlemanDirectory(label.getRepository()));
      return NestedSetBuilder.create(Order.STABLE_ORDER, prerequisiteStampFile);
    }

    /**
     * This class helps create efficient flattened transitive sets across all transitive
     * dependencies. For very sparsely populated items, this can be more efficient both in terms of
     * CPU and in terms of memory than NestedSets. Merged transitive set will be returned as a flat
     * list to be memory efficient. As a further optimization, if a single dependencies contains a
     * superset of all other dependencies, its list is simply re-used.
     */
    private static class TransitiveSetHelper<E> {
      private final Set<E> all = CompactHashSet.create();
      private ImmutableList<E> largestTransitive = ImmutableList.of();

      public void add(E element) {
        all.add(element);
      }

      public void addAll(Iterable<E> elements) {
        Iterables.addAll(all, elements);
      }

      public void addTransitive(ImmutableList<E> transitive) {
        all.addAll(transitive);
        if (transitive.size() > largestTransitive.size()) {
          largestTransitive = transitive;
        }
      }

      public ImmutableList<E> getMergedResult() {
        ImmutableList<E> allAsList = ImmutableList.copyOf(all);
        return allAsList.equals(largestTransitive) ? largestTransitive : allAsList;
      }
    }
  }

  /**
   * Gathers data about the PIC and no-PIC .pcm files belonging to this context and the associated
   * information about the headers, e.g. modular vs. textual headers and pre-grepped header files.
   *
   * <p>This also implements a data structure very similar to NestedSet, but chosing slightly
   * different trade-offs to account for the specific data stored in here, specifically, we know
   * that there is going to be a single entry in every node of the DAG. Contrary to NestedSet, we
   * reuse memoization data from dependencies to conserve both runtime and memory. Experiments have
   * shown that >90% of a node's flattened transitive deps come from the largest dependency.
   *
   * <p>The order of elements is stable, although not necessarily the same as a STABLE NestedSet.
   * The transitive collection can be iterated without materialization in memory.
   */
  @Immutable
  private static final class HeaderInfo {
    /**
     * The modules built for this context. If null, then no module is being compiled for this
     * context.
     */
    private final DerivedArtifact headerModule;

    private final DerivedArtifact picHeaderModule;

    /** All public header files that are compiled into this module. */
    private final ImmutableList<Artifact> modularPublicHeaders;

    /** All private header files that are compiled into this module. */
    private final ImmutableList<Artifact> modularPrivateHeaders;

    /** All header files that are compiled into this module. */
    private final ImmutableList<Artifact> modularHeaders;

    /** All textual header files that are contained in this module. */
    private final ImmutableList<Artifact> textualHeaders;

    /** Headers that can be compiled into a separate, smaller module for performance reasons. */
    private final ImmutableList<Artifact> separateModuleHeaders;

    private final DerivedArtifact separateModule;
    private final DerivedArtifact separatePicModule;

    /** HeaderInfos of direct dependencies of C++ target represented by this context. */
    private final ImmutableList<HeaderInfo> deps;

    /** Collection representing the memoized form of transitive information, set by flatten(). */
    private TransitiveHeaderCollection memo = null;

    HeaderInfo(
        DerivedArtifact headerModule,
        DerivedArtifact picHeaderModule,
        ImmutableList<Artifact> modularPublicHeaders,
        ImmutableList<Artifact> modularPrivateHeaders,
        ImmutableList<Artifact> textualHeaders,
        ImmutableList<Artifact> separateModuleHeaders,
        DerivedArtifact separateModule,
        DerivedArtifact separatePicModule,
        ImmutableList<HeaderInfo> deps) {
      this.headerModule = headerModule;
      this.picHeaderModule = picHeaderModule;
      this.modularPublicHeaders = modularPublicHeaders;
      this.modularPrivateHeaders = modularPrivateHeaders;
      this.modularHeaders =
          ImmutableList.copyOf(Iterables.concat(modularPublicHeaders, modularPrivateHeaders));
      this.textualHeaders = textualHeaders;
      this.separateModuleHeaders = separateModuleHeaders;
      this.separateModule = separateModule;
      this.separatePicModule = separatePicModule;
      this.deps = deps;
    }

    DerivedArtifact getModule(boolean pic) {
      return pic ? picHeaderModule : headerModule;
    }

    DerivedArtifact getSeparateModule(boolean pic) {
      return pic ? separatePicModule : separateModule;
    }

    public Collection<HeaderInfo> getTransitiveCollection() {
      if (deps.isEmpty()) {
        return ImmutableList.of(this);
      }
      if (memo == null) {
        flatten();
      }
      return memo;
    }

    private synchronized void flatten() {
      if (memo != null) {
        return; // Some other thread has flattened this list while we waited for the lock.
      }
      Collection<HeaderInfo> largestDepList = ImmutableList.of();
      HeaderInfo largestDep = null;
      for (HeaderInfo dep : deps) {
        Collection<HeaderInfo> depList = dep.getTransitiveCollection();
        if (depList.size() > largestDepList.size()) {
          largestDepList = depList;
          largestDep = dep;
        }
      }
      CompactHashSet<HeaderInfo> result = CompactHashSet.create(largestDepList);
      result.add(this);
      ArrayList<HeaderInfo> additionalDeps = new ArrayList<>();
      for (HeaderInfo dep : deps) {
        dep.addOthers(result, additionalDeps);
      }
      memo = new TransitiveHeaderCollection(result.size(), largestDep, additionalDeps);
    }

    private void addOthers(Set<HeaderInfo> result, List<HeaderInfo> additionalDeps) {
      if (result.add(this)) {
        additionalDeps.add(this);
        for (HeaderInfo dep : deps) {
          dep.addOthers(result, additionalDeps);
        }
      }
    }

    /** Represents the memoized transitive information for a HeaderInfo instance. */
    private class TransitiveHeaderCollection extends AbstractCollection<HeaderInfo> {
      private final int size;
      private final HeaderInfo largestDep;
      private final ImmutableList<HeaderInfo> additionalDeps;

      TransitiveHeaderCollection(int size, HeaderInfo largestDep, List<HeaderInfo> additionalDeps) {
        this.size = size;
        this.largestDep = largestDep;
        this.additionalDeps = ImmutableList.copyOf(additionalDeps);
      }

      @Override
      public int size() {
        return size;
      }

      @Override
      public Iterator<HeaderInfo> iterator() {
        return new TransitiveHeaderIterator(HeaderInfo.this);
      }
    }

    /** Iterates over memoized transitive information, without materializing it in memory. */
    private static class TransitiveHeaderIterator implements Iterator<HeaderInfo> {
      private HeaderInfo headerInfo;
      private int pos = -1;

      public TransitiveHeaderIterator(HeaderInfo headerInfo) {
        this.headerInfo = headerInfo;
      }

      @Override
      public boolean hasNext() {
        return !headerInfo.deps.isEmpty();
      }

      @Override
      public HeaderInfo next() {
        pos++;
        if (pos > headerInfo.memo.additionalDeps.size()) {
          pos = 0;
          headerInfo = headerInfo.memo.largestDep;
        }
        if (pos == 0) {
          return headerInfo;
        }
        return headerInfo.memo.additionalDeps.get(pos - 1);
      }
    }

    /** Builder class for {@link HeaderInfo}. */
    public static class Builder {
      private DerivedArtifact headerModule = null;
      private DerivedArtifact picHeaderModule = null;
      private final Set<Artifact> modularPublicHeaders = new HashSet<>();
      private final Set<Artifact> modularPrivateHeaders = new HashSet<>();
      private final Set<Artifact> textualHeaders = new HashSet<>();
      private Collection<Artifact> separateModuleHeaders = ImmutableList.of();
      private DerivedArtifact separateModule = null;
      private DerivedArtifact separatePicModule = null;
      private final List<HeaderInfo> deps = new ArrayList<>();

      @CanIgnoreReturnValue
      Builder setHeaderModule(DerivedArtifact headerModule) {
        this.headerModule = headerModule;
        return this;
      }

      @CanIgnoreReturnValue
      Builder setPicHeaderModule(DerivedArtifact headerModule) {
        this.picHeaderModule = headerModule;
        return this;
      }

      @CanIgnoreReturnValue
      public Builder addDep(HeaderInfo dep) {
        deps.add(dep);
        return this;
      }

      @CanIgnoreReturnValue
      public Builder addPublicHeaders(Collection<Artifact> headers) {
        // TODO(djasper): CPP_TEXTUAL_INCLUDEs are currently special cased here and in
        // CppModuleMapAction. These should be moved to a place earlier in the Action construction.
        for (Artifact header : headers) {
          if (header.isFileType(CppFileTypes.CPP_TEXTUAL_INCLUDE)) {
            this.textualHeaders.add(header);
          } else {
            this.modularPublicHeaders.add(header);
          }
        }
        return this;
      }

      @CanIgnoreReturnValue
      public Builder addPrivateHeaders(Collection<Artifact> headers) {
        // TODO(djasper): CPP_TEXTUAL_INCLUDEs are currently special cased here and in
        // CppModuleMapAction. These should be moved to a place earlier in the Action construction.
        for (Artifact header : headers) {
          if (header.isFileType(CppFileTypes.CPP_TEXTUAL_INCLUDE)) {
            this.textualHeaders.add(header);
          } else {
            this.modularPrivateHeaders.add(header);
          }
        }
        return this;
      }

      @CanIgnoreReturnValue
      public Builder addTextualHeaders(Collection<Artifact> headers) {
        this.textualHeaders.addAll(headers);
        return this;
      }

      @CanIgnoreReturnValue
      public Builder setSeparateModuleHdrs(
          Collection<Artifact> headers,
          DerivedArtifact separateModule,
          DerivedArtifact separatePicModule) {
        this.separateModuleHeaders = headers;
        this.separateModule = separateModule;
        this.separatePicModule = separatePicModule;
        return this;
      }

      /** Adds the headers of the given {@code HeaderInfo} into the one being built. */
      @CanIgnoreReturnValue
      public Builder mergeHeaderInfo(HeaderInfo otherHeaderInfo) {
        this.modularPublicHeaders.addAll(otherHeaderInfo.modularPublicHeaders);
        this.modularPrivateHeaders.addAll(otherHeaderInfo.modularPrivateHeaders);
        this.textualHeaders.addAll(otherHeaderInfo.textualHeaders);
        return this;
      }

      @SuppressWarnings("LenientFormatStringValidation")
      public HeaderInfo build() {
        // Expected 0 args, but got 2.
        Preconditions.checkState(
            (separateModule == null || headerModule != null)
                && (separatePicModule == null || picHeaderModule != null),
            "Separate module cannot be used without main module",
            separateModule,
            separatePicModule);
        return new HeaderInfo(
            headerModule,
            picHeaderModule,
            ImmutableList.copyOf(modularPublicHeaders),
            ImmutableList.copyOf(modularPrivateHeaders),
            ImmutableList.copyOf(textualHeaders),
            ImmutableList.copyOf(separateModuleHeaders),
            separateModule,
            separatePicModule,
            ImmutableList.copyOf(deps));
      }
    }
  }
}
