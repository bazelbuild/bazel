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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.bugreport.BugReport;
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
import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.eval.Tuple;

/**
 * Immutable store of information needed for C++ compilation that is aggregated across dependencies.
 */
@Immutable
public final class CcCompilationContext implements CcCompilationContextApi<Artifact, CppModuleMap> {
  /** An empty {@code CcCompilationContext}. */
  public static final CcCompilationContext EMPTY =
      create(
          new CommandLineCcCompilationContext(
              /* includeDirs= */ ImmutableList.of(),
              /* quoteIncludeDirs= */ ImmutableList.of(),
              /* systemIncludeDirs= */ ImmutableList.of(),
              /* frameworkIncludeDirs= */ ImmutableList.of(),
              /* externalIncludeDirs= */ ImmutableList.of(),
              /* defines= */ ImmutableList.of(),
              /* localDefines= */ ImmutableList.of()),
          /* declaredIncludeSrcs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* nonCodeInputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          HeaderInfo.EMPTY,
          /* transitiveModules= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* transitivePicModules= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* directModuleMaps= */ ImmutableList.of(),
          /* exportingModuleMaps= */ ImmutableList.of(),
          /* cppModuleMap= */ null,
          /* virtualToOriginalHeaders= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* headerTokens= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER));

  private final CommandLineCcCompilationContext commandLineCcCompilationContext;

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

  // Derived from depsContexts.

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
      NestedSet<Artifact> declaredIncludeSrcs,
      NestedSet<Artifact> nonCodeInputs,
      HeaderInfo headerInfo,
      NestedSet<Artifact> transitiveModules,
      NestedSet<Artifact> transitivePicModules,
      ImmutableList<Artifact> directModuleMaps,
      ImmutableList<CppModuleMap> exportingModuleMaps,
      CppModuleMap cppModuleMap,
      NestedSet<Tuple> virtualToOriginalHeaders,
      NestedSet<Artifact> headerTokens) {
    Preconditions.checkNotNull(commandLineCcCompilationContext);
    this.commandLineCcCompilationContext = commandLineCcCompilationContext;
    this.declaredIncludeSrcs = declaredIncludeSrcs;
    this.directModuleMaps = directModuleMaps;
    this.exportingModuleMaps = exportingModuleMaps;
    this.headerInfo = headerInfo;
    this.transitiveModules = transitiveModules;
    this.transitivePicModules = transitivePicModules;
    this.cppModuleMap = cppModuleMap;
    this.nonCodeInputs = nonCodeInputs;
    this.virtualToOriginalHeaders = virtualToOriginalHeaders;
    this.transitiveHeaderCount = -1;
    this.headerTokens = headerTokens;
  }

  public static CcCompilationContext create(
      CommandLineCcCompilationContext commandLineCcCompilationContext,
      NestedSet<Artifact> declaredIncludeSrcs,
      NestedSet<Artifact> nonCodeInputs,
      HeaderInfo headerInfo,
      NestedSet<Artifact> transitiveModules,
      NestedSet<Artifact> transitivePicModules,
      ImmutableList<Artifact> directModuleMaps,
      ImmutableList<CppModuleMap> exportingModuleMaps,
      CppModuleMap cppModuleMap,
      NestedSet<Tuple> virtualToOriginalHeaders,
      NestedSet<Artifact> headerTokens) {
    return new CcCompilationContext(
        commandLineCcCompilationContext,
        declaredIncludeSrcs,
        nonCodeInputs,
        headerInfo,
        transitiveModules,
        transitivePicModules,
        directModuleMaps,
        exportingModuleMaps,
        cppModuleMap,
        virtualToOriginalHeaders,
        headerTokens);
  }

  public static CcCompilationContext createAndMerge(
      SymbolGenerator.Symbol<?> identityToken,
      CcCompilationContext single,
      Sequence<CcCompilationContext> exportedDeps,
      Sequence<CcCompilationContext> deps) {
    Preconditions.checkState(single.getHeaderInfo().deps.isEmpty());
    Preconditions.checkArgument(single.transitiveModules.isEmpty());
    Preconditions.checkArgument(single.transitivePicModules.isEmpty());
    Preconditions.checkArgument(single.directModuleMaps.isEmpty());
    Preconditions.checkArgument(single.exportingModuleMaps.isEmpty());

    // CommandLineCcCompilationContext fields
    TransitiveSetHelper<PathFragment> includeDirs = new TransitiveSetHelper<>();
    TransitiveSetHelper<PathFragment> quoteIncludeDirs = new TransitiveSetHelper<>();
    TransitiveSetHelper<PathFragment> systemIncludeDirs = new TransitiveSetHelper<>();
    TransitiveSetHelper<PathFragment> frameworkIncludeDirs = new TransitiveSetHelper<>();
    TransitiveSetHelper<PathFragment> externalIncludeDirs = new TransitiveSetHelper<>();
    TransitiveSetHelper<String> allDefines = new TransitiveSetHelper<>();

    // CcCompilationContext fields
    NestedSetBuilder<Artifact> declaredIncludeSrcs = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> nonCodeInputs = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> transitiveModules = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> transitivePicModules = NestedSetBuilder.stableOrder();
    Set<Artifact> directModuleMaps = new LinkedHashSet<>();
    Set<CppModuleMap> exportingModuleMaps = new LinkedHashSet<>();
    NestedSetBuilder<Tuple> virtualToOriginalHeaders = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> headerTokens = NestedSetBuilder.stableOrder();

    // Transitive part of HeaderInfo
    ImmutableList.Builder<HeaderInfo> depHeaderInfos = ImmutableList.builder();
    ImmutableList.Builder<HeaderInfo> mergedHeaderInfos = ImmutableList.builder();

    // Merge in single
    includeDirs.addTransitive(single.getIncludeDirs());
    quoteIncludeDirs.addTransitive(single.getQuoteIncludeDirs());
    systemIncludeDirs.addTransitive(single.getSystemIncludeDirs());
    frameworkIncludeDirs.addTransitive(single.getFrameworkIncludeDirs());
    externalIncludeDirs.addTransitive(single.getExternalIncludeDirs());
    declaredIncludeSrcs.addAll(single.getDeclaredIncludeSrcs().toList());
    nonCodeInputs.addAll(single.getNonCodeInputs().toList());
    virtualToOriginalHeaders.addTransitive(single.getVirtualToOriginalHeaders());
    headerTokens.addTransitive(single.getHeaderTokens());

    // Merge the compilation contexts.
    for (CcCompilationContext otherCcCompilationContext : Iterables.concat(exportedDeps, deps)) {
      includeDirs.addTransitive(otherCcCompilationContext.getIncludeDirs());
      quoteIncludeDirs.addTransitive(otherCcCompilationContext.getQuoteIncludeDirs());
      systemIncludeDirs.addTransitive(otherCcCompilationContext.getSystemIncludeDirs());
      frameworkIncludeDirs.addTransitive(otherCcCompilationContext.getFrameworkIncludeDirs());
      externalIncludeDirs.addTransitive(otherCcCompilationContext.getExternalIncludeDirs());
      allDefines.addTransitive(otherCcCompilationContext.getDefines());
      declaredIncludeSrcs.addTransitive(otherCcCompilationContext.getDeclaredIncludeSrcs());
      nonCodeInputs.addTransitive(otherCcCompilationContext.getNonCodeInputs());

      transitiveModules.addTransitive(otherCcCompilationContext.getTransitiveModules(false));
      addIfNotNull(transitiveModules, otherCcCompilationContext.getHeaderInfo().getModule(false));
      addIfNotNull(
          transitiveModules, otherCcCompilationContext.getHeaderInfo().getSeparateModule(false));

      transitivePicModules.addTransitive(otherCcCompilationContext.getTransitiveModules(true));
      addIfNotNull(transitivePicModules, otherCcCompilationContext.getHeaderInfo().getModule(true));
      addIfNotNull(
          transitivePicModules, otherCcCompilationContext.getHeaderInfo().getSeparateModule(true));

      // All module maps of direct dependencies are inputs to the current compile independently of
      // the build type.
      if (otherCcCompilationContext.getCppModuleMap() != null) {
        directModuleMaps.add(otherCcCompilationContext.getCppModuleMap().getArtifact());
      }
      // Likewise, module maps re-exported from dependencies are inputs to the current compile.
      for (CppModuleMap moduleMap : otherCcCompilationContext.getExportingModuleMaps()) {
        directModuleMaps.add(moduleMap.getArtifact());
      }

      virtualToOriginalHeaders.addTransitive(
          otherCcCompilationContext.getVirtualToOriginalHeaders());
      headerTokens.addTransitive(otherCcCompilationContext.getHeaderTokens());

      depHeaderInfos.add(otherCcCompilationContext.getHeaderInfo());
    }

    for (CcCompilationContext ccCompilationContext : exportedDeps) {
      // For each of the exported contexts, re-export its own module map and all of the module
      // maps that it exports.
      CppModuleMap moduleMap = ccCompilationContext.getCppModuleMap();
      if (moduleMap != null) {
        exportingModuleMaps.add(moduleMap);
      }
      exportingModuleMaps.addAll(ccCompilationContext.getExportingModuleMaps());

      // Merge the modular and textual headers from the compilation context so that they are also
      // re-exported.
      mergedHeaderInfos.add(ccCompilationContext.getHeaderInfo());
    }

    // Merge direct defines last.
    allDefines.addAll(single.getDefines());

    HeaderInfo headerInfo =
        HeaderInfo.create(
            identityToken,
            single.getHeaderInfo().headerModule,
            single.getHeaderInfo().picHeaderModule,
            single.getHeaderInfo().modularPublicHeaders,
            single.getHeaderInfo().modularPrivateHeaders,
            single.getHeaderInfo().textualHeaders,
            single.getHeaderInfo().separateModuleHeaders,
            single.getHeaderInfo().separateModule,
            single.getHeaderInfo().separatePicModule,
            depHeaderInfos.build(),
            mergedHeaderInfos.build());

    return new CcCompilationContext(
        new CommandLineCcCompilationContext(
            includeDirs.getMergedResult(),
            quoteIncludeDirs.getMergedResult(),
            systemIncludeDirs.getMergedResult(),
            frameworkIncludeDirs.getMergedResult(),
            externalIncludeDirs.getMergedResult(),
            allDefines.getMergedResult(),
            single.getNonTransitiveDefines()),
        declaredIncludeSrcs.build(),
        nonCodeInputs.build(),
        headerInfo,
        transitiveModules.build(),
        transitivePicModules.build(),
        ImmutableList.copyOf(directModuleMaps),
        ImmutableList.copyOf(exportingModuleMaps),
        single.getCppModuleMap(),
        virtualToOriginalHeaders.build(),
        headerTokens.build());
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
  public Depset getStarlarkValidationArtifacts() {
    return Depset.of(Artifact.class, getHeaderTokens());
  }

  @Override
  public Depset getStarlarkVirtualToOriginalHeaders() {
    return Depset.of(Tuple.class, getVirtualToOriginalHeaders());
  }

  @Override
  @Nullable
  public CppModuleMap getStarlarkModuleMap() {
    return getCppModuleMap();
  }

  @Override
  public StarlarkList<CppModuleMap> getStarlarkExportingModuleMaps() {
    return StarlarkList.immutableCopyOf(getExportingModuleMaps());
  }

  /** Returns the command line compilation context. */
  public CommandLineCcCompilationContext getCommandLineCcCompilationContext() {
    return commandLineCcCompilationContext;
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
   * Returns the immutable set of headers that have been declared in the {@code srcs} or {@code
   * hdrs} attribute (possibly empty but never null).
   *
   * <p>Those are exactly transitive compilation prerequisites needed by all reverse dependencies;
   * note that these do specifically not include any compilation prerequisites that are only needed
   * by the rule itself (for example, compiled source files from the {@code srcs} attribute).
   *
   * <p>The returned set can be empty if there are no prerequisites.
   */
  public NestedSet<Artifact> getDeclaredIncludeSrcs() {
    return declaredIncludeSrcs;
  }

  public NestedSet<Artifact> getNonCodeInputs() {
    return nonCodeInputs;
  }

  @Override
  public Depset getNonCodeInputsForStarlark() {
    return Depset.of(Artifact.class, nonCodeInputs);
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

  @VisibleForTesting
  public HeaderInfo getHeaderInfo() {
    return headerInfo;
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
          transitiveHeaderInfo.modularPublicHeaders,
          pathToLegalArtifact,
          treeArtifacts,
          isModule,
          modularHeaders);
      handleHeadersForIncludeScanning(
          transitiveHeaderInfo.modularPrivateHeaders,
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
    removeArtifactsFromSet(modularHeaders, headerInfo.modularPublicHeaders);
    removeArtifactsFromSet(modularHeaders, headerInfo.modularPrivateHeaders);
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
      for (int i = 0; i < transitiveHeaderInfo.modularPublicHeaders.size(); i++) {
        Artifact header = transitiveHeaderInfo.modularPublicHeaders.get(i);
        if (includes.contains(header)) {
          modules.add(module);
          break;
        }
      }
      for (int i = 0; i < transitiveHeaderInfo.modularPrivateHeaders.size(); i++) {
        Artifact header = transitiveHeaderInfo.modularPrivateHeaders.get(i);
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
  public Depset getStarlarkTransitiveModules() {
    return Depset.of(Artifact.class, getTransitiveModules(false));
  }

  @Override
  public Depset getStarlarkTransitivePicModules() {
    return Depset.of(Artifact.class, getTransitiveModules(true));
  }

  /** Adds additional transitive inputs needed for compilation to builder. */
  void addAdditionalInputs(NestedSetBuilder<Artifact> builder) {
    builder.addAll(directModuleMaps);
    builder.addTransitive(nonCodeInputs);
    if (cppModuleMap != null) {
      builder.add(cppModuleMap.getArtifact());
    }
  }

  /** Returns modules maps from direct dependencies. */
  public ImmutableList<Artifact> getDirectModuleMaps() {
    return directModuleMaps;
  }

  @Override
  public StarlarkList<Artifact> getDirectModuleMapsForStarlark() {
    return StarlarkList.immutableCopyOf(getDirectModuleMaps());
  }

  @StarlarkMethod(name = "_direct_module_maps_set", structField = true, documented = false)
  public Depset getDirectModuleMapsSetForStarlark() {
    return Depset.of(
        Artifact.class, NestedSetBuilder.wrap(Order.STABLE_ORDER, getDirectModuleMaps()));
  }

  DerivedArtifact getHeaderModule(boolean usePic) {
    return headerInfo.getModule(usePic);
  }

  DerivedArtifact getSeparateHeaderModule(boolean usePic) {
    return headerInfo.getSeparateModule(usePic);
  }

  /**
   * Returns all declared headers of the current module if the current target is compiled as a
   * module.
   */
  ImmutableList<Artifact> getHeaderModuleSrcs(boolean separateModule) {
    if (separateModule) {
      return headerInfo.separateModuleHeaders;
    }
    return new ImmutableSet.Builder<Artifact>()
        .addAll(headerInfo.modularPublicHeaders)
        .addAll(headerInfo.modularPrivateHeaders)
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

  /** Returns the C++ module map of the owner. */
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
  static final class CommandLineCcCompilationContext {
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

  private static void addIfNotNull(
      NestedSetBuilder<Artifact> builder, @Nullable Artifact artifact) {
    if (artifact != null) {
      builder.add(artifact);
    }
  }

  /**
   * This class helps create efficient flattened transitive sets across all transitive dependencies.
   * For very sparsely populated items, this can be more efficient both in terms of CPU and in terms
   * of memory than NestedSets. Merged transitive set will be returned as a flat list to be memory
   * efficient. As a further optimization, if a single dependencies contains a superset of all other
   * dependencies, its list is simply re-used.
   */
  private static class TransitiveSetHelper<E> {
    private final Set<E> all = CompactHashSet.create();
    private ImmutableList<E> largestTransitive = ImmutableList.of();

    void addAll(Iterable<E> elements) {
      Iterables.addAll(all, elements);
    }

    void addTransitive(ImmutableList<E> transitive) {
      all.addAll(transitive);
      if (transitive.size() > largestTransitive.size()) {
        largestTransitive = transitive;
      }
    }

    ImmutableList<E> getMergedResult() {
      ImmutableList<E> allAsList = ImmutableList.copyOf(all);
      return allAsList.equals(largestTransitive) ? largestTransitive : allAsList;
    }
  }

  /**
   * Gathers data about the PIC and no-PIC .pcm files belonging to this context and the associated
   * information about the headers, e.g. modular vs. textual headers and pre-grepped header files.
   *
   * <p>This also implements a data structure very similar to NestedSet, but choosing slightly
   * different trade-offs to account for the specific data stored in here, specifically, we know
   * that there is going to be a single entry in every node of the DAG. Contrary to NestedSet, we
   * reuse memoization data from dependencies to conserve both runtime and memory. Experiments have
   * shown that >90% of a node's flattened transitive deps come from the largest dependency.
   *
   * <p>The order of elements is stable, although not necessarily the same as a STABLE NestedSet.
   * The transitive collection can be iterated without materialization in memory.
   */
  @Immutable
  @VisibleForTesting
  public static final class HeaderInfo {
    // This class has non-private visibility testing and HeaderInfoCodec.

    final SymbolGenerator.Symbol<?> identityToken;

    /**
     * The modules built for this context. If null, then no module is being compiled for this
     * context.
     */
    final DerivedArtifact headerModule;

    final DerivedArtifact picHeaderModule;

    /** All public header files that are compiled into this module. */
    final ImmutableList<Artifact> modularPublicHeaders;

    /** All private header files that are compiled into this module. */
    final ImmutableList<Artifact> modularPrivateHeaders;

    /** All textual header files that are contained in this module. */
    final ImmutableList<Artifact> textualHeaders;

    /** Headers that can be compiled into a separate, smaller module for performance reasons. */
    final ImmutableList<Artifact> separateModuleHeaders;

    final DerivedArtifact separateModule;
    final DerivedArtifact separatePicModule;

    /** HeaderInfos of direct dependencies of C++ target represented by this context. */
    final ImmutableList<HeaderInfo> deps;

    public static final HeaderInfo EMPTY =
        HeaderInfo.create(
            SymbolGenerator.CONSTANT_SYMBOL,
            /* headerModule= */ null,
            /* picHeaderModule= */ null,
            /* publicHeaders= */ ImmutableList.of(),
            /* privateHeaders= */ ImmutableList.of(),
            /* textualHeaders= */ ImmutableList.of(),
            /* separateModuleHeaders= */ ImmutableList.of(),
            /* separateModule= */ null,
            /* separatePicModule= */ null,
            /* deps= */ ImmutableList.of(),
            /* mergedDeps= */ ImmutableList.of());

    /** Collection representing the memoized form of transitive information, set by flatten(). */
    private TransitiveHeaderCollection memo = null;

    HeaderInfo(
        SymbolGenerator.Symbol<?> identityToken,
        DerivedArtifact headerModule,
        DerivedArtifact picHeaderModule,
        ImmutableList<Artifact> modularPublicHeaders,
        ImmutableList<Artifact> modularPrivateHeaders,
        ImmutableList<Artifact> textualHeaders,
        ImmutableList<Artifact> separateModuleHeaders,
        DerivedArtifact separateModule,
        DerivedArtifact separatePicModule,
        ImmutableList<HeaderInfo> deps) {
      this.identityToken = identityToken;
      this.headerModule = headerModule;
      this.picHeaderModule = picHeaderModule;
      this.modularPublicHeaders = modularPublicHeaders;
      this.modularPrivateHeaders = modularPrivateHeaders;
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

    @VisibleForTesting
    public ImmutableList<Artifact> modularPublicHeaders() {
      return modularPublicHeaders;
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

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof HeaderInfo that)) {
        return false;
      }
      return identityToken.equals(that.identityToken);
    }

    @Override
    public int hashCode() {
      return identityToken.hashCode();
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

    /**
     * Creates a new {@link HeaderInfo} instance.
     *
     * @param identityToken The identity token for the HeaderInfo.
     * @param headerModule The .pcm file generated for this library.
     * @param picHeaderModule The .pic.pcm file generated for this library.
     * @param publicHeaders All public header files that are compiled into this module.
     * @param privateHeaders All private header files that are compiled into this module.
     * @param textualHeaders All textual header files that are contained in this module.
     * @param separateModuleHeaders Headers that can be compiled into a separate, smaller module for
     *     performance reasons.
     * @param separateModule The .pcm file generated for the separate module.
     * @param separatePicModule The .pic.pcm file generated for the separate module.
     * @param deps HeaderInfos of direct dependencies of C++ target represented by this context.
     * @param mergedDeps HeaderInfos to merge into this one.
     */
    public static HeaderInfo create(
        SymbolGenerator.Symbol<?> identityToken,
        @Nullable DerivedArtifact headerModule,
        @Nullable DerivedArtifact picHeaderModule,
        Collection<Artifact> publicHeaders,
        Collection<Artifact> privateHeaders,
        Collection<Artifact> textualHeaders,
        ImmutableList<Artifact> separateModuleHeaders,
        @Nullable DerivedArtifact separateModule,
        @Nullable DerivedArtifact separatePicModule,
        ImmutableList<HeaderInfo> deps,
        ImmutableList<HeaderInfo> mergedDeps) {
      Preconditions.checkState(
          (separateModule == null || headerModule != null)
              && (separatePicModule == null || picHeaderModule != null),
          "Separate module ('%s', '%s') cannot be used without main module",
          separateModule,
          separatePicModule);
      ImmutableSet.Builder<Artifact> modularPublicHeaders = ImmutableSet.builder();
      ImmutableSet.Builder<Artifact> modularPrivateHeaders = ImmutableSet.builder();
      ImmutableSet.Builder<Artifact> allTextualHeaders = ImmutableSet.builder();
      allTextualHeaders.addAll(textualHeaders);
      // TODO(djasper): CPP_TEXTUAL_INCLUDEs are currently special cased here and in
      // CppModuleMapAction. These should be moved to a place earlier in the Action construction.
      for (Artifact header : publicHeaders) {
        if (header.isFileType(CppFileTypes.CPP_TEXTUAL_INCLUDE)) {
          allTextualHeaders.add(header);
        } else {
          modularPublicHeaders.add(header);
        }
      }
      for (Artifact header : privateHeaders) {
        if (header.isFileType(CppFileTypes.CPP_TEXTUAL_INCLUDE)) {
          allTextualHeaders.add(header);
        } else {
          modularPrivateHeaders.add(header);
        }
      }
      for (HeaderInfo otherHeaderInfo : mergedDeps) {
        modularPublicHeaders.addAll(otherHeaderInfo.modularPublicHeaders);
        modularPrivateHeaders.addAll(otherHeaderInfo.modularPrivateHeaders);
        allTextualHeaders.addAll(otherHeaderInfo.textualHeaders);
      }
      return new HeaderInfo(
          identityToken,
          headerModule,
          picHeaderModule,
          modularPublicHeaders.build().asList(),
          modularPrivateHeaders.build().asList(),
          allTextualHeaders.build().asList(),
          separateModuleHeaders,
          separateModule,
          separatePicModule,
          deps);
    }
  }
}
