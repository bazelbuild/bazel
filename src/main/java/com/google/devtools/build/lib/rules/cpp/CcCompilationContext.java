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
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.collect.compacthashmap.CompactHashMap;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScanningHeaderData;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.eval.Tuple;

/**
 * Immutable store of information needed for C++ compilation that is aggregated across dependencies.
 */
@Immutable
public final class CcCompilationContext {

  private final StarlarkInfo starlarkInfo;

  private CcCompilationContext(StarlarkInfo starlarkInfo) {
    this.starlarkInfo = starlarkInfo;
  }

  public static CcCompilationContext of(StarlarkInfo starlarkInfo) {
    return new CcCompilationContext(starlarkInfo);
  }

  public Depset getStarlarkHeaders() {
    try {
      return starlarkInfo.getValue("headers", Depset.class);
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  private ImmutableList<PathFragment> getPathFragmentList(String fieldName) {
    try {
      return starlarkInfo.getValue(fieldName, Depset.class).getSet(String.class).toList().stream()
          .map(PathFragment::create)
          .collect(ImmutableList.toImmutableList());
    } catch (EvalException | Depset.TypeException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Returns the immutable list of include directories to be added with "-I" (possibly empty but
   * never null). This includes the include dirs from the transitive deps closure of the target.
   * This list does not contain duplicates. All fragments are either absolute or relative to the
   * exec root (see {@link
   * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot(String)}).
   */
  public ImmutableList<PathFragment> getIncludeDirs() {
    return getPathFragmentList("includes");
  }

  /**
   * Returns the immutable list of include directories to be added with "-iquote" (possibly empty
   * but never null). This includes the include dirs from the transitive deps closure of the target.
   * This list does not contain duplicates. All fragments are either absolute or relative to the
   * exec root (see {@link
   * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot(String)}).
   */
  public ImmutableList<PathFragment> getQuoteIncludeDirs() {
    return getPathFragmentList("quote_includes");
  }

  /**
   * Returns the immutable list of include directories to be added with "-isystem" (possibly empty
   * but never null). This includes the include dirs from the transitive deps closure of the target.
   * This list does not contain duplicates. All fragments are either absolute or relative to the
   * exec root (see {@link
   * com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot(String)}).
   */
  public ImmutableList<PathFragment> getSystemIncludeDirs() {
    return getPathFragmentList("system_includes");
  }

  /**
   * Returns the immutable list of include directories to be added with "-F" (possibly empty but
   * never null). This includes the include dirs from the transitive deps closure of the target.
   * This list does not contain duplicates. All fragments are either absolute or relative to the
   * exec root (see {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot}).
   */
  public ImmutableList<PathFragment> getFrameworkIncludeDirs() {
    return getPathFragmentList("framework_includes");
  }

  /**
   * Returns the immutable list of external include directories (possibly empty but never null).
   * This includes the include dirs from the transitive deps closure of the target. This list does
   * not contain duplicates. All fragments are either absolute or relative to the exec root (see
   * {@link com.google.devtools.build.lib.analysis.BlazeDirectories#getExecRoot(String)}).
   */
  public ImmutableList<PathFragment> getExternalIncludeDirs() {
    return getPathFragmentList("external_includes");
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
    try {
      return starlarkInfo.getValue("headers", Depset.class).getSet(Artifact.class);
    } catch (EvalException | Depset.TypeException e) {
      throw new IllegalStateException(e);
    }
  }

  public NestedSet<Artifact> getNonCodeInputs() {
    try {
      return starlarkInfo.getValue("_non_code_inputs", Depset.class).getSet(Artifact.class);
    } catch (EvalException | Depset.TypeException e) {
      throw new IllegalStateException(e);
    }
  }

  /** Returns headers given as textual_hdrs in this target. */
  @SuppressWarnings("unchecked")
  public ImmutableList<Artifact> getTextualHdrs() {
    try {
      return starlarkInfo.getValue("direct_textual_headers", StarlarkList.class).getImmutableList();
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  /** Returns public headers (given as {@code hdrs}) in this target. */
  @SuppressWarnings("unchecked")
  public ImmutableList<Artifact> getDirectPublicHdrs() {
    try {
      return starlarkInfo.getValue("direct_public_headers", StarlarkList.class).getImmutableList();
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  /** Returns private headers (given as {@code srcs}) in this target. */
  @SuppressWarnings("unchecked")
  public ImmutableList<Artifact> getDirectPrivateHdrs() {
    try {
      return starlarkInfo.getValue("direct_private_headers", StarlarkList.class).getImmutableList();
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  public NestedSet<Artifact> getHeaderTokens() {
    try {
      return starlarkInfo.getValue("validation_artifacts", Depset.class).getSet(Artifact.class);
    } catch (EvalException | Depset.TypeException e) {
      throw new IllegalStateException(e);
    }
  }

  @VisibleForTesting
  public HeaderInfo getHeaderInfo() {
    try {
      return starlarkInfo.getValue("_header_info", HeaderInfo.class);
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
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
    HeaderInfo headerInfo = getHeaderInfo();
    Collection<HeaderInfo> transitiveHeaderInfos = headerInfo.getTransitiveCollection();
    ArrayList<Artifact> treeArtifacts = new ArrayList<>();
    // We'd prefer for these types to use ImmutableSet/ImmutableMap. However, constructing these is
    // substantially more costly in a way that shows up in profiles.
    Map<PathFragment, Artifact> pathToLegalArtifact =
        CompactHashMap.createWithExpectedSize(transitiveHeaderInfos.size());
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
    HeaderInfo headerInfo = getHeaderInfo();
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
    try {
      return starlarkInfo
          .getValue(usePic ? "_transitive_pic_modules" : "_transitive_modules", Depset.class)
          .getSet(Artifact.class);
    } catch (EvalException | Depset.TypeException e) {
      throw new IllegalStateException(e);
    }
  }

  /** Adds additional transitive inputs needed for compilation to builder. */
  void addAdditionalInputs(NestedSetBuilder<Artifact> builder) {
    builder.addTransitive(getDirectModuleMaps());
    builder.addTransitive(getNonCodeInputs());
    if (getCppModuleMap() != null) {
      builder.add(getCppModuleMap().getArtifact());
    }
  }

  /** Returns modules maps from direct dependencies. */
  public NestedSet<Artifact> getDirectModuleMaps() {
    try {
      return starlarkInfo.getValue("_direct_module_maps", Depset.class).getSet(Artifact.class);
    } catch (EvalException | Depset.TypeException e) {
      throw new IllegalStateException(e);
    }
  }

  DerivedArtifact getHeaderModule(boolean usePic) {
    return getHeaderInfo().getModule(usePic);
  }

  DerivedArtifact getSeparateHeaderModule(boolean usePic) {
    return getHeaderInfo().getSeparateModule(usePic);
  }

  /**
   * Returns all declared headers of the current module if the current target is compiled as a
   * module.
   */
  ImmutableList<Artifact> getHeaderModuleSrcs(boolean separateModule) {
    HeaderInfo headerInfo = getHeaderInfo();
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
    try {
      return starlarkInfo.getValue("defines", Depset.class).getSet(String.class).toList();
    } catch (EvalException | Depset.TypeException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Returns the set of defines needed to compile this target. This doesn't include definitions from
   * the transitive deps closure for the target.
   */
  ImmutableList<String> getNonTransitiveDefines() {
    try {
      return starlarkInfo.getValue("local_defines", Depset.class).getSet(String.class).toList();
    } catch (EvalException | Depset.TypeException e) {
      throw new IllegalStateException(e);
    }
  }

  /** Returns the C++ module map of the owner. */
  @Nullable
  public CppModuleMap getCppModuleMap() {
    try {
      StarlarkInfo moduleMap = starlarkInfo.getNoneableValue("_module_map", StarlarkInfo.class);
      return moduleMap == null ? null : new CppModuleMap(moduleMap);
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  /** Returns the list of dependencies' C++ module maps re-exported by this compilation context. */
  @SuppressWarnings("unchecked")
  public ImmutableList<CppModuleMap> getExportingModuleMaps() {
    try {
      return ((StarlarkList<StarlarkInfo>)
              starlarkInfo.getValue("_exporting_module_maps", StarlarkList.class))
          .stream().map(CppModuleMap::new).collect(toImmutableList());
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  public NestedSet<Tuple> getVirtualToOriginalHeaders() {
    try {
      return starlarkInfo
          .getValue("_virtual_to_original_headers", Depset.class)
          .getSet(Tuple.class);
    } catch (EvalException | Depset.TypeException e) {
      throw new IllegalStateException(e);
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
  public static final class HeaderInfo implements StarlarkValue {
    // This class has non-private visibility testing and HeaderInfoCodec.

    final SymbolGenerator.Symbol<?> identityToken;

    /**
     * The modules built for this context. If null, then no module is being compiled for this
     * context.
     */
    @Nullable final DerivedArtifact headerModule;

    @Nullable final DerivedArtifact picHeaderModule;

    /** All public header files that are compiled into this module. */
    final ImmutableList<Artifact> modularPublicHeaders;

    /** All private header files that are compiled into this module. */
    final ImmutableList<Artifact> modularPrivateHeaders;

    /** All textual header files that are contained in this module. */
    final ImmutableList<Artifact> textualHeaders;

    /** Headers that can be compiled into a separate, smaller module for performance reasons. */
    final ImmutableList<Artifact> separateModuleHeaders;

    @Nullable final DerivedArtifact separateModule;
    @Nullable final DerivedArtifact separatePicModule;

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

    @StarlarkMethod(
        name = "separate_module",
        documented = false,
        allowReturnNones = true,
        structField = true)
    @Nullable
    public DerivedArtifact getSeparateModule() {
      return separateModule;
    }

    @StarlarkMethod(
        name = "header_module",
        documented = false,
        allowReturnNones = true,
        structField = true)
    @Nullable
    public DerivedArtifact getHeaderModule() {
      return headerModule;
    }

    @StarlarkMethod(
        name = "pic_header_module",
        documented = false,
        allowReturnNones = true,
        structField = true)
    @Nullable
    public DerivedArtifact getPicHeaderModule() {
      return picHeaderModule;
    }

    @StarlarkMethod(name = "modular_public_headers", documented = false, structField = true)
    public StarlarkList<Artifact> getModularPublicHeaders() {
      return StarlarkList.immutableCopyOf(modularPublicHeaders);
    }

    @StarlarkMethod(name = "modular_private_headers", documented = false, structField = true)
    public StarlarkList<Artifact> getModularPrivateHeaders() {
      return StarlarkList.immutableCopyOf(modularPrivateHeaders);
    }

    @StarlarkMethod(name = "textual_headers", documented = false, structField = true)
    public StarlarkList<Artifact> getTextualHeaders() {
      return StarlarkList.immutableCopyOf(textualHeaders);
    }

    @StarlarkMethod(name = "separate_module_headers", documented = false, structField = true)
    public StarlarkList<Artifact> getSeparateModuleHeaders() {
      return StarlarkList.immutableCopyOf(separateModuleHeaders);
    }

    @StarlarkMethod(
        name = "separate_pic_module",
        documented = false,
        allowReturnNones = true,
        structField = true)
    @Nullable
    public DerivedArtifact getSeparatePicModule() {
      return separatePicModule;
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

    @Override
    public boolean isImmutable() {
      return true;
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
