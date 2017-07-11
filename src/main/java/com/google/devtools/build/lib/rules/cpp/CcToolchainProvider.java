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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Information about a C++ compiler used by the <code>cc_*</code> rules.
 */
@SkylarkModule(
    name = "CcToolchainInfo",
    doc = "Information about the C++ compiler being used."
)
@Immutable
public final class CcToolchainProvider
    extends SkylarkClassObject implements TransitiveInfoProvider {
  public static final String SKYLARK_NAME = "CcToolchainInfo";

  public static final NativeClassObjectConstructor<CcToolchainProvider> SKYLARK_CONSTRUCTOR =
      new NativeClassObjectConstructor<CcToolchainProvider>(
          CcToolchainProvider.class, SKYLARK_NAME) {};

  /** An empty toolchain to be returned in the error case (instead of null). */
  public static final CcToolchainProvider EMPTY_TOOLCHAIN_IS_ERROR =
      new CcToolchainProvider(
          null,
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          null,
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          null,
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          null,
          PathFragment.EMPTY_FRAGMENT,
          CppCompilationContext.EMPTY,
          false,
          false,
          ImmutableMap.<String, String>of(),
          ImmutableList.<Artifact>of(),
          NestedSetBuilder.<Pair<String, String>>emptySet(Order.COMPILE_ORDER),
          null,
          ImmutableMap.<String, String>of(),
          ImmutableList.<PathFragment>of(),
          null);

  @Nullable private final CppConfiguration cppConfiguration;
  private final NestedSet<Artifact> crosstool;
  private final NestedSet<Artifact> crosstoolMiddleman;
  private final NestedSet<Artifact> compile;
  private final NestedSet<Artifact> strip;
  private final NestedSet<Artifact> objCopy;
  private final NestedSet<Artifact> link;
  private final Artifact interfaceSoBuilder;
  private final NestedSet<Artifact> dwp;
  private final NestedSet<Artifact> coverage;
  private final NestedSet<Artifact> libcLink;
  private final NestedSet<Artifact> staticRuntimeLinkInputs;
  @Nullable private final Artifact staticRuntimeLinkMiddleman;
  private final NestedSet<Artifact> dynamicRuntimeLinkInputs;
  @Nullable private final Artifact dynamicRuntimeLinkMiddleman;
  private final PathFragment dynamicRuntimeSolibDir;
  private final CppCompilationContext cppCompilationContext;
  private final boolean supportsParamFiles;
  private final boolean supportsHeaderParsing;
  private final ImmutableMap<String, String> buildVariables;
  private final ImmutableList<Artifact> builtinIncludeFiles;
  private final NestedSet<Pair<String, String>> coverageEnvironment;
  @Nullable private final Artifact linkDynamicLibraryTool;
  private final ImmutableMap<String, String> environment;
  private final ImmutableList<PathFragment> builtInIncludeDirectories;
  @Nullable private final PathFragment sysroot;

  public CcToolchainProvider(
      @Nullable CppConfiguration cppConfiguration,
      NestedSet<Artifact> crosstool,
      NestedSet<Artifact> crosstoolMiddleman,
      NestedSet<Artifact> compile,
      NestedSet<Artifact> strip,
      NestedSet<Artifact> objCopy,
      NestedSet<Artifact> link,
      Artifact interfaceSoBuilder,
      NestedSet<Artifact> dwp,
      NestedSet<Artifact> coverage,
      NestedSet<Artifact> libcLink,
      NestedSet<Artifact> staticRuntimeLinkInputs,
      @Nullable Artifact staticRuntimeLinkMiddleman,
      NestedSet<Artifact> dynamicRuntimeLinkInputs,
      @Nullable Artifact dynamicRuntimeLinkMiddleman,
      PathFragment dynamicRuntimeSolibDir,
      CppCompilationContext cppCompilationContext,
      boolean supportsParamFiles,
      boolean supportsHeaderParsing,
      Map<String, String> buildVariables,
      ImmutableList<Artifact> builtinIncludeFiles,
      NestedSet<Pair<String, String>> coverageEnvironment,
      Artifact linkDynamicLibraryTool,
      ImmutableMap<String, String> environment,
      ImmutableList<PathFragment> builtInIncludeDirectories,
      @Nullable PathFragment sysroot) {
    super(SKYLARK_CONSTRUCTOR, ImmutableMap.<String, Object>of());
    this.cppConfiguration = cppConfiguration;
    this.crosstool = Preconditions.checkNotNull(crosstool);
    this.crosstoolMiddleman = Preconditions.checkNotNull(crosstoolMiddleman);
    this.compile = Preconditions.checkNotNull(compile);
    this.strip = Preconditions.checkNotNull(strip);
    this.objCopy = Preconditions.checkNotNull(objCopy);
    this.link = Preconditions.checkNotNull(link);
    this.interfaceSoBuilder = interfaceSoBuilder;
    this.dwp = Preconditions.checkNotNull(dwp);
    this.coverage = Preconditions.checkNotNull(coverage);
    this.libcLink = Preconditions.checkNotNull(libcLink);
    this.staticRuntimeLinkInputs = Preconditions.checkNotNull(staticRuntimeLinkInputs);
    this.staticRuntimeLinkMiddleman = staticRuntimeLinkMiddleman;
    this.dynamicRuntimeLinkInputs = Preconditions.checkNotNull(dynamicRuntimeLinkInputs);
    this.dynamicRuntimeLinkMiddleman = dynamicRuntimeLinkMiddleman;
    this.dynamicRuntimeSolibDir = Preconditions.checkNotNull(dynamicRuntimeSolibDir);
    this.cppCompilationContext = Preconditions.checkNotNull(cppCompilationContext);
    this.supportsParamFiles = supportsParamFiles;
    this.supportsHeaderParsing = supportsHeaderParsing;
    this.buildVariables = ImmutableMap.copyOf(buildVariables);
    this.builtinIncludeFiles = builtinIncludeFiles;
    this.coverageEnvironment = coverageEnvironment;
    this.linkDynamicLibraryTool = linkDynamicLibraryTool;
    this.environment = environment;
    this.builtInIncludeDirectories = builtInIncludeDirectories;
    this.sysroot = sysroot;
  }

  @SkylarkCallable(
      name = "built_in_include_directories",
      doc = "Returns the list of built-in directories of the compiler.",
      structField = true
  )
  public ImmutableList<PathFragment> getBuiltInIncludeDirectories() {
    return builtInIncludeDirectories;
  }

  /**
   * Returns all the files in Crosstool. Is not a middleman.
   */
  public NestedSet<Artifact> getCrosstool() {
    return crosstool;
  }

  /**
   * Returns a middleman for all the files in Crosstool.
   */
  public NestedSet<Artifact> getCrosstoolMiddleman() {
    return crosstoolMiddleman;
  }

  /**
   * Returns the files necessary for compilation.
   */
  public NestedSet<Artifact> getCompile() {
    return compile;
  }

  /**
   * Returns the files necessary for a 'strip' invocation.
   */
  public NestedSet<Artifact> getStrip() {
    return strip;
  }

  /**
   * Returns the files necessary for an 'objcopy' invocation.
   */
  public NestedSet<Artifact> getObjcopy() {
    return objCopy;
  }

  /**
   * Returns the files necessary for linking, including the files needed for libc.
   */
  public NestedSet<Artifact> getLink() {
    return link;
  }

  public NestedSet<Artifact> getDwp() {
    return dwp;
  }

  /**
   * Returns the files necessary for capturing code coverage.
   */
  public NestedSet<Artifact> getCoverage() {
    return coverage;
  }

  public NestedSet<Artifact> getLibcLink() {
    return libcLink;
  }

  /**
   * Returns the static runtime libraries.
   */
  public NestedSet<Artifact> getStaticRuntimeLinkInputs() {
    return staticRuntimeLinkInputs;
  }

  /**
   * Returns an aggregating middleman that represents the static runtime libraries.
   */
  @Nullable public Artifact getStaticRuntimeLinkMiddleman() {
    return staticRuntimeLinkMiddleman;
  }

  /**
   * Returns the dynamic runtime libraries.
   */
  public NestedSet<Artifact> getDynamicRuntimeLinkInputs() {
    return dynamicRuntimeLinkInputs;
  }

  /**
   * Returns an aggregating middleman that represents the dynamic runtime libraries.
   */
  @Nullable public Artifact getDynamicRuntimeLinkMiddleman() {
    return dynamicRuntimeLinkMiddleman;
  }

  /**
   * Returns the name of the directory where the solib symlinks for the dynamic runtime libraries
   * live. The directory itself will be under the root of the host configuration in the 'bin'
   * directory.
   */
  public PathFragment getDynamicRuntimeSolibDir() {
    return dynamicRuntimeSolibDir;
  }

  /**
   * Returns the C++ compilation context for the toolchain.
   */
  public CppCompilationContext getCppCompilationContext() {
    return cppCompilationContext;
  }

  /**
   * Whether the toolchains supports parameter files.
   */
  public boolean supportsParamFiles() {
    return supportsParamFiles;
  }

  /**
   * Whether the toolchains supports header parsing.
   */
  public boolean supportsHeaderParsing() {
    return supportsHeaderParsing;
  }
  
  /**
   * Returns the configured features of the toolchain.
   */
  @Nullable
  public CcToolchainFeatures getFeatures() {
    return cppConfiguration == null ? null : cppConfiguration.getFeatures();
  }
  
  /**
   * Returns the compilation mode.
   */
  @Nullable
  public CompilationMode getCompilationMode() {
    return cppConfiguration == null ? null : cppConfiguration.getCompilationMode();
  }

  @Nullable
  public CppConfiguration getCppConfiguration() {
    return cppConfiguration;
  }
  
  /**
   * Returns build variables to be templated into the crosstool.
   */
  public ImmutableMap<String, String> getBuildVariables() {
    return buildVariables;
  }

  /**
   * Return the set of include files that may be included even if they are not mentioned in the
   * source file or any of the headers included by it.
   */
  public ImmutableList<Artifact> getBuiltinIncludeFiles() {
    return builtinIncludeFiles;
  }

  /**
   * Returns the environment variables that need to be added to tests that collect code coverage.
   */
  public NestedSet<Pair<String, String>> getCoverageEnvironment() {
    return coverageEnvironment;
  }

  public ImmutableMap<String, String> getEnvironment() {
    return environment;
  }

  /**
   * Returns the tool which should be used for linking dynamic libraries, or in case it's not
   * specified by the crosstool this will be @tools_repository/tools/cpp:link_dynamic_library
   */
  public Artifact getLinkDynamicLibraryTool() {
    return linkDynamicLibraryTool;
  }

  /**
   * Returns the tool that builds interface libraries from dynamic libraries.
   */
  public Artifact getInterfaceSoBuilder() {
    return interfaceSoBuilder;
  }

  @SkylarkCallable(
    name = "sysroot",
    structField = true,
    doc =
        "Returns the sysroot to be used. If the toolchain compiler does not support "
            + "different sysroots, or the sysroot is the same as the default sysroot, then "
            + "this method returns <code>None</code>."
  )
  public PathFragment getSysroot() {
    return sysroot;
  }

  @SkylarkCallable(
    name = "unfiltered_compiler_options_do_not_use",
    doc =
        "Returns the default list of options which cannot be filtered by BUILD "
            + "rules. These should be appended to the command line after filtering."
  )
  public ImmutableList<String> getUnfilteredCompilerOptions(Iterable<String> features) {
    return cppConfiguration.getUnfilteredCompilerOptions(features, sysroot);
  }

  @SkylarkCallable(
    name = "link_options_do_not_use",
    structField = true,
    doc =
        "Returns the set of command-line linker options, including any flags "
            + "inferred from the command-line options."
  )
  public ImmutableList<String> getLinkOptions() {
    return cppConfiguration.getLinkOptions(sysroot);
  }

  // Not all of CcToolchainProvider is exposed to Skylark, which makes implementing deep equality
  // impossible: if Java-only parts are considered, the behavior is surprising in Skylark, if they
  // are not, the behavior is surprising in Java. Thus, object identity it is.
  @Override
  public boolean equals(Object other) {
    return other == this;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }
}
