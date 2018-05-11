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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.rules.cpp.FdoSupport.FdoMode;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import java.util.Map;
import javax.annotation.Nullable;

/** Information about a C++ compiler used by the <code>cc_*</code> rules. */
@SkylarkModule(name = "CcToolchainInfo", doc = "Information about the C++ compiler being used.")
@Immutable
@AutoCodec
public final class CcToolchainProvider extends ToolchainInfo {
  public static final String SKYLARK_NAME = "CcToolchainInfo";

  /** An empty toolchain to be returned in the error case (instead of null). */
  public static final CcToolchainProvider EMPTY_TOOLCHAIN_IS_ERROR =
      new CcToolchainProvider(
          /* values= */ ImmutableMap.of(),
          /* cppConfiguration= */ null,
          /* toolchainInfo= */ null,
          /* crosstoolTopPathFragment= */ null,
          /* crosstool= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* crosstoolMiddleman= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* compile= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* strip= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* objCopy= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* as= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* ar= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* link= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* interfaceSoBuilder= */ null,
          /* dwp= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* coverage= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* libcLink= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* staticRuntimeLinkInputs= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* staticRuntimeLinkMiddleman= */ null,
          /* dynamicRuntimeLinkInputs= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          /* dynamicRuntimeLinkMiddleman= */ null,
          /* dynamicRuntimeSolibDir= */ PathFragment.EMPTY_FRAGMENT,
          CcCompilationContextInfo.EMPTY,
          /* supportsParamFiles= */ false,
          /* supportsHeaderParsing= */ false,
          Variables.EMPTY,
          /* builtinIncludeFiles= */ ImmutableList.<Artifact>of(),
          /* coverageEnvironment= */ NestedSetBuilder.emptySet(Order.COMPILE_ORDER),
          /* linkDynamicLibraryTool= */ null,
          /* builtInIncludeDirectories= */ ImmutableList.<PathFragment>of(),
          /* sysroot= */ null,
          FdoMode.OFF,
          /* useLLVMCoverageMapFormat= */ false,
          /* codeCoverageEnabled= */ false,
          /* isHostConfiguration= */ false);

  @Nullable private final CppConfiguration cppConfiguration;
  private final CppToolchainInfo toolchainInfo;
  private final PathFragment crosstoolTopPathFragment;
  private final NestedSet<Artifact> crosstool;
  private final NestedSet<Artifact> crosstoolMiddleman;
  private final NestedSet<Artifact> compile;
  private final NestedSet<Artifact> strip;
  private final NestedSet<Artifact> objCopy;
  private final NestedSet<Artifact> as;
  private final NestedSet<Artifact> ar;
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
  private final CcCompilationContextInfo ccCompilationContextInfo;
  private final boolean supportsParamFiles;
  private final boolean supportsHeaderParsing;
  private final Variables buildVariables;
  private final ImmutableList<Artifact> builtinIncludeFiles;
  private final NestedSet<Pair<String, String>> coverageEnvironment;
  @Nullable private final Artifact linkDynamicLibraryTool;
  private final ImmutableList<PathFragment> builtInIncludeDirectories;
  @Nullable private final PathFragment sysroot;
  private final FdoMode fdoMode;
  private final boolean useLLVMCoverageMapFormat;
  private final boolean codeCoverageEnabled;
  private final boolean isHostConfiguration;
  private final boolean forcePic;
  private final boolean shouldStripBinaries;

  public CcToolchainProvider(
      ImmutableMap<String, Object> values,
      @Nullable CppConfiguration cppConfiguration,
      CppToolchainInfo toolchainInfo,
      PathFragment crosstoolTopPathFragment,
      NestedSet<Artifact> crosstool,
      NestedSet<Artifact> crosstoolMiddleman,
      NestedSet<Artifact> compile,
      NestedSet<Artifact> strip,
      NestedSet<Artifact> objCopy,
      NestedSet<Artifact> as,
      NestedSet<Artifact> ar,
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
      CcCompilationContextInfo ccCompilationContextInfo,
      boolean supportsParamFiles,
      boolean supportsHeaderParsing,
      Variables buildVariables,
      ImmutableList<Artifact> builtinIncludeFiles,
      NestedSet<Pair<String, String>> coverageEnvironment,
      Artifact linkDynamicLibraryTool,
      ImmutableList<PathFragment> builtInIncludeDirectories,
      @Nullable PathFragment sysroot,
      FdoMode fdoMode,
      boolean useLLVMCoverageMapFormat,
      boolean codeCoverageEnabled,
      boolean isHostConfiguration) {
    super(values, Location.BUILTIN);
    this.cppConfiguration = cppConfiguration;
    this.toolchainInfo = toolchainInfo;
    this.crosstoolTopPathFragment = crosstoolTopPathFragment;
    this.crosstool = Preconditions.checkNotNull(crosstool);
    this.crosstoolMiddleman = Preconditions.checkNotNull(crosstoolMiddleman);
    this.compile = Preconditions.checkNotNull(compile);
    this.strip = Preconditions.checkNotNull(strip);
    this.objCopy = Preconditions.checkNotNull(objCopy);
    this.as = Preconditions.checkNotNull(as);
    this.ar = Preconditions.checkNotNull(ar);
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
    this.ccCompilationContextInfo = Preconditions.checkNotNull(ccCompilationContextInfo);
    this.supportsParamFiles = supportsParamFiles;
    this.supportsHeaderParsing = supportsHeaderParsing;
    this.buildVariables = buildVariables;
    this.builtinIncludeFiles = builtinIncludeFiles;
    this.coverageEnvironment = coverageEnvironment;
    this.linkDynamicLibraryTool = linkDynamicLibraryTool;
    this.builtInIncludeDirectories = builtInIncludeDirectories;
    this.sysroot = sysroot;
    this.fdoMode = fdoMode;
    this.useLLVMCoverageMapFormat = useLLVMCoverageMapFormat;
    this.codeCoverageEnabled = codeCoverageEnabled;
    this.isHostConfiguration = isHostConfiguration;
    if (cppConfiguration != null) {
      this.forcePic = cppConfiguration.forcePic();
      this.shouldStripBinaries = cppConfiguration.shouldStripBinaries();
    } else {
      this.forcePic = false;
      this.shouldStripBinaries = false;
    }
  }

  /** Returns c++ Make variables. */
  public static Map<String, String> getCppBuildVariables(
      Function<Tool, PathFragment> getToolPathFragment,
      String targetLibc,
      String compiler,
      String targetCpu,
      PathFragment crosstoolTopPathFragment,
      String abiGlibcVersion,
      String abi,
      Map<String, String> additionalMakeVariables) {
    ImmutableMap.Builder<String, String> result = ImmutableMap.builder();

    // hardcoded CC->gcc setting for unit tests
    result.put("CC", getToolPathFragment.apply(Tool.GCC).getPathString());

    // Make variables provided by crosstool/gcc compiler suite.
    result.put("AR", getToolPathFragment.apply(Tool.AR).getPathString());
    result.put("NM", getToolPathFragment.apply(Tool.NM).getPathString());
    result.put("LD", getToolPathFragment.apply(Tool.LD).getPathString());
    PathFragment objcopyTool = getToolPathFragment.apply(Tool.OBJCOPY);
    if (objcopyTool != null) {
      // objcopy is optional in Crosstool
      result.put("OBJCOPY", objcopyTool.getPathString());
    }
    result.put("STRIP", getToolPathFragment.apply(Tool.STRIP).getPathString());

    PathFragment gcovtool = getToolPathFragment.apply(Tool.GCOVTOOL);
    if (gcovtool != null) {
      // gcov-tool is optional in Crosstool
      result.put("GCOVTOOL", gcovtool.getPathString());
    }

    if (targetLibc.startsWith("glibc-")) {
      result.put("GLIBC_VERSION", targetLibc.substring("glibc-".length()));
    } else {
      result.put("GLIBC_VERSION", targetLibc);
    }

    result.put("C_COMPILER", compiler);
    result.put("TARGET_CPU", targetCpu);

    // Deprecated variables

    // TODO(bazel-team): delete all of these.
    result.put("CROSSTOOLTOP", crosstoolTopPathFragment.getPathString());

    // TODO(kmensah): Remove when skylark dependencies can be updated to rely on
    // CcToolchainProvider.
    result.putAll(additionalMakeVariables);

    result.put("ABI_GLIBC_VERSION", abiGlibcVersion);
    result.put("ABI", abi);

    return result.build();
  }

  /**
   * Returns true if Fission is specified and supported by the CROSSTOOL for the build implied by
   * the given configuration and toolchain.
   */
  public boolean useFission() {
    return Preconditions.checkNotNull(cppConfiguration).fissionIsActiveForCurrentCompilationMode()
        && supportsFission();
  }

  /**
   * Returns true if Fission and PER_OBJECT_DEBUG_INFO are specified and supported by the CROSSTOOL
   * for the build implied by the given configuration, toolchain and feature configuration.
   */
  public boolean shouldCreatePerObjectDebugInfo(FeatureConfiguration featureConfiguration) {
    return useFission() && featureConfiguration.isEnabled(CppRuleClasses.PER_OBJECT_DEBUG_INFO);
  }

  @Override
  public void addGlobalMakeVariables(ImmutableMap.Builder<String, String> globalMakeEnvBuilder) {
    globalMakeEnvBuilder.putAll(
        getCppBuildVariables(
            this::getToolPathFragment,
            getTargetLibc(),
            getCompiler(),
            getTargetCpu(),
            crosstoolTopPathFragment,
            getAbiGlibcVersion(),
            getAbi(),
            getAdditionalMakeVariables()));
  }

  @SkylarkCallable(
      name = "built_in_include_directories",
      doc = "Returns the list of built-in directories of the compiler.",
      structField = true
  )
  public ImmutableList<PathFragment> getBuiltInIncludeDirectories() {
    return builtInIncludeDirectories;
  }

  /** Returns the identifier of the toolchain as specified in the {@code CToolchain} proto. */
  public String getToolchainIdentifier() {
    return toolchainInfo.getToolchainIdentifier();
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
   * Returns the files necessary for an 'as' invocation.  May be empty if the CROSSTOOL
   * file does not define as_files.
   */
  public NestedSet<Artifact> getAs() {
    return as;
  }

  /**
   * Returns the files necessary for an 'ar' invocation.  May be empty if the CROSSTOOL
   * file does not define ar_files.
   */
  public NestedSet<Artifact> getAr() {
    return ar;
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
   * Returns true if the featureConfiguration includes statically linking the cpp runtimes.
   *
   * @param featureConfiguration the relevant FeatureConfiguration.
   */
  public boolean shouldStaticallyLinkCppRuntimes(FeatureConfiguration featureConfiguration) {
    return featureConfiguration.isEnabled(CppRuleClasses.STATIC_LINK_CPP_RUNTIMES);
  }

  /** Returns the static runtime libraries. */
  public NestedSet<Artifact> getStaticRuntimeLinkInputs(FeatureConfiguration featureConfiguration) {
    if (shouldStaticallyLinkCppRuntimes(featureConfiguration)) {
      return staticRuntimeLinkInputs;
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  /** Returns an aggregating middleman that represents the static runtime libraries. */
  @Nullable
  public Artifact getStaticRuntimeLinkMiddleman(FeatureConfiguration featureConfiguration) {
    if (shouldStaticallyLinkCppRuntimes(featureConfiguration)) {
      return staticRuntimeLinkMiddleman;
    } else {
      return null;
    }
  }

  /** Returns the dynamic runtime libraries. */
  public NestedSet<Artifact> getDynamicRuntimeLinkInputs(
      FeatureConfiguration featureConfiguration) {
    if (shouldStaticallyLinkCppRuntimes(featureConfiguration)) {
      return dynamicRuntimeLinkInputs;
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  /** Returns an aggregating middleman that represents the dynamic runtime libraries. */
  @Nullable
  public Artifact getDynamicRuntimeLinkMiddleman(FeatureConfiguration featureConfiguration) {
    if (shouldStaticallyLinkCppRuntimes(featureConfiguration)) {
      return dynamicRuntimeLinkMiddleman;
    } else {
      return null;
    }
  }

  /**
   * Returns the name of the directory where the solib symlinks for the dynamic runtime libraries
   * live. The directory itself will be under the root of the host configuration in the 'bin'
   * directory.
   */
  public PathFragment getDynamicRuntimeSolibDir() {
    return dynamicRuntimeSolibDir;
  }

  /** Returns the {@code CcCompilationContextInfo} for the toolchain. */
  public CcCompilationContextInfo getCcCompilationContextInfo() {
    return ccCompilationContextInfo;
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
    return toolchainInfo.getFeatures();
  }

  public Label getCcToolchainLabel() {
    return toolchainInfo.getCcToolchainLabel();
  }

  /**
   * Returns whether shared libraries must be compiled with position independent code on this
   * platform.
   */
  public boolean toolchainNeedsPic() {
    return toolchainInfo.toolchainNeedsPic();
  }

  /**
   * Returns the run time sysroot, which is where the dynamic linker and system libraries are found
   * at runtime. This is usually an absolute path. If the toolchain compiler does not support
   * sysroots, then this method returns <code>null</code>.
   */
  public PathFragment getRuntimeSysroot() {
    return toolchainInfo.getRuntimeSysroot();
  }

  /**
   * Return the name of the directory (relative to the bin directory) that holds mangled links to
   * shared libraries. This name is always set to the '{@code _solib_<cpu_archictecture_name>}.
   */
  public String getSolibDirectory() {
    return toolchainInfo.getSolibDirectory();
  }

  /**
   * Returns the compilation mode.
   */
  @Nullable
  public CompilationMode getCompilationMode() {
    return cppConfiguration == null ? null : cppConfiguration.getCompilationMode();
  }

  /**
   * Returns whether the toolchain supports the gold linker.
   */
  public boolean supportsGoldLinker() {
    return toolchainInfo.supportsGoldLinker();
  }

  /**
   * Returns whether the toolchain supports dynamic linking.
   */
  public boolean supportsDynamicLinker() {
    return toolchainInfo.supportsDynamicLinker();
  }

  /**
   * Returns whether the toolchain supports linking C/C++ runtime libraries
   * supplied inside the toolchain distribution.
   */
  public boolean supportsEmbeddedRuntimes() {
    return toolchainInfo.supportsEmbeddedRuntimes();
  }

  /**
   * Returns whether the toolchain supports EXEC_ORIGIN libraries resolution.
   */
  public boolean supportsExecOrigin() {
    // We're rolling out support for this in the same release that also supports embedded runtimes.
    return toolchainInfo.supportsEmbeddedRuntimes();
  }

  /** Returns whether the toolchain supports the --start-lib/--end-lib options. */
  public boolean supportsStartEndLib() {
    return toolchainInfo.supportsStartEndLib();
  }

  /**
   * Returns whether this toolchain supports interface shared objects.
   *
   * <p>Should be true if this toolchain generates ELF objects.
   */
  public boolean supportsInterfaceSharedObjects() {
    return toolchainInfo.supportsInterfaceSharedObjects();
  }

  @Nullable
  public CppConfiguration getCppConfiguration() {
    return cppConfiguration;
  }

  /** Returns build variables to be templated into the crosstool. */
  public Variables getBuildVariables() {
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

  /**
   * Returns the path fragment that is either absolute or relative to the execution root that can be
   * used to execute the given tool.
   */
  public PathFragment getToolPathFragment(CppConfiguration.Tool tool) {
    return toolchainInfo.getToolPathFragment(tool);
  }

  /**
   * Returns the abi we're using, which is a gcc version. E.g.: "gcc-3.4". Note that in practice we
   * might be using gcc-3.4 as ABI even when compiling with gcc-4.1.0, because ABIs are backwards
   * compatible.
   */
  // TODO(bazel-team): The javadoc should clarify how this is used in Blaze.
  public String getAbi() {
    return toolchainInfo.getAbi();
  }

  /**
   * Returns the glibc version used by the abi we're using. This is a glibc version number (e.g.,
   * "2.2.2"). Note that in practice we might be using glibc 2.2.2 as ABI even when compiling with
   * gcc-4.2.2, gcc-4.3.1, or gcc-4.4.0 (which use glibc 2.3.6), because ABIs are backwards
   * compatible.
   */
  // TODO(bazel-team): The javadoc should clarify how this is used in Blaze.
  public String getAbiGlibcVersion() {
    return toolchainInfo.getAbiGlibcVersion();
  }

  /**
   * Returns a label that references the library files needed to statically
   * link the C++ runtime (i.e. libgcc.a, libgcc_eh.a, libstdc++.a) for the
   * target architecture.
   */
  public Label getStaticRuntimeLibsLabel() {
    return toolchainInfo.getStaticRuntimeLibsLabel();
  }

  /**
   * Returns a label that references the library files needed to dynamically
   * link the C++ runtime (i.e. libgcc_s.so, libstdc++.so) for the target
   * architecture.
   */
  public Label getDynamicRuntimeLibsLabel() {
    return toolchainInfo.getDynamicRuntimeLibsLabel();
  }

  /** Returns the compiler version string (e.g. "gcc-4.1.1"). */
  @SkylarkCallable(name = "compiler", structField = true, doc = "C++ compiler.",
      allowReturnNones = true)
  public String getCompiler() {
    return toolchainInfo == null ? null : toolchainInfo.getCompiler();
  }

  /** Returns the libc version string (e.g. "glibc-2.2.2"). */
  @SkylarkCallable(name = "libc", structField = true, doc = "libc version string.",
      allowReturnNones = true)
  public String getTargetLibc() {
    return toolchainInfo == null ? null : toolchainInfo.getTargetLibc();
  }

  /** Returns the target architecture using blaze-specific constants (e.g. "piii"). */
  @SkylarkCallable(name = "cpu", structField = true, doc = "Target CPU of the C++ toolchain.",
      allowReturnNones = true)
  public String getTargetCpu() {
    return toolchainInfo == null ? null : toolchainInfo.getTargetCpu();
  }

  /**
   * Returns a map of additional make variables for use by {@link BuildConfiguration}. These are to
   * used to allow some build rules to avoid the limits on stack frame sizes and variable-length
   * arrays.
   *
   * <p>The returned map must contain an entry for {@code STACK_FRAME_UNLIMITED}, though the entry
   * may be an empty string.
   */
  public ImmutableMap<String, String> getAdditionalMakeVariables() {
    return toolchainInfo.getAdditionalMakeVariables();
  }

  /**
   * Returns whether the toolchain supports "Fission" C++ builds, i.e. builds where compilation
   * partitions object code and debug symbols into separate output files.
   */
  public boolean supportsFission() {
    return toolchainInfo.supportsFission();
  }

  @SkylarkCallable(
      name = "unfiltered_compiler_options",
      doc =
          "Returns the default list of options which cannot be filtered by BUILD "
              + "rules. These should be appended to the command line after filtering.")
  // TODO(b/24373706): Remove this method once new C++ toolchain API is available
  public ImmutableList<String> getUnfilteredCompilerOptionsWithSysroot(
      Iterable<String> featuresNotUsedAnymore) {
    return toolchainInfo.getUnfilteredCompilerOptions(sysroot);
  }

  public ImmutableList<String> getUnfilteredCompilerOptions() {
    return toolchainInfo.getUnfilteredCompilerOptions(/* sysroot= */ null);
  }

  /**
   * Unused, for compatibility with things internal to Google.
   *
   * <p>Deprecated: Use platforms.
   */
  @Deprecated
  public String getTargetOS() {
    return toolchainInfo.getTargetOS();
  }

  @SkylarkCallable(
    name = "link_options_do_not_use",
    structField = true,
    doc =
        "Returns the set of command-line linker options, including any flags "
            + "inferred from the command-line options."
  )
  public ImmutableList<String> getLinkOptionsWithSysroot() {
    return cppConfiguration == null
        ? ImmutableList.of()
        : cppConfiguration.getLinkOptionsDoNotUse(sysroot);
  }

  public ImmutableList<String> getLinkOptions() {
    return cppConfiguration.getLinkOptionsDoNotUse(/* sysroot= */ null);
  }

  /**
   * Returns test-only link options such that certain test-specific features can be configured
   * separately (e.g. lazy binding).
   */
  public ImmutableList<String> getTestOnlyLinkOptions() {
    return toolchainInfo.getTestOnlyLinkOptions();
  }

  /** Returns the system name which is required by the toolchain to run. */
  public String getHostSystemName() {
    return toolchainInfo.getHostSystemName();
  }

  /**
   * Returns the list of options to be used with 'objcopy' when converting binary files to object
   * files, or {@code null} if this operation is not supported.
   */
  public ImmutableList<String> getObjCopyOptionsForEmbedding() {
    return toolchainInfo.getObjCopyOptionsForEmbedding();
  }

  /**
   * Returns the list of options to be used with 'ld' when converting binary files to object files,
   * or {@code null} if this operation is not supported.
   */
  public ImmutableList<String> getLdOptionsForEmbedding() {
    return toolchainInfo.getLdOptionsForEmbedding();
  }

  /**
   * Returns link options for the specified flag list, combined with universal options for all
   * shared libraries (regardless of link staticness).
   */
  ImmutableList<String> getSharedLibraryLinkOptions(FlagList flags) {
    return toolchainInfo.getSharedLibraryLinkOptions(flags);
  }

  /** Returns compiler flags arising from the {@link CToolchain}. */
  ImmutableList<String> getToolchainCompilerFlags() {
    return toolchainInfo.getCompilerFlags();
  }

  /** Returns additional compiler flags for C++ arising from the {@link CToolchain} */
  ImmutableList<String> getToolchainCxxFlags() {
    return toolchainInfo.getCxxFlags();
  }

  /**
   * Returns compiler flags arising from the {@link CToolchain} for C compilation by compilation
   * mode.
   */
  ImmutableListMultimap<CompilationMode, String> getCFlagsByCompilationMode() {
    return toolchainInfo.getCFlagsByCompilationMode();
  }

  /**
   * Returns compiler flags arising from the {@link CToolchain} for C++ compilation by compilation
   * mode.
   */
  ImmutableListMultimap<CompilationMode, String> getCxxFlagsByCompilationMode() {
    return toolchainInfo.getCxxFlagsByCompilationMode();
  }

  /** Returns compiler flags arising from the {@link CToolchain} for C compilation by lipo mode. */
  ImmutableListMultimap<LipoMode, String> getLipoCFlags() {
    return toolchainInfo.getLipoCFlags();
  }

  /**
   * Returns compiler flags arising from the {@link CToolchain} for C++ compilation by lipo mode.
   */
  ImmutableListMultimap<LipoMode, String> getLipoCxxFlags() {
    return toolchainInfo.getLipoCxxFlags();
  }

  /** Returns linker flags for fully statically linked outputs. */
  FlagList getLegacyFullyStaticLinkFlags(CompilationMode compilationMode, LipoMode lipoMode) {
    return new FlagList(
        configureAllLegacyLinkOptions(compilationMode, lipoMode, LinkingMode.LEGACY_FULLY_STATIC),
        ImmutableList.<String>of());
  }

  /** Returns linker flags for mostly static linked outputs. */
  FlagList getLegacyMostlyStaticLinkFlags(CompilationMode compilationMode, LipoMode lipoMode) {
    return new FlagList(
        configureAllLegacyLinkOptions(compilationMode, lipoMode, LinkingMode.STATIC),
        ImmutableList.<String>of());
  }

  /** Returns linker flags for mostly static shared linked outputs. */
  FlagList getLegacyMostlyStaticSharedLinkFlags(
      CompilationMode compilationMode, LipoMode lipoMode) {
    return new FlagList(
        configureAllLegacyLinkOptions(
            compilationMode, lipoMode, LinkingMode.LEGACY_MOSTLY_STATIC_LIBRARIES),
        ImmutableList.<String>of());
  }

  /** Returns linker flags for artifacts that are not fully or mostly statically linked. */
  FlagList getLegacyDynamicLinkFlags(CompilationMode compilationMode, LipoMode lipoMode) {
    return new FlagList(
        configureAllLegacyLinkOptions(compilationMode, lipoMode, LinkingMode.DYNAMIC),
        ImmutableList.of());
  }

  /**
   * Return all flags coming from naked {@code linker_flag} fields in the crosstool. {@code
   * linker_flag}s coming from linking_mode_flags and compilation_mode_flags are not included. If
   * you need all possible linker flags, use {@link #configureAllLegacyLinkOptions(CompilationMode,
   * LipoMode, LinkingMode)}.
   */
  public ImmutableList<String> getLegacyLinkOptions() {
    return toolchainInfo.getLegacyLinkOptions();
  }

  /**
   * Return all flags coming from {@code compiler_flag} crosstool fields excluding flags coming from
   * --copt options and copts attribute.
   */
  public ImmutableList<String> getLegacyCompileOptions() {
    ImmutableList.Builder<String> coptsBuilder =
        ImmutableList.<String>builder()
            .addAll(getToolchainCompilerFlags())
            .addAll(getCFlagsByCompilationMode().get(cppConfiguration.getCompilationMode()))
            .addAll(getLipoCFlags().get(cppConfiguration.getLipoMode()));

    if (cppConfiguration.isOmitfp()) {
      coptsBuilder.add("-fomit-frame-pointer");
      coptsBuilder.add("-fasynchronous-unwind-tables");
      coptsBuilder.add("-DNO_FRAME_POINTER");
    }

    return coptsBuilder.build();
  }

  public ImmutableList<String> getLegacyCompileOptionsWithCopts() {
    return ImmutableList.<String>builder()
        .addAll(getLegacyCompileOptions())
        .addAll(cppConfiguration.getCopts())
        .build();
  }

  /** Return all possible {@code linker_flag} flags from the crosstool. */
  ImmutableList<String> configureAllLegacyLinkOptions(
      CompilationMode compilationMode, LipoMode lipoMode, LinkingMode linkingMode) {
    return toolchainInfo.configureAllLegacyLinkOptions(compilationMode, lipoMode, linkingMode);
  }

  /** Returns the GNU System Name */
  @SkylarkCallable(
    name = "target_gnu_system_name",
    structField = true,
    doc = "The GNU System Name.",
    allowReturnNones = true
  )
  public String getTargetGnuSystemName() {
    return toolchainInfo == null ? null : toolchainInfo.getTargetGnuSystemName();
  }

  /** Returns the architecture component of the GNU System Name */
  public String getGnuSystemArch() {
    return toolchainInfo.getGnuSystemArch();
  }

  public final boolean isLLVMCompiler() {
    return toolchainInfo.isLLVMCompiler();
  }

  public FdoMode getFdoMode() {
    return fdoMode;
  }

  /**
   * WARNING: This method is only added to allow incremental migration of existing users. Please do
   * not use in new code. Will be removed soon as part of the new Skylark API to the C++ toolchain.
   */
  @SkylarkCallable(
      name = "compiler_options",
      doc =
          "Returns the default options to use for compiling C, C++, and assembler. "
              + "This is just the options that should be used for all three languages. "
              + "There may be additional C-specific or C++-specific options that should be used, "
              + "in addition to the ones returned by this method"
  )
  public ImmutableList<String> getCompilerOptions() {
    return getLegacyCompileOptionsWithCopts();
  }

  /**
   * WARNING: This method is only added to allow incremental migration of existing users. Please do
   * not use in new code. Will be removed soon as part of the new Skylark API to the C++ toolchain.
   *
   * <p>Returns the list of additional C-specific options to use for compiling C. These should be go
   * on the command line after the common options returned by {@link
   * CcToolchainProvider#getLegacyCompileOptionsWithCopts()}.
   */
  @SkylarkCallable(
      name = "c_options",
      doc =
          "Returns the list of additional C-specific options to use for compiling C. "
              + "These should be go on the command line after the common options returned by "
              + "<code>compiler_options</code>")
  public ImmutableList<String> getCOptions() {
    return cppConfiguration.getCOptions();
  }

  /**
   * WARNING: This method is only added to allow incremental migration of existing users. Please do
   * not use in new code. Will be removed soon as part of the new Skylark API to the C++ toolchain.
   *
   * <p>Returns the list of additional C++-specific options to use for compiling C++. These should
   * be on the command line after the common options returned by {@link #getCompilerOptions}.
   */
  @SkylarkCallable(
      name = "cxx_options",
      doc =
          "Returns the list of additional C++-specific options to use for compiling C++. "
              + "These should be go on the command line after the common options returned by "
              + "<code>compiler_options</code>")
  @Deprecated
  public ImmutableList<String> getCxxOptionsWithCopts() {
    return ImmutableList.<String>builder()
        .addAll(getLegacyCxxOptions())
        .addAll(cppConfiguration.getCxxopts())
        .build();
  }

  public ImmutableList<String> getLegacyCxxOptions() {
    return ImmutableList.<String>builder()
        .addAll(getToolchainCxxFlags())
        .addAll(getCxxFlagsByCompilationMode().get(cppConfiguration.getCompilationMode()))
        .addAll(getLipoCxxFlags().get(cppConfiguration.getLipoMode()))
        .build();
  }

  /**
   * WARNING: This method is only added to allow incremental migration of existing users. Please do
   * not use in new code. Will be removed soon as part of the new Skylark API to the C++ toolchain.
   *
   * <p>Returns the immutable list of linker options for fully statically linked outputs. Does not
   * include command-line options passed via --linkopt or --linkopts.
   *
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   */
  @SkylarkCallable(
      name = "fully_static_link_options",
      doc =
          "Returns the immutable list of linker options for fully statically linked "
              + "outputs. Does not include command-line options passed via --linkopt or "
              + "--linkopts.")
  @Deprecated
  public ImmutableList<String> getFullyStaticLinkOptions(Boolean sharedLib) throws EvalException {
    if (!sharedLib) {
      throw new EvalException(
          Location.BUILTIN, "fully_static_link_options is deprecated, new uses are not allowed.");
    }
    return CppHelper.getFullyStaticLinkOptions(cppConfiguration, this, sharedLib);
  }

  /**
   * WARNING: This method is only added to allow incremental migration of existing users. Please do
   * not use in new code. Will be removed soon as part of the new Skylark API to the C++ toolchain.
   *
   * Returns the immutable list of linker options for mostly statically linked outputs. Does not
   * include command-line options passed via --linkopt or --linkopts.
   *
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   */
  @SkylarkCallable(
      name = "mostly_static_link_options",
      doc =
          "Returns the immutable list of linker options for mostly statically linked "
              + "outputs. Does not include command-line options passed via --linkopt or "
              + "--linkopts.")
  @Deprecated
  public ImmutableList<String> getMostlyStaticLinkOptions(Boolean sharedLib) {
    return CppHelper.getMostlyStaticLinkOptions(
        cppConfiguration, this, sharedLib, /* shouldStaticallyLinkCppRuntimes= */ true);
  }

  /**
   * WARNING: This method is only added to allow incremental migration of existing users. Please do
   * not use in new code. Will be removed soon as part of the new Skylark API to the C++ toolchain.
   *
   * Returns the immutable list of linker options for artifacts that are not fully or mostly
   * statically linked. Does not include command-line options passed via --linkopt or --linkopts.
   *
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   */
  @SkylarkCallable(
      name = "dynamic_link_options",
      doc =
          "Returns the immutable list of linker options for artifacts that are not "
              + "fully or mostly statically linked. Does not include command-line options "
              + "passed via --linkopt or --linkopts."
  )
  @Deprecated
  public ImmutableList<String> getDynamicLinkOptions(Boolean sharedLib) {
    return CppHelper.getDynamicLinkOptions(cppConfiguration, this, sharedLib);
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

  public boolean useLLVMCoverageMapFormat() {
    return useLLVMCoverageMapFormat;
  }

  public boolean isCodeCoverageEnabled() {
    return codeCoverageEnabled;
  }

  public boolean isHostConfiguration() {
    return isHostConfiguration;
  }

  public boolean getForcePic() {
    return forcePic;
  }

  public boolean getShouldStripBinaries() {
    return shouldStripBinaries;
  }
}

