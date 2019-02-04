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
import com.google.devtools.build.lib.analysis.LicensesProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.rules.cpp.LibraryToLinkWrapper.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcToolchainProviderApi;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import java.util.Map;
import javax.annotation.Nullable;

/** Information about a C++ compiler used by the <code>cc_*</code> rules. */
@Immutable
@AutoCodec
public final class CcToolchainProvider extends ToolchainInfo
    implements CcToolchainProviderApi<FeatureConfiguration>, HasCcToolchainLabel {

  /** An empty toolchain to be returned in the error case (instead of null). */
  public static final CcToolchainProvider EMPTY_TOOLCHAIN_IS_ERROR =
      new CcToolchainProvider(
          /* values= */ ImmutableMap.of(),
          /* cppConfiguration= */ null,
          /* toolchainInfo= */ null,
          /* crosstoolTopPathFragment= */ null,
          /* allFiles= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* allFilesMiddleman= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* compilerFiles= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* compilerFilesWithoutIncludes= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* stripFiles= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* objcopyFiles= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* asFiles= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* arFiles= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* linkerFiles= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* interfaceSoBuilder= */ null,
          /* dwpFiles= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* coverageFiles= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* libcLink= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* staticRuntimeLinkInputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* staticRuntimeLinkMiddleman= */ null,
          /* dynamicRuntimeLinkInputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* dynamicRuntimeLinkMiddleman= */ null,
          /* dynamicRuntimeSolibDir= */ PathFragment.EMPTY_FRAGMENT,
          CcCompilationContext.EMPTY,
          /* supportsParamFiles= */ false,
          /* supportsHeaderParsing= */ false,
          CcToolchainVariables.EMPTY,
          /* builtinIncludeFiles= */ ImmutableList.of(),
          /* coverageEnvironment= */ NestedSetBuilder.emptySet(Order.COMPILE_ORDER),
          /* linkDynamicLibraryTool= */ null,
          /* builtInIncludeDirectories= */ ImmutableList.of(),
          /* sysroot= */ null,
          /* fdoContext= */ null,
          /* useLLVMCoverageMapFormat= */ false,
          /* codeCoverageEnabled= */ false,
          /* isHostConfiguration= */ false,
          /* licensesProvider= */ null);

  @Nullable private final CppConfiguration cppConfiguration;
  private final CppToolchainInfo toolchainInfo;
  private final PathFragment crosstoolTopPathFragment;
  private final NestedSet<Artifact> allFiles;
  private final NestedSet<Artifact> allFilesMiddleman;
  private final NestedSet<Artifact> compilerFiles;
  private final NestedSet<Artifact> compilerFilesWithoutIncludes;
  private final NestedSet<Artifact> stripFiles;
  private final NestedSet<Artifact> objcopyFiles;
  private final NestedSet<Artifact> asFiles;
  private final NestedSet<Artifact> arFiles;
  private final NestedSet<Artifact> linkerFiles;
  private final Artifact interfaceSoBuilder;
  private final NestedSet<Artifact> dwpFiles;
  private final NestedSet<Artifact> coverageFiles;
  private final NestedSet<Artifact> libcLink;
  @Nullable private final NestedSet<Artifact> staticRuntimeLinkInputs;
  @Nullable private final Artifact staticRuntimeLinkMiddleman;
  @Nullable private final NestedSet<Artifact> dynamicRuntimeLinkInputs;
  @Nullable private final Artifact dynamicRuntimeLinkMiddleman;
  private final PathFragment dynamicRuntimeSolibDir;
  private final CcInfo ccInfo;
  private final boolean supportsParamFiles;
  private final boolean supportsHeaderParsing;
  private final CcToolchainVariables buildVariables;
  private final ImmutableList<Artifact> builtinIncludeFiles;
  private final NestedSet<Pair<String, String>> coverageEnvironment;
  @Nullable private final Artifact linkDynamicLibraryTool;
  private final ImmutableList<PathFragment> builtInIncludeDirectories;
  @Nullable private final PathFragment sysroot;
  private final boolean useLLVMCoverageMapFormat;
  private final boolean codeCoverageEnabled;
  private final boolean isHostConfiguration;
  private final boolean forcePic;
  private final boolean shouldStripBinaries;
  /**
   * WARNING: We don't like {@link FdoContext}. Its {@link FdoContext#fdoProfilePath} is pure path
   * and that is horrible as it breaks many Bazel assumptions! Don't do bad stuff with it, don't
   * take inspiration from it.
   */
  private final FdoContext fdoContext;

  private final LicensesProvider licensesProvider;

  public CcToolchainProvider(
      ImmutableMap<String, Object> values,
      @Nullable CppConfiguration cppConfiguration,
      CppToolchainInfo toolchainInfo,
      PathFragment crosstoolTopPathFragment,
      NestedSet<Artifact> allFiles,
      NestedSet<Artifact> allFilesMiddleman,
      NestedSet<Artifact> compilerFiles,
      NestedSet<Artifact> compilerFilesWithoutIncludes,
      NestedSet<Artifact> stripFiles,
      NestedSet<Artifact> objcopyFiles,
      NestedSet<Artifact> asFiles,
      NestedSet<Artifact> arFiles,
      NestedSet<Artifact> linkerFiles,
      Artifact interfaceSoBuilder,
      NestedSet<Artifact> dwpFiles,
      NestedSet<Artifact> coverageFiles,
      NestedSet<Artifact> libcLink,
      NestedSet<Artifact> staticRuntimeLinkInputs,
      @Nullable Artifact staticRuntimeLinkMiddleman,
      NestedSet<Artifact> dynamicRuntimeLinkInputs,
      @Nullable Artifact dynamicRuntimeLinkMiddleman,
      PathFragment dynamicRuntimeSolibDir,
      CcCompilationContext ccCompilationContext,
      boolean supportsParamFiles,
      boolean supportsHeaderParsing,
      CcToolchainVariables buildVariables,
      ImmutableList<Artifact> builtinIncludeFiles,
      NestedSet<Pair<String, String>> coverageEnvironment,
      Artifact linkDynamicLibraryTool,
      ImmutableList<PathFragment> builtInIncludeDirectories,
      @Nullable PathFragment sysroot,
      FdoContext fdoContext,
      boolean useLLVMCoverageMapFormat,
      boolean codeCoverageEnabled,
      boolean isHostConfiguration,
      LicensesProvider licensesProvider) {
    super(values, Location.BUILTIN);
    this.cppConfiguration = cppConfiguration;
    this.toolchainInfo = toolchainInfo;
    this.crosstoolTopPathFragment = crosstoolTopPathFragment;
    this.allFiles = Preconditions.checkNotNull(allFiles);
    this.allFilesMiddleman = Preconditions.checkNotNull(allFilesMiddleman);
    this.compilerFiles = Preconditions.checkNotNull(compilerFiles);
    this.compilerFilesWithoutIncludes = Preconditions.checkNotNull(compilerFilesWithoutIncludes);
    this.stripFiles = Preconditions.checkNotNull(stripFiles);
    this.objcopyFiles = Preconditions.checkNotNull(objcopyFiles);
    this.asFiles = Preconditions.checkNotNull(asFiles);
    this.arFiles = Preconditions.checkNotNull(arFiles);
    this.linkerFiles = Preconditions.checkNotNull(linkerFiles);
    this.interfaceSoBuilder = interfaceSoBuilder;
    this.dwpFiles = Preconditions.checkNotNull(dwpFiles);
    this.coverageFiles = Preconditions.checkNotNull(coverageFiles);
    this.libcLink = Preconditions.checkNotNull(libcLink);
    this.staticRuntimeLinkInputs = staticRuntimeLinkInputs;
    this.staticRuntimeLinkMiddleman = staticRuntimeLinkMiddleman;
    this.dynamicRuntimeLinkInputs = dynamicRuntimeLinkInputs;
    this.dynamicRuntimeLinkMiddleman = dynamicRuntimeLinkMiddleman;
    this.dynamicRuntimeSolibDir = Preconditions.checkNotNull(dynamicRuntimeSolibDir);
    this.ccInfo =
        CcInfo.builder()
            .setCcCompilationContext(Preconditions.checkNotNull(ccCompilationContext))
            .setCcLinkingContext(CcLinkingContext.EMPTY)
            .build();
    this.supportsParamFiles = supportsParamFiles;
    this.supportsHeaderParsing = supportsHeaderParsing;
    this.buildVariables = buildVariables;
    this.builtinIncludeFiles = builtinIncludeFiles;
    this.coverageEnvironment = coverageEnvironment;
    this.linkDynamicLibraryTool = linkDynamicLibraryTool;
    this.builtInIncludeDirectories = builtInIncludeDirectories;
    this.sysroot = sysroot;
    this.fdoContext = fdoContext == null ? FdoContext.getDisabledContext() : fdoContext;
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
    this.licensesProvider = licensesProvider;
  }

  /** Returns c++ Make variables. */
  public static Map<String, String> getCppBuildVariables(
      Function<Tool, PathFragment> getToolPathFragment,
      String targetLibc,
      String compiler,
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
   * Determines if we should apply -fPIC for this rule's C++ compilations. This determination is
   * generally made by the global C++ configuration settings "needsPic" and "usePicForBinaries".
   * However, an individual rule may override these settings by applying -fPIC" to its "nocopts"
   * attribute. This allows incompatible rules to "opt out" of global PIC settings (see bug:
   * "Provide a way to turn off -fPIC for targets that can't be built that way").
   *
   * @return true if this rule's compilations should apply -fPIC, false otherwise
   */
  @Override
  public boolean usePicForDynamicLibraries(FeatureConfiguration featureConfiguration) {
    return forcePic
        || toolchainNeedsPic()
        || featureConfiguration.isEnabled(CppRuleClasses.SUPPORTS_PIC);
  }

  /**
   * Deprecated since it uses legacy crosstool fields.
   *
   * <p>See {link {@link #usePicForDynamicLibraries(FeatureConfiguration)} for docs}
   *
   * @return
   */
  @Deprecated
  @Override
  public boolean usePicForDynamicLibrariesUsingLegacyFields() {
    return forcePic
        || toolchainNeedsPic()
        || FeatureConfiguration.EMPTY.isEnabled(CppRuleClasses.SUPPORTS_PIC);
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
   * Returns true if PER_OBJECT_DEBUG_INFO are specified and supported by the CROSSTOOL for the
   * build implied by the given configuration, toolchain and feature configuration.
   */
  public boolean shouldCreatePerObjectDebugInfo(FeatureConfiguration featureConfiguration) {
    return cppConfiguration.fissionIsActiveForCurrentCompilationMode()
        && featureConfiguration.isEnabled(CppRuleClasses.PER_OBJECT_DEBUG_INFO);
  }

  /** Whether the toolchains supports header parsing. */
  public boolean supportsHeaderParsing() {
    return supportsHeaderParsing;
  }

  /**
   * Returns true if headers should be parsed in this build.
   *
   * <p>This means headers in 'srcs' and 'hdrs' will be "compiled" using {@link CppCompileAction}).
   * It will run compiler's parser to ensure the header is self-contained. This is required for
   * layering_check to work.
   */
  public boolean shouldProcessHeaders(FeatureConfiguration featureConfiguration) {
    // If parse_headers_verifies_modules is switched on, we verify that headers are
    // self-contained by building the module instead.
    return !cppConfiguration.getParseHeadersVerifiesModules()
        && featureConfiguration.isEnabled(CppRuleClasses.PARSE_HEADERS);
  }

  @Override
  public void addGlobalMakeVariables(ImmutableMap.Builder<String, String> globalMakeEnvBuilder) {
    globalMakeEnvBuilder.putAll(
        getCppBuildVariables(
            this::getToolPathFragment,
            getTargetLibc(),
            getCompiler(),
            crosstoolTopPathFragment,
            getAbiGlibcVersion(),
            getAbi(),
            getAdditionalMakeVariables()));
  }

  @Override
  public ImmutableList<String> getBuiltInIncludeDirectoriesAsStrings() {
    return builtInIncludeDirectories
        .stream()
        .map(PathFragment::getSafePathString)
        .collect(ImmutableList.toImmutableList());
  }

  public ImmutableList<PathFragment> getBuiltInIncludeDirectories() {
    return builtInIncludeDirectories;
  }

  /** Returns the identifier of the toolchain as specified in the {@code CToolchain} proto. */
  public String getToolchainIdentifier() {
    return toolchainInfo.getToolchainIdentifier();
  }

  /** Returns all the files in Crosstool. Is not a middleman. */
  public NestedSet<Artifact> getAllFiles() {
    return allFiles;
  }

  /** Returns a middleman for all the files in Crosstool. */
  public NestedSet<Artifact> getAllFilesMiddleman() {
    return allFilesMiddleman;
  }

  /** Returns the files necessary for compilation. */
  public NestedSet<Artifact> getCompilerFiles() {
    return compilerFiles;
  }

  /**
   * Returns the files necessary for compilation excluding headers, assuming that included files
   * will be discovered by input discovery. If the toolchain does not provide this fileset, falls
   * back to {@link #getCompilerFiles()}.
   */
  public NestedSet<Artifact> getCompilerFilesWithoutIncludes() {
    if (compilerFilesWithoutIncludes.isEmpty()) {
      return getCompilerFiles();
    }
    return compilerFilesWithoutIncludes;
  }

  /** Returns the files necessary for a 'strip' invocation. */
  public NestedSet<Artifact> getStripFiles() {
    return stripFiles;
  }

  /** Returns the files necessary for an 'objcopy' invocation. */
  public NestedSet<Artifact> getObjcopyFiles() {
    return objcopyFiles;
  }

  /**
   * Returns the files necessary for an 'as' invocation. May be empty if the CROSSTOOL file does not
   * define as_files.
   */
  public NestedSet<Artifact> getAsFiles() {
    return asFiles;
  }

  /**
   * Returns the files necessary for an 'ar' invocation. May be empty if the CROSSTOOL file does not
   * define ar_files.
   */
  public NestedSet<Artifact> getArFiles() {
    return arFiles;
  }

  /** Returns the files necessary for linking, including the files needed for libc. */
  public NestedSet<Artifact> getLinkerFiles() {
    return linkerFiles;
  }

  public NestedSet<Artifact> getDwpFiles() {
    return dwpFiles;
  }

  /** Returns the files necessary for capturing code coverage. */
  public NestedSet<Artifact> getCoverageFiles() {
    return coverageFiles;
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
  public NestedSet<Artifact> getStaticRuntimeLinkInputs(
      RuleContext ruleContext, FeatureConfiguration featureConfiguration)
      throws RuleErrorException {
    if (shouldStaticallyLinkCppRuntimes(featureConfiguration)) {
      if (staticRuntimeLinkInputs == null) {
        throw ruleContext.throwWithRuleError(
            "Toolchain supports embedded runtimes, but didn't "
                + "provide static_runtime_lib attribute.");
      }
      return staticRuntimeLinkInputs;
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  /** Returns an aggregating middleman that represents the static runtime libraries. */
  @Nullable
  public Artifact getStaticRuntimeLinkMiddleman(
      RuleContext ruleContext, FeatureConfiguration featureConfiguration)
      throws RuleErrorException {
    if (shouldStaticallyLinkCppRuntimes(featureConfiguration)) {
      if (staticRuntimeLinkInputs == null) {
        throw ruleContext.throwWithRuleError(
            "Toolchain supports embedded runtimes, but didn't "
                + "provide static_runtime_lib attribute.");
      }
      return staticRuntimeLinkMiddleman;
    } else {
      return null;
    }
  }

  /** Returns the dynamic runtime libraries. */
  public NestedSet<Artifact> getDynamicRuntimeLinkInputs(
      RuleErrorConsumer ruleContext, FeatureConfiguration featureConfiguration)
      throws RuleErrorException {
    if (shouldStaticallyLinkCppRuntimes(featureConfiguration)) {
      if (dynamicRuntimeLinkInputs == null) {
        throw ruleContext.throwWithRuleError(
            "Toolchain supports embedded runtimes, but didn't "
                + "provide dynamic_runtime_lib attribute.");
      }
      return dynamicRuntimeLinkInputs;
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  /** Returns an aggregating middleman that represents the dynamic runtime libraries. */
  @Nullable
  public Artifact getDynamicRuntimeLinkMiddleman(
      RuleErrorConsumer ruleContext, FeatureConfiguration featureConfiguration)
      throws RuleErrorException {
    if (shouldStaticallyLinkCppRuntimes(featureConfiguration)) {
      if (dynamicRuntimeLinkInputs == null) {
        throw ruleContext.throwWithRuleError(
            "Toolchain supports embedded runtimes, but didn't "
                + "provide dynamic_runtime_lib attribute.");
      }
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

  /** Returns the {@code CcCompilationContext} for the toolchain. */
  public CcCompilationContext getCcCompilationContext() {
    return ccInfo.getCcCompilationContext();
  }

  /** Returns the {@code CcInfo} for the toolchain. */
  public CcInfo getCcInfo() {
    return ccInfo;
  }

  /**
   * Whether the toolchains supports parameter files.
   */
  public boolean supportsParamFiles() {
    return supportsParamFiles;
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

  /** Returns whether the toolchain supports dynamic linking. */
  public boolean supportsDynamicLinker(FeatureConfiguration featureConfiguration) {
    return toolchainInfo.supportsDynamicLinker()
        || featureConfiguration.isEnabled(CppRuleClasses.SUPPORTS_DYNAMIC_LINKER);
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
  public boolean supportsStartEndLib(FeatureConfiguration featureConfiguration) {
    return toolchainInfo.supportsStartEndLib()
        || featureConfiguration.isEnabled(CppRuleClasses.SUPPORTS_START_END_LIB);
  }

  /** Returns whether this toolchain supports interface shared libraries. */
  public boolean supportsInterfaceSharedLibraries(FeatureConfiguration featureConfiguration) {
    return toolchainInfo.supportsInterfaceSharedLibraries()
        || featureConfiguration.isEnabled(CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES);
  }

  @Nullable
  public CppConfiguration getCppConfiguration() {
    return cppConfiguration;
  }

  /** Returns build variables to be templated into the crosstool. */
  public CcToolchainVariables getBuildVariables() {
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
   * Returns the environment variables that need to be added to tests that collect code
   * coverageFiles.
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

  @Override
  @Nullable
  public String getSysroot() {
    return sysroot != null ? sysroot.getPathString() : null;
  }

  public PathFragment getSysrootPathFragment() {
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
  @Override
  public String getCompiler() {
    return toolchainInfo == null ? null : toolchainInfo.getCompiler();
  }

  /** Returns the libc version string (e.g. "glibc-2.2.2"). */
  @Override
  public String getTargetLibc() {
    return toolchainInfo == null ? null : toolchainInfo.getTargetLibc();
  }

  /** Returns the target architecture using blaze-specific constants (e.g. "piii"). */
  @Override
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
   * Returns the legacy value of the CC_FLAGS Make variable.
   *
   * @deprecated Use the CC_FLAGS from feature configuration instead.
   */
  // TODO(b/65151735): Remove when cc_flags is entirely from features.
  @Deprecated
  public String getLegacyCcFlagsMakeVariable() {
    return toolchainInfo.getLegacyCcFlagsMakeVariable();
  }

  public FdoContext getFdoContext() {
    return fdoContext;
  }

  /**
   * Returns whether the toolchain supports "Fission" C++ builds, i.e. builds where compilation
   * partitions object code and debug symbols into separate output files.
   */
  public boolean supportsFission() {
    return toolchainInfo.supportsFission();
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
  ImmutableList<String> getSharedLibraryLinkOptions(ImmutableList<String> flags) {
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

  /** Returns linker flags for fully statically linked outputs. */
  ImmutableList<String> getLegacyFullyStaticLinkFlags(CompilationMode compilationMode) {
    return configureAllLegacyLinkOptions(compilationMode, LinkingMode.LEGACY_FULLY_STATIC);
  }

  /** Returns linker flags for mostly static linked outputs. */
  ImmutableList<String> getLegacyMostlyStaticLinkFlags(CompilationMode compilationMode) {
    return configureAllLegacyLinkOptions(compilationMode, LinkingMode.STATIC);
  }

  /** Returns linker flags for mostly static shared linked outputs. */
  ImmutableList<String> getLegacyMostlyStaticSharedLinkFlags(CompilationMode compilationMode) {
    return configureAllLegacyLinkOptions(
        compilationMode, LinkingMode.LEGACY_MOSTLY_STATIC_LIBRARIES);
  }

  /** Returns linker flags for artifacts that are not fully or mostly statically linked. */
  ImmutableList<String> getLegacyDynamicLinkFlags(CompilationMode compilationMode) {
    return configureAllLegacyLinkOptions(compilationMode, LinkingMode.DYNAMIC);
  }

  /**
   * Return all flags coming from naked {@code linker_flag} fields in the crosstool. {@code
   * linker_flag}s coming from linking_mode_flags and compilation_mode_flags are not included. If
   * you need all possible linker flags, use {@link #configureAllLegacyLinkOptions(CompilationMode,
   * LinkingMode)}.
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
            .addAll(getCFlagsByCompilationMode().get(cppConfiguration.getCompilationMode()));

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
      CompilationMode compilationMode, LinkingMode linkingMode) {
    return toolchainInfo.configureAllLegacyLinkOptions(compilationMode, linkingMode);
  }

  /** Returns the GNU System Name */
  @Override
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

  public ImmutableList<String> getLegacyCxxOptions() {
    return ImmutableList.<String>builder()
        .addAll(getToolchainCxxFlags())
        .addAll(getCxxFlagsByCompilationMode().get(cppConfiguration.getCompilationMode()))
        .build();
  }

  /**
   * WARNING: This method is only added to allow incremental migration of existing users. Please do
   * not use in new code. Will be removed soon as part of the new Skylark API to the C++ toolchain.
   *
   * Returns the execution path to the linker binary to use for this build. Relative paths are
   * relative to the execution root.
   */
  @Override
  public String getLdExecutableForSkylark() {
    PathFragment ldExecutable = getToolPathFragment(CppConfiguration.Tool.LD);
    return ldExecutable != null ? ldExecutable.getPathString() : "";
  }

  /**
   * WARNING: This method is only added to allow incremental migration of existing users. Please do
   * not use in new code. Will be removed soon as part of the new Skylark API to the C++ toolchain.
   *
   * Returns the path to the GNU binutils 'objcopy' binary to use for this build. (Corresponds to
   * $(OBJCOPY) in make-dbg.) Relative paths are relative to the execution root.
   */
  @Override
  public String getObjCopyExecutableForSkylark() {
    PathFragment objCopyExecutable = getToolPathFragment(Tool.OBJCOPY);
    return objCopyExecutable != null ? objCopyExecutable.getPathString() : "";
  }

  @Override
  public String getCppExecutableForSkylark() {
    PathFragment cppExecutable = getToolPathFragment(Tool.GCC);
    return cppExecutable != null ? cppExecutable.getPathString() : "";
  }

  @Override
  public String getCpreprocessorExecutableForSkylark() {
    PathFragment cpreprocessorExecutable = getToolPathFragment(Tool.CPP);
    return cpreprocessorExecutable != null ? cpreprocessorExecutable.getPathString() : "";
  }

  @Override
  public String getNmExecutableForSkylark() {
    PathFragment nmExecutable = getToolPathFragment(Tool.NM);
    return nmExecutable != null ? nmExecutable.getPathString() : "";
  }

  @Override
  public String getObjdumpExecutableForSkylark() {
    PathFragment objdumpExecutable = getToolPathFragment(Tool.OBJDUMP);
    return objdumpExecutable != null ? objdumpExecutable.getPathString() : "";
  }

  @Override
  public String getArExecutableForSkylark() {
    PathFragment arExecutable = getToolPathFragment(Tool.AR);
    return arExecutable != null ? arExecutable.getPathString() : "";
  }

  @Override
  public String getStripExecutableForSkylark() {
    PathFragment stripExecutable = getToolPathFragment(Tool.STRIP);
    return stripExecutable != null ? stripExecutable.getPathString() : "";
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

  public LicensesProvider getLicensesProvider() {
    return licensesProvider;
  }
}

