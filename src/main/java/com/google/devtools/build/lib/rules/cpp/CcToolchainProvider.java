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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.LicensesProvider;
import com.google.devtools.build.lib.analysis.PackageSpecificationProvider;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.rules.cpp.FdoContext.BranchFdoProfile;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcToolchainProviderApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkThread;

/** Information about a C++ compiler used by the <code>cc_*</code> rules. */
@Immutable
public final class CcToolchainProvider extends NativeInfo
    implements CcToolchainProviderApi<
            FeatureConfigurationForStarlark,
            BranchFdoProfile,
            FdoContext,
            ConstraintValueInfo,
            StarlarkRuleContext,
            InvalidConfigurationException,
            CppConfiguration,
            CcToolchainVariables>,
        HasCcToolchainLabel {

  public static final BuiltinProvider<CcToolchainProvider> PROVIDER =
      new BuiltinProvider<CcToolchainProvider>("CcToolchainInfo", CcToolchainProvider.class) {};

  @Nullable private final CppConfiguration cppConfiguration;
  private final PathFragment crosstoolTopPathFragment;
  private final NestedSet<Artifact> allFiles;
  private final NestedSet<Artifact> allFilesIncludingLibc;
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
  private final NestedSet<Artifact> targetLibcLink;
  @Nullable private final NestedSet<Artifact> staticRuntimeLinkInputs;
  @Nullable private final NestedSet<Artifact> dynamicRuntimeLinkInputs;
  private final PathFragment dynamicRuntimeSolibDir;
  private final CcInfo ccInfo;
  private final boolean supportsParamFiles;
  private final boolean supportsHeaderParsing;
  private final BuildOptions buildOptions;
  private final CcToolchainVariables buildVariables;
  private final ImmutableList<Artifact> builtinIncludeFiles;
  private final ImmutableList<Artifact> targetBuiltinIncludeFiles;
  @Nullable private final Artifact linkDynamicLibraryTool;
  @Nullable private final Artifact grepIncludes;
  private final ImmutableList<PathFragment> builtInIncludeDirectories;
  @Nullable private final PathFragment sysroot;
  private final PathFragment targetSysroot;
  private final boolean isToolConfiguration;
  private final ImmutableMap<String, PathFragment> toolPaths;
  private final CcToolchainFeatures toolchainFeatures;
  private final String toolchainIdentifier;
  private final String compiler;
  private final String targetCpu;
  private final String targetOS;
  private final PathFragment defaultSysroot;
  private final PathFragment runtimeSysroot;
  private final String abiGlibcVersion;
  private final String abi;
  private final String targetLibc;
  private final String targetSystemName;
  private final Label ccToolchainLabel;
  private final String solibDirectory;

  private final ImmutableMap<String, String> additionalMakeVariables;
  // TODO(b/65151735): Remove when cc_flags is entirely from features.
  private final String legacyCcFlagsMakeVariable;
  /**
   * WARNING: We don't like {@link FdoContext}. Its {@link FdoContext#fdoProfilePath} is pure path
   * and that is horrible as it breaks many Bazel assumptions! Don't do bad stuff with it, don't
   * take inspiration from it.
   */
  private final FdoContext fdoContext;

  private final LicensesProvider licensesProvider;
  private final PackageSpecificationProvider allowlistForLayeringCheck;
  private final PackageSpecificationProvider allowListForLooseHeaderCheck;

  private final String objcopyExecutable;
  private final String compilerExecutable;
  private final String preprocessorExecutable;
  private final String nmExecutable;
  private final String objdumpExecutable;
  private final String arExecutable;
  private final String stripExecutable;
  private final String ldExecutable;
  private final String gcovExecutable;
  private final StarlarkFunction ccToolchainBuildVariablesFunc;

  public CcToolchainProvider(
      @Nullable CppConfiguration cppConfiguration,
      CcToolchainFeatures toolchainFeatures,
      PathFragment crosstoolTopPathFragment,
      NestedSet<Artifact> allFiles,
      NestedSet<Artifact> allFilesIncludingLibc,
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
      NestedSet<Artifact> targetLibcLink,
      NestedSet<Artifact> staticRuntimeLinkInputs,
      NestedSet<Artifact> dynamicRuntimeLinkInputs,
      PathFragment dynamicRuntimeSolibDir,
      CcCompilationContext ccCompilationContext,
      boolean supportsParamFiles,
      boolean supportsHeaderParsing,
      BuildOptions buildOptions,
      CcToolchainVariables buildVariables,
      ImmutableList<Artifact> builtinIncludeFiles,
      ImmutableList<Artifact> targetBuiltinIncludeFiles,
      Artifact linkDynamicLibraryTool,
      @Nullable Artifact grepIncludes,
      ImmutableList<PathFragment> builtInIncludeDirectories,
      @Nullable PathFragment sysroot,
      @Nullable PathFragment targetSysroot,
      FdoContext fdoContext,
      boolean isToolConfiguration,
      LicensesProvider licensesProvider,
      ImmutableMap<String, PathFragment> toolPaths,
      String toolchainIdentifier,
      String compiler,
      String abiGlibcVersion,
      String targetCpu,
      String targetOS,
      PathFragment defaultSysroot,
      PathFragment runtimeSysroot,
      String targetLibc,
      Label ccToolchainLabel,
      String solibDirectory,
      String abi,
      String targetSystemName,
      ImmutableMap<String, String> additionalMakeVariables,
      String legacyCcFlagsMakeVariable,
      PackageSpecificationProvider allowlistForLayeringCheck,
      PackageSpecificationProvider allowListForLooseHeaderCheck,
      String objcopyExecutable,
      String compilerExecutable,
      String preprocessorExecutable,
      String nmExecutable,
      String objdumpExecutable,
      String arExecutable,
      String stripExecutable,
      String ldExecutable,
      String gcovExecutable,
      StarlarkFunction ccToolchainBuildVariablesFunc) {
    super();
    this.cppConfiguration = cppConfiguration;
    this.crosstoolTopPathFragment = crosstoolTopPathFragment;
    this.allFiles = Preconditions.checkNotNull(allFiles);
    this.allFilesIncludingLibc = Preconditions.checkNotNull(allFilesIncludingLibc);
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
    this.targetLibcLink = Preconditions.checkNotNull(targetLibcLink);
    this.staticRuntimeLinkInputs = staticRuntimeLinkInputs;
    this.dynamicRuntimeLinkInputs = dynamicRuntimeLinkInputs;
    this.dynamicRuntimeSolibDir = Preconditions.checkNotNull(dynamicRuntimeSolibDir);
    this.ccInfo =
        CcInfo.builder()
            .setCcCompilationContext(Preconditions.checkNotNull(ccCompilationContext))
            .build();
    this.supportsParamFiles = supportsParamFiles;
    this.supportsHeaderParsing = supportsHeaderParsing;
    this.buildOptions = buildOptions;
    this.buildVariables = buildVariables;
    this.builtinIncludeFiles = builtinIncludeFiles;
    this.targetBuiltinIncludeFiles = targetBuiltinIncludeFiles;
    this.linkDynamicLibraryTool = linkDynamicLibraryTool;
    this.builtInIncludeDirectories = builtInIncludeDirectories;
    this.sysroot = sysroot;
    this.targetSysroot = targetSysroot;
    this.defaultSysroot = defaultSysroot;
    this.runtimeSysroot = runtimeSysroot;
    this.fdoContext = fdoContext == null ? FdoContext.getDisabledContext() : fdoContext;
    this.isToolConfiguration = isToolConfiguration;
    this.licensesProvider = licensesProvider;
    this.toolPaths = toolPaths;
    this.toolchainFeatures = toolchainFeatures;
    this.toolchainIdentifier = toolchainIdentifier;
    this.compiler = compiler;
    this.abiGlibcVersion = abiGlibcVersion;
    this.targetCpu = targetCpu;
    this.targetOS = targetOS;
    this.targetLibc = targetLibc;
    this.ccToolchainLabel = ccToolchainLabel;
    this.solibDirectory = solibDirectory;
    this.abi = abi;
    this.targetSystemName = targetSystemName;
    this.additionalMakeVariables = additionalMakeVariables;
    this.legacyCcFlagsMakeVariable = legacyCcFlagsMakeVariable;
    this.allowlistForLayeringCheck = allowlistForLayeringCheck;
    this.allowListForLooseHeaderCheck = allowListForLooseHeaderCheck;

    this.objcopyExecutable = objcopyExecutable;
    this.compilerExecutable = compilerExecutable;
    this.preprocessorExecutable = preprocessorExecutable;
    this.nmExecutable = nmExecutable;
    this.objdumpExecutable = objdumpExecutable;
    this.arExecutable = arExecutable;
    this.stripExecutable = stripExecutable;
    this.ldExecutable = ldExecutable;
    this.gcovExecutable = gcovExecutable;
    this.grepIncludes = grepIncludes;
    this.ccToolchainBuildVariablesFunc = ccToolchainBuildVariablesFunc;
  }

  @Override
  public BuiltinProvider<CcToolchainProvider> getProvider() {
    return PROVIDER;
  }

  /**
   * See {@link #usePicForDynamicLibraries}. This method is there only to serve Starlark callers.
   */
  @Override
  public boolean usePicForDynamicLibrariesFromStarlark(
      FeatureConfigurationForStarlark featureConfiguration) {
    return usePicForDynamicLibraries(
        featureConfiguration
            .getCppConfigurationFromFeatureConfigurationCreatedForStarlark_andIKnowWhatImDoing(),
        featureConfiguration.getFeatureConfiguration());
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
  public boolean usePicForDynamicLibraries(
      CppConfiguration cppConfiguration, FeatureConfiguration featureConfiguration) {
    return cppConfiguration.forcePic()
        || featureConfiguration.isEnabled(CppRuleClasses.SUPPORTS_PIC);
  }

  /**
   * Returns true if PER_OBJECT_DEBUG_INFO are specified and supported by the CROSSTOOL for the
   * build implied by the given configuration, toolchain and feature configuration.
   */
  public boolean shouldCreatePerObjectDebugInfo(
      FeatureConfiguration featureConfiguration, CppConfiguration cppConfiguration) {
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
  public boolean shouldProcessHeaders(
      FeatureConfiguration featureConfiguration, CppConfiguration cppConfiguration) {
    return featureConfiguration.isEnabled(CppRuleClasses.PARSE_HEADERS);
  }

  public void addGlobalMakeVariables(ImmutableMap.Builder<String, String> globalMakeEnvBuilder) {
    ImmutableMap.Builder<String, String> result = ImmutableMap.builder();

    // hardcoded CC->gcc setting for unit tests
    result.put("CC", getToolPathStringOrNull(Tool.GCC));

    // Make variables provided by crosstool/gcc compiler suite.
    result.put("AR", getToolPathStringOrNull(Tool.AR));
    result.put("NM", getToolPathStringOrNull(Tool.NM));
    result.put("LD", getToolPathStringOrNull(Tool.LD));
    String objcopyTool = getToolPathStringOrNull(Tool.OBJCOPY);
    if (objcopyTool != null) {
      // objcopy is optional in Crosstool
      result.put("OBJCOPY", objcopyTool);
    }
    result.put("STRIP", getToolPathStringOrNull(Tool.STRIP));

    String gcovtool = getToolPathStringOrNull(Tool.GCOVTOOL);
    if (gcovtool != null) {
      // gcov-tool is optional in Crosstool
      result.put("GCOVTOOL", gcovtool);
    }

    if (getTargetLibc().startsWith("glibc-")) {
      result.put("GLIBC_VERSION", getTargetLibc().substring("glibc-".length()));
    } else {
      result.put("GLIBC_VERSION", getTargetLibc());
    }

    result.put("C_COMPILER", getCompiler());

    // Deprecated variables

    // TODO(bazel-team): delete all of these.
    result.put("CROSSTOOLTOP", crosstoolTopPathFragment.getPathString());

    // TODO(bazel-team): Remove when Starlark dependencies can be updated to rely on
    // CcToolchainProvider.
    result.putAll(getAdditionalMakeVariables());

    String abiGlibcVersion = getAbiGlibcVersion();
    if (abiGlibcVersion != null) {
      result.put("ABI_GLIBC_VERSION", getAbiGlibcVersion());
    }

    String abi = getAbi();
    if (abi != null) {
      result.put("ABI", getAbi());
    }

    globalMakeEnvBuilder.putAll(result.buildOrThrow());
  }

  /**
   * Returns the path fragment that is either absolute or relative to the execution root that can be
   * used to execute the given tool.
   *
   * @throws RuleErrorException when the tool is not specified by the toolchain.
   */
  public PathFragment getToolPathFragment(
      CppConfiguration.Tool tool, RuleErrorConsumer ruleErrorConsumer) throws RuleErrorException {
    PathFragment toolPathFragment = getToolPathFragmentOrNull(tool);
    if (toolPathFragment == null) {
      throw ruleErrorConsumer.throwWithRuleError(
          String.format(
              "cc_toolchain '%s' with identifier '%s' doesn't define a tool path for '%s'",
              getCcToolchainLabel(), getToolchainIdentifier(), tool.getNamePart()));
    }
    return toolPathFragment;
  }

  /**
   * Returns the path fragment that is either absolute or relative to the execution root that can be
   * used to execute the given tool.
   */
  @Nullable
  public String getToolPathStringOrNull(Tool tool) {
    PathFragment toolPathFragment = getToolPathFragmentOrNull(tool);
    return toolPathFragment == null ? null : toolPathFragment.getPathString();
  }

  @Nullable
  @Override
  public String getToolPathStringOrNoneForStarlark(String toolString, StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    Tool tool = Tool.valueOf(toolString);
    PathFragment toolPathFragment = getToolPathFragmentOrNull(tool);
    return toolPathFragment == null ? null : toolPathFragment.getPathString();
  }

  /**
   * Returns the path fragment that is either absolute or relative to the execution root that can be
   * used to execute the given tool.
   */
  @Nullable
  public PathFragment getToolPathFragmentOrNull(CppConfiguration.Tool tool) {
    return toolPaths.get(tool.getNamePart());
  }

  @Override
  public ImmutableList<String> getBuiltInIncludeDirectoriesAsStrings() {
    return builtInIncludeDirectories.stream()
        .map(PathFragment::getSafePathString)
        .collect(ImmutableList.toImmutableList());
  }

  @Override
  public Depset getAllFilesForStarlark() {
    return Depset.of(Artifact.class, getAllFiles());
  }

  @Override
  public Depset getStaticRuntimeLibForStarlark(
      FeatureConfigurationForStarlark featureConfigurationForStarlark) throws EvalException {
    return Depset.of(
        Artifact.class,
        getStaticRuntimeLinkInputs(featureConfigurationForStarlark.getFeatureConfiguration()));
  }

  @Override
  public Depset getDynamicRuntimeLibForStarlark(
      FeatureConfigurationForStarlark featureConfigurationForStarlark) throws EvalException {
    return Depset.of(
        Artifact.class,
        getDynamicRuntimeLinkInputs(featureConfigurationForStarlark.getFeatureConfiguration()));
  }

  public ImmutableList<PathFragment> getBuiltInIncludeDirectories() {
    return builtInIncludeDirectories;
  }

  /** Returns the identifier of the toolchain as specified in the {@code CToolchain} proto. */
  @Override
  public String getToolchainIdentifier() {
    return toolchainIdentifier;
  }

  /** Returns all the files in Crosstool. */
  public NestedSet<Artifact> getAllFiles() {
    return allFiles;
  }

  /** Returns all the files in Crosstool + libc. */
  public NestedSet<Artifact> getAllFilesIncludingLibc() {
    return allFilesIncludingLibc;
  }

  @Override
  public Depset getAllFilesIncludingLibcForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Artifact.class, getAllFilesIncludingLibc());
  }

  /** Returns the files necessary for compilation. */
  public NestedSet<Artifact> getCompilerFiles() {
    return compilerFiles;
  }

  @Override
  public Depset getCompilerFilesForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Artifact.class, getCompilerFiles());
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

  @Override
  public Depset getStripFilesForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Artifact.class, getStripFiles());
  }

  /** Returns the files necessary for an 'objcopy' invocation. */
  public NestedSet<Artifact> getObjcopyFiles() {
    return objcopyFiles;
  }

  @Override
  public Depset getObjcopyFilesForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Artifact.class, getObjcopyFiles());
  }

  /**
   * Returns the files necessary for an 'as' invocation. May be empty if the CROSSTOOL file does not
   * define as_files.
   */
  public NestedSet<Artifact> getAsFiles() {
    return asFiles;
  }

  @Override
  public Depset getAsFilesForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Artifact.class, getAsFiles());
  }

  /**
   * Returns the files necessary for an 'ar' invocation. May be empty if the CROSSTOOL file does not
   * define ar_files.
   */
  public NestedSet<Artifact> getArFiles() {
    return arFiles;
  }

  @Override
  public Depset getArFilesForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Artifact.class, getArFiles());
  }

  /** Returns the files necessary for linking, including the files needed for libc. */
  public NestedSet<Artifact> getLinkerFiles() {
    return linkerFiles;
  }

  @Override
  public Depset getLinkerFilesForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Artifact.class, getLinkerFiles());
  }

  public NestedSet<Artifact> getDwpFiles() {
    return dwpFiles;
  }

  @Override
  public Depset getDwpFilesForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Artifact.class, getDwpFiles());
  }

  /** Returns the files necessary for capturing code coverage. */
  public NestedSet<Artifact> getCoverageFiles() {
    return coverageFiles;
  }

  @Override
  public Depset getCoverageFilesForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Depset.of(Artifact.class, getCoverageFiles());
  }

  public NestedSet<Artifact> getLibcLink(CppConfiguration cppConfiguration) {
    if (cppConfiguration.equals(getCppConfigurationEvenThoughItCanBeDifferentThanWhatTargetHas())) {
      return libcLink;
    } else {
      return targetLibcLink;
    }
  }

  @Override
  public String getArtifactNameForCategory(
      String category, String outputName, StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return toolchainFeatures.getArtifactNameForCategory(
        ArtifactCategory.valueOf(category), outputName);
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
  public NestedSet<Artifact> getStaticRuntimeLinkInputs(FeatureConfiguration featureConfiguration)
      throws EvalException {
    if (shouldStaticallyLinkCppRuntimes(featureConfiguration)) {
      if (staticRuntimeLinkInputs == null) {
        throw Starlark.errorf(
            "Toolchain supports embedded runtimes, but didn't provide static_runtime_lib"
                + " attribute.");
      }
      return staticRuntimeLinkInputs;
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  /** Returns the dynamic runtime libraries. */
  public NestedSet<Artifact> getDynamicRuntimeLinkInputs(FeatureConfiguration featureConfiguration)
      throws EvalException {
    if (shouldStaticallyLinkCppRuntimes(featureConfiguration)) {
      if (dynamicRuntimeLinkInputs == null) {
        throw new EvalException(
            "Toolchain supports embedded runtimes, but didn't provide dynamic_runtime_lib"
                + " attribute.");
      }
      return dynamicRuntimeLinkInputs;
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  /**
   * Returns the name of the directory where the solib symlinks for the dynamic runtime libraries
   * live. The directory itself will be under the root of the exec configuration in the 'bin'
   * directory.
   */
  public PathFragment getDynamicRuntimeSolibDir() {
    return dynamicRuntimeSolibDir;
  }

  @Override
  public String getDynamicRuntimeSolibDirForStarlark() {
    return getDynamicRuntimeSolibDir().getPathString();
  }

  /** Returns the {@code CcCompilationContext} for the toolchain. */
  public CcCompilationContext getCcCompilationContext() {
    return ccInfo.getCcCompilationContext();
  }

  /** Returns the {@code CcInfo} for the toolchain. */
  public CcInfo getCcInfo() {
    return ccInfo;
  }

  /** Whether the toolchains supports parameter files. */
  public boolean supportsParamFiles() {
    return supportsParamFiles;
  }

  /** Returns the configured features of the toolchain. */
  @Nullable
  public CcToolchainFeatures getFeatures() {
    return toolchainFeatures;
  }

  @Override
  public Label getCcToolchainLabel() {
    return ccToolchainLabel;
  }

  /**
   * Returns the run time sysroot, which is where the dynamic linker and system libraries are found
   * at runtime. This is usually an absolute path. If the toolchain compiler does not support
   * sysroots, then this method returns <code>null</code>.
   */
  public PathFragment getRuntimeSysroot() {
    return runtimeSysroot;
  }

  @Override
  @Nullable
  public String getRuntimeSysrootForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    PathFragment runtimeSysroot = getRuntimeSysroot();
    return runtimeSysroot != null ? runtimeSysroot.getPathString() : null;
  }

  /**
   * Return the name of the directory (relative to the bin directory) that holds mangled links to
   * shared libraries. This name is always set to the '{@code _solib_<cpu_archictecture_name>}.
   */
  public String getSolibDirectory() {
    return solibDirectory;
  }

  @Override
  public String getSolibDirectoryForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getSolibDirectory();
  }

  /** Returns whether the toolchain supports dynamic linking. */
  public boolean supportsDynamicLinker(FeatureConfiguration featureConfiguration) {
    return featureConfiguration.isEnabled(CppRuleClasses.SUPPORTS_DYNAMIC_LINKER);
  }

  /** Returns whether the toolchain supports the --start-lib/--end-lib options. */
  public boolean supportsStartEndLib(FeatureConfiguration featureConfiguration) {
    return featureConfiguration.isEnabled(CppRuleClasses.SUPPORTS_START_END_LIB);
  }

  /** Returns whether this toolchain supports interface shared libraries. */
  public boolean supportsInterfaceSharedLibraries(FeatureConfiguration featureConfiguration) {
    return featureConfiguration.isEnabled(CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES);
  }

  /**
   * Return CppConfiguration instance that was used to configure CcToolchain.
   *
   * <p>If C++ rules use platforms/toolchains without
   * https://github.com/bazelbuild/proposals/blob/master/designs/2019-02-12-toolchain-transitions.md
   * implemented, CcToolchain is analyzed in the exec configuration. This configuration is not what
   * should be used by rules using the toolchain. This method should only be used to access stuff
   * from CppConfiguration that is identical between exec and target (e.g. incompatible flag
   * values). Don't use it if you don't know what you're doing.
   *
   * <p>Once toolchain transitions are implemented, we can safely use the CppConfiguration from the
   * toolchain in rules.
   */
  CppConfiguration getCppConfigurationEvenThoughItCanBeDifferentThanWhatTargetHas() {
    return cppConfiguration;
  }

  /** Return context-sensitive fdo instrumentation path. */
  public String getCSFdoInstrument() {
    return cppConfiguration.getCSFdoInstrument();
  }

  /** Returns build variables to be templated into the crosstool. */
  public CcToolchainVariables getBuildVariables(
      StarlarkThread thread, BuildOptions buildOptions, CppConfiguration cppConfiguration)
      throws EvalException, InterruptedException {
    if (!cppConfiguration.enableCcToolchainResolution()) {
      return buildVariables;
    }
    // With platforms, cc toolchain is analyzed in the exec configuration, so we can only reuse the
    // same build variables instance if the inputs to the construction match.
    PathFragment sysroot = getSysrootPathFragment(cppConfiguration);
    String minOsVersion = cppConfiguration.getMinimumOsVersion();
    if (Objects.equals(sysroot, this.sysroot)
        && Objects.equals(minOsVersion, this.cppConfiguration.getMinimumOsVersion())
        && (ccToolchainBuildVariablesFunc.getName().equals("cc_toolchain_build_variables")
            || buildOptions.equals(this.buildOptions))) {
      return buildVariables;
    }
    // With platforms, cc toolchain is analyzed in the exec configuration, so we cannot reuse
    // build variables instance.
    AppleConfiguration appleConfiguration = new AppleConfiguration(buildOptions);
    ApplePlatform platform = appleConfiguration.getSingleArchPlatform();
    Object ccToolchainVariables =
        Starlark.call(
            thread,
            ccToolchainBuildVariablesFunc,
            ImmutableList.of(
                /* platform */ platform,
                /* cpu */ buildOptions.get(CoreOptions.class).cpu,
                /* cpp_config */ cppConfiguration,
                /* sysroot */ (sysroot != null ? sysroot.getPathString() : Starlark.NONE)),
            ImmutableMap.of());
    return (CcToolchainVariables) ccToolchainVariables;
  }

  @Override
  public CcToolchainVariables getBuildVariablesForStarlark(
      StarlarkRuleContext starlarkRuleContext,
      CppConfiguration cppConfiguration,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getBuildVariables(
        starlarkRuleContext.getRuleContext().getStarlarkThread(),
        starlarkRuleContext.getRuleContext().getConfiguration().getOptions(),
        cppConfiguration);
  }

  /**
   * Return the set of include files that may be included even if they are not mentioned in the
   * source file or any of the headers included by it.
   */
  public ImmutableList<Artifact> getBuiltinIncludeFiles(CppConfiguration cppConfiguration) {
    if (cppConfiguration.equals(getCppConfigurationEvenThoughItCanBeDifferentThanWhatTargetHas())) {
      return builtinIncludeFiles;
    } else {
      return targetBuiltinIncludeFiles;
    }
  }

  /**
   * Returns the tool which should be used for linking dynamic libraries, or in case it's not
   * specified by the crosstool this will be @tools_repository/tools/cpp:link_dynamic_library
   */
  public Artifact getLinkDynamicLibraryTool() {
    return linkDynamicLibraryTool;
  }

  /** Returns the grep-includes tool which is needing during linking because of linkstamping. */
  @Nullable
  public Artifact getGrepIncludes() {
    return grepIncludes;
  }

  /** Returns the tool that builds interface libraries from dynamic libraries. */
  public Artifact getInterfaceSoBuilder() {
    return interfaceSoBuilder;
  }

  @Override
  @Nullable
  public String getSysroot() {
    return sysroot != null ? sysroot.getPathString() : null;
  }

  public PathFragment getSysrootPathFragment(CppConfiguration cppConfiguration) {
    if (cppConfiguration.equals(getCppConfigurationEvenThoughItCanBeDifferentThanWhatTargetHas())) {
      return sysroot;
    } else {
      return targetSysroot;
    }
  }

  /**
   * Returns the abi we're using, which is a gcc version. E.g.: "gcc-3.4". Note that in practice we
   * might be using gcc-3.4 as ABI even when compiling with gcc-4.1.0, because ABIs are backwards
   * compatible.
   */
  // TODO(bazel-team): The javadoc should clarify how this is used in Blaze.
  public String getAbi() {
    return abi;
  }

  @Override
  public String getAbiForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getAbi();
  }

  /**
   * Returns the glibc version used by the abi we're using. This is a glibc version number (e.g.,
   * "2.2.2"). Note that in practice we might be using glibc 2.2.2 as ABI even when compiling with
   * gcc-4.2.2, gcc-4.3.1, or gcc-4.4.0 (which use glibc 2.3.6), because ABIs are backwards
   * compatible.
   */
  // TODO(bazel-team): The javadoc should clarify how this is used in Blaze.
  public String getAbiGlibcVersion() {
    return abiGlibcVersion;
  }

  @Override
  public String getAbiGlibcVersionForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getAbiGlibcVersion();
  }

  @Override
  public String getCrosstoolTopPathForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return crosstoolTopPathFragment.getPathString();
  }

  /** Returns the compiler version string (e.g. "gcc-4.1.1"). */
  @Override
  public String getCompiler() {
    return compiler;
  }

  /** Returns the libc version string (e.g. "glibc-2.2.2"). */
  @Override
  public String getTargetLibc() {
    return targetLibc;
  }

  /** Returns the target architecture using blaze-specific constants (e.g. "piii"). */
  @Override
  public String getTargetCpu() {
    return targetCpu;
  }

  /**
   * Returns a map of additional make variables for use by {@link BuildConfigurationValue}. These
   * are to used to allow some build rules to avoid the limits on stack frame sizes and
   * variable-length arrays.
   *
   * <p>The returned map must contain an entry for {@code STACK_FRAME_UNLIMITED}, though the entry
   * may be an empty string.
   */
  public ImmutableMap<String, String> getAdditionalMakeVariables() {
    return additionalMakeVariables;
  }

  @Override
  public Dict<String, String> getAdditionalMakeVariablesForStarlark(StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Dict.immutableCopyOf(getAdditionalMakeVariables());
  }

  /**
   * Returns the legacy value of the CC_FLAGS Make variable.
   *
   * @deprecated Use the CC_FLAGS from feature configuration instead.
   */
  // TODO(b/65151735): Remove when cc_flags is entirely from features.
  @Deprecated
  public String getLegacyCcFlagsMakeVariable() {
    return legacyCcFlagsMakeVariable;
  }

  @Override
  public String getLegacyCcFlagsMakeVariableForStarlark(StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return legacyCcFlagsMakeVariable;
  }

  public FdoContext getFdoContext() {
    return fdoContext;
  }

  @Override
  public FdoContext getFdoContextForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return fdoContext;
  }

  @Override
  public String objcopyExecutable() {
    return objcopyExecutable;
  }

  @Override
  public String compilerExecutable() {
    return compilerExecutable;
  }

  @Override
  public String preprocessorExecutable() {
    return preprocessorExecutable;
  }

  @Override
  public String nmExecutable() {
    return nmExecutable;
  }

  @Override
  public String objdumpExecutable() {
    return objdumpExecutable;
  }

  @Override
  public String arExecutable() {
    return arExecutable;
  }

  @Override
  public String stripExecutable() {
    return stripExecutable;
  }

  @Override
  public String ldExecutable() {
    return ldExecutable;
  }

  @Override
  public String gcovExecutable() {
    return gcovExecutable;
  }

  /**
   * Unused, for compatibility with things internal to Google.
   *
   * @deprecated use platforms
   */
  @Deprecated
  public String getTargetOS() {
    return targetOS;
  }

  /** Returns the GNU System Name */
  @Override
  public String getTargetGnuSystemName() {
    return targetSystemName;
  }

  /** Returns the architecture component of the GNU System Name */
  public String getGnuSystemArch() {
    if (targetSystemName.indexOf('-') == -1) {
      return targetSystemName;
    }
    return targetSystemName.substring(0, targetSystemName.indexOf('-'));
  }

  // Not all of CcToolchainProvider is exposed to Starlark, which makes implementing deep equality
  // impossible: if Java-only parts are considered, the behavior is surprising in Starlark, if they
  // are not, the behavior is surprising in Java. Thus, object identity it is.
  @Override
  public boolean equals(Object other) {
    return other == this;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }

  public boolean isToolConfiguration() {
    return isToolConfiguration;
  }

  public LicensesProvider getLicensesProvider() {
    return licensesProvider;
  }

  public PathFragment getDefaultSysroot() {
    return defaultSysroot;
  }

  public boolean requireCtxInConfigureFeatures() {
    return getCppConfigurationEvenThoughItCanBeDifferentThanWhatTargetHas()
        .requireCtxInConfigureFeatures();
  }

  @VisibleForTesting
  NestedSet<Artifact> getStaticRuntimeLibForTesting() {
    return staticRuntimeLinkInputs;
  }

  @VisibleForTesting
  NestedSet<Artifact> getDynamicRuntimeLibForTesting() {
    return dynamicRuntimeLinkInputs;
  }

  public PackageSpecificationProvider getAllowlistForLayeringCheck() {
    return allowlistForLayeringCheck;
  }

  public PackageSpecificationProvider getAllowlistForLooseHeaderCheck() {
    return allowListForLooseHeaderCheck;
  }
}
