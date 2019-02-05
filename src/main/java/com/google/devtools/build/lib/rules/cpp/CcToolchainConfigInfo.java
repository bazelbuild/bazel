// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.rules.cpp.CppConfiguration.getLegacyCrosstoolFieldErrorMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ActionConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ArtifactNamePattern;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.EnvEntry;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.EnvSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Feature;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FlagGroup;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FlagSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Tool;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.VariableWithValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.WithFeatureSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.Expandable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcToolchainConfigInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CompilationModeFlags;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LinkingModeFlags;

/** Information describing C++ toolchain derived from CROSSTOOL file. */
@Immutable
public class CcToolchainConfigInfo extends NativeInfo implements CcToolchainConfigInfoApi {
  public static final NativeProvider<CcToolchainConfigInfo> PROVIDER =
      new NativeProvider<CcToolchainConfigInfo>(
          CcToolchainConfigInfo.class, "CcToolchainConfigInfo") {};

  private final ImmutableList<ActionConfig> actionConfigs;
  private final ImmutableList<Feature> features;
  private final ImmutableList<ArtifactNamePattern> artifactNamePatterns;
  private final ImmutableList<String> cxxBuiltinIncludeDirectories;

  private final String toolchainIdentifier;
  private final String hostSystemName;
  private final String targetSystemName;
  private final String targetCpu;
  private final String targetLibc;
  private final String compiler;
  private final String abiVersion;
  private final String abiLibcVersion;
  private final boolean supportsStartEndLib;
  private final boolean supportsInterfaceSharedLibraries;
  private final boolean supportsEmbeddedRuntimes;
  private final String staticRuntimesFilegroup;
  private final String dynamicRuntimesFilegroup;
  private final boolean supportsFission;
  private final boolean needsPic;
  private final ImmutableList<Pair<String, String>> toolPaths;
  private final ImmutableList<String> compilerFlags;
  private final ImmutableList<String> cxxFlags;
  private final ImmutableList<String> unfilteredCxxFlags;
  private final ImmutableList<String> linkerFlags;
  private final ImmutableList<String> dynamicLibraryLinkerFlags;
  private final ImmutableList<String> testOnlyLinkerFlags;
  private final ImmutableList<String> objcopyEmbedFlags;
  private final ImmutableList<String> ldEmbedFlags;
  private final ImmutableMap<CompilationMode, ImmutableList<String>> compilationModeCompilerFlags;
  private final ImmutableMap<CompilationMode, ImmutableList<String>> compilationModeCxxFlags;
  private final ImmutableMap<CompilationMode, ImmutableList<String>> compilationModeLinkerFlags;
  private final ImmutableList<String> mostlyStaticLinkingModeFlags;
  private final ImmutableList<String> dynamicLinkingModeFlags;
  private final ImmutableList<String> fullyStaticLinkingModeFlags;
  private final ImmutableList<String> mostlyStaticLibrariesLinkingModeFlags;
  private final ImmutableList<Pair<String, String>> makeVariables;
  private final String builtinSysroot;
  private final String defaultLibcTop;
  private final String ccTargetOs;
  private final boolean hasDynamicLinkingModeFlags;
  private final String proto;

  @AutoCodec.Instantiator
  public CcToolchainConfigInfo(
      ImmutableList<ActionConfig> actionConfigs,
      ImmutableList<Feature> features,
      ImmutableList<ArtifactNamePattern> artifactNamePatterns,
      ImmutableList<String> cxxBuiltinIncludeDirectories,
      String toolchainIdentifier,
      String hostSystemName,
      String targetSystemName,
      String targetCpu,
      String targetLibc,
      String compiler,
      String abiVersion,
      String abiLibcVersion,
      boolean supportsStartEndLib,
      boolean supportsInterfaceSharedLibraries,
      boolean supportsEmbeddedRuntimes,
      String staticRuntimesFilegroup,
      String dynamicRuntimesFilegroup,
      boolean supportsFission,
      boolean needsPic,
      ImmutableList<Pair<String, String>> toolPaths,
      ImmutableList<String> compilerFlags,
      ImmutableList<String> cxxFlags,
      ImmutableList<String> unfilteredCxxFlags,
      ImmutableList<String> linkerFlags,
      ImmutableList<String> dynamicLibraryLinkerFlags,
      ImmutableList<String> testOnlyLinkerFlags,
      ImmutableList<String> objcopyEmbedFlags,
      ImmutableList<String> ldEmbedFlags,
      ImmutableMap<CompilationMode, ImmutableList<String>> compilationModeCompilerFlags,
      ImmutableMap<CompilationMode, ImmutableList<String>> compilationModeCxxFlags,
      ImmutableMap<CompilationMode, ImmutableList<String>> compilationModeLinkerFlags,
      ImmutableList<String> mostlyStaticLinkingModeFlags,
      ImmutableList<String> dynamicLinkingModeFlags,
      ImmutableList<String> fullyStaticLinkingModeFlags,
      ImmutableList<String> mostlyStaticLibrariesLinkingModeFlags,
      ImmutableList<Pair<String, String>> makeVariables,
      String builtinSysroot,
      String defaultLibcTop,
      String ccTargetOs,
      boolean hasDynamicLinkingModeFlags,
      String proto) {
    super(PROVIDER);
    this.actionConfigs = actionConfigs;
    this.features = features;
    this.artifactNamePatterns = artifactNamePatterns;
    this.cxxBuiltinIncludeDirectories = cxxBuiltinIncludeDirectories;
    this.toolchainIdentifier = toolchainIdentifier;
    this.hostSystemName = hostSystemName;
    this.targetSystemName = targetSystemName;
    this.targetCpu = targetCpu;
    this.targetLibc = targetLibc;
    this.compiler = compiler;
    this.abiVersion = abiVersion;
    this.abiLibcVersion = abiLibcVersion;
    this.supportsStartEndLib = supportsStartEndLib;
    this.supportsInterfaceSharedLibraries = supportsInterfaceSharedLibraries;
    this.supportsEmbeddedRuntimes = supportsEmbeddedRuntimes;
    this.staticRuntimesFilegroup = staticRuntimesFilegroup;
    this.dynamicRuntimesFilegroup = dynamicRuntimesFilegroup;
    this.supportsFission = supportsFission;
    this.needsPic = needsPic;
    this.toolPaths = toolPaths;
    this.compilerFlags = compilerFlags;
    this.cxxFlags = cxxFlags;
    this.unfilteredCxxFlags = unfilteredCxxFlags;
    this.linkerFlags = linkerFlags;
    this.dynamicLibraryLinkerFlags = dynamicLibraryLinkerFlags;
    this.testOnlyLinkerFlags = testOnlyLinkerFlags;
    this.objcopyEmbedFlags = objcopyEmbedFlags;
    this.ldEmbedFlags = ldEmbedFlags;
    this.compilationModeCompilerFlags = compilationModeCompilerFlags;
    this.compilationModeCxxFlags = compilationModeCxxFlags;
    this.compilationModeLinkerFlags = compilationModeLinkerFlags;
    this.mostlyStaticLinkingModeFlags = mostlyStaticLinkingModeFlags;
    this.dynamicLinkingModeFlags = dynamicLinkingModeFlags;
    this.fullyStaticLinkingModeFlags = fullyStaticLinkingModeFlags;
    this.mostlyStaticLibrariesLinkingModeFlags = mostlyStaticLibrariesLinkingModeFlags;
    this.makeVariables = makeVariables;
    this.builtinSysroot = builtinSysroot;
    this.defaultLibcTop = defaultLibcTop;
    this.ccTargetOs = ccTargetOs;
    this.hasDynamicLinkingModeFlags = hasDynamicLinkingModeFlags;
    this.proto = proto;
  }

  public static CcToolchainConfigInfo fromToolchain(RuleContext ruleContext, CToolchain toolchain)
      throws EvalException {
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    boolean disableLegacyCrosstoolFields = cppConfiguration.disableLegacyCrosstoolFields();
    boolean disableExpandIfAllAvailableInFlagSet =
        cppConfiguration.disableExpandIfAllAvailableInFlagSet();
    ImmutableList.Builder<ActionConfig> actionConfigBuilder = ImmutableList.builder();
    for (CToolchain.ActionConfig actionConfig : toolchain.getActionConfigList()) {
      actionConfigBuilder.add(new ActionConfig(actionConfig));
    }

    ImmutableList.Builder<Feature> featureBuilder = ImmutableList.builder();
    for (CToolchain.Feature feature : toolchain.getFeatureList()) {
      featureBuilder.add(new Feature(feature));
    }

    ImmutableList.Builder<ArtifactNamePattern> artifactNamePatternBuilder = ImmutableList.builder();
    for (CToolchain.ArtifactNamePattern artifactNamePattern :
        toolchain.getArtifactNamePatternList()) {
      artifactNamePatternBuilder.add(new ArtifactNamePattern(artifactNamePattern));
    }

    ImmutableList.Builder<String> optCompilationModeCompilerFlags = ImmutableList.builder();
    ImmutableList.Builder<String> optCompilationModeCxxFlags = ImmutableList.builder();
    ImmutableList.Builder<String> optCompilationModeLinkerFlags = ImmutableList.builder();
    ImmutableList.Builder<String> dbgCompilationModeCompilerFlags = ImmutableList.builder();
    ImmutableList.Builder<String> dbgCompilationModeCxxFlags = ImmutableList.builder();
    ImmutableList.Builder<String> dbgCompilationModeLinkerFlags = ImmutableList.builder();
    ImmutableList.Builder<String> fastbuildCompilationModeCompilerFlags = ImmutableList.builder();
    ImmutableList.Builder<String> fastbuildCompilationModeCxxFlags = ImmutableList.builder();
    ImmutableList.Builder<String> fastbuildCompilationModeLinkerFlags = ImmutableList.builder();

    ImmutableList.Builder<String> fullyStaticLinkerFlags = ImmutableList.builder();
    ImmutableList.Builder<String> mostlyStaticLinkerFlags = ImmutableList.builder();
    ImmutableList.Builder<String> dynamicLinkerFlags = ImmutableList.builder();
    ImmutableList.Builder<String> mostlyStaticLibrariesLinkerFlags = ImmutableList.builder();

    if (disableLegacyCrosstoolFields) {
      if (toolchain.getCompilationModeFlagsCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("compilation_mode_flags"));
      }
      if (toolchain.getLinkingModeFlagsCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("linking_mode_flags"));
      }
      if (toolchain.getArFlagCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("ar_flag"));
      }
      if (toolchain.getArThinArchivesFlagCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("ar_thin_archives_flag"));
      }
      if (toolchain.getCompilerFlagCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("compiler_flag"));
      }
      if (toolchain.getCxxFlagCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("cxx_flag"));
      }
      if (toolchain.getDebianExtraRequiresCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("debian_extra_requires"));
      }
      if (toolchain.hasDefaultPythonTop()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("default_python_top"));
      }
      if (toolchain.hasDefaultPythonVersion()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("default_python_version"));
      }
      if (toolchain.getDynamicLibraryLinkerFlagCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("dynamic_library_linker_flag"));
      }
      if (toolchain.getGccPluginCompilerFlagCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("gcc_plugin_compiler_flag"));
      }
      if (toolchain.getGccPluginHeaderDirectoryCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("gcc_plugin_header_directory"));
      }
      if (toolchain.getLdEmbedFlagCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("ld_embed_flag"));
      }
      if (toolchain.getLinkerFlagCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("linker_flag"));
      }
      if (toolchain.getMaoPluginHeaderDirectoryCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("mao_plugin_header_directory"));
      }
      if (toolchain.hasNeedsPic()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("needsPic"));
      }
      if (toolchain.getObjcopyEmbedFlagCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("objcopy_embed_flag"));
      }
      if (toolchain.hasPythonPreloadSwigdeps()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("python_preload_swigdeps"));
      }
      if (toolchain.hasStaticRuntimesFilegroup()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("static_runtimes_filegroup"));
      }
      if (toolchain.hasDynamicRuntimesFilegroup()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("dynamic_runtimes_filegroup"));
      }
      if (toolchain.hasSupportsDsym()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("supports_dsym"));
      }
      if (toolchain.hasSupportsEmbeddedRuntimes()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("supports_embedded_runtimes"));
      }
      if (toolchain.hasSupportsFission()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("supports_fission"));
      }
      if (toolchain.hasSupportsGoldLinker()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("supports_gold_linker"));
      }
      if (toolchain.hasSupportsIncrementalLinker()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("supports_incremental_linker"));
      }
      if (toolchain.hasSupportsInterfaceSharedObjects()) {
        ruleContext.ruleError(
            getLegacyCrosstoolFieldErrorMessage("supports_interface_shared_objects"));
      }
      if (toolchain.hasSupportsNormalizingAr()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("supports_normalizing_ar"));
      }
      if (toolchain.hasSupportsStartEndLib()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("supports_start_end_lib"));
      }
      if (toolchain.hasSupportsThinArchives()) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("supports_thin_archives"));
      }
      if (toolchain.getTestOnlyLinkerFlagCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("test_only_linker_flag"));
      }
      if (toolchain.getUnfilteredCxxFlagCount() != 0) {
        ruleContext.ruleError(getLegacyCrosstoolFieldErrorMessage("unfiltered_cxx_flag"));
      }
    }

    if (disableExpandIfAllAvailableInFlagSet) {
      toolchain
          .getFeatureList()
          .forEach(
              (f) -> {
                if (f.getFlagSetList().stream()
                    .anyMatch((s) -> s.getExpandIfAllAvailableCount() != 0)) {
                  ruleContext.ruleError(
                      String.format(
                          "Feature '%s' defines a flag_set with expand_if_all_available set. "
                              + "This is disabled by "
                              + "--incompatible_disable_expand_if_all_available_in_flag_set, "
                              + "please migrate your CROSSTOOL (see "
                              + "https://github.com/bazelbuild/bazel/issues/7008 "
                              + "for migration instructions).",
                          f.getName()));
                }
              });
    }

    for (CompilationModeFlags flag : toolchain.getCompilationModeFlagsList()) {
      switch (flag.getMode()) {
        case OPT:
          optCompilationModeCompilerFlags.addAll(flag.getCompilerFlagList());
          optCompilationModeCxxFlags.addAll(flag.getCxxFlagList());
          optCompilationModeLinkerFlags.addAll(flag.getLinkerFlagList());
          break;
        case DBG:
          dbgCompilationModeCompilerFlags.addAll(flag.getCompilerFlagList());
          dbgCompilationModeCxxFlags.addAll(flag.getCxxFlagList());
          dbgCompilationModeLinkerFlags.addAll(flag.getLinkerFlagList());
          break;
        case FASTBUILD:
          fastbuildCompilationModeCompilerFlags.addAll(flag.getCompilerFlagList());
          fastbuildCompilationModeCxxFlags.addAll(flag.getCxxFlagList());
          fastbuildCompilationModeLinkerFlags.addAll(flag.getLinkerFlagList());
          break;
        default:
          // CompilationMode.COVERAGE is ignored
      }
    }

    ImmutableMap.Builder<CompilationMode, ImmutableList<String>> compilationModeCompilerFlags =
        ImmutableMap.builder();
    compilationModeCompilerFlags.put(CompilationMode.OPT, optCompilationModeCompilerFlags.build());
    compilationModeCompilerFlags.put(CompilationMode.DBG, dbgCompilationModeCompilerFlags.build());
    compilationModeCompilerFlags.put(
        CompilationMode.FASTBUILD, fastbuildCompilationModeCompilerFlags.build());

    ImmutableMap.Builder<CompilationMode, ImmutableList<String>> compilationModeCxxFlags =
        ImmutableMap.builder();
    compilationModeCxxFlags.put(CompilationMode.OPT, optCompilationModeCxxFlags.build());
    compilationModeCxxFlags.put(CompilationMode.DBG, dbgCompilationModeCxxFlags.build());
    compilationModeCxxFlags.put(
        CompilationMode.FASTBUILD, fastbuildCompilationModeCxxFlags.build());

    ImmutableMap.Builder<CompilationMode, ImmutableList<String>> compilationModeLinkerFlags =
        ImmutableMap.builder();
    compilationModeLinkerFlags.put(CompilationMode.OPT, optCompilationModeLinkerFlags.build());
    compilationModeLinkerFlags.put(CompilationMode.DBG, dbgCompilationModeLinkerFlags.build());
    compilationModeLinkerFlags.put(
        CompilationMode.FASTBUILD, fastbuildCompilationModeLinkerFlags.build());

    boolean hasDynamicLinkingModeFlags = false;
    for (LinkingModeFlags flag : toolchain.getLinkingModeFlagsList()) {
      switch (flag.getMode()) {
        case FULLY_STATIC:
          fullyStaticLinkerFlags.addAll(flag.getLinkerFlagList());
          break;
        case MOSTLY_STATIC:
          mostlyStaticLinkerFlags.addAll(flag.getLinkerFlagList());
          break;
        case DYNAMIC:
          hasDynamicLinkingModeFlags = true;
          dynamicLinkerFlags.addAll(flag.getLinkerFlagList());
          break;
        case MOSTLY_STATIC_LIBRARIES:
          mostlyStaticLibrariesLinkerFlags.addAll(flag.getLinkerFlagList());
          break;
      }
    }

    return new CcToolchainConfigInfo(
        actionConfigBuilder.build(),
        featureBuilder.build(),
        artifactNamePatternBuilder.build(),
        ImmutableList.copyOf(toolchain.getCxxBuiltinIncludeDirectoryList()),
        toolchain.getToolchainIdentifier(),
        toolchain.getHostSystemName(),
        toolchain.getTargetSystemName(),
        toolchain.getTargetCpu(),
        toolchain.getTargetLibc(),
        toolchain.getCompiler(),
        toolchain.getAbiVersion(),
        toolchain.getAbiLibcVersion(),
        toolchain.getSupportsStartEndLib(),
        toolchain.getSupportsInterfaceSharedObjects(),
        toolchain.getSupportsEmbeddedRuntimes(),
        toolchain.getStaticRuntimesFilegroup(),
        toolchain.getDynamicRuntimesFilegroup(),
        toolchain.getSupportsFission(),
        toolchain.getNeedsPic(),
        toolchain.getToolPathList().stream()
            .map(a -> Pair.of(a.getName(), a.getPath()))
            .collect(ImmutableList.toImmutableList()),
        ImmutableList.copyOf(toolchain.getCompilerFlagList()),
        ImmutableList.copyOf(toolchain.getCxxFlagList()),
        ImmutableList.copyOf(toolchain.getUnfilteredCxxFlagList()),
        ImmutableList.copyOf(toolchain.getLinkerFlagList()),
        ImmutableList.copyOf(toolchain.getDynamicLibraryLinkerFlagList()),
        ImmutableList.copyOf(toolchain.getTestOnlyLinkerFlagList()),
        ImmutableList.copyOf(toolchain.getObjcopyEmbedFlagList()),
        ImmutableList.copyOf(toolchain.getLdEmbedFlagList()),
        compilationModeCompilerFlags.build(),
        compilationModeCxxFlags.build(),
        compilationModeLinkerFlags.build(),
        mostlyStaticLinkerFlags.build(),
        dynamicLinkerFlags.build(),
        fullyStaticLinkerFlags.build(),
        mostlyStaticLibrariesLinkerFlags.build(),
        toolchain.getMakeVariableList().stream()
            .map(makeVariable -> Pair.of(makeVariable.getName(), makeVariable.getValue()))
            .collect(ImmutableList.toImmutableList()),
        toolchain.getBuiltinSysroot(),
        toolchain.getDefaultGrteTop(),
        toolchain.getCcTargetOs(),
        hasDynamicLinkingModeFlags,
        /* proto= */ "");
  }

  public ImmutableList<ActionConfig> getActionConfigs() {
    return actionConfigs;
  }

  public ImmutableList<Feature> getFeatures() {
    return features;
  }

  public ImmutableList<ArtifactNamePattern> getArtifactNamePatterns() {
    return artifactNamePatterns;
  }

  public ImmutableList<String> getCxxBuiltinIncludeDirectories() {
    return cxxBuiltinIncludeDirectories;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getToolchainIdentifier() {
    return toolchainIdentifier;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getHostSystemName() {
    return hostSystemName;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getTargetSystemName() {
    return targetSystemName;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getTargetCpu() {
    return targetCpu;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getTargetLibc() {
    return targetLibc;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getCompiler() {
    return compiler;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getAbiVersion() {
    return abiVersion;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getAbiLibcVersion() {
    return abiLibcVersion;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public boolean supportsStartEndLib() {
    return supportsStartEndLib;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public boolean supportsInterfaceSharedLibraries() {
    return supportsInterfaceSharedLibraries;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public boolean supportsEmbeddedRuntimes() {
    return supportsEmbeddedRuntimes;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getStaticRuntimesFilegroup() {
    return staticRuntimesFilegroup;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getDynamicRuntimesFilegroup() {
    return dynamicRuntimesFilegroup;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public boolean supportsFission() {
    return supportsFission;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public boolean needsPic() {
    return needsPic;
  }

  /** Returns a list of paths of the tools in the form Pair<toolName, path>. */
  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<Pair<String, String>> getToolPaths() {
    return toolPaths;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getCompilerFlags() {
    return compilerFlags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getCxxFlags() {
    return cxxFlags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getUnfilteredCxxFlags() {
    return unfilteredCxxFlags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getLinkerFlags() {
    return linkerFlags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getDynamicLibraryLinkerFlags() {
    return dynamicLibraryLinkerFlags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getTestOnlyLinkerFlags() {
    return testOnlyLinkerFlags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getObjcopyEmbedFlags() {
    return objcopyEmbedFlags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getLdEmbedFlags() {
    return ldEmbedFlags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getMostlyStaticLinkingModeFlags() {
    return mostlyStaticLinkingModeFlags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getDynamicLinkingModeFlags() {
    return dynamicLinkingModeFlags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getFullyStaticLinkingModeFlags() {
    return fullyStaticLinkingModeFlags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getMostlyStaticLibrariesLinkingModeFlags() {
    return mostlyStaticLibrariesLinkingModeFlags;
  }

  /** Returns a list of make variables that have the form Pair<name, value>. */
  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<Pair<String, String>> getMakeVariables() {
    return makeVariables;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getBuiltinSysroot() {
    return builtinSysroot;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getDefaultLibcTop() {
    return defaultLibcTop;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getOptCompilationModeCompilerFlags() {
    ImmutableList<String> flags = compilationModeCompilerFlags.get(CompilationMode.OPT);
    return flags == null ? ImmutableList.of() : flags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getOptCompilationModeCxxFlags() {
    ImmutableList<String> flags = compilationModeCxxFlags.get(CompilationMode.OPT);
    return flags == null ? ImmutableList.of() : flags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getOptCompilationModeLinkerFlags() {
    ImmutableList<String> flags = compilationModeLinkerFlags.get(CompilationMode.OPT);
    return flags == null ? ImmutableList.of() : flags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getDbgCompilationModeCompilerFlags() {
    ImmutableList<String> flags = compilationModeCompilerFlags.get(CompilationMode.DBG);
    return flags == null ? ImmutableList.of() : flags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getDbgCompilationModeCxxFlags() {
    ImmutableList<String> flags = compilationModeCxxFlags.get(CompilationMode.DBG);
    return flags == null ? ImmutableList.of() : flags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getDbgCompilationModeLinkerFlags() {
    ImmutableList<String> flags = compilationModeLinkerFlags.get(CompilationMode.DBG);
    return flags == null ? ImmutableList.of() : flags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getFastbuildCompilationModeCompilerFlags() {
    ImmutableList<String> flags = compilationModeCompilerFlags.get(CompilationMode.FASTBUILD);
    return flags == null ? ImmutableList.of() : flags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getFastbuildCompilationModeCxxFlags() {
    ImmutableList<String> flags = compilationModeCxxFlags.get(CompilationMode.FASTBUILD);
    return flags == null ? ImmutableList.of() : flags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public ImmutableList<String> getFastbuildCompilationModeLinkerFlags() {
    ImmutableList<String> flags = compilationModeLinkerFlags.get(CompilationMode.FASTBUILD);
    return flags == null ? ImmutableList.of() : flags;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public String getCcTargetOs() {
    return ccTargetOs;
  }

  // TODO(b/65151735): Remove once this field is migrated to features.
  @Deprecated
  public boolean hasDynamicLinkingModeFlags() {
    return hasDynamicLinkingModeFlags;
  }

  @Override
  public String getProto() {
    return proto;
  }

  private static CToolchain.WithFeatureSet withFeatureSetToProto(WithFeatureSet withFeatureSet) {
    return CToolchain.WithFeatureSet.newBuilder()
        .addAllFeature(withFeatureSet.getFeatures())
        .addAllNotFeature(withFeatureSet.getNotFeatures())
        .build();
  }

  private static CToolchain.EnvEntry envEntryToProto(EnvEntry envEntry) {
    return CToolchain.EnvEntry.newBuilder()
        .setKey(envEntry.getKey())
        .setValue(envEntry.getValue())
        .build();
  }

  private static CToolchain.EnvSet envSetToProto(EnvSet envSet) {
    return CToolchain.EnvSet.newBuilder()
        .addAllAction(envSet.getActions())
        .addAllEnvEntry(
            envSet.getEnvEntries().stream()
                .map(envEntry -> envEntryToProto(envEntry))
                .collect(ImmutableList.toImmutableList()))
        .addAllWithFeature(
            envSet.getWithFeatureSets().stream()
                .map(withFeatureSet -> withFeatureSetToProto(withFeatureSet))
                .collect(ImmutableList.toImmutableList()))
        .build();
  }

  private static CToolchain.FlagGroup flagGroupToProto(FlagGroup flagGroup) {
    ImmutableList.Builder<CToolchain.FlagGroup> flagGroups = ImmutableList.builder();
    ImmutableList.Builder<String> flags = ImmutableList.builder();
    for (Expandable expandable : flagGroup.getExpandables()) {
      if (expandable instanceof FlagGroup) {
        flagGroups.add(flagGroupToProto((FlagGroup) expandable));
      } else {
        flags.add(((CcToolchainFeatures.Flag) expandable).getString());
      }
    }

    CToolchain.FlagGroup.Builder flagGroupBuilder = CToolchain.FlagGroup.newBuilder();
    if (flagGroup.getIterateOverVariable() != null) {
      flagGroupBuilder.setIterateOver(flagGroup.getIterateOverVariable());
    }
    if (flagGroup.getExpandIfTrue() != null) {
      flagGroupBuilder.setExpandIfTrue(flagGroup.getExpandIfTrue());
    }
    if (flagGroup.getExpandIfFalse() != null) {
      flagGroupBuilder.setExpandIfFalse(flagGroup.getExpandIfFalse());
    }
    if (flagGroup.getExpandIfEqual() != null) {
      flagGroupBuilder.setExpandIfEqual(variableWithValueFromProto(flagGroup.getExpandIfEqual()));
    }
    return flagGroupBuilder
        .addAllExpandIfAllAvailable(flagGroup.getExpandIfAllAvailable())
        .addAllExpandIfNoneAvailable(flagGroup.getExpandIfNoneAvailable())
        .addAllFlagGroup(flagGroups.build())
        .addAllFlag(flags.build())
        .build();
  }

  private static CToolchain.VariableWithValue variableWithValueFromProto(
      VariableWithValue variable) {
    return CToolchain.VariableWithValue.newBuilder()
        .setValue(variable.getValue())
        .setVariable(variable.getVariable())
        .build();
  }

  static CToolchain.Feature featureToProto(Feature feature) {
    return CToolchain.Feature.newBuilder()
        .setName(feature.getName())
        .setEnabled(feature.isEnabled())
        .addAllFlagSet(
            feature.getFlagSets().stream()
                .map(flagSet -> flagSetToProto(flagSet, /* forActionConfig= */ false))
                .collect(ImmutableList.toImmutableList()))
        .addAllEnvSet(
            feature.getEnvSets().stream()
                .map(envSet -> envSetToProto(envSet))
                .collect(ImmutableList.toImmutableList()))
        .addAllRequires(
            feature.getRequires().stream()
                .map(featureSet -> featureSetToProto(featureSet))
                .collect(ImmutableList.toImmutableList()))
        .addAllImplies(feature.getImplies())
        .addAllProvides(feature.getProvides())
        .build();
  }

  private static CToolchain.FlagSet flagSetToProto(FlagSet flagSet, boolean forActionConfig) {
    CToolchain.FlagSet.Builder flagSetBuilder =
        CToolchain.FlagSet.newBuilder()
            .addAllFlagGroup(
                flagSet.getFlagGroups().stream()
                    .map(flagGroup -> flagGroupToProto(flagGroup))
                    .collect(ImmutableList.toImmutableList()))
            .addAllWithFeature(
                flagSet.getWithFeatureSets().stream()
                    .map(withFeatureSet -> withFeatureSetToProto(withFeatureSet))
                    .collect(ImmutableList.toImmutableList()))
            .addAllExpandIfAllAvailable(flagSet.getExpandIfAllAvailable());
    if (!forActionConfig) {
      flagSetBuilder.addAllAction(flagSet.getActions());
    }
    return flagSetBuilder.build();
  }

  private static CToolchain.FeatureSet featureSetToProto(ImmutableSet<String> features) {
    return CToolchain.FeatureSet.newBuilder().addAllFeature(features).build();
  }

  private static CToolchain.Tool toolToProto(Tool tool) {
    return CToolchain.Tool.newBuilder()
        .setToolPath(tool.getToolPathFragment().toString())
        .addAllWithFeature(
            tool.getWithFeatureSetSets().stream()
                .map(withFeatureSet -> withFeatureSetToProto(withFeatureSet))
                .collect(ImmutableList.toImmutableList()))
        .addAllExecutionRequirement(tool.getExecutionRequirements())
        .build();
  }

  static CToolchain.ActionConfig actionConfigToProto(ActionConfig actionConfig) {
    return CToolchain.ActionConfig.newBuilder()
        .setConfigName(actionConfig.getName())
        .setActionName(actionConfig.getActionName())
        .setEnabled(actionConfig.isEnabled())
        .addAllTool(
            actionConfig.getTools().stream()
                .map(tool -> toolToProto(tool))
                .collect(ImmutableList.toImmutableList()))
        .addAllFlagSet(
            actionConfig.getFlagSets().stream()
                .map(flagSet -> flagSetToProto(flagSet, /* forActionConfig= */ true))
                .collect(ImmutableList.toImmutableList()))
        .addAllImplies(actionConfig.getImplies())
        .build();
  }
}
