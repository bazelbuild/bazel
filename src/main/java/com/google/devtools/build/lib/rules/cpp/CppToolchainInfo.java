// Copyright 2017 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.rules.cpp.CppActionConfigs.CppPlatform;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain.ArtifactNamePattern;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath;
import com.google.protobuf.Descriptors.FieldDescriptor;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Information describing the C++ compiler derived from the CToolchain proto.
 *
 * <p>This wrapper class is used to re-plumb information so that it's eventually accessed through
 * {@link CcToolchainProvider} instead of {@link CppConfiguration}.
 */
@AutoCodec
@Immutable
public final class CppToolchainInfo {
  private final CcToolchainConfigInfo ccToolchainConfigInfo;
  private final PathFragment crosstoolTopPathFragment;
  private final String toolchainIdentifier;
  private final CcToolchainFeatures toolchainFeatures;

  private final ImmutableMap<String, PathFragment> toolPaths;
  private final String compiler;
  private final String abiGlibcVersion;

  private final String targetCpu;
  private final String targetOS;

  private final ImmutableList<String> rawBuiltInIncludeDirectories;
  private final PathFragment defaultSysroot;
  private final PathFragment runtimeSysroot;

  private final String targetLibc;
  private final String hostSystemName;
  private final ImmutableList<String> dynamicLibraryLinkFlags;
  private final ImmutableList<String> legacyLinkOptions;
  private final ImmutableListMultimap<LinkingMode, String> legacyLinkOptionsFromLinkingMode;
  private final ImmutableListMultimap<CompilationMode, String> legacyLinkOptionsFromCompilationMode;
  private final ImmutableList<String> testOnlyLinkFlags;
  private final ImmutableList<String> ldOptionsForEmbedding;
  private final ImmutableList<String> objCopyOptionsForEmbedding;

  private final Label ccToolchainLabel;
  private final Label staticRuntimeLibsLabel;
  private final Label dynamicRuntimeLibsLabel;
  private final String solibDirectory;
  private final String abi;
  private final String targetSystemName;

  private final ImmutableMap<String, String> additionalMakeVariables;

  private final ImmutableList<String> crosstoolCompilerFlags;
  private final ImmutableList<String> crosstoolCxxFlags;

  private final ImmutableListMultimap<CompilationMode, String> cFlagsByCompilationMode;
  private final ImmutableListMultimap<CompilationMode, String> cxxFlagsByCompilationMode;

  private final ImmutableList<String> unfilteredCompilerFlags;

  private final boolean supportsFission;
  private final boolean supportsStartEndLib;
  private final boolean supportsEmbeddedRuntimes;
  private final boolean supportsDynamicLinker;
  private final boolean supportsInterfaceSharedObjects;
  private final boolean supportsGoldLinker;
  private final boolean toolchainNeedsPic;

  /**
   * Creates a CppToolchainInfo from CROSSTOOL info encapsulated in {@link CcToolchainConfigInfo}.
   */
  public static CppToolchainInfo create(
      PathFragment crosstoolTopPathFragment,
      Label toolchainLabel,
      CcToolchainConfigInfo ccToolchainConfigInfo,
      boolean disableLegacyCrosstoolFields,
      boolean disableCompilationModeFlags,
      boolean disableLinkingModeFlags)
      throws EvalException {
    ImmutableMap<String, PathFragment> toolPaths =
        computeToolPaths(ccToolchainConfigInfo, crosstoolTopPathFragment);
    PathFragment defaultSysroot =
        CppConfiguration.computeDefaultSysroot(ccToolchainConfigInfo.getBuiltinSysroot());

    ImmutableListMultimap.Builder<LinkingMode, String> linkOptionsFromLinkingModeBuilder =
        ImmutableListMultimap.builder();

    boolean haveDynamicMode = false;
    if (!disableLinkingModeFlags) {
      // If a toolchain supports dynamic libraries at all, there must be at least one
      // of the following:
      // - a "DYNAMIC" section in linking_mode_flags (even if no flags are needed)
      // - a non-empty list in one of the dynamicLibraryLinkerFlag fields
      // If none of the above contain data, then the toolchain can't do dynamic linking.
      haveDynamicMode = ccToolchainConfigInfo.hasDynamicLinkingModeFlags();
      linkOptionsFromLinkingModeBuilder.putAll(
          LinkingMode.DYNAMIC, ccToolchainConfigInfo.getDynamicLinkingModeFlags());
      linkOptionsFromLinkingModeBuilder.putAll(
          LinkingMode.LEGACY_FULLY_STATIC, ccToolchainConfigInfo.getFullyStaticLinkingModeFlags());
      linkOptionsFromLinkingModeBuilder.putAll(
          LinkingMode.STATIC, ccToolchainConfigInfo.getMostlyStaticLinkingModeFlags());
      linkOptionsFromLinkingModeBuilder.putAll(
          LinkingMode.LEGACY_MOSTLY_STATIC_LIBRARIES,
          ccToolchainConfigInfo.getMostlyStaticLibrariesLinkingModeFlags());
    }

    ImmutableListMultimap.Builder<CompilationMode, String> cFlagsBuilder =
        ImmutableListMultimap.builder();
    ImmutableListMultimap.Builder<CompilationMode, String> cxxFlagsBuilder =
        ImmutableListMultimap.builder();
    ImmutableListMultimap.Builder<CompilationMode, String> linkOptionsFromCompilationModeBuilder =
        ImmutableListMultimap.builder();

    if (!disableCompilationModeFlags) {
      cFlagsBuilder.putAll(
          importCompilationMode(CrosstoolConfig.CompilationMode.OPT),
          ccToolchainConfigInfo.getOptCompilationModeCompilerFlags());
      cxxFlagsBuilder.putAll(
          importCompilationMode(CrosstoolConfig.CompilationMode.OPT),
          ccToolchainConfigInfo.getOptCompilationModeCxxFlags());
      linkOptionsFromCompilationModeBuilder.putAll(
          importCompilationMode(CrosstoolConfig.CompilationMode.OPT),
          ccToolchainConfigInfo.getOptCompilationModeLinkerFlags());
      cFlagsBuilder.putAll(
          importCompilationMode(CrosstoolConfig.CompilationMode.DBG),
          ccToolchainConfigInfo.getDbgCompilationModeCompilerFlags());
      cxxFlagsBuilder.putAll(
          importCompilationMode(CrosstoolConfig.CompilationMode.DBG),
          ccToolchainConfigInfo.getDbgCompilationModeCxxFlags());
      linkOptionsFromCompilationModeBuilder.putAll(
          importCompilationMode(CrosstoolConfig.CompilationMode.DBG),
          ccToolchainConfigInfo.getDbgCompilationModeLinkerFlags());
      cFlagsBuilder.putAll(
          importCompilationMode(CrosstoolConfig.CompilationMode.FASTBUILD),
          ccToolchainConfigInfo.getFastbuildCompilationModeCompilerFlags());
      cxxFlagsBuilder.putAll(
          importCompilationMode(CrosstoolConfig.CompilationMode.FASTBUILD),
          ccToolchainConfigInfo.getFastbuildCompilationModeCxxFlags());
      linkOptionsFromCompilationModeBuilder.putAll(
          importCompilationMode(CrosstoolConfig.CompilationMode.FASTBUILD),
          ccToolchainConfigInfo.getFastbuildCompilationModeLinkerFlags());
    }

    try {
      return new CppToolchainInfo(
          ccToolchainConfigInfo,
          crosstoolTopPathFragment,
          ccToolchainConfigInfo.getToolchainIdentifier(),
          toolPaths,
          ccToolchainConfigInfo.getCompiler(),
          ccToolchainConfigInfo.getAbiLibcVersion(),
          ccToolchainConfigInfo.getTargetCpu(),
          ccToolchainConfigInfo.getCcTargetOs(),
          ccToolchainConfigInfo.getCxxBuiltinIncludeDirectories(),
          defaultSysroot,
          // The runtime sysroot should really be set from --grte_top. However, currently libc has
          // no way to set the sysroot. The CROSSTOOL file does set the runtime sysroot, in the
          // builtin_sysroot field. This implies that you can not arbitrarily mix and match
          // Crosstool and libc versions, you must always choose compatible ones.
          defaultSysroot,
          ccToolchainConfigInfo.getTargetLibc(),
          ccToolchainConfigInfo.getHostSystemName(),
          disableLegacyCrosstoolFields
              ? ImmutableList.of()
              : ccToolchainConfigInfo.getDynamicLibraryLinkerFlags(),
          disableLegacyCrosstoolFields
              ? ImmutableList.of()
              : ccToolchainConfigInfo.getLinkerFlags(),
          linkOptionsFromLinkingModeBuilder.build(),
          linkOptionsFromCompilationModeBuilder.build(),
          disableLegacyCrosstoolFields
              ? ImmutableList.of()
              : ccToolchainConfigInfo.getTestOnlyLinkerFlags(),
          ccToolchainConfigInfo.getLdEmbedFlags(),
          ccToolchainConfigInfo.getObjcopyEmbedFlags(),
          toolchainLabel,
          toolchainLabel.getRelativeWithRemapping(
              !ccToolchainConfigInfo.getStaticRuntimesFilegroup().isEmpty()
                  ? ccToolchainConfigInfo.getStaticRuntimesFilegroup()
                  : "static-runtime-libs-" + ccToolchainConfigInfo.getTargetCpu(),
              ImmutableMap.of()),
          toolchainLabel.getRelativeWithRemapping(
              !ccToolchainConfigInfo.getDynamicRuntimesFilegroup().isEmpty()
                  ? ccToolchainConfigInfo.getDynamicRuntimesFilegroup()
                  : "dynamic-runtime-libs-" + ccToolchainConfigInfo.getTargetCpu(),
              ImmutableMap.of()),
          "_solib_" + ccToolchainConfigInfo.getTargetCpu(),
          ccToolchainConfigInfo.getAbiVersion(),
          ccToolchainConfigInfo.getTargetSystemName(),
          computeAdditionalMakeVariables(ccToolchainConfigInfo),
          disableLegacyCrosstoolFields
              ? ImmutableList.of()
              : ccToolchainConfigInfo.getCompilerFlags(),
          disableLegacyCrosstoolFields ? ImmutableList.of() : ccToolchainConfigInfo.getCxxFlags(),
          cFlagsBuilder.build(),
          cxxFlagsBuilder.build(),
          disableLegacyCrosstoolFields
              ? ImmutableList.of()
              : ccToolchainConfigInfo.getUnfilteredCxxFlags(),
          ccToolchainConfigInfo.supportsFission(),
          ccToolchainConfigInfo.supportsStartEndLib(),
          ccToolchainConfigInfo.supportsEmbeddedRuntimes(),
          haveDynamicMode || !ccToolchainConfigInfo.getDynamicLibraryLinkerFlags().isEmpty(),
          ccToolchainConfigInfo.supportsInterfaceSharedObjects(),
          ccToolchainConfigInfo.supportsGoldLinker(),
          ccToolchainConfigInfo.needsPic());
    } catch (LabelSyntaxException e) {
      // All of the above label.getRelativeWithRemapping() calls are valid labels, and the
      // crosstool_top was already checked earlier in the process.
      throw new EvalException(Location.BUILTIN, e);
    }
  }

  @AutoCodec.Instantiator
  CppToolchainInfo(
      CcToolchainConfigInfo ccToolchainConfigInfo,
      PathFragment crosstoolTopPathFragment,
      String toolchainIdentifier,
      ImmutableMap<String, PathFragment> toolPaths,
      String compiler,
      String abiGlibcVersion,
      String targetCpu,
      String targetOS,
      ImmutableList<String> rawBuiltInIncludeDirectories,
      PathFragment defaultSysroot,
      PathFragment runtimeSysroot,
      String targetLibc,
      String hostSystemName,
      ImmutableList<String> dynamicLibraryLinkFlags,
      ImmutableList<String> legacyLinkOptions,
      ImmutableListMultimap<LinkingMode, String> legacyLinkOptionsFromLinkingMode,
      ImmutableListMultimap<CompilationMode, String> legacyLinkOptionsFromCompilationMode,
      ImmutableList<String> testOnlyLinkFlags,
      ImmutableList<String> ldOptionsForEmbedding,
      ImmutableList<String> objCopyOptionsForEmbedding,
      Label ccToolchainLabel,
      Label staticRuntimeLibsLabel,
      Label dynamicRuntimeLibsLabel,
      String solibDirectory,
      String abi,
      String targetSystemName,
      ImmutableMap<String, String> additionalMakeVariables,
      ImmutableList<String> crosstoolCompilerFlags,
      ImmutableList<String> crosstoolCxxFlags,
      ImmutableListMultimap<CompilationMode, String> cFlagsByCompilationMode,
      ImmutableListMultimap<CompilationMode, String> cxxFlagsByCompilationMode,
      ImmutableList<String> unfilteredCompilerFlags,
      boolean supportsFission,
      boolean supportsStartEndLib,
      boolean supportsEmbeddedRuntimes,
      boolean supportsDynamicLinker,
      boolean supportsInterfaceSharedObjects,
      boolean supportsGoldLinker,
      boolean toolchainNeedsPic)
      throws EvalException {
    this.ccToolchainConfigInfo = ccToolchainConfigInfo;
    this.crosstoolTopPathFragment = crosstoolTopPathFragment;
    this.toolchainIdentifier = toolchainIdentifier;
    // Since this field can be derived from `crosstoolInfo`, it is re-derived instead of serialized.
    this.toolchainFeatures =
        new CcToolchainFeatures(ccToolchainConfigInfo, crosstoolTopPathFragment);
    this.toolPaths = toolPaths;
    this.compiler = compiler;
    this.abiGlibcVersion = abiGlibcVersion;
    this.targetCpu = targetCpu;
    this.targetOS = targetOS;
    this.rawBuiltInIncludeDirectories = rawBuiltInIncludeDirectories;
    this.defaultSysroot = defaultSysroot;
    this.runtimeSysroot = runtimeSysroot;
    this.targetLibc = targetLibc;
    this.hostSystemName = hostSystemName;
    this.dynamicLibraryLinkFlags = dynamicLibraryLinkFlags;
    this.legacyLinkOptions = legacyLinkOptions;
    this.legacyLinkOptionsFromLinkingMode = legacyLinkOptionsFromLinkingMode;
    this.legacyLinkOptionsFromCompilationMode = legacyLinkOptionsFromCompilationMode;
    this.testOnlyLinkFlags = testOnlyLinkFlags;
    this.ldOptionsForEmbedding = ldOptionsForEmbedding;
    this.objCopyOptionsForEmbedding = objCopyOptionsForEmbedding;
    this.ccToolchainLabel = ccToolchainLabel;
    this.staticRuntimeLibsLabel = staticRuntimeLibsLabel;
    this.dynamicRuntimeLibsLabel = dynamicRuntimeLibsLabel;
    this.solibDirectory = solibDirectory;
    this.abi = abi;
    this.targetSystemName = targetSystemName;
    this.additionalMakeVariables = additionalMakeVariables;
    this.crosstoolCompilerFlags = crosstoolCompilerFlags;
    this.crosstoolCxxFlags = crosstoolCxxFlags;
    this.cFlagsByCompilationMode = cFlagsByCompilationMode;
    this.cxxFlagsByCompilationMode = cxxFlagsByCompilationMode;
    this.unfilteredCompilerFlags = unfilteredCompilerFlags;
    this.supportsFission = supportsFission;
    this.supportsStartEndLib = supportsStartEndLib;
    this.supportsEmbeddedRuntimes = supportsEmbeddedRuntimes;
    this.supportsDynamicLinker =
        supportsDynamicLinker
            || toolchainFeatures
                .getActivatableNames()
                .contains(CppRuleClasses.DYNAMIC_LINKING_MODE);
    this.supportsInterfaceSharedObjects = supportsInterfaceSharedObjects;
    this.supportsGoldLinker = supportsGoldLinker;
    this.toolchainNeedsPic = toolchainNeedsPic;
  }

  @VisibleForTesting
  static CompilationMode importCompilationMode(CrosstoolConfig.CompilationMode mode) {
    return CompilationMode.valueOf(mode.name());
  }

  // TODO(bazel-team): Remove this once bazel supports all crosstool flags through
  // feature configuration, and all crosstools have been converted.
  public static CToolchain addLegacyFeatures(
      CToolchain toolchain, PathFragment crosstoolTopPathFragment) {
    CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();

    Set<ArtifactCategory> definedCategories = new HashSet<>();
    for (ArtifactNamePattern pattern : toolchainBuilder.getArtifactNamePatternList()) {
      try {
        definedCategories.add(
            ArtifactCategory.valueOf(pattern.getCategoryName().toUpperCase(Locale.ENGLISH)));
      } catch (IllegalArgumentException e) {
        // Invalid category name, will be detected later.
        continue;
      }
    }

    for (ArtifactCategory category : ArtifactCategory.values()) {
      if (!definedCategories.contains(category) && category.getDefaultPrefix() != null
          && category.getDefaultExtension() != null) {
        toolchainBuilder.addArtifactNamePattern(
            ArtifactNamePattern.newBuilder()
                .setCategoryName(category.toString().toLowerCase())
                .setPrefix(category.getDefaultPrefix())
                .setExtension(category.getDefaultExtension())
                .build());
      }
    }

    ImmutableSet<String> featureNames =
        toolchain
            .getFeatureList()
            .stream()
            .map(feature -> feature.getName())
            .collect(ImmutableSet.toImmutableSet());
    if (!featureNames.contains(CppRuleClasses.NO_LEGACY_FEATURES)) {
        String gccToolPath = "DUMMY_GCC_TOOL";
        String linkerToolPath = "DUMMY_LINKER_TOOL";
        String arToolPath = "DUMMY_AR_TOOL";
        String stripToolPath = "DUMMY_STRIP_TOOL";
        for (ToolPath tool : toolchain.getToolPathList()) {
          if (tool.getName().equals(CppConfiguration.Tool.GCC.getNamePart())) {
            gccToolPath = tool.getPath();
            linkerToolPath =
                crosstoolTopPathFragment
                    .getRelative(PathFragment.create(tool.getPath()))
                    .getPathString();
          }
          if (tool.getName().equals(CppConfiguration.Tool.AR.getNamePart())) {
            arToolPath = tool.getPath();
          }
          if (tool.getName().equals(CppConfiguration.Tool.STRIP.getNamePart())) {
            stripToolPath = tool.getPath();
          }
        }

        // TODO(b/30109612): Remove fragile legacyCompileFlags shuffle once there are no legacy
        // crosstools.
        // Existing projects depend on flags from legacy toolchain fields appearing first on the
        // compile command line. 'legacy_compile_flags' feature contains all these flags, and so it
        // needs to appear before other features from {@link CppActionConfigs}.
        CToolchain.Feature legacyCompileFlagsFeature =
            toolchain
                .getFeatureList()
                .stream()
                .filter(feature -> feature.getName().equals(CppRuleClasses.LEGACY_COMPILE_FLAGS))
                .findFirst()
                .orElse(null);
        if (legacyCompileFlagsFeature != null) {
          toolchainBuilder.addFeature(legacyCompileFlagsFeature);
          toolchain = removeLegacyCompileFlagsFeatureFromToolchain(toolchain);
        }

      CppPlatform platform =
          toolchain.getTargetLibc().equals("macosx") ? CppPlatform.MAC : CppPlatform.LINUX;

      toolchainBuilder.addAllActionConfig(
          CppActionConfigs.getLegacyActionConfigs(
              platform,
              gccToolPath,
              arToolPath,
              stripToolPath,
              toolchain.getSupportsInterfaceSharedObjects()));

      toolchainBuilder.addAllFeature(
          CppActionConfigs.getLegacyFeatures(
              platform,
              featureNames,
              linkerToolPath,
              toolchain.getSupportsEmbeddedRuntimes(),
              toolchain.getSupportsInterfaceSharedObjects()));
    }

    toolchainBuilder.mergeFrom(toolchain);

    if (!featureNames.contains(CppRuleClasses.NO_LEGACY_FEATURES)) {
      toolchainBuilder.addAllFeature(
          CppActionConfigs.getFeaturesToAppearLastInFeaturesList(featureNames));
    }

    return toolchainBuilder.build();
  }

  private static CToolchain removeLegacyCompileFlagsFeatureFromToolchain(CToolchain toolchain) {
    FieldDescriptor featuresFieldDescriptor = CToolchain.getDescriptor().findFieldByName("feature");
    return toolchain
        .toBuilder()
        .setField(
            featuresFieldDescriptor,
            toolchain
                .getFeatureList()
                .stream()
                .filter(feature -> !feature.getName().equals(CppRuleClasses.LEGACY_COMPILE_FLAGS))
                .collect(ImmutableList.toImmutableList()))
        .build();
  }

  /** @see CcToolchainProvider#getLegacyLinkOptions(). */
  public ImmutableList<String> getLegacyLinkOptions() {
    return legacyLinkOptions;
  }

  /** @see CcToolchainProvider#configureAllLegacyLinkOptions(CompilationMode, LinkingMode). */
  ImmutableList<String> configureAllLegacyLinkOptions(
      CompilationMode compilationMode, LinkingMode linkingMode) {
    List<String> result = new ArrayList<>();
    result.addAll(legacyLinkOptions);

    result.addAll(legacyLinkOptionsFromCompilationMode.get(compilationMode));
    result.addAll(legacyLinkOptionsFromLinkingMode.get(linkingMode));
    return ImmutableList.copyOf(result);
  }

  /**
   * Returns the {@link CcToolchainConfigInfo} instance that was used to initialize this {@link
   * CppToolchainInfo}.
   */
  public CcToolchainConfigInfo getCcToolchainConfigInfo() {
    return ccToolchainConfigInfo;
  }

  /**
   * Returns the toolchain identifier, which uniquely identifies the compiler version, target libc
   * version, and target cpu.
   */
  public String getToolchainIdentifier() {
    return toolchainIdentifier;
  }

  /** Returns the path of the crosstool. */
  public PathFragment getCrosstoolTopPathFragment() {
    return crosstoolTopPathFragment;
  }

  /** Returns the system name which is required by the toolchain to run. */
  public String getHostSystemName() {
    return hostSystemName;
  }

  @Override
  public String toString() {
    return toolchainIdentifier;
  }

  /** Returns the compiler version string (e.g. "gcc-4.1.1"). */
  public String getCompiler() {
    return compiler;
  }

  /** Returns the libc version string (e.g. "glibc-2.2.2"). */
  public String getTargetLibc() {
    return targetLibc;
  }

  /**
   * Returns the target architecture using blaze-specific constants (e.g. "piii").
   *
   * <p>Equivalent to {@link BuildConfiguration#getCpu()}
   */
  public String getTargetCpu() {
    return targetCpu;
  }

  /** Unused, for compatibility with things internal to Google. */
  public String getTargetOS() {
    return targetOS;
  }

  /**
   * Returns the path fragment that is either absolute or relative to the execution root that can be
   * used to execute the given tool.
   */
  public PathFragment getToolPathFragment(CppConfiguration.Tool tool) {
    return getToolPathFragment(toolPaths, tool);
  }

  /** Returns a label that references the current cc_toolchain target. */
  public Label getCcToolchainLabel() {
    return ccToolchainLabel;
  }

  /**
   * Returns a label that references the library files needed to statically link the C++ runtime
   * (i.e. libgcc.a, libgcc_eh.a, libstdc++.a) for the target architecture.
   */
  public Label getStaticRuntimeLibsLabel() {
    return supportsEmbeddedRuntimes() ? staticRuntimeLibsLabel : null;
  }

  /**
   * Returns a label that references the library files needed to dynamically link the C++ runtime
   * (i.e. libgcc_s.so, libstdc++.so) for the target architecture.
   */
  public Label getDynamicRuntimeLibsLabel() {
    return supportsEmbeddedRuntimes() ? dynamicRuntimeLibsLabel : null;
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

  /**
   * Returns the configured features of the toolchain. Rules should not call this directly, but
   * instead use {@code CcToolchainProvider.getFeatures}.
   */
  public CcToolchainFeatures getFeatures() {
    return toolchainFeatures;
  }

  /** Returns whether the toolchain supports the gold linker. */
  public boolean supportsGoldLinker() {
    return supportsGoldLinker;
  }

  /** Returns whether the toolchain supports the --start-lib/--end-lib options. */
  public boolean supportsStartEndLib() {
    return supportsStartEndLib;
  }

  /** Returns whether the toolchain supports dynamic linking. */
  public boolean supportsDynamicLinker() {
    return supportsDynamicLinker;
  }

  /**
   * Returns whether this toolchain supports interface shared objects.
   *
   * <p>Should be true if this toolchain generates ELF objects.
   */
  public boolean supportsInterfaceSharedObjects() {
    return supportsInterfaceSharedObjects;
  }

  /**
   * Returns whether the toolchain supports linking C/C++ runtime libraries supplied inside the
   * toolchain distribution.
   */
  public boolean supportsEmbeddedRuntimes() {
    return supportsEmbeddedRuntimes;
  }

  /**
   * Returns whether the toolchain supports "Fission" C++ builds, i.e. builds where compilation
   * partitions object code and debug symbols into separate output files.
   */
  public boolean supportsFission() {
    return supportsFission;
  }

  /**
   * Returns whether shared libraries must be compiled with position independent code on this
   * platform.
   */
  public boolean toolchainNeedsPic() {
    return toolchainNeedsPic;
  }

  /**
   * Returns the run time sysroot, which is where the dynamic linker and system libraries are found
   * at runtime. This is usually an absolute path. If the toolchain compiler does not support
   * sysroots, then this method returns <code>null</code>.
   */
  public PathFragment getRuntimeSysroot() {
    return runtimeSysroot;
  }

  /**
   * Returns link options for the specified flag list, combined with universal options for all
   * shared libraries (regardless of link staticness).
   */
  ImmutableList<String> getSharedLibraryLinkOptions(ImmutableList<String> flags) {
    return ImmutableList.<String>builder().addAll(flags).addAll(dynamicLibraryLinkFlags).build();
  }

  /**
   * Returns test-only link options such that certain test-specific features can be configured
   * separately (e.g. lazy binding).
   */
  public ImmutableList<String> getTestOnlyLinkOptions() {
    return testOnlyLinkFlags;
  }

  /**
   * Returns the list of options to be used with 'objcopy' when converting binary files to object
   * files, or {@code null} if this operation is not supported.
   */
  public ImmutableList<String> getObjCopyOptionsForEmbedding() {
    return objCopyOptionsForEmbedding;
  }

  /**
   * Returns the list of options to be used with 'ld' when converting binary files to object files,
   * or {@code null} if this operation is not supported.
   */
  public ImmutableList<String> getLdOptionsForEmbedding() {
    return ldOptionsForEmbedding;
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
    return additionalMakeVariables;
  }

  public final boolean isLLVMCompiler() {
    // TODO(tmsriram): Checking for "llvm" does not handle all the cases.  This
    // is temporary until the crosstool configuration is modified to add fields that
    // indicate which flavor of fdo is being used.
    return toolchainIdentifier.contains("llvm");
  }

  /**
   * Return the name of the directory (relative to the bin directory) that holds mangled links to
   * shared libraries. This name is always set to the '{@code _solib_<cpu_archictecture_name>}.
   */
  public String getSolibDirectory() {
    return solibDirectory;
  }

  /** Returns the architecture component of the GNU System Name */
  public String getGnuSystemArch() {
    if (targetSystemName.indexOf('-') == -1) {
      return targetSystemName;
    }
    return targetSystemName.substring(0, targetSystemName.indexOf('-'));
  }

  /** Returns the GNU System Name */
  public String getTargetGnuSystemName() {
    return targetSystemName;
  }

  public PathFragment getDefaultSysroot() {
    return defaultSysroot;
  }

  /** Returns built-in include directories. */
  public ImmutableList<String> getRawBuiltInIncludeDirectories() {
    return rawBuiltInIncludeDirectories;
  }

  /** Returns compiler flags for C/C++/Asm compilation. */
  public ImmutableList<String> getCompilerFlags() {
    return crosstoolCompilerFlags;
  }

  /** Returns additional compiler flags for C++ compilation. */
  public ImmutableList<String> getCxxFlags() {
    return crosstoolCxxFlags;
  }

  /** Returns compiler flags for C compilation by compilation mode. */
  public ImmutableListMultimap<CompilationMode, String> getCFlagsByCompilationMode() {
    return cFlagsByCompilationMode;
  }

  /** Returns compiler flags for C++ compilation, by compilation mode. */
  public ImmutableListMultimap<CompilationMode, String> getCxxFlagsByCompilationMode() {
    return cxxFlagsByCompilationMode;
  }

  /** Returns unfiltered compiler options for C++ from this toolchain. */
  public ImmutableList<String> getUnfilteredCompilerOptions(@Nullable PathFragment sysroot) {
    if (sysroot == null) {
      return unfilteredCompilerFlags;
    }
    return ImmutableList.<String>builder()
        .add("--sysroot=" + sysroot)
        .addAll(unfilteredCompilerFlags)
        .build();
  }

  private static ImmutableMap<String, String> computeAdditionalMakeVariables(
      CcToolchainConfigInfo ccToolchainConfigInfo) {
    Map<String, String> makeVariablesBuilder = new HashMap<>();
    // The following are to be used to allow some build rules to avoid the limits on stack frame
    // sizes and variable-length arrays. Ensure that these are always set.
    makeVariablesBuilder.put("STACK_FRAME_UNLIMITED", "");
    makeVariablesBuilder.put(CppConfiguration.CC_FLAGS_MAKE_VARIABLE_NAME, "");
    for (Pair<String, String> variable : ccToolchainConfigInfo.getMakeVariables()) {
      makeVariablesBuilder.put(variable.getFirst(), variable.getSecond());
    }
    return ImmutableMap.copyOf(makeVariablesBuilder);
  }

  private static ImmutableMap<String, PathFragment> computeToolPaths(
      CcToolchainConfigInfo ccToolchainConfigInfo, PathFragment crosstoolTopPathFragment)
      throws EvalException {
    Map<String, PathFragment> toolPathsCollector = Maps.newHashMap();
    for (Pair<String, String> tool : ccToolchainConfigInfo.getToolPaths()) {
      String pathStr = tool.getSecond();
      if (!PathFragment.isNormalized(pathStr)) {
        throw new IllegalArgumentException("The include path '" + pathStr + "' is not normalized.");
      }
      PathFragment path = PathFragment.create(pathStr);
      toolPathsCollector.put(tool.getFirst(), crosstoolTopPathFragment.getRelative(path));
    }

    if (toolPathsCollector.isEmpty()) {
      // If no paths are specified, we just use the names of the tools as the path.
      for (CppConfiguration.Tool tool : CppConfiguration.Tool.values()) {
        toolPathsCollector.put(
            tool.getNamePart(), crosstoolTopPathFragment.getRelative(tool.getNamePart()));
      }
    } else {
      Iterable<CppConfiguration.Tool> neededTools =
          Iterables.filter(
              EnumSet.allOf(CppConfiguration.Tool.class),
              tool -> {
                if (tool == CppConfiguration.Tool.DWP) {
                  // When fission is unsupported, don't check for the dwp tool.
                  return ccToolchainConfigInfo.supportsFission();
                } else if (tool == CppConfiguration.Tool.LLVM_PROFDATA) {
                  // TODO(tmsriram): Fix this to check if this is a llvm crosstool
                  // and return true.  This needs changes to crosstool_config.proto.
                  return false;
                } else if (tool == CppConfiguration.Tool.GCOVTOOL
                    || tool == CppConfiguration.Tool.OBJCOPY) {
                  // gcov-tool and objcopy are optional, don't check whether they're present
                  return false;
                } else {
                  return true;
                }
              });
      for (CppConfiguration.Tool tool : neededTools) {
        if (!toolPathsCollector.containsKey(tool.getNamePart())) {
          throw new EvalException(
              Location.BUILTIN, "Tool path for '" + tool.getNamePart() + "' is missing");
        }
      }
    }
    return ImmutableMap.copyOf(toolPathsCollector);
  }

  private static PathFragment getToolPathFragment(
      ImmutableMap<String, PathFragment> toolPaths, CppConfiguration.Tool tool) {
    return toolPaths.get(tool.getNamePart());
  }
}
