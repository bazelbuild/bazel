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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.rules.cpp.CppActionConfigs.CppPlatform;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain.ArtifactNamePattern;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath;
import com.google.protobuf.Descriptors.FieldDescriptor;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Information describing the C++ compiler derived from the CToolchain proto.
 *
 * <p>This wrapper class is used to re-plumb information so that it's eventually accessed through
 * {@link CcToolchainProvider} instead of {@link CppConfiguration}.
 */
@AutoCodec
@Immutable
public final class CppToolchainInfo {
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

  private final Label ccToolchainLabel;
  private final String solibDirectory;
  private final String abi;
  private final String targetSystemName;

  private final ImmutableMap<String, String> additionalMakeVariables;
  // TODO(b/65151735): Remove when cc_flags is entirely from features.
  private final String legacyCcFlagsMakeVariable;

  /**
   * Creates a CppToolchainInfo from CROSSTOOL info encapsulated in {@link CcToolchainConfigInfo}.
   */
  public static CppToolchainInfo create(
      Label toolchainLabel, CcToolchainConfigInfo ccToolchainConfigInfo) throws EvalException {
    ImmutableMap<String, PathFragment> toolPaths =
        computeToolPaths(ccToolchainConfigInfo, getToolsDirectory(toolchainLabel));
    PathFragment defaultSysroot =
        CppConfiguration.computeDefaultSysroot(ccToolchainConfigInfo.getBuiltinSysroot());

    return new CppToolchainInfo(
        ccToolchainConfigInfo.getToolchainIdentifier(),
        new CcToolchainFeatures(ccToolchainConfigInfo, getToolsDirectory(toolchainLabel)),
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
        toolchainLabel,
        "_solib_" + ccToolchainConfigInfo.getTargetCpu(),
        ccToolchainConfigInfo.getAbiVersion(),
        ccToolchainConfigInfo.getTargetSystemName(),
        computeAdditionalMakeVariables(ccToolchainConfigInfo),
        computeLegacyCcFlagsMakeVariable(ccToolchainConfigInfo));
  }

  @AutoCodec.Instantiator
  CppToolchainInfo(
      String toolchainIdentifier,
      CcToolchainFeatures toolchainFeatures,
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
      Label ccToolchainLabel,
      String solibDirectory,
      String abi,
      String targetSystemName,
      ImmutableMap<String, String> additionalMakeVariables,
      String legacyCcFlagsMakeVariable)
      throws EvalException {
    this.toolchainIdentifier = toolchainIdentifier;
    // Since this field can be derived from `crosstoolInfo`, it is re-derived instead of serialized.
    this.toolchainFeatures = toolchainFeatures;
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
    this.ccToolchainLabel = ccToolchainLabel;
    this.solibDirectory = solibDirectory;
    this.abi = abi;
    this.targetSystemName = targetSystemName;
    this.additionalMakeVariables = additionalMakeVariables;
    this.legacyCcFlagsMakeVariable = legacyCcFlagsMakeVariable;
  }

  @VisibleForTesting
  static CompilationMode importCompilationMode(CrosstoolConfig.CompilationMode mode) {
    return CompilationMode.valueOf(mode.name());
  }

  // TODO(bazel-team): Remove this once bazel supports all crosstool flags through
  // feature configuration, and all crosstools have been converted.
  public static CToolchain addLegacyFeatures(
      CToolchain toolchain,
      boolean doNotSplitLinkingCmdLine,
      PathFragment crosstoolTopPathFragment) {
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
      if (!definedCategories.contains(category)
          && category.getDefaultPrefix() != null
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
        toolchain.getFeatureList().stream()
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
      if (featureNames.contains(CppRuleClasses.LEGACY_COMPILE_FLAGS)) {
        CToolchain.Feature legacyCompileFlags =
            toolchain.getFeatureList().stream()
                .filter(feature -> feature.getName().equals(CppRuleClasses.LEGACY_COMPILE_FLAGS))
                .findFirst()
                .get();
        if (legacyCompileFlags != null) {
          toolchainBuilder.addFeature(legacyCompileFlags);
        }
      }
      if (featureNames.contains(CppRuleClasses.DEFAULT_COMPILE_FLAGS)) {
        CToolchain.Feature defaultCompileFlags =
            toolchain.getFeatureList().stream()
                .filter(feature -> feature.getName().equals(CppRuleClasses.DEFAULT_COMPILE_FLAGS))
                .findFirst()
                .get();
        if (defaultCompileFlags != null) {
          toolchainBuilder.addFeature(defaultCompileFlags);
        }
      }
      toolchain = removeSpecialFeatureFromToolchain(toolchain);

      CppPlatform platform =
          toolchain.getTargetLibc().equals(CppActionConfigs.MACOS_TARGET_LIBC)
              ? CppPlatform.MAC
              : CppPlatform.LINUX;

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
              toolchain.getSupportsInterfaceSharedObjects(),
              doNotSplitLinkingCmdLine));
    }

    toolchainBuilder.mergeFrom(toolchain);

    if (!featureNames.contains(CppRuleClasses.NO_LEGACY_FEATURES)) {
      toolchainBuilder.addAllFeature(
          CppActionConfigs.getFeaturesToAppearLastInFeaturesList(
              featureNames, doNotSplitLinkingCmdLine));
    }

    return toolchainBuilder.build();
  }

  private static CToolchain removeSpecialFeatureFromToolchain(CToolchain toolchain) {
    FieldDescriptor featuresFieldDescriptor = CToolchain.getDescriptor().findFieldByName("feature");
    return toolchain
        .toBuilder()
        .setField(
            featuresFieldDescriptor,
            toolchain.getFeatureList().stream()
                .filter(feature -> !feature.getName().equals(CppRuleClasses.LEGACY_COMPILE_FLAGS))
                .filter(feature -> !feature.getName().equals(CppRuleClasses.DEFAULT_COMPILE_FLAGS))
                .collect(ImmutableList.toImmutableList()))
        .build();
  }

  /**
   * Returns the toolchain identifier, which uniquely identifies the compiler version, target libc
   * version, and target cpu.
   */
  public String getToolchainIdentifier() {
    return toolchainIdentifier;
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

  /**
   * Returns the run time sysroot, which is where the dynamic linker and system libraries are found
   * at runtime. This is usually an absolute path. If the toolchain compiler does not support
   * sysroots, then this method returns <code>null</code>.
   */
  public PathFragment getRuntimeSysroot() {
    return runtimeSysroot;
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

  private static ImmutableMap<String, String> computeAdditionalMakeVariables(
      CcToolchainConfigInfo ccToolchainConfigInfo) {
    Map<String, String> makeVariablesBuilder = new HashMap<>();
    // The following are to be used to allow some build rules to avoid the limits on stack frame
    // sizes and variable-length arrays.
    // These variables are initialized here, but may be overridden by the getMakeVariables() checks.
    makeVariablesBuilder.put("STACK_FRAME_UNLIMITED", "");
    makeVariablesBuilder.put(CppConfiguration.CC_FLAGS_MAKE_VARIABLE_NAME, "");
    for (Pair<String, String> variable : ccToolchainConfigInfo.getMakeVariables()) {
      makeVariablesBuilder.put(variable.getFirst(), variable.getSecond());
    }
    makeVariablesBuilder.remove(CppConfiguration.CC_FLAGS_MAKE_VARIABLE_NAME);

    return ImmutableMap.copyOf(makeVariablesBuilder);
  }

  // TODO(b/65151735): Remove when cc_flags is entirely from features.
  private static String computeLegacyCcFlagsMakeVariable(
      CcToolchainConfigInfo ccToolchainConfigInfo) {
    String legacyCcFlags = "";
    // Needs to ensure the last value with the name is used, to match the previous logic in
    // computeAdditionalMakeVariables.
    for (Pair<String, String> variable : ccToolchainConfigInfo.getMakeVariables()) {
      if (variable.getFirst().equals(CppConfiguration.CC_FLAGS_MAKE_VARIABLE_NAME)) {
        legacyCcFlags = variable.getSecond();
      }
    }

    return legacyCcFlags;
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
                  // TODO(hlopko): check dwp tool in analysis when per_object_debug_info is enabled.
                  return false;
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

  public PathFragment getToolsDirectory() {
    return getToolsDirectory(ccToolchainLabel);
  }

  static PathFragment getToolsDirectory(Label ccToolchainLabel) {
    return ccToolchainLabel.getPackageIdentifier().getPathUnderExecRoot();
  }
}
