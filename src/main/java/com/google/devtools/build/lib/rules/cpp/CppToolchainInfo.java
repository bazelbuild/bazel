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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CppActionConfigs.CppPlatform;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.FlagList;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain.ArtifactNamePattern;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LinkingModeFlags;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath;
import com.google.protobuf.Descriptors.FieldDescriptor;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Information describing the C++ compiler derived from the CToolchain proto.
 *
 * <p>This wrapper class is used to re-plumb information so that it's eventually accessed through
 * {@link CcToolchainProvider} instead of {@link CppConfiguration}.
 */
@Immutable
public final class CppToolchainInfo {

  private final PathFragment crosstoolTopPathFragment;
  private final String toolchainIdentifier;
  private final CcToolchainFeatures toolchainFeatures;

  private final ImmutableMap<String, PathFragment> toolPaths;
  private final String compiler;
  private final PathFragment ldExecutable;
  private final String abiGlibcVersion;

  private final String targetCpu;
  private final String targetOS;

  private final ImmutableList<String> rawBuiltInIncludeDirectories;
  private final PathFragment defaultSysroot;
  private final PathFragment runtimeSysroot;

  private final String targetLibc;
  private final String hostSystemName;
  private final FlagList dynamicLibraryLinkFlags;
  private final ImmutableList<String> commonLinkOptions;
  private final ImmutableListMultimap<LinkingMode, String> linkOptionsFromLinkingMode;
  private final ImmutableListMultimap<LipoMode, String> linkOptionsFromLipoMode;
  private final ImmutableListMultimap<CompilationMode, String> linkOptionsFromCompilationMode;
  private final ImmutableList<String> testOnlyLinkFlags;
  private final ImmutableList<String> ldOptions;
  private final ImmutableList<String> objcopyOptions;

  private final Label staticRuntimeLibsLabel;
  private final Label dynamicRuntimeLibsLabel;
  private final String solibDirectory;
  private final String abi;
  private final String targetSystemName;

  private final ImmutableMap<String, String> additionalMakeVariables;

  private final boolean supportsFission;
  private final boolean supportsStartEndLib;
  private final boolean supportsEmbeddedRuntimes;
  private final boolean supportsDynamicLinker;
  private final boolean supportsInterfaceSharedObjects;
  private final boolean supportsGoldLinker;
  private final boolean toolchainNeedsPic;

  /** Creates a CppToolchainInfo from a toolchain. */
  public CppToolchainInfo(
      CToolchain cToolchain, PathFragment crosstoolTopPathFragment, Label toolchainLabel)
      throws InvalidConfigurationException {
    CToolchain toolchain = cToolchain;
    this.crosstoolTopPathFragment = crosstoolTopPathFragment;
    this.hostSystemName = toolchain.getHostSystemName();
    this.compiler = toolchain.getCompiler();
    this.targetCpu = toolchain.getTargetCpu();
    this.targetSystemName = toolchain.getTargetSystemName();
    this.targetLibc = toolchain.getTargetLibc();
    this.targetOS = toolchain.getCcTargetOs();

    try {
      this.staticRuntimeLibsLabel =
          toolchainLabel.getRelative(
              toolchain.hasStaticRuntimesFilegroup()
                  ? toolchain.getStaticRuntimesFilegroup()
                  : "static-runtime-libs-" + targetCpu);
      this.dynamicRuntimeLibsLabel =
          toolchainLabel.getRelative(
              toolchain.hasDynamicRuntimesFilegroup()
                  ? toolchain.getDynamicRuntimesFilegroup()
                  : "dynamic-runtime-libs-" + targetCpu);
    } catch (LabelSyntaxException e) {
      // All of the above label.getRelative() calls are valid labels, and the crosstool_top
      // was already checked earlier in the process.
      throw new AssertionError(e);
    }

    // Needs to be set before the first call to isLLVMCompiler().
    this.toolchainIdentifier = toolchain.getToolchainIdentifier();

    CrosstoolConfigurationIdentifier crosstoolConfig =
        CrosstoolConfigurationIdentifier.fromToolchain(toolchain);
    Preconditions.checkState(crosstoolConfig.getCpu().equals(targetCpu));
    Preconditions.checkState(crosstoolConfig.getCompiler().equals(compiler));
    Preconditions.checkState(crosstoolConfig.getLibc().equals(targetLibc));

    this.solibDirectory = "_solib_" + targetCpu;

    this.supportsEmbeddedRuntimes = toolchain.getSupportsEmbeddedRuntimes();
    toolchain = addLegacyFeatures(toolchain);
    this.toolchainFeatures = new CcToolchainFeatures(toolchain);
    this.supportsGoldLinker = toolchain.getSupportsGoldLinker();
    this.supportsStartEndLib = toolchain.getSupportsStartEndLib();
    this.supportsInterfaceSharedObjects = toolchain.getSupportsInterfaceSharedObjects();
    this.supportsFission = toolchain.getSupportsFission();

    Map<String, PathFragment> toolPathsCollector = Maps.newHashMap();
    for (CrosstoolConfig.ToolPath tool : toolchain.getToolPathList()) {
      PathFragment path = PathFragment.create(tool.getPath());
      if (!path.isNormalized()) {
        throw new IllegalArgumentException(
            "The include path '" + tool.getPath() + "' is not normalized.");
      }
      toolPathsCollector.put(tool.getName(), crosstoolTopPathFragment.getRelative(path));
    }

    if (toolPathsCollector.isEmpty()) {
      // If no paths are specified, we just use the names of the tools as the path.
      for (Tool tool : Tool.values()) {
        toolPathsCollector.put(
            tool.getNamePart(), crosstoolTopPathFragment.getRelative(tool.getNamePart()));
      }
    } else {
      Iterable<Tool> neededTools =
          Iterables.filter(
              EnumSet.allOf(Tool.class),
              tool -> {
                if (tool == Tool.DWP) {
                  // When fission is unsupported, don't check for the dwp tool.
                  return supportsFission();
                } else if (tool == Tool.LLVM_PROFDATA) {
                  // TODO(tmsriram): Fix this to check if this is a llvm crosstool
                  // and return true.  This needs changes to crosstool_config.proto.
                  return false;
                } else if (tool == Tool.GCOVTOOL || tool == Tool.OBJCOPY) {
                  // gcov-tool and objcopy are optional, don't check whether they're present
                  return false;
                } else {
                  return true;
                }
              });
      for (Tool tool : neededTools) {
        if (!toolPathsCollector.containsKey(tool.getNamePart())) {
          throw new IllegalArgumentException(
              "Tool path for '" + tool.getNamePart() + "' is missing");
        }
      }
    }
    this.toolPaths = ImmutableMap.copyOf(toolPathsCollector);

    ImmutableListMultimap.Builder<CompilationMode, String> linkOptionsFromCompilationModeBuilder =
        ImmutableListMultimap.builder();
    for (CrosstoolConfig.CompilationModeFlags flags : toolchain.getCompilationModeFlagsList()) {
      // Remove this when CROSSTOOL files no longer contain 'coverage'.
      if (flags.getMode() == CrosstoolConfig.CompilationMode.COVERAGE) {
        continue;
      }
      CompilationMode realmode = CppConfiguration.importCompilationMode(flags.getMode());
      linkOptionsFromCompilationModeBuilder.putAll(realmode, flags.getLinkerFlagList());
    }
    linkOptionsFromCompilationMode = linkOptionsFromCompilationModeBuilder.build();

    ImmutableListMultimap.Builder<LipoMode, String> linkOptionsFromLipoModeBuilder =
        ImmutableListMultimap.builder();
    for (CrosstoolConfig.LipoModeFlags flags : toolchain.getLipoModeFlagsList()) {
      LipoMode realmode = flags.getMode();
      linkOptionsFromLipoModeBuilder.putAll(realmode, flags.getLinkerFlagList());
    }
    linkOptionsFromLipoMode = linkOptionsFromLipoModeBuilder.build();

    ImmutableListMultimap.Builder<LinkingMode, String> linkOptionsFromLinkingModeBuilder =
        ImmutableListMultimap.builder();

    // If a toolchain supports dynamic libraries at all, there must be at least one
    // of the following:
    // - a "DYNAMIC" section in linking_mode_flags (even if no flags are needed)
    // - a non-empty list in one of the dynamicLibraryLinkerFlag fields
    // If none of the above contain data, then the toolchain can't do dynamic linking.
    boolean haveDynamicMode = false;

    for (LinkingModeFlags flags : toolchain.getLinkingModeFlagsList()) {
      LinkingMode realmode = CppConfiguration.importLinkingMode(flags.getMode());
      if (realmode == LinkingMode.DYNAMIC) {
        haveDynamicMode = true;
      }
      linkOptionsFromLinkingModeBuilder.putAll(realmode, flags.getLinkerFlagList());
    }
    linkOptionsFromLinkingMode = linkOptionsFromLinkingModeBuilder.build();

    this.commonLinkOptions = ImmutableList.copyOf(toolchain.getLinkerFlagList());
    List<String> linkerFlagList = toolchain.getDynamicLibraryLinkerFlagList();
    List<CToolchain.OptionalFlag> optionalLinkerFlagList =
        toolchain.getOptionalDynamicLibraryLinkerFlagList();
    if (!linkerFlagList.isEmpty() || !optionalLinkerFlagList.isEmpty()) {
      haveDynamicMode = true;
    }
    this.supportsDynamicLinker = haveDynamicMode;
    dynamicLibraryLinkFlags =
        new FlagList(
            ImmutableList.copyOf(linkerFlagList),
            CppConfiguration.convertOptionalOptions(optionalLinkerFlagList),
            ImmutableList.<String>of());

    this.objcopyOptions = ImmutableList.copyOf(toolchain.getObjcopyEmbedFlagList());
    this.ldOptions = ImmutableList.copyOf(toolchain.getLdEmbedFlagList());

    this.abi = toolchain.getAbiVersion();
    this.abiGlibcVersion = toolchain.getAbiLibcVersion();

    this.defaultSysroot = CppConfiguration.computeDefaultSysroot(toolchain);

    this.rawBuiltInIncludeDirectories =
        ImmutableList.copyOf(toolchain.getCxxBuiltinIncludeDirectoryList());

    // The default value for optional string attributes is the empty string.

    // The runtime sysroot should really be set from --grte_top. However, currently libc has no
    // way to set the sysroot. The CROSSTOOL file does set the runtime sysroot, in the
    // builtin_sysroot field. This implies that you can not arbitrarily mix and match Crosstool
    // and libc versions, you must always choose compatible ones.
    runtimeSysroot = defaultSysroot;

    this.testOnlyLinkFlags = ImmutableList.copyOf(toolchain.getTestOnlyLinkerFlagList());
    this.toolchainNeedsPic = toolchain.getNeedsPic();

    Map<String, String> makeVariablesBuilder = new HashMap<>();
    // The following are to be used to allow some build rules to avoid the limits on stack frame
    // sizes and variable-length arrays. Ensure that these are always set.
    makeVariablesBuilder.put("STACK_FRAME_UNLIMITED", "");
    makeVariablesBuilder.put(CppConfiguration.CC_FLAGS_MAKE_VARIABLE_NAME, "");
    for (CrosstoolConfig.MakeVariable variable : toolchain.getMakeVariableList()) {
      makeVariablesBuilder.put(variable.getName(), variable.getValue());
    }
    this.additionalMakeVariables = ImmutableMap.copyOf(makeVariablesBuilder);

    this.ldExecutable = getToolPathFragment(CppConfiguration.Tool.LD);
  }

  // TODO(bazel-team): Remove this once bazel supports all crosstool flags through
  // feature configuration, and all crosstools have been converted.
  private CToolchain addLegacyFeatures(CToolchain toolchain) {
    CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();

    Set<ArtifactCategory> definedCategories = new HashSet<>();
    for (ArtifactNamePattern pattern : toolchainBuilder.getArtifactNamePatternList()) {
      try {
        definedCategories.add(ArtifactCategory.valueOf(pattern.getCategoryName().toUpperCase()));
      } catch (IllegalArgumentException e) {
        // Invalid category name, will be detected later.
        continue;
      }
    }

    for (ArtifactCategory category : ArtifactCategory.values()) {
      if (!definedCategories.contains(category) && category.getDefaultPattern() != null) {
        toolchainBuilder.addArtifactNamePattern(
            ArtifactNamePattern.newBuilder()
                .setCategoryName(category.toString().toLowerCase())
                .setPattern(category.getDefaultPattern())
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
      try {
        String gccToolPath = "DUMMY_GCC_TOOL";
        String linkerToolPath = "DUMMY_LINKER_TOOL";
        String arToolPath = "DUMMY_AR_TOOL";
        String stripToolPath = "DUMMY_STRIP_TOOL";
        for (ToolPath tool : toolchain.getToolPathList()) {
          if (tool.getName().equals(Tool.GCC.getNamePart())) {
            gccToolPath = tool.getPath();
            linkerToolPath =
                crosstoolTopPathFragment
                    .getRelative(PathFragment.create(tool.getPath()))
                    .getPathString();
          }
          if (tool.getName().equals(Tool.AR.getNamePart())) {
            arToolPath = tool.getPath();
          }
          if (tool.getName().equals(Tool.STRIP.getNamePart())) {
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

        TextFormat.merge(
            CppActionConfigs.getCppActionConfigs(
                getTargetLibc().equals("macosx") ? CppPlatform.MAC : CppPlatform.LINUX,
                featureNames,
                gccToolPath,
                linkerToolPath,
                arToolPath,
                stripToolPath,
                supportsEmbeddedRuntimes,
                toolchain.getSupportsInterfaceSharedObjects()),
            toolchainBuilder);
      } catch (ParseException e) {
        // Can only happen if we change the proto definition without changing our
        // configuration above.
        throw new RuntimeException(e);
      }
    }

    toolchainBuilder.mergeFrom(toolchain);

    if (!featureNames.contains(CppRuleClasses.NO_LEGACY_FEATURES)) {
      try {
        TextFormat.merge(
            CppActionConfigs.getFeaturesToAppearLastInToolchain(featureNames), toolchainBuilder);
      } catch (ParseException e) {
        // Can only happen if we change the proto definition without changing our
        // configuration above.
        throw new RuntimeException(e);
      }
    }
    return toolchainBuilder.build();
  }

  private CToolchain removeLegacyCompileFlagsFeatureFromToolchain(CToolchain toolchain) {
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

  ImmutableList<String> configureLinkerOptions(
      CompilationMode compilationMode,
      LipoMode lipoMode,
      LinkingMode linkingMode,
      PathFragment ldExecutable) {
    List<String> result = new ArrayList<>();
    result.addAll(commonLinkOptions);

    result.addAll(linkOptionsFromCompilationMode.get(compilationMode));
    result.addAll(linkOptionsFromLipoMode.get(lipoMode));
    result.addAll(linkOptionsFromLinkingMode.get(linkingMode));
    return ImmutableList.copyOf(result);
  }

  /**
   * Returns the toolchain identifier, which uniquely identifies the compiler version, target libc
   * version, target cpu, and LIPO linkage.
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
   *
   * <p>Note that you must not use this method to get the linker location, but use {@link
   * #getLdExecutable} instead!
   */
  public PathFragment getToolPathFragment(CppConfiguration.Tool tool) {
    return toolPaths.get(tool.getNamePart());
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
  ImmutableList<String> getSharedLibraryLinkOptions(FlagList flags, Iterable<String> features) {
    return ImmutableList.<String>builder()
        .addAll(flags.evaluate(features))
        .addAll(dynamicLibraryLinkFlags.evaluate(features))
        .build();
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
    return objcopyOptions;
  }

  /**
   * Returns the list of options to be used with 'ld' when converting binary files to object files,
   * or {@code null} if this operation is not supported.
   */
  public ImmutableList<String> getLdOptionsForEmbedding() {
    return ldOptions;
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

  public PathFragment getLdExecutable() {
    return ldExecutable;
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
}
