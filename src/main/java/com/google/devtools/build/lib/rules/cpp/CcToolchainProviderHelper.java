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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.CompilationHelper;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcSkyframeCrosstoolSupportFunction.CcSkyframeCrosstoolSupportException;
import com.google.devtools.build.lib.rules.cpp.CcToolchain.AdditionalBuildVariablesComputer;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/** Helper responsible for creating CcToolchainProvider */
public class CcToolchainProviderHelper {

  /**
   * These files (found under the sysroot) may be unconditionally included in every C/C++
   * compilation.
   */
  static final ImmutableList<PathFragment> BUILTIN_INCLUDE_FILE_SUFFIXES =
      ImmutableList.of(PathFragment.create("include/stdc-predef.h"));

  private static final String SYSROOT_START = "%sysroot%/";
  private static final String WORKSPACE_START = "%workspace%/";
  private static final String CROSSTOOL_START = "%crosstool_top%/";
  private static final String PACKAGE_START = "%package(";
  private static final String PACKAGE_END = ")%";

  public static CcToolchainProvider getCcToolchainProvider(
      RuleContext ruleContext,
      CcToolchainAttributesProvider attributes,
      CrosstoolRelease crosstoolFromCcToolchainSuiteProtoAttribute)
      throws RuleErrorException, InterruptedException {
    BuildConfiguration configuration = Preconditions.checkNotNull(ruleContext.getConfiguration());
    CppConfiguration cppConfiguration =
        Preconditions.checkNotNull(configuration.getFragment(CppConfiguration.class));

    CToolchain toolchain = null;
    CrosstoolRelease crosstoolFromCrosstoolFile = null;

    if (cppConfiguration.disableCrosstool() && attributes.getCcToolchainConfigInfo() == null) {
      ruleContext.ruleError(
          "cc_toolchain.toolchain_config attribute must be specified. See "
              + "https://github.com/bazelbuild/bazel/issues/7320 for details.");
    }

    if (attributes.getCcToolchainConfigInfo() == null) {
      // Is there a toolchain proto available on the target directly?
      toolchain = parseToolchainFromAttributes(ruleContext, attributes);
      PackageIdentifier packageWithCrosstoolInIt = null;
      if (toolchain == null && crosstoolFromCcToolchainSuiteProtoAttribute == null) {
        packageWithCrosstoolInIt = ruleContext.getLabel().getPackageIdentifier();
      }
      if (packageWithCrosstoolInIt != null) {
        SkyKey crosstoolKey = CcSkyframeCrosstoolSupportValue.key(packageWithCrosstoolInIt);
        SkyFunction.Environment skyframeEnv = ruleContext.getAnalysisEnvironment().getSkyframeEnv();
        try {
          CcSkyframeCrosstoolSupportValue ccSkyframeCrosstoolSupportValue =
              (CcSkyframeCrosstoolSupportValue)
                  skyframeEnv.getValueOrThrow(
                      crosstoolKey, CcSkyframeCrosstoolSupportException.class);
          if (skyframeEnv.valuesMissing()) {
            return null;
          }
          crosstoolFromCrosstoolFile = ccSkyframeCrosstoolSupportValue.getCrosstoolRelease();
        } catch (CcSkyframeCrosstoolSupportException e) {
          throw ruleContext.throwWithRuleError(e.getMessage());
        }
      }
    }

    CppToolchainInfo toolchainInfo =
        getCppToolchainInfo(
            ruleContext,
            cppConfiguration.getTransformedCpuFromOptions(),
            cppConfiguration.getCompilerFromOptions(),
            attributes,
            crosstoolFromCrosstoolFile,
            toolchain,
            crosstoolFromCcToolchainSuiteProtoAttribute);

    FdoContext fdoContext =
        FdoHelper.getFdoContext(
            ruleContext, attributes, configuration, cppConfiguration, toolchainInfo);
    if (fdoContext == null) {
      return null;
    }

    String purposePrefix = attributes.getPurposePrefix();
    String runtimeSolibDirBase = attributes.getRuntimeSolibDirBase();
    final PathFragment runtimeSolibDir =
        configuration.getBinFragment().getRelative(runtimeSolibDirBase);

    // Static runtime inputs.
    TransitiveInfoCollection staticRuntimeLib = attributes.getStaticRuntimeLib();
    final NestedSet<Artifact> staticRuntimeLinkInputs;
    final Artifact staticRuntimeLinkMiddleman;

    if (staticRuntimeLib != null) {
      staticRuntimeLinkInputs = staticRuntimeLib.getProvider(FileProvider.class).getFilesToBuild();
      if (!staticRuntimeLinkInputs.isEmpty()) {
        NestedSet<Artifact> staticRuntimeLinkMiddlemanSet =
            CompilationHelper.getAggregatingMiddleman(
                ruleContext, purposePrefix + "static_runtime_link", staticRuntimeLib);
        staticRuntimeLinkMiddleman =
            staticRuntimeLinkMiddlemanSet.isEmpty()
                ? null
                : Iterables.getOnlyElement(staticRuntimeLinkMiddlemanSet);
      } else {
        staticRuntimeLinkMiddleman = null;
      }
      Preconditions.checkState(
          (staticRuntimeLinkMiddleman == null) == staticRuntimeLinkInputs.isEmpty());
    } else {
      staticRuntimeLinkInputs = null;
      staticRuntimeLinkMiddleman = null;
    }

    // Dynamic runtime inputs.
    TransitiveInfoCollection dynamicRuntimeLib = attributes.getDynamicRuntimeLib();
    NestedSet<Artifact> dynamicRuntimeLinkSymlinks;
    List<Artifact> dynamicRuntimeLinkInputs = new ArrayList<>();
    Artifact dynamicRuntimeLinkMiddleman;
    if (dynamicRuntimeLib != null) {
      NestedSetBuilder<Artifact> dynamicRuntimeLinkSymlinksBuilder = NestedSetBuilder.stableOrder();
      for (Artifact artifact :
          dynamicRuntimeLib.getProvider(FileProvider.class).getFilesToBuild()) {
        if (CppHelper.SHARED_LIBRARY_FILETYPES.matches(artifact.getFilename())) {
          dynamicRuntimeLinkInputs.add(artifact);
          dynamicRuntimeLinkSymlinksBuilder.add(
              SolibSymlinkAction.getCppRuntimeSymlink(
                  ruleContext, artifact, toolchainInfo.getSolibDirectory(), runtimeSolibDirBase));
        }
      }
      if (dynamicRuntimeLinkInputs.isEmpty()) {
        dynamicRuntimeLinkSymlinks = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      } else {
        dynamicRuntimeLinkSymlinks = dynamicRuntimeLinkSymlinksBuilder.build();
      }

    } else {
      dynamicRuntimeLinkSymlinks = null;
    }

    if (!dynamicRuntimeLinkInputs.isEmpty()) {
      List<Artifact> dynamicRuntimeLinkMiddlemanSet =
          CppHelper.getAggregatingMiddlemanForCppRuntimes(
              ruleContext,
              purposePrefix + "dynamic_runtime_link",
              dynamicRuntimeLinkInputs,
              toolchainInfo.getSolibDirectory(),
              runtimeSolibDirBase,
              configuration);
      dynamicRuntimeLinkMiddleman =
          dynamicRuntimeLinkMiddlemanSet.isEmpty()
              ? null
              : Iterables.getOnlyElement(dynamicRuntimeLinkMiddlemanSet);
    } else {
      dynamicRuntimeLinkMiddleman = null;
    }

    Preconditions.checkState(
        (dynamicRuntimeLinkMiddleman == null)
            == (dynamicRuntimeLinkSymlinks == null || dynamicRuntimeLinkSymlinks.isEmpty()));

    CcCompilationContext.Builder ccCompilationContextBuilder =
        CcCompilationContext.builder(
            ruleContext, ruleContext.getConfiguration(), ruleContext.getLabel());
    CppModuleMap moduleMap = createCrosstoolModuleMap(attributes);
    if (moduleMap != null) {
      ccCompilationContextBuilder.setCppModuleMap(moduleMap);
    }
    final CcCompilationContext ccCompilationContext = ccCompilationContextBuilder.build();

    PathFragment sysroot =
        calculateSysroot(attributes.getLibcTopLabel(), toolchainInfo.getDefaultSysroot());
    PathFragment targetSysroot =
        calculateSysroot(attributes.getTargetLibcTopLabel(), toolchainInfo.getDefaultSysroot());

    ImmutableList.Builder<PathFragment> builtInIncludeDirectoriesBuilder = ImmutableList.builder();
    for (String s : toolchainInfo.getRawBuiltInIncludeDirectories()) {
      try {
        builtInIncludeDirectoriesBuilder.add(
            resolveIncludeDir(s, sysroot, toolchainInfo.getToolsDirectory()));
      } catch (InvalidConfigurationException e) {
        ruleContext.ruleError(e.getMessage());
      }
    }
    ImmutableList<PathFragment> builtInIncludeDirectories =
        builtInIncludeDirectoriesBuilder.build();

    return new CcToolchainProvider(
        getToolchainForSkylark(toolchainInfo),
        cppConfiguration,
        toolchainInfo,
        toolchainInfo.getToolsDirectory(),
        attributes.getAllFiles(),
        attributes.getFullInputsForCrosstool(),
        attributes.getCompilerFiles(),
        attributes.getCompilerFilesWithoutIncludes(),
        attributes.getStripFiles(),
        attributes.getObjcopyFiles(),
        attributes.getAsFiles(),
        attributes.getArFiles(),
        attributes.getFullInputsForLink(),
        attributes.getIfsoBuilder(),
        attributes.getDwpFiles(),
        attributes.getCoverage(),
        attributes.getLibc(),
        attributes.getTargetLibc(),
        staticRuntimeLinkInputs,
        staticRuntimeLinkMiddleman,
        dynamicRuntimeLinkSymlinks,
        dynamicRuntimeLinkMiddleman,
        runtimeSolibDir,
        ccCompilationContext,
        attributes.isSupportsParamFiles(),
        attributes.isSupportsHeaderParsing(),
        attributes.getAdditionalBuildVariablesComputer(),
        getBuildVariables(
            ruleContext.getConfiguration().getOptions(),
            cppConfiguration,
            sysroot,
            attributes.getAdditionalBuildVariablesComputer()),
        getBuiltinIncludes(attributes.getLibc()),
        getBuiltinIncludes(attributes.getTargetLibc()),
        attributes.getLinkDynamicLibraryTool(),
        builtInIncludeDirectories,
        sysroot,
        targetSysroot,
        fdoContext,
        configuration.isHostConfiguration(),
        attributes.getLicensesProvider());
  }

  /**
   * Resolve the given include directory.
   *
   * <p>If it starts with %sysroot%/, that part is replaced with the actual sysroot.
   *
   * <p>If it starts with %workspace%/, that part is replaced with the empty string (essentially
   * making it relative to the build directory).
   *
   * <p>If it starts with %crosstool_top%/ or is any relative path, it is interpreted relative to
   * the crosstool top. The use of assumed-crosstool-relative specifications is considered
   * deprecated, and all such uses should eventually be replaced by "%crosstool_top%/".
   *
   * <p>If it is of the form %package(@repository//my/package)%/folder, then it is interpreted as
   * the named folder in the appropriate package. All of the normal package syntax is supported. The
   * /folder part is optional.
   *
   * <p>It is illegal if it starts with a % and does not match any of the above forms to avoid
   * accidentally silently ignoring misspelled prefixes.
   *
   * <p>If it is absolute, it remains unchanged.
   */
  static PathFragment resolveIncludeDir(
      String s, PathFragment sysroot, PathFragment crosstoolTopPathFragment)
      throws InvalidConfigurationException {
    PathFragment pathPrefix;
    String pathString;
    int packageEndIndex = s.indexOf(PACKAGE_END);
    if (packageEndIndex != -1 && s.startsWith(PACKAGE_START)) {
      String packageString = s.substring(PACKAGE_START.length(), packageEndIndex);
      try {
        pathPrefix = PackageIdentifier.parse(packageString).getSourceRoot();
      } catch (LabelSyntaxException e) {
        throw new InvalidConfigurationException("The package '" + packageString + "' is not valid");
      }
      int pathStartIndex = packageEndIndex + PACKAGE_END.length();
      if (pathStartIndex + 1 < s.length()) {
        if (s.charAt(pathStartIndex) != '/') {
          throw new InvalidConfigurationException(
              "The path in the package for '" + s + "' is not valid");
        }
        pathString = s.substring(pathStartIndex + 1, s.length());
      } else {
        pathString = "";
      }
    } else if (s.startsWith(SYSROOT_START)) {
      if (sysroot == null) {
        throw new InvalidConfigurationException(
            "A %sysroot% prefix is only allowed if the " + "default_sysroot option is set");
      }
      pathPrefix = sysroot;
      pathString = s.substring(SYSROOT_START.length(), s.length());
    } else if (s.startsWith(WORKSPACE_START)) {
      pathPrefix = PathFragment.EMPTY_FRAGMENT;
      pathString = s.substring(WORKSPACE_START.length(), s.length());
    } else {
      pathPrefix = crosstoolTopPathFragment;
      if (s.startsWith(CROSSTOOL_START)) {
        pathString = s.substring(CROSSTOOL_START.length(), s.length());
      } else if (s.startsWith("%")) {
        throw new InvalidConfigurationException(
            "The include path '" + s + "' has an " + "unrecognized %prefix%");
      } else {
        pathString = s;
      }
    }

    if (!PathFragment.isNormalized(pathString)) {
      throw new InvalidConfigurationException("The include path '" + s + "' is not normalized.");
    }
    PathFragment path = PathFragment.create(pathString);
    return pathPrefix.getRelative(path);
  }

  private static String getSkylarkValueForTool(Tool tool, CppToolchainInfo cppToolchainInfo) {
    PathFragment toolPath = cppToolchainInfo.getToolPathFragment(tool);
    return toolPath != null ? toolPath.getPathString() : "";
  }

  private static ImmutableMap<String, Object> getToolchainForSkylark(
      CppToolchainInfo cppToolchainInfo) {
    return ImmutableMap.<String, Object>builder()
        .put("objcopy_executable", getSkylarkValueForTool(Tool.OBJCOPY, cppToolchainInfo))
        .put("compiler_executable", getSkylarkValueForTool(Tool.GCC, cppToolchainInfo))
        .put("preprocessor_executable", getSkylarkValueForTool(Tool.CPP, cppToolchainInfo))
        .put("nm_executable", getSkylarkValueForTool(Tool.NM, cppToolchainInfo))
        .put("objdump_executable", getSkylarkValueForTool(Tool.OBJDUMP, cppToolchainInfo))
        .put("ar_executable", getSkylarkValueForTool(Tool.AR, cppToolchainInfo))
        .put("strip_executable", getSkylarkValueForTool(Tool.STRIP, cppToolchainInfo))
        .put("ld_executable", getSkylarkValueForTool(Tool.LD, cppToolchainInfo))
        .build();
  }

  private static PathFragment calculateSysroot(Label libcTopLabel, PathFragment defaultSysroot) {
    if (libcTopLabel == null) {
      return defaultSysroot;
    }

    return libcTopLabel.getPackageFragment();
  }

  /** Finds an appropriate {@link CppToolchainInfo} for this target. */
  private static CppToolchainInfo getCppToolchainInfo(
      RuleContext ruleContext,
      String cpuFromOptions,
      String compilerFromOptions,
      CcToolchainAttributesProvider attributes,
      CrosstoolRelease crosstoolFromCrosstoolFile,
      CToolchain toolchainFromCcToolchainAttribute,
      CrosstoolRelease crosstoolFromCcToolchainSuiteProtoAttribute)
      throws RuleErrorException, InterruptedException {

    CcToolchainConfigInfo configInfo = attributes.getCcToolchainConfigInfo();

    if (configInfo != null) {
      try {
        return CppToolchainInfo.create(ruleContext.getLabel(), configInfo);
      } catch (EvalException e) {
        throw ruleContext.throwWithRuleError(e.getMessage());
      }
    }

    // Attempt to find a toolchain based on the target attributes, not the configuration.
    CToolchain toolchain = toolchainFromCcToolchainAttribute;
    if (toolchain == null) {
      toolchain =
          getToolchainFromAttributes(
              ruleContext,
              attributes,
              cpuFromOptions,
              compilerFromOptions,
              crosstoolFromCcToolchainSuiteProtoAttribute,
              crosstoolFromCrosstoolFile);
    }

    // If we found a toolchain, use it.
    try {
      toolchain =
          CppToolchainInfo.addLegacyFeatures(
              toolchain,
              ruleContext
                  .getAnalysisEnvironment()
                  .getSkylarkSemantics()
                  .incompatibleDoNotSplitLinkingCmdline(),
              CppToolchainInfo.getToolsDirectory(attributes.getCcToolchainLabel()));
      CcToolchainConfigInfo ccToolchainConfigInfo = CcToolchainConfigInfo.fromToolchain(toolchain);
      return CppToolchainInfo.create(attributes.getCcToolchainLabel(), ccToolchainConfigInfo);
    } catch (EvalException e) {
      throw ruleContext.throwWithRuleError(e.getMessage());
    }
  }

  @Nullable
  private static CToolchain parseToolchainFromAttributes(
      RuleContext ruleContext, CcToolchainAttributesProvider attributes) throws RuleErrorException {
    String protoAttribute = StringUtil.emptyToNull(attributes.getProto());
    if (protoAttribute == null) {
      return null;
    }

    CToolchain.Builder builder = CToolchain.newBuilder();
    try {
      TextFormat.merge(protoAttribute, builder);
      return builder.build();
    } catch (ParseException e) {
      throw ruleContext.throwWithAttributeError("proto", "Could not parse CToolchain data");
    }
  }

  @Nullable
  private static CToolchain getToolchainFromAttributes(
      RuleContext ruleContext,
      CcToolchainAttributesProvider attributes,
      String cpuFromOptions,
      String compilerFromOptions,
      CrosstoolRelease crosstoolFromCcToolchainSuiteProtoAttribute,
      CrosstoolRelease crosstoolFromCrosstoolFile)
      throws RuleErrorException {
    try {
      CrosstoolRelease crosstoolRelease;
      if (crosstoolFromCcToolchainSuiteProtoAttribute != null) {
        // We have cc_toolchain_suite.proto attribute set, let's use it
        crosstoolRelease = crosstoolFromCcToolchainSuiteProtoAttribute;
      } else {
        // We use the proto from the CROSSTOOL file
        crosstoolRelease = crosstoolFromCrosstoolFile;
      }

      return CToolchainSelectionUtils.selectCToolchain(
          attributes.getToolchainIdentifier(),
          attributes.getCpu(),
          attributes.getCompiler(),
          cpuFromOptions,
          compilerFromOptions,
          crosstoolRelease);
    } catch (InvalidConfigurationException e) {
      ruleContext.throwWithRuleError(
          String.format("Error while selecting cc_toolchain: %s", e.getMessage()));
      return null;
    }
  }

  private static ImmutableList<Artifact> getBuiltinIncludes(NestedSet<Artifact> libc) {
    ImmutableList.Builder<Artifact> result = ImmutableList.builder();
    for (Artifact artifact : libc) {
      for (PathFragment suffix : BUILTIN_INCLUDE_FILE_SUFFIXES) {
        if (artifact.getExecPath().endsWith(suffix)) {
          result.add(artifact);
          break;
        }
      }
    }

    return result.build();
  }

  private static CppModuleMap createCrosstoolModuleMap(CcToolchainAttributesProvider attributes) {
    if (attributes.getModuleMap() == null) {
      return null;
    }
    Artifact moduleMapArtifact = attributes.getModuleMapArtifact();
    if (moduleMapArtifact == null) {
      return null;
    }
    return new CppModuleMap(moduleMapArtifact, "crosstool");
  }

  /**
   * Returns {@link CcToolchainVariables} instance with build variables that only depend on the
   * toolchain.
   *
   * @throws RuleErrorException if there are configuration errors making it impossible to resolve
   *     certain build variables of this toolchain
   */
  static CcToolchainVariables getBuildVariables(
      BuildOptions buildOptions,
      CppConfiguration cppConfiguration,
      PathFragment sysroot,
      AdditionalBuildVariablesComputer additionalBuildVariablesComputer) {
    CcToolchainVariables.Builder variables = CcToolchainVariables.builder();

    String minOsVersion = cppConfiguration.getMinimumOsVersion();
    if (minOsVersion != null) {
      variables.addStringVariable(CcCommon.MINIMUM_OS_VERSION_VARIABLE_NAME, minOsVersion);
    }

    if (sysroot != null) {
      variables.addStringVariable(CcCommon.SYSROOT_VARIABLE_NAME, sysroot.getPathString());
    }

    variables.addAllNonTransitive(additionalBuildVariablesComputer.apply(buildOptions));

    return variables.build();
  }
}
