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
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcSkyframeSupportFunction.CcSkyframeSupportException;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.rules.cpp.FdoProvider.FdoMode;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
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
   * This file (found under the sysroot) may be unconditionally included in every C/C++ compilation.
   */
  static final PathFragment BUILTIN_INCLUDE_FILE_SUFFIX =
      PathFragment.create("include/stdc-predef.h");

  private static final String SYSROOT_START = "%sysroot%/";
  private static final String WORKSPACE_START = "%workspace%/";
  private static final String CROSSTOOL_START = "%crosstool_top%/";
  private static final String PACKAGE_START = "%package(";
  private static final String PACKAGE_END = ")%";

  /**
   * Returns the profile name with the same file name as fdoProfile and an extension that matches
   * {@link FileType}.
   */
  private static String getLLVMProfileFileName(FdoInputFile fdoProfile, FileType type) {
    if (type.matches(fdoProfile)) {
      return fdoProfile.getBasename();
    } else {
      return FileSystemUtils.removeExtension(fdoProfile.getBasename())
          + type.getExtensions().get(0);
    }
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

  private static PathFragment calculateSysroot(
      CcToolchainAttributesProvider attributes, PathFragment defaultSysroot) {
    TransitiveInfoCollection sysrootTarget = attributes.getLibcTop();
    if (sysrootTarget == null) {
      return defaultSysroot;
    }

    return sysrootTarget.getLabel().getPackageFragment();
  }

  private static Artifact getPrefetchHintsArtifact(
      FdoInputFile prefetchHintsFile, RuleContext ruleContext) {
    if (prefetchHintsFile == null) {
      return null;
    }
    Artifact prefetchHintsArtifact = prefetchHintsFile.getArtifact();
    if (prefetchHintsArtifact != null) {
      return prefetchHintsArtifact;
    }

    prefetchHintsArtifact =
        ruleContext.getUniqueDirectoryArtifact(
            "fdo",
            prefetchHintsFile.getAbsolutePath().getBaseName(),
            ruleContext.getBinOrGenfilesDirectory());
    ruleContext.registerAction(SymlinkAction.toAbsolutePath(
        ruleContext.getActionOwner(),
        PathFragment.create(prefetchHintsFile.getAbsolutePath().getPathString()),
        prefetchHintsArtifact,
        "Symlinking LLVM Cache Prefetch Hints Profile "
            + prefetchHintsFile.getAbsolutePath().getPathString()));
    return prefetchHintsArtifact;
  }

  private static void symlinkTo(
      RuleContext ruleContext,
      Artifact symlink,
      FdoInputFile fdoInputFile,
      String progressMessage) {
    if (fdoInputFile.getArtifact() != null) {
      ruleContext.registerAction(
          SymlinkAction.toArtifact(
              ruleContext.getActionOwner(), fdoInputFile.getArtifact(), symlink, progressMessage));
    } else {
      ruleContext.registerAction(
          SymlinkAction.toAbsolutePath(
              ruleContext.getActionOwner(),
              fdoInputFile.getAbsolutePath(),
              symlink,
              progressMessage));
    }
  }

  /*
   * This function checks the format of the input profile data and converts it to
   * the indexed format (.profdata) if necessary.
   */
  private static Artifact convertLLVMRawProfileToIndexed(
      CcToolchainAttributesProvider attributes,
      FdoInputFile fdoProfile,
      CppToolchainInfo toolchainInfo,
      RuleContext ruleContext) {

    Artifact profileArtifact =
        ruleContext.getUniqueDirectoryArtifact(
            "fdo",
            getLLVMProfileFileName(fdoProfile, CppFileTypes.LLVM_PROFILE),
            ruleContext.getBinOrGenfilesDirectory());

    // If the profile file is already in the desired format, symlink to it and return.
    if (CppFileTypes.LLVM_PROFILE.matches(fdoProfile)) {
      symlinkTo(
          ruleContext,
          profileArtifact,
          fdoProfile,
          "Symlinking LLVM Profile " + fdoProfile.getBasename());
      return profileArtifact;
    }

    Artifact rawProfileArtifact;

    if (CppFileTypes.LLVM_PROFILE_ZIP.matches(fdoProfile)) {
      // Get the zipper binary for unzipping the profile.
      Artifact zipperBinaryArtifact = attributes.getZipper();
      if (zipperBinaryArtifact == null) {
        ruleContext.ruleError("Cannot find zipper binary to unzip the profile");
        return null;
      }

      // TODO(zhayu): find a way to avoid hard-coding cpu architecture here (b/65582760)
      String rawProfileFileName = "fdocontrolz_profile.profraw";
      String cpu = toolchainInfo.getTargetCpu();
      if (!"k8".equals(cpu)) {
        rawProfileFileName = "fdocontrolz_profile-" + cpu + ".profraw";
      }
      rawProfileArtifact =
          ruleContext.getUniqueDirectoryArtifact(
              "fdo", rawProfileFileName, ruleContext.getBinOrGenfilesDirectory());

      // Symlink to the zipped profile file to extract the contents.
      Artifact zipProfileArtifact =
          ruleContext.getUniqueDirectoryArtifact(
              "fdo", fdoProfile.getBasename(), ruleContext.getBinOrGenfilesDirectory());
      symlinkTo(
          ruleContext,
          zipProfileArtifact,
          fdoProfile,
          "Symlinking LLVM ZIP Profile " + fdoProfile.getBasename());

      // Unzip the profile.
      ruleContext.registerAction(
          new SpawnAction.Builder()
              .addInput(zipProfileArtifact)
              .addInput(zipperBinaryArtifact)
              .addOutput(rawProfileArtifact)
              .useDefaultShellEnvironment()
              .setExecutable(zipperBinaryArtifact)
              .setProgressMessage(
                  "LLVMUnzipProfileAction: Generating %s", rawProfileArtifact.prettyPrint())
              .setMnemonic("LLVMUnzipProfileAction")
              .addCommandLine(
                  CustomCommandLine.builder()
                      .addExecPath("xf", zipProfileArtifact)
                      .add(
                          "-d",
                          rawProfileArtifact.getExecPath().getParentDirectory().getSafePathString())
                      .build())
              .build(ruleContext));
    } else {
      rawProfileArtifact =
          ruleContext.getUniqueDirectoryArtifact(
              "fdo",
              getLLVMProfileFileName(fdoProfile, CppFileTypes.LLVM_PROFILE_RAW),
              ruleContext.getBinOrGenfilesDirectory());
      symlinkTo(
          ruleContext,
          rawProfileArtifact,
          fdoProfile,
          "Symlinking LLVM Raw Profile " + fdoProfile.getBasename());
    }

    if (toolchainInfo.getToolPathFragment(Tool.LLVM_PROFDATA) == null) {
      ruleContext.ruleError(
          "llvm-profdata not available with this crosstool, needed for profile conversion");
      return null;
    }

    // Convert LLVM raw profile to indexed format.
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addInput(rawProfileArtifact)
            .addTransitiveInputs(attributes.getCrosstoolMiddleman())
            .addOutput(profileArtifact)
            .useDefaultShellEnvironment()
            .setExecutable(toolchainInfo.getToolPathFragment(Tool.LLVM_PROFDATA))
            .setProgressMessage("LLVMProfDataAction: Generating %s", profileArtifact.prettyPrint())
            .setMnemonic("LLVMProfDataAction")
            .addCommandLine(
                CustomCommandLine.builder()
                    .add("merge")
                    .add("-o")
                    .addExecPath(profileArtifact)
                    .addExecPath(rawProfileArtifact)
                    .build())
            .build(ruleContext));

    return profileArtifact;
  }

  static CcToolchainProvider getCcToolchainProvider(
      RuleContext ruleContext, CcToolchainAttributesProvider attributes)
      throws RuleErrorException, InterruptedException {
    BuildConfiguration configuration = Preconditions.checkNotNull(ruleContext.getConfiguration());
    CppConfiguration cppConfiguration =
        Preconditions.checkNotNull(configuration.getFragment(CppConfiguration.class));

    PathFragment fdoZip = null;
    FdoInputFile fdoInputFile = null;
    FdoInputFile prefetchHints = null;
    Artifact protoProfileArtifact = null;
    boolean allowInference = true;
    if (configuration.getCompilationMode() == CompilationMode.OPT) {
      if (cppConfiguration.getFdoPrefetchHintsLabel() != null) {
        FdoPrefetchHintsProvider provider = attributes.getFdoPrefetch();
        prefetchHints = provider.getInputFile();
      }
      if (cppConfiguration.getFdoPath() != null) {
        fdoZip = cppConfiguration.getFdoPath();
      } else if (cppConfiguration.getFdoOptimizeLabel() != null) {
        // If fdo_profile rule is used, do not allow inferring proto.profile from AFDO profile.
        allowInference = false;
        FdoProfileProvider fdoProfileProvider = attributes.getFdoOptimizeProvider();
        if (fdoProfileProvider != null) {
          fdoInputFile = fdoProfileProvider.getInputFile();
          protoProfileArtifact = fdoProfileProvider.getProtoProfileArtifact();
        } else {
          fdoInputFile = fdoInputFileFromArtifacts(ruleContext, attributes);
        }
      } else if (cppConfiguration.getFdoProfileLabel() != null) {
        FdoProfileProvider fdoProvider = attributes.getFdoProfileProvider();
        fdoInputFile = fdoProvider.getInputFile();
        protoProfileArtifact = fdoProvider.getProtoProfileArtifact();
      }
    }

    if (ruleContext.hasErrors()) {
      return null;
    }

    // Is there a toolchain proto available on the target directly?
    CToolchain toolchain = parseToolchainFromAttributes(ruleContext, attributes);
    Label ccToolchainSuiteLabelIfNeeded = null;
    if (toolchain == null && cppConfiguration.getCrosstoolFromCcToolchainProtoAttribute() == null) {
      ccToolchainSuiteLabelIfNeeded = cppConfiguration.getCrosstoolTop();
    }

    SkyKey ccSupportKey = CcSkyframeSupportValue.key(fdoZip, ccToolchainSuiteLabelIfNeeded);

    SkyFunction.Environment skyframeEnv = ruleContext.getAnalysisEnvironment().getSkyframeEnv();
    CcSkyframeSupportValue ccSkyframeSupportValue;
    try {
      ccSkyframeSupportValue =
          (CcSkyframeSupportValue)
              skyframeEnv.getValueOrThrow(ccSupportKey, CcSkyframeSupportException.class);
    } catch (CcSkyframeSupportException e) {
      ruleContext.throwWithRuleError(e.getMessage());
      throw new IllegalStateException("Should not be reached");
    }
    if (skyframeEnv.valuesMissing()) {
      return null;
    }

    if (fdoZip != null) {
      // fdoZip should be set if the profile is a path, fdoInputFile if it is an artifact, but never
      // both
      Preconditions.checkState(fdoInputFile == null);
      fdoInputFile =
          FdoInputFile.fromAbsolutePath(ccSkyframeSupportValue.getFdoZipPath().asFragment());
    }

    FdoMode fdoMode;
    if (fdoInputFile == null) {
      fdoMode = FdoMode.OFF;
    } else if (CppFileTypes.GCC_AUTO_PROFILE.matches(fdoInputFile)) {
      fdoMode = FdoMode.AUTO_FDO;
    } else if (CppFileTypes.XBINARY_PROFILE.matches(fdoInputFile)) {
      fdoMode = FdoMode.XBINARY_FDO;
    } else if (CppFileTypes.LLVM_PROFILE.matches(fdoInputFile)) {
      fdoMode = FdoMode.LLVM_FDO;
    } else if (CppFileTypes.LLVM_PROFILE_RAW.matches(fdoInputFile)) {
      fdoMode = FdoMode.LLVM_FDO;
    } else if (CppFileTypes.LLVM_PROFILE_ZIP.matches(fdoInputFile)) {
      fdoMode = FdoMode.LLVM_FDO;
    } else {
      ruleContext.ruleError("invalid extension for FDO profile file.");
      return null;
    }

    CppToolchainInfo toolchainInfo =
        getCppToolchainInfo(
            ruleContext, cppConfiguration, attributes, ccSkyframeSupportValue, toolchain);

    String purposePrefix = attributes.getPurposePrefix();
    String runtimeSolibDirBase = attributes.getRuntimeSolibDirBase();
    final PathFragment runtimeSolibDir =
        configuration.getBinFragment().getRelative(runtimeSolibDirBase);

    // Static runtime inputs.
    TransitiveInfoCollection staticRuntimeLibDep =
        selectDep(attributes.getStaticRuntimesLibs(), toolchainInfo.getStaticRuntimeLibsLabel());
    final NestedSet<Artifact> staticRuntimeLinkInputs;
    final Artifact staticRuntimeLinkMiddleman;
    if (toolchainInfo.supportsEmbeddedRuntimes()) {
      staticRuntimeLinkInputs =
          staticRuntimeLibDep.getProvider(FileProvider.class).getFilesToBuild();
    } else {
      staticRuntimeLinkInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    if (!staticRuntimeLinkInputs.isEmpty()) {
      NestedSet<Artifact> staticRuntimeLinkMiddlemanSet =
          CompilationHelper.getAggregatingMiddleman(
              ruleContext, purposePrefix + "static_runtime_link", staticRuntimeLibDep);
      staticRuntimeLinkMiddleman =
          staticRuntimeLinkMiddlemanSet.isEmpty()
              ? null
              : Iterables.getOnlyElement(staticRuntimeLinkMiddlemanSet);
    } else {
      staticRuntimeLinkMiddleman = null;
    }

    Preconditions.checkState(
        (staticRuntimeLinkMiddleman == null) == staticRuntimeLinkInputs.isEmpty());

    // Dynamic runtime inputs.
    TransitiveInfoCollection dynamicRuntimeLibDep =
        selectDep(attributes.getDynamicRuntimesLibs(), toolchainInfo.getDynamicRuntimeLibsLabel());
    NestedSet<Artifact> dynamicRuntimeLinkSymlinks;
    List<Artifact> dynamicRuntimeLinkInputs = new ArrayList<>();
    Artifact dynamicRuntimeLinkMiddleman;
    if (toolchainInfo.supportsEmbeddedRuntimes()) {
      NestedSetBuilder<Artifact> dynamicRuntimeLinkSymlinksBuilder = NestedSetBuilder.stableOrder();
      for (Artifact artifact :
          dynamicRuntimeLibDep.getProvider(FileProvider.class).getFilesToBuild()) {
        if (CppHelper.SHARED_LIBRARY_FILETYPES.matches(artifact.getFilename())) {
          dynamicRuntimeLinkInputs.add(artifact);
          dynamicRuntimeLinkSymlinksBuilder.add(
              SolibSymlinkAction.getCppRuntimeSymlink(
                  ruleContext,
                  artifact,
                  toolchainInfo.getSolibDirectory(),
                  runtimeSolibDirBase,
                  configuration));
        }
      }
      dynamicRuntimeLinkSymlinks = dynamicRuntimeLinkSymlinksBuilder.build();
    } else {
      dynamicRuntimeLinkSymlinks = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
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
        (dynamicRuntimeLinkMiddleman == null) == dynamicRuntimeLinkSymlinks.isEmpty());

    CcCompilationContext.Builder ccCompilationContextBuilder =
        new CcCompilationContext.Builder(ruleContext);
    CppModuleMap moduleMap = createCrosstoolModuleMap(attributes);
    if (moduleMap != null) {
      ccCompilationContextBuilder.setCppModuleMap(moduleMap);
    }
    final CcCompilationContext ccCompilationContext = ccCompilationContextBuilder.build();

    NestedSetBuilder<Pair<String, String>> coverageEnvironment = NestedSetBuilder.compileOrder();

    PathFragment sysroot = calculateSysroot(attributes, toolchainInfo.getDefaultSysroot());

    ImmutableList.Builder<PathFragment> builtInIncludeDirectoriesBuilder = ImmutableList.builder();
    for (String s : toolchainInfo.getRawBuiltInIncludeDirectories()) {
      try {
        builtInIncludeDirectoriesBuilder.add(
            resolveIncludeDir(s, sysroot, toolchainInfo.getCrosstoolTopPathFragment()));
      } catch (InvalidConfigurationException e) {
        ruleContext.ruleError(e.getMessage());
      }
    }
    ImmutableList<PathFragment> builtInIncludeDirectories =
        builtInIncludeDirectoriesBuilder.build();

    coverageEnvironment.add(
        Pair.of(
            "COVERAGE_GCOV_PATH", toolchainInfo.getToolPathFragment(Tool.GCOV).getPathString()));
    if (cppConfiguration.getFdoInstrument() != null) {
      coverageEnvironment.add(Pair.of("FDO_DIR", cppConfiguration.getFdoInstrument()));
    }

    // This tries to convert LLVM profiles to the indexed format if necessary.
    Artifact profileArtifact = null;
    if (fdoMode == FdoMode.LLVM_FDO) {
      profileArtifact =
          convertLLVMRawProfileToIndexed(attributes, fdoInputFile, toolchainInfo, ruleContext);
      if (ruleContext.hasErrors()) {
        return null;
      }
    } else if (fdoMode == FdoMode.AUTO_FDO || fdoMode == FdoMode.XBINARY_FDO) {
      profileArtifact =
          ruleContext.getUniqueDirectoryArtifact(
              "fdo", fdoInputFile.getBasename(), ruleContext.getBinOrGenfilesDirectory());
      symlinkTo(
          ruleContext,
          profileArtifact,
          fdoInputFile,
          "Symlinking FDO profile " + fdoInputFile.getBasename());
    }

    Artifact prefetchHintsArtifact = getPrefetchHintsArtifact(prefetchHints, ruleContext);

    // This relies on ccSkyframeSupportValue containing a Path if the FDO profile is not an
    // artifact. Not nice, but will do until the Path#exists() call on proto.profile can go away and
    // we can forget about this nightmare.
    Path fdoZipPath =
        fdoInputFile == null
            ? null
            : fdoInputFile.getArtifact() != null
                ? fdoInputFile.getArtifact().getPath()
                : ccSkyframeSupportValue.getFdoZipPath();

    reportInvalidOptions(ruleContext, toolchainInfo);
    return new CcToolchainProvider(
        getToolchainForSkylark(toolchainInfo),
        cppConfiguration,
        toolchainInfo,
        cppConfiguration.getCrosstoolTopPathFragment(),
        attributes.getCrosstool(),
        attributes.getFullInputsForCrosstool(),
        attributes.getCompile(),
        attributes.getCompileWithoutIncludes(),
        attributes.getStrip(),
        attributes.getObjcopy(),
        attributes.getAs(),
        attributes.getAr(),
        attributes.getFullInputsForLink(),
        attributes.getIfsoBuilder(),
        attributes.getDwp(),
        attributes.getCoverage(),
        attributes.getLibc(),
        staticRuntimeLinkInputs,
        staticRuntimeLinkMiddleman,
        dynamicRuntimeLinkSymlinks,
        dynamicRuntimeLinkMiddleman,
        runtimeSolibDir,
        ccCompilationContext,
        attributes.isSupportsParamFiles(),
        attributes.isSupportsHeaderParsing(),
        getBuildVariables(
            ruleContext,
            attributes,
            toolchainInfo.getDefaultSysroot(),
            attributes.getAdditionalBuildVariables()),
        getBuiltinIncludes(attributes.getLibc()),
        coverageEnvironment.build(),
        toolchainInfo.supportsInterfaceSharedObjects()
            ? attributes.getLinkDynamicLibraryTool()
            : null,
        builtInIncludeDirectories,
        sysroot,
        fdoMode,
        new FdoProvider(
            fdoZipPath,
            fdoMode,
            cppConfiguration.getFdoInstrument(),
            profileArtifact,
            prefetchHintsArtifact,
            protoProfileArtifact,
            allowInference),
        cppConfiguration.useLLVMCoverageMapFormat(),
        configuration.isCodeCoverageEnabled(),
        configuration.isHostConfiguration());
  }

  private static FdoInputFile fdoInputFileFromArtifacts(
      RuleContext ruleContext, CcToolchainAttributesProvider attributes) {
    ImmutableList<Artifact> fdoArtifacts = attributes.getFdoOptimizeArtifacts();
    if (fdoArtifacts.size() != 1) {
      ruleContext.ruleError("--fdo_optimize does not point to a single target");
      return null;
    }

    Artifact fdoArtifact = fdoArtifacts.get(0);
    if (!fdoArtifact.isSourceArtifact()) {
      ruleContext.ruleError("--fdo_optimize points to a target that is not an input file");
      return null;
    }

    Label fdoLabel = attributes.getFdoOptimize().getLabel();
    if (!fdoLabel
        .getPackageIdentifier()
        .getPathUnderExecRoot()
        .getRelative(fdoLabel.getName())
        .equals(fdoArtifact.getExecPath())) {
      ruleContext.ruleError("--fdo_optimize points to a target that is not an input file");
      return null;
    }

    return FdoInputFile.fromArtifact(fdoArtifact);
  }

  /** Finds an appropriate {@link CppToolchainInfo} for this target. */
  private static CppToolchainInfo getCppToolchainInfo(
      RuleContext ruleContext,
      CppConfiguration cppConfiguration,
      CcToolchainAttributesProvider attributes,
      CcSkyframeSupportValue ccSkyframeSupportValue,
      CToolchain toolchainFromCcToolchainAttribute)
      throws RuleErrorException {

    if (cppConfiguration.enableCcToolchainConfigInfoFromSkylark()) {
      // Attempt to obtain CppToolchainInfo from the 'toolchain_config' attribute of cc_toolchain.
      CcToolchainConfigInfo configInfo = attributes.getCcToolchainConfigInfo();

      if (configInfo != null) {
        try {
          return CppToolchainInfo.create(
              ruleContext.getRepository().getPathUnderExecRoot(),
              ruleContext.getLabel(),
              configInfo,
              cppConfiguration.disableLegacyCrosstoolFields(),
              cppConfiguration.disableCompilationModeFlags(),
              cppConfiguration.disableLinkingModeFlags());
        } catch (InvalidConfigurationException e) {
          throw ruleContext.throwWithRuleError(e.getMessage());
        }
      }
    }

    // Attempt to find a toolchain based on the target attributes, not the configuration.
    CToolchain toolchain = toolchainFromCcToolchainAttribute;
    if (toolchain == null) {
      toolchain =
          getToolchainFromAttributes(
              ruleContext, attributes, cppConfiguration, ccSkyframeSupportValue);
    }

    // If we found a toolchain, use it.
    try {
      toolchain =
          CppToolchainInfo.addLegacyFeatures(
              toolchain, cppConfiguration.getCrosstoolTopPathFragment());
      CcToolchainConfigInfo ccToolchainConfigInfo = CcToolchainConfigInfo.fromToolchain(toolchain);
      return CppToolchainInfo.create(
          cppConfiguration.getCrosstoolTopPathFragment(),
          attributes.getCcToolchainLabel(),
          ccToolchainConfigInfo,
          cppConfiguration.disableLegacyCrosstoolFields(),
          cppConfiguration.disableCompilationModeFlags(),
          cppConfiguration.disableLinkingModeFlags());
    } catch (InvalidConfigurationException e) {
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

  private static void reportInvalidOptions(RuleContext ruleContext, CppToolchainInfo toolchain) {
    CppOptions options = ruleContext.getConfiguration().getOptions().get(CppOptions.class);
    CppConfiguration config = ruleContext.getFragment(CppConfiguration.class);
    if (options.fissionModes.contains(config.getCompilationMode())
        && !toolchain.supportsFission()) {
      ruleContext.ruleWarning(
          "Fission is not supported by this crosstool.  Please use a "
              + "supporting crosstool to enable fission");
    }
    if (options.buildTestDwp
        && !(toolchain.supportsFission() && config.fissionIsActiveForCurrentCompilationMode())) {
      ruleContext.ruleWarning(
          "Test dwp file requested, but Fission is not enabled.  To generate a "
              + "dwp for the test executable, use '--fission=yes' with a toolchain that supports "
              + "Fission to build statically.");
    }
  }

  @Nullable
  private static CToolchain getToolchainFromAttributes(
      RuleContext ruleContext,
      CcToolchainAttributesProvider attributes,
      CppConfiguration cppConfiguration,
      CcSkyframeSupportValue ccSkyframeSupportValue)
      throws RuleErrorException {
    try {
      CrosstoolRelease crosstoolRelease;
      if (cppConfiguration.getCrosstoolFromCcToolchainProtoAttribute() != null) {
        // We have cc_toolchain_suite.proto attribute set, let's use it
        crosstoolRelease = cppConfiguration.getCrosstoolFromCcToolchainProtoAttribute();
      } else {
        // We use the proto from the CROSSTOOL file
        crosstoolRelease = ccSkyframeSupportValue.getCrosstoolRelease();
      }

      return CToolchainSelectionUtils.selectCToolchain(
          attributes.getToolchainIdentifier(),
          attributes.getCpu(),
          attributes.getCompiler(),
          cppConfiguration.getTransformedCpuFromOptions(),
          cppConfiguration.getCompilerFromOptions(),
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
      if (artifact.getExecPath().endsWith(BUILTIN_INCLUDE_FILE_SUFFIX)) {
        result.add(artifact);
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

  static TransitiveInfoCollection selectDep(
      ImmutableList<? extends TransitiveInfoCollection> deps, Label label) {
    for (TransitiveInfoCollection dep : deps) {
      if (dep.getLabel().equals(label)) {
        return dep;
      }
    }

    return deps.get(0);
  }

  /**
   * Returns {@link CcToolchainVariables} instance with build variables that only depend on the
   * toolchain.
   *
   * @param ruleContext the rule context
   * @param defaultSysroot the default sysroot
   * @param additionalBuildVariables
   * @throws RuleErrorException if there are configuration errors making it impossible to resolve
   *     certain build variables of this toolchain
   */
  private static final CcToolchainVariables getBuildVariables(
      RuleContext ruleContext,
      CcToolchainAttributesProvider attributes,
      PathFragment defaultSysroot,
      CcToolchainVariables additionalBuildVariables) {
    CcToolchainVariables.Builder variables = new CcToolchainVariables.Builder();

    CppConfiguration cppConfiguration =
        Preconditions.checkNotNull(ruleContext.getFragment(CppConfiguration.class));
    String minOsVersion = cppConfiguration.getMinimumOsVersion();
    if (minOsVersion != null) {
      variables.addStringVariable(CcCommon.MINIMUM_OS_VERSION_VARIABLE_NAME, minOsVersion);
    }

    PathFragment sysroot = calculateSysroot(attributes, defaultSysroot);
    if (sysroot != null) {
      variables.addStringVariable(CcCommon.SYSROOT_VARIABLE_NAME, sysroot.getPathString());
    }

    variables.addAllNonTransitive(additionalBuildVariables);

    return variables.build();
  }
}
