// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.rules.android;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.AndroidNdkCrosstools;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.AndroidNdkCrosstools.NdkCrosstoolsException;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.ApiLevel;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkPaths;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkRelease;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.StlImpl;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.StlImpls;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.skyframe.FileSymlinkException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Implementation of the {@code android_ndk_repository} rule.
 */
public class AndroidNdkRepositoryFunction extends RepositoryFunction {

  private static final String TOOLCHAIN_NAME_PREFIX = "toolchain-";
  private static final String PATH_ENV_VAR = "ANDROID_NDK_HOME";
  
  private static final class CrosstoolStlPair {

    private final CrosstoolRelease crosstoolRelease;
    private final StlImpl stlImpl;

    private CrosstoolStlPair(CrosstoolRelease crosstoolRelease, StlImpl stlImpl) {
      this.crosstoolRelease = crosstoolRelease;
      this.stlImpl = stlImpl;
    }
  }

  @Override
  public boolean isLocal(Rule rule) {
    return true;
  }

  @Override
  public RepositoryDirectoryValue.Builder fetch(Rule rule, Path outputDirectory,
      BlazeDirectories directories, Environment env, Map<String, String> markerData)
      throws InterruptedException, RepositoryFunctionException {
    prepareLocalRepositorySymlinkTree(rule, outputDirectory);
    WorkspaceAttributeMapper attributes = WorkspaceAttributeMapper.of(rule);
    PathFragment pathFragment;
    if (attributes.isAttributeValueExplicitlySpecified("path")) {
      pathFragment = getTargetPath(rule, directories.getWorkspace());
    } else if (clientEnvironment.containsKey(PATH_ENV_VAR)) {
      pathFragment = getAndroidNdkHomeEnvironmentVar(directories.getWorkspace(), clientEnvironment);
    } else {
      throw new RepositoryFunctionException(
          new EvalException(
              rule.getLocation(),
              "Either the path attribute of android_ndk_repository or the ANDROID_NDK_HOME "
                  + " environment variable must be set."),
          Transience.PERSISTENT);
    }

    Path ndkSymlinkTreeDirectory = outputDirectory.getRelative("ndk");
    try {
      ndkSymlinkTreeDirectory.createDirectory();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    if (!symlinkLocalRepositoryContents(ndkSymlinkTreeDirectory,
        directories.getOutputBase().getFileSystem().getPath(pathFragment))) {
      return null;
    }

    String ruleName = rule.getName();

    NdkRelease ndkRelease = getNdkRelease(outputDirectory, env);
    if (env.valuesMissing()) {
      return null;
    }

    ApiLevel apiLevel;
    try {
      String apiLevelAttr = attributes.get("api_level", Type.INTEGER).toString();
      apiLevel = ApiLevel.getApiLevel(ndkRelease, env.getListener(), ruleName, apiLevelAttr);
    } catch (EvalException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }

    if (!ndkRelease.isValid) {
      env.getListener().handle(Event.warn(String.format(
          "The revision of the Android NDK given in android_ndk_repository rule '%s' could not be "
          + "determined (the revision string found is '%s'). "
          + "Defaulting to Android NDK revision %s", ruleName, ndkRelease.rawRelease,
          AndroidNdkCrosstools.LATEST_KNOWN_REVISION)));
    }

    if (!AndroidNdkCrosstools.isKnownNDKRevision(ndkRelease)) {
      env.getListener().handle(Event.warn(String.format(
          "Bazel Android NDK crosstools are based on Android NDK revision %s. "
          + "The revision of the Android NDK given in android_ndk_repository rule '%s' is '%s'",
          AndroidNdkCrosstools.LATEST_KNOWN_REVISION, ruleName, ndkRelease.rawRelease)));
    }
    
    ImmutableList.Builder<CrosstoolStlPair> crosstoolsAndStls = ImmutableList.builder();
    try {

      String hostPlatform = AndroidNdkCrosstools.getHostPlatform(ndkRelease);
      NdkPaths ndkPaths = new NdkPaths(ruleName, hostPlatform, apiLevel);

      for (StlImpl stlImpl : StlImpls.get(ndkPaths)) {

        CrosstoolRelease crosstoolRelease = AndroidNdkCrosstools.create(
            ndkRelease,
            ndkPaths,
            stlImpl,
            hostPlatform);

        crosstoolsAndStls.add(new CrosstoolStlPair(crosstoolRelease, stlImpl));
      }

    } catch (NdkCrosstoolsException e) {
      throw new RepositoryFunctionException(new IOException(e), Transience.PERSISTENT);
    }

    String buildFile = createBuildFile(ruleName, crosstoolsAndStls.build());
    writeBuildFile(outputDirectory, buildFile);
    return RepositoryDirectoryValue.builder().setPath(outputDirectory);
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return AndroidNdkRepositoryRule.class;
  }

  private static PathFragment getAndroidNdkHomeEnvironmentVar(
      Path workspace, Map<String, String> env) {
    return workspace.getRelative(new PathFragment(env.get(PATH_ENV_VAR))).asFragment();
  }

  private static String createBuildFile(String ruleName, List<CrosstoolStlPair> crosstools) {

    String buildFileTemplate = getTemplate("android_ndk_build_file_template.txt");
    String ccToolchainSuiteTemplate = getTemplate("android_ndk_cc_toolchain_suite_template.txt");
    String ccToolchainTemplate = getTemplate("android_ndk_cc_toolchain_template.txt");
    String stlFilegroupTemplate = getTemplate("android_ndk_stl_filegroup_template.txt");

    StringBuilder ccToolchainSuites = new StringBuilder();
    StringBuilder ccToolchainRules = new StringBuilder();
    StringBuilder stlFilegroups = new StringBuilder();
    for (CrosstoolStlPair crosstoolStlPair : crosstools) {

      // Create the cc_toolchain_suite rule
      CrosstoolRelease crosstool = crosstoolStlPair.crosstoolRelease;

      StringBuilder toolchainMap = new StringBuilder();
      for (CToolchain toolchain : crosstool.getToolchainList()) {
        toolchainMap.append(String.format("      \"%s|%s\": \":%s\",\n",
            toolchain.getTargetCpu(),
            toolchain.getCompiler(),
            toolchain.getToolchainIdentifier()));
      }

      String toolchainName = createToolchainName(crosstoolStlPair.stlImpl.getName());
      
      ccToolchainSuites.append(ccToolchainSuiteTemplate
          .replace("%toolchainName%", toolchainName)
          .replace("%toolchainMap%", toolchainMap.toString().trim())
          .replace("%crosstoolReleaseProto%", crosstool.toString()));

      // Create the cc_toolchain rules
      for (CToolchain toolchain : crosstool.getToolchainList()) {
        ccToolchainRules.append(createCcToolchainRule(ccToolchainTemplate, toolchain));
      }

      // Create the STL file group rules
      for (Map.Entry<String, String> entry :
        crosstoolStlPair.stlImpl.getFilegroupNamesAndFilegroupFileGlobPatterns().entrySet()) {

        stlFilegroups.append(stlFilegroupTemplate
            .replace("%name%", entry.getKey())
            .replace("%fileGlobPattern%", entry.getValue()));
      }
    }

    return buildFileTemplate
        .replace("%ruleName%", ruleName)
        .replace("%ccToolchainSuites%", ccToolchainSuites)
        .replace("%ccToolchainRules%", ccToolchainRules)
        .replace("%stlFilegroups%", stlFilegroups);
  }

  static String createToolchainName(String stlName) {
    return TOOLCHAIN_NAME_PREFIX + stlName;
  }
  
  private static String createCcToolchainRule(String ccToolchainTemplate, CToolchain toolchain) {

    // TODO(bazel-team): It's unfortunate to have to extract data from a CToolchain proto like this.
    // It would be better to have a higher-level construction (like an AndroidToolchain class)
    // from which the CToolchain proto and rule information needed here can be created.
    // Alternatively it would be nicer to just be able to glob the entire NDK and add that one glob
    // to each cc_toolchain rule, and then the complexities in the method and the templates can
    // go away, but globbing the entire NDK takes ~60 seconds, mostly because of MD5ing all the
    // binary files in the NDK (eg the .so / .a / .o files).

    // This also includes the files captured with cxx_builtin_include_directory.
    // Use gcc specifically because clang toolchains will have both gcc and llvm toolchain paths,
    // but the gcc tool will actually be clang.
    ToolPath gcc = null;
    for (ToolPath toolPath : toolchain.getToolPathList()) {
      if ("gcc".equals(toolPath.getName())) {
        gcc = toolPath;
      }
    }
    checkNotNull(gcc, "gcc not found in crosstool toolpaths");
    String toolchainDirectory = NdkPaths.getToolchainDirectoryFromToolPath(gcc.getPath());

    // Create file glob patterns for the various files that the toolchain references.

    String androidPlatformIncludes =
        NdkPaths.stripRepositoryPrefix(toolchain.getBuiltinSysroot()) + "/**/*";

    List<String> toolchainFileGlobPatterns = new ArrayList<>();
    toolchainFileGlobPatterns.add(androidPlatformIncludes);

    for (String cxxFlag : toolchain.getUnfilteredCxxFlagList()) {
      if (!cxxFlag.startsWith("-")) { // Skip flag names
        toolchainFileGlobPatterns.add(NdkPaths.stripRepositoryPrefix(cxxFlag) + "/**/*");
      }
    }

    // If this is a clang toolchain, also add the corresponding gcc toolchain to the globs.
    int gccToolchainIndex = toolchain.getCompilerFlagList().indexOf("-gcc-toolchain");
    if (gccToolchainIndex > -1) {
      String gccToolchain = toolchain.getCompilerFlagList().get(gccToolchainIndex + 1);
      toolchainFileGlobPatterns.add(NdkPaths.stripRepositoryPrefix(gccToolchain) + "/**/*");
    }

    StringBuilder toolchainFileGlobs = new StringBuilder();
    for (String toolchainFileGlobPattern : toolchainFileGlobPatterns) {
      toolchainFileGlobs.append(String.format(
          "        \"%s\",\n", toolchainFileGlobPattern));
    }

    return ccToolchainTemplate
        .replace("%toolchainName%", toolchain.getToolchainIdentifier())
        .replace("%cpu%", toolchain.getTargetCpu())
        .replace("%dynamicRuntimeLibs%", toolchain.getDynamicRuntimesFilegroup())
        .replace("%staticRuntimeLibs%", toolchain.getStaticRuntimesFilegroup())
        .replace("%toolchainDirectory%", toolchainDirectory)
        .replace("%toolchainFileGlobs%", toolchainFileGlobs.toString().trim());
  }

  private static NdkRelease getNdkRelease(Path directory, Environment env)
      throws RepositoryFunctionException, InterruptedException {

    // For NDK r11+
    Path releaseFilePath = directory.getRelative("ndk/source.properties");
    if (!releaseFilePath.exists()) {
      // For NDK r10e
      releaseFilePath = directory.getRelative("ndk/RELEASE.TXT");
    }
    
    SkyKey releaseFileKey = FileValue.key(
        RootedPath.toRootedPath(directory, releaseFilePath));
    
    String releaseFileContent;
    try {
      env.getValueOrThrow(releaseFileKey,
          IOException.class,
          FileSymlinkException.class,
          InconsistentFilesystemException.class);

      releaseFileContent = new String(FileSystemUtils.readContent(releaseFilePath));
    } catch (IOException | FileSymlinkException | InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not read " + releaseFilePath.getBaseName() + " in Android NDK: "
              + e.getMessage()), Transience.PERSISTENT);
    }

    return NdkRelease.create(releaseFileContent.trim());
  }

  private static String getTemplate(String templateFile) {
    try {
      return ResourceFileLoader.loadResource(AndroidNdkRepositoryFunction.class, templateFile);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
}
