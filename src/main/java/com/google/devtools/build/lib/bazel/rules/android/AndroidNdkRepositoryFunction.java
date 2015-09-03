// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.repository.RepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.AndroidNdkCrosstools;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.AndroidNdkCrosstools.NdkCrosstoolsException;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkPaths;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkRelease;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.FileSymlinkException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.DefaultCpuToolchain;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Implementation of the {@code android_ndk_repository} rule.
 */
public class AndroidNdkRepositoryFunction extends RepositoryFunction {

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = getRule(repositoryName, AndroidNdkRepositoryRule.NAME, env);
    if (rule == null) {
      return null;
    }

    FileValue directoryValue = prepareLocalRepositorySymlinkTree(rule, env);
    if (directoryValue == null) {
      return null;
    }

    PathFragment pathFragment = getTargetPath(rule);
    Path ndkSymlinkTreeDirectory = directoryValue.realRootedPath().asPath().getRelative("ndk");
    try {
      ndkSymlinkTreeDirectory.createDirectory();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    if (!symlinkLocalRepositoryContents(
        ndkSymlinkTreeDirectory, getOutputBase().getFileSystem().getPath(pathFragment), env)) {
      return null;
    }

    AttributeMap attributes = NonconfigurableAttributeMapper.of(rule);
    String ruleName = rule.getName();
    String apiLevel = attributes.get("api_level", Type.INTEGER).toString();

    CrosstoolRelease androidCrosstoolRelease;
    try {
      androidCrosstoolRelease = AndroidNdkCrosstools.createCrosstoolRelease(
          env.getListener(), ruleName, apiLevel, getNdkRelease(directoryValue, env));
    } catch (NdkCrosstoolsException e) {
      throw new RepositoryFunctionException(new IOException(e), Transience.PERSISTENT);
    }

    String buildFile = createBuildFile(androidCrosstoolRelease, ruleName);
    return writeBuildFile(directoryValue, buildFile);
  }

  @Override
  public SkyFunctionName getSkyFunctionName() {
    return SkyFunctionName.create(AndroidNdkRepositoryRule.NAME.toUpperCase());
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return AndroidNdkRepositoryRule.class;
  }

  private static String createBuildFile(CrosstoolRelease androidCrosstoolRelease, String ruleName) {

    String ccToolchainSuiteTemplate = getTemplate("android_ndk_cc_toolchain_suite_template.txt");
    String ccToolchainTemplate = getTemplate("android_ndk_cc_toolchain_template.txt");

    StringBuilder toolchainMap = new StringBuilder();
    for (DefaultCpuToolchain defaultToolchain : androidCrosstoolRelease.getDefaultToolchainList()) {
      toolchainMap.append(String.format("      \"%s\": \":%s\",\n",
          defaultToolchain.getCpu(), defaultToolchain.getToolchainIdentifier()));
    }

    StringBuilder ccToolchainRules = new StringBuilder();
    for (CToolchain toolchain : androidCrosstoolRelease.getToolchainList()) {
      ccToolchainRules.append(createCcToolchainRule(ccToolchainTemplate, toolchain));
    }

    String crosstoolReleaseProto = androidCrosstoolRelease.toString();

    return ccToolchainSuiteTemplate
        .replace("%ruleName%", ruleName)
        .replace("%toolchainMap%", toolchainMap.toString().trim())
        .replace("%crosstoolReleaseProto%", crosstoolReleaseProto)
        .replace("%ccToolchainRules%", ccToolchainRules);
  }

  private static String createCcToolchainRule(String ccToolchainTemplate, CToolchain toolchain) {

    // TODO(bazel-team): It's unfortunate to have to extract data from a CToolchain proto like this.
    // It would be better to have a higher-level construction (like an AndroidToolchain class)
    // from which the CToolchain proto and rule information needed here can be created.
    // Alternatively it would be nicer to just be able to glob the entire NDK and add that one glob
    // to each cc_toolchain rule, and then the complexities in the method and the templates can
    // go away, but globbing the entire NDK takes ~60 seconds, mostly because of MD5ing all the
    // binary files in the NDK (eg the .so / .a / .o files).

    // This also includes the files captured with cxx_builtin_include_directory
    String toolchainDirectory = NdkPaths.getToolchainDirectoryFromToolPath(
        toolchain.getToolPathList().get(0).getPath());

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

    StringBuilder toolchainFileGlobs = new StringBuilder();
    for (String toolchainFileGlobPattern : toolchainFileGlobPatterns) {
      toolchainFileGlobs.append(String.format(
          "        \"%s\",\n", toolchainFileGlobPattern));
    }

    return ccToolchainTemplate
        .replace("%toolchainName%", toolchain.getToolchainIdentifier())
        .replace("%cpu%", toolchain.getTargetCpu())
        .replace("%toolchainDirectory%", toolchainDirectory)
        .replace("%toolchainFileGlobs%", toolchainFileGlobs.toString().trim());
  }

  private static NdkRelease getNdkRelease(FileValue directoryValue, Environment env)
      throws RepositoryFunctionException {

    Path releaseFilePath = directoryValue.realRootedPath().asPath().getRelative("ndk/RELEASE.TXT");
    
    SkyKey releaseFileKey = FileValue.key(RootedPath.toRootedPath(
        releaseFilePath, PathFragment.EMPTY_FRAGMENT));

    String releaseFileContent;
    try {
      env.getValueOrThrow(releaseFileKey,
          IOException.class,
          FileSymlinkException.class,
          InconsistentFilesystemException.class);

      releaseFileContent = new String(FileSystemUtils.readContent(releaseFilePath));
    } catch (IOException | FileSymlinkException | InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not read RELEASE.TXT in Android NDK: " + e.getMessage()),
              Transience.PERSISTENT);
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
