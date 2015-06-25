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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.repository.RepositoryFunction;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.List;

/**
 * Implementation of the {@code android_ndk} repository rule.
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

    if (!symlinkLocalRepositoryContents(
        directoryValue, getOutputBase().getFileSystem().getPath(pathFragment), env)) {
      return null;
    }

    AttributeMap attributes = NonconfigurableAttributeMapper.of(rule);
    String ruleName = rule.getName();
    String apiLevel = attributes.get("api_level", Type.INTEGER).toString();
    List<String> cpus = ImmutableList.of("arm");  // TODO(bazel-team): autodetect
    String abi = "armeabi-v7a";  // TODO(bazel-team): Should this be an attribute on the rule?
    String compiler = "4.9";  // TODO(bazel-team): Should this be an attribute on the rule?

    String ccToolchainSuiteTemplate;
    String ccToolchainTemplate;
    String toolchainTemplate;

    try {
      ccToolchainSuiteTemplate = ResourceFileLoader.loadResource(
          AndroidNdkRepositoryFunction.class, "android_ndk_cc_toolchain_suite_template.txt");
      ccToolchainTemplate = ResourceFileLoader.loadResource(
          AndroidNdkRepositoryFunction.class, "android_ndk_cc_toolchain_template.txt");
      toolchainTemplate = ResourceFileLoader.loadResource(
          AndroidNdkRepositoryFunction.class, "android_ndk_toolchain_template.txt");
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }

    StringBuilder toolchainMap = new StringBuilder();
    StringBuilder toolchainProtos = new StringBuilder();
    StringBuilder toolchains = new StringBuilder();

    for (String cpu : cpus) {
      toolchainMap.append(String.format("\"%s\": \":cc-compiler-%s\", ", cpu, cpu));
      toolchainProtos.append(toolchainTemplate
          .replace("%repository%", ruleName)
          .replace("%cpu%", cpu)
          .replace("%abi%", abi)
          .replace("%api_level%", apiLevel)
          .replace("%compiler%", compiler));
      toolchains.append(ccToolchainTemplate
          .replace("%repository%", ruleName)
          .replace("%cpu%", cpu)
          .replace("%abi%", abi)
          .replace("%api_level%", apiLevel)
          .replace("%compiler%", compiler));
    }

    String buildFile = ccToolchainSuiteTemplate
        .replace("%toolchain_map%", toolchainMap)
        .replace("%toolchain_protos%", toolchainProtos)
        .replace("%toolchains%", toolchains)
        .replace("%default_cpu%", cpus.get(0));

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
}
