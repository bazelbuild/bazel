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
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

/**
 * Implementation of the {@code android_sdk_repository} rule.
 */
public class AndroidSdkRepositoryFunction extends RepositoryFunction {
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = getRule(repositoryName, AndroidSdkRepositoryRule.NAME, env);
    if (rule == null) {
      return null;
    }

    FileValue directoryValue = prepareLocalRepositorySymlinkTree(rule, env);
    if (directoryValue == null) {
      return null;
    }

    PathFragment pathFragment = getTargetPath(rule);

    if (!symlinkLocalRepositoryContents(
        directoryValue.realRootedPath().asPath(),
        getOutputBase().getFileSystem().getPath(pathFragment),
        env)) {
      return null;
    }

    AttributeMap attributes = NonconfigurableAttributeMapper.of(rule);
    String buildToolsVersion = attributes.get("build_tools_version", Type.STRING);
    Integer apiLevel = attributes.get("api_level", Type.INTEGER);

    String template;
    try {
      template = ResourceFileLoader.loadResource(
        AndroidSdkRepositoryFunction.class, "android_sdk_repository_template.txt");
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }

    String buildFile = template
        .replaceAll("%build_tools_version%", buildToolsVersion)
        .replaceAll("%api_level%", apiLevel.toString());

    return writeBuildFile(directoryValue, buildFile);
  }

  /**
   * @see RepositoryFunction#getRule(RepositoryName, String, Environment)
   */
  @Override
  public SkyFunctionName getSkyFunctionName() {
    return SkyFunctionName.create(AndroidSdkRepositoryRule.NAME.toUpperCase());
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return AndroidSdkRepositoryRule.class;
  }
}
