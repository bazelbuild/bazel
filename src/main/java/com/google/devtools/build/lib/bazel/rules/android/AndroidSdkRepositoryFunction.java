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

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

/**
 * Implementation of the {@code android_sdk_repository} rule.
 */
public class AndroidSdkRepositoryFunction extends RepositoryFunction {
  @Override
  public boolean isLocal() {
    return true;
  }

  @Override
  public SkyValue fetch(Rule rule, Path outputDirectory, Environment env)
      throws SkyFunctionException {
    prepareLocalRepositorySymlinkTree(rule, outputDirectory);
    PathFragment pathFragment = getTargetPath(rule, getWorkspace());

    if (!symlinkLocalRepositoryContents(
        outputDirectory, getOutputBase().getFileSystem().getPath(pathFragment))) {
      return null;
    }

    AttributeMap attributes = NonconfigurableAttributeMapper.of(rule);
    String buildToolsVersion = attributes.get("build_tools_version", Type.STRING);
    Integer apiLevel = attributes.get("api_level", Type.INTEGER);

    String template = getStringResource("android_sdk_repository_template.txt");

    // Android 23 removed most of org.apache.http from android.jar and moved it
    // to a separate jar, but this jar exists only with version 23 and above.
    // Not sure when this jar will be removed.
    String orgApacheHttpLegacyImport = "";
    if (apiLevel >= 23) {
      orgApacheHttpLegacyImport =
          getStringResource("android_sdk_org_apache_http_legacy_import_template.txt")
              .replaceAll("%api_level%", apiLevel.toString());
    }

    String buildFile = template
        .replaceAll("%repository_name%", rule.getName())
        .replaceAll("%build_tools_version%", buildToolsVersion)
        .replaceAll("%api_level%", apiLevel.toString())
        .replaceAll("%org_apache_http_legacy_import%", orgApacheHttpLegacyImport);

    writeBuildFile(outputDirectory, buildFile);
    return RepositoryDirectoryValue.create(outputDirectory);
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return AndroidSdkRepositoryRule.class;
  }

  private static String getStringResource(String name) {
    try {
      return ResourceFileLoader.loadResource(
          AndroidSdkRepositoryFunction.class, name);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
}
