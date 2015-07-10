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
import com.google.devtools.build.lib.bazel.repository.DecompressorValue;
import com.google.devtools.build.lib.bazel.repository.HttpDownloadFunction;
import com.google.devtools.build.lib.bazel.repository.HttpDownloadValue;
import com.google.devtools.build.lib.bazel.repository.RepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidRepositoryRules.AndroidHttpToolsRepositoryRule;
import com.google.devtools.build.lib.packages.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.RepositoryValue;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

/**
 * Implementation of the {@code android_http_tools_repository} workspace rule.
 */
public class AndroidHttpToolsRepositoryFunction extends RepositoryFunction {

  @Override
  public SkyFunctionName getSkyFunctionName() {
    return SkyFunctionName.create(
        AndroidHttpToolsRepositoryRule.NAME.toUpperCase());
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return AndroidHttpToolsRepositoryRule.class;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = RepositoryFunction.getRule(
        repositoryName, AndroidHttpToolsRepositoryRule.NAME, env);
    if (rule == null) {
      return null;
    }

    Path outputDirectory = getExternalRepositoryDirectory().getRelative(rule.getName());
    try {
      FileSystemUtils.createDirectoryAndParents(outputDirectory);
    } catch (IOException e1) {
      throw new RepositoryFunctionException(e1, Transience.TRANSIENT);
    }
    FileValue directoryValue = getRepositoryDirectory(outputDirectory, env);
    if (directoryValue == null) {
      return null;
    }

    try {
      HttpDownloadValue downloadValue = (HttpDownloadValue) env.getValueOrThrow(
          HttpDownloadFunction.key(rule, outputDirectory), IOException.class);
      if (downloadValue == null) {
        return null;
      }

      DecompressorValue value = (DecompressorValue) env.getValueOrThrow(DecompressorValue.key(
          rule.getTargetKind(), rule.getName(), downloadValue.getPath(), outputDirectory),
          IOException.class);
      if (value == null) {
        return null;
      }
    } catch (IOException e) {
      // Assumes all IO errors transient.
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
    return RepositoryValue.create(directoryValue);
  }
}
