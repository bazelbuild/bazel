// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpArchiveRule;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
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
 * Downloads a file over HTTP.
 */
public class HttpArchiveFunction extends RepositoryFunction {

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = RepositoryFunction.getRule(repositoryName, HttpArchiveRule.NAME, env);
    if (rule == null) {
      return null;
    }

    return compute(env, rule);
  }

  protected void createDirectory(Path path)
      throws RepositoryFunctionException {
    try {
      FileSystemUtils.createDirectoryAndParents(path);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  protected SkyValue compute(Environment env, Rule rule)
      throws RepositoryFunctionException {
    // The output directory is always under .external-repository (to stay out of the way of
    // artifacts from this repository) and uses the rule's name to avoid conflicts with other
    // remote repository rules. For example, suppose you had the following WORKSPACE file:
    //
    // http_archive(name = "png", url = "http://example.com/downloads/png.tar.gz", sha256 = "...")
    //
    // This would download png.tar.gz to .external-repository/png/png.tar.gz.
    Path outputDirectory = getExternalRepositoryDirectory().getRelative(rule.getName());
    createDirectory(outputDirectory);
    try {
      HttpDownloadValue downloadValue = (HttpDownloadValue) env.getValueOrThrow(
          HttpDownloadFunction.key(rule, outputDirectory), IOException.class);
      if (downloadValue == null) {
        return null;
      }

      DecompressorValue value = (DecompressorValue) env.getValueOrThrow(
          decompressorValueKey(rule, downloadValue.getPath(), outputDirectory), IOException.class);
      if (value == null) {
        return null;
      }
    } catch (IOException e) {
      // Assumes all IO errors transient.
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
    return RepositoryValue.create(outputDirectory);
  }

  protected SkyKey decompressorValueKey(Rule rule, Path downloadPath, Path outputDirectory)
      throws IOException {
    return DecompressorValue.key(DecompressorDescriptor.builder()
        .setTargetKind(rule.getTargetKind())
        .setTargetName(rule.getName())
        .setArchivePath(downloadPath)
        .setRepositoryPath(outputDirectory)
        .build());
  }

  @Override
  public SkyFunctionName getSkyFunctionName() {
    return SkyFunctionName.create(HttpArchiveRule.NAME.toUpperCase());
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return HttpArchiveRule.class;
  }
}
