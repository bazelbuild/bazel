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
import com.google.devtools.build.lib.bazel.rules.workspace.HttpFileRule;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

/**
 * Downloads a jar file from a URL.
 */
public class HttpFileFunction extends HttpArchiveFunction {

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = RepositoryFunction.getRule(repositoryName, HttpFileRule.NAME, env);
    if (rule == null) {
      return null;
    }
    return compute(env, rule);
  }

  @Override
  protected SkyKey decompressorValueKey(Rule rule, Path downloadPath, Path outputDirectory)
      throws IOException {
    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    boolean executable = (mapper.has("executable", Type.BOOLEAN)
        && mapper.get("executable", Type.BOOLEAN));
    return DecompressorValue.key(FileFunction.NAME, DecompressorDescriptor.builder()
        .setTargetKind(rule.getTargetKind())
        .setTargetName(rule.getName())
        .setArchivePath(downloadPath)
        .setRepositoryPath(outputDirectory)
        .setExecutable(executable)
        .build());
  }

  @Override
  public SkyFunctionName getSkyFunctionName() {
    return SkyFunctionName.create(HttpFileRule.NAME.toUpperCase());
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return HttpFileRule.class;
  }
}
