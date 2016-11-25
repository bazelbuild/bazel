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
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpFileRule;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;

/**
 * Downloads a jar file from a URL.
 */
public class HttpFileFunction extends HttpArchiveFunction {

  public HttpFileFunction(HttpDownloader httpDownloader) {
    super(httpDownloader);
  }

  @Override
  protected DecompressorDescriptor getDescriptor(Rule rule, Path downloadPath, Path outputDirectory)
      throws RepositoryFunctionException {
    WorkspaceAttributeMapper mapper = WorkspaceAttributeMapper.of(rule);
    boolean executable = false;
    try {
      executable = (mapper.isAttributeValueExplicitlySpecified("executable")
          && mapper.get("executable", Type.BOOLEAN));
    } catch (EvalException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }
    return DecompressorDescriptor.builder()
        .setDecompressor(FileDecompressor.INSTANCE)
        .setTargetKind(rule.getTargetKind())
        .setTargetName(rule.getName())
        .setArchivePath(downloadPath)
        .setRepositoryPath(outputDirectory)
        .setExecutable(executable)
        .build();
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return HttpFileRule.class;
  }
}
