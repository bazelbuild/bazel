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
import com.google.devtools.build.lib.bazel.rules.workspace.HttpJarRule;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.vfs.Path;

/**
 * Downloads a jar file from a URL.
 */
public class HttpJarFunction extends HttpArchiveFunction {

  public HttpJarFunction(HttpDownloader httpDownloader) {
    super(httpDownloader);
  }

  @Override
  protected DecompressorDescriptor getDescriptor(Rule rule, Path downloadPath, Path outputDirectory)
      throws RepositoryFunctionException {
    return DecompressorDescriptor.builder()
        .setDecompressor(JarDecompressor.INSTANCE)
        .setTargetKind(rule.getTargetKind())
        .setTargetName(rule.getName())
        .setArchivePath(downloadPath)
        .setRepositoryPath(outputDirectory)
        .build();
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return HttpJarRule.class;
  }
}
