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

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.NewRepositoryBuildFileHandler;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Downloads an archive from a URL, decompresses it, creates a WORKSPACE file, and adds a BUILD
 * file for it.
 */
public class NewHttpArchiveFunction extends HttpArchiveFunction {

  public NewHttpArchiveFunction(HttpDownloader httpDownloader) {
    super(httpDownloader);
  }

  @Nullable
  @Override
  public SkyValue fetch(
      Rule rule, Path outputDirectory, BlazeDirectories directories, Environment env)
          throws RepositoryFunctionException, InterruptedException {
    NewRepositoryBuildFileHandler buildFileHandler =
        new NewRepositoryBuildFileHandler(directories.getWorkspace());
    if (!buildFileHandler.prepareBuildFile(rule, env)) {
      return null;
    }

    try {
      FileSystemUtils.createDirectoryAndParents(outputDirectory);
    } catch (IOException e) {
      throw new RepositoryFunctionException(new IOException("Could not create directory for "
          + rule.getName() + ": " + e.getMessage()), Transience.TRANSIENT);
    }

    // Download.
    Path downloadedPath = downloader.download(
        rule, outputDirectory, env.getListener(), clientEnvironment);

    // Decompress.
    Path decompressed;
    WorkspaceAttributeMapper mapper = WorkspaceAttributeMapper.of(rule);
    String prefix = null;
    if (mapper.isAttributeValueExplicitlySpecified("strip_prefix")) {
      try {
        prefix = mapper.get("strip_prefix", Type.STRING);
      } catch (EvalException e) {
        throw new RepositoryFunctionException(e, Transience.PERSISTENT);
      }
    }
    decompressed = DecompressorValue.decompress(DecompressorDescriptor.builder()
        .setTargetKind(rule.getTargetKind())
        .setTargetName(rule.getName())
        .setArchivePath(downloadedPath)
        .setRepositoryPath(outputDirectory)
        .setPrefix(prefix)
        .build());

    // Finally, write WORKSPACE and BUILD files.
    createWorkspaceFile(decompressed, rule.getTargetKind(), rule.getName());
    buildFileHandler.finishBuildFile(outputDirectory);

    return RepositoryDirectoryValue.create(outputDirectory);
  }
}
