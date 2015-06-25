// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.bazel.rules.workspace.NewHttpArchiveRule;
import com.google.devtools.build.lib.packages.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

import javax.annotation.Nullable;

/**
 * Downloads an archive from a URL, decompresses it, creates a WORKSPACE file, and adds a BUILD
 * file for it.
 */
public class NewHttpArchiveFunction extends HttpArchiveFunction {

  @Override
  public SkyFunctionName getSkyFunctionName() {
    return SkyFunctionName.create(NewHttpArchiveRule.NAME);
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, SkyFunction.Environment env)
      throws RepositoryFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = getRule(repositoryName, NewHttpArchiveRule.NAME, env);
    if (rule == null) {
      return null;
    }
    Path outputDirectory = getExternalRepositoryDirectory().getRelative(rule.getName());
    try {
      FileSystemUtils.createDirectoryAndParents(outputDirectory);
    } catch (IOException e) {
      throw new RepositoryFunctionException(new IOException("Could not create directory for "
          + rule.getName() + ": " + e.getMessage()), Transience.TRANSIENT);
    }
    FileValue repositoryDirectory = getRepositoryDirectory(outputDirectory, env);
    if (repositoryDirectory == null) {
      return null;
    }

    // Download.
    HttpDownloadValue downloadedFileValue;
    try {
      downloadedFileValue = (HttpDownloadValue) env.getValueOrThrow(
          HttpDownloadFunction.key(rule, outputDirectory), IOException.class);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }
    if (downloadedFileValue == null) {
      return null;
    }

    // Decompress.
    DecompressorValue decompressed;
    try {
      decompressed = (DecompressorValue) env.getValueOrThrow(
          DecompressorValue.key(rule.getTargetKind(), rule.getName(),
              downloadedFileValue.getPath(), outputDirectory), IOException.class);
      if (decompressed == null) {
        return null;
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(e.getMessage()), Transience.TRANSIENT);
    }

    // Add WORKSPACE and BUILD files.
    createWorkspaceFile(decompressed.getDirectory(), rule);
    return symlinkBuildFile(rule, getWorkspace(), repositoryDirectory, env);
  }
}
