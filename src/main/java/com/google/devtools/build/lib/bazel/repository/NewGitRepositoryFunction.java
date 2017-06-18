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

package com.google.devtools.build.lib.bazel.repository;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.NewRepositoryFileHandler;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.util.Map;

/**
 * Clones a Git repository, creates a WORKSPACE file, and adds a BUILD file for it.
 */
public class NewGitRepositoryFunction extends GitRepositoryFunction {
  public NewGitRepositoryFunction(HttpDownloader httpDownloader) {
    super(httpDownloader);
  }

  @Override
  public RepositoryDirectoryValue.Builder fetch(Rule rule, Path outputDirectory,
      BlazeDirectories directories, Environment env, Map<String, String> markerData)
      throws InterruptedException, RepositoryFunctionException {
    NewRepositoryFileHandler fileHandler = new NewRepositoryFileHandler(directories.getWorkspace());
    if (!fileHandler.prepareFile(rule, env)) {
      return null;
    }

    createDirectory(outputDirectory, rule);
    GitCloner.clone(rule, outputDirectory, env.getListener(), clientEnvironment, downloader);
    fileHandler.finishFile(outputDirectory);

    return RepositoryDirectoryValue.builder().setPath(outputDirectory);
  }
}
