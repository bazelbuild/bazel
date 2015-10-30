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

import com.google.devtools.build.lib.bazel.rules.workspace.NewGitRepositoryRule;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

/**
 * Clones a Git repository, creates a WORKSPACE file, and adds a BUILD file for it.
 */
public class NewGitRepositoryFunction extends GitRepositoryFunction {
  @Override
  public SkyFunctionName getSkyFunctionName() {
    return SkyFunctionName.create(NewGitRepositoryRule.NAME.toUpperCase());
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = RepositoryFunction.getRule(repositoryName, NewGitRepositoryRule.NAME, env);
    if (rule == null) {
      return null;
    }

    Path outputDirectory = getExternalRepositoryDirectory().getRelative(rule.getName());
    FileValue directoryValue = createDirectory(outputDirectory, env, rule);
    if (directoryValue == null) {
      return null;
    }

    try {
      HttpDownloadValue value = (HttpDownloadValue) env.getValueOrThrow(
          GitCloneFunction.key(rule, outputDirectory), IOException.class);
      if (value == null) {
        return null;
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    createWorkspaceFile(outputDirectory, rule);
    return symlinkBuildFile(rule, getWorkspace(), directoryValue, env);
  }
}
