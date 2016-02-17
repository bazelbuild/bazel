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

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.rules.workspace.GitRepositoryRule;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

/**
 * Clones a Git repository.
 */
public class GitRepositoryFunction extends RepositoryFunction {
  @Override
  public boolean isLocal(Rule rule) {
    return false;
  }

  @Override
  public SkyValue fetch(Rule rule, Path outputDirectory, Environment env)
      throws SkyFunctionException {
    createDirectory(outputDirectory, rule);
    GitCloner.clone(rule, outputDirectory, env.getListener());
    return RepositoryDirectoryValue.create(outputDirectory);
  }

  protected void createDirectory(Path path, Rule rule)
      throws RepositoryFunctionException {
    try {
      FileSystemUtils.createDirectoryAndParents(path);
    } catch (IOException e) {
      throw new RepositoryFunctionException(new IOException("Could not create directory for "
          + rule.getName() + ": " + e.getMessage()), Transience.TRANSIENT);
    }
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return GitRepositoryRule.class;
  }
}
