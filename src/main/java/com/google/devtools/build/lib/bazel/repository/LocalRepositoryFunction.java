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

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.rules.workspace.LocalRepositoryRule;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.RepositoryValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

/**
 * Access a repository on the local filesystem.
 */
public class LocalRepositoryFunction extends RepositoryFunction {

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = RepositoryFunction.getRule(repositoryName, LocalRepositoryRule.NAME, env);
    if (rule == null) {
      return null;
    }

    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    String path = mapper.get("path", Type.STRING);
    PathFragment pathFragment = new PathFragment(path);
    if (!pathFragment.isAbsolute()) {
      throw new RepositoryFunctionException(
          new EvalException(
              rule.getLocation(),
              "In " + rule + " the 'path' attribute must specify an absolute path"),
          Transience.PERSISTENT);
    }
    Path repositoryPath = getExternalRepositoryDirectory().getRelative(rule.getName());
    try {
      FileSystemUtils.createDirectoryAndParents(repositoryPath.getParentDirectory());
      if (repositoryPath.exists()) {
        repositoryPath.delete();
      }
      repositoryPath.createSymbolicLink(pathFragment);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not create symlink to repository " + pathFragment + ": "
              + e.getMessage()), Transience.TRANSIENT);
    }
    FileValue repositoryValue = getRepositoryDirectory(repositoryPath, env);
    if (repositoryValue == null) {
      // TODO(bazel-team): If this returns null, we unnecessarily recreate the symlink above on the
      // second execution.
      return null;
    }

    if (!repositoryValue.isDirectory()) {
      throw new RepositoryFunctionException(
          new IOException(rule + " must specify an existing directory"), Transience.TRANSIENT);
    }

    return RepositoryValue.create(repositoryPath, repositoryValue);
  }

  @Override
  public SkyFunctionName getSkyFunctionName() {
    return SkyFunctionName.create(LocalRepositoryRule.NAME.toUpperCase());
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return LocalRepositoryRule.class;
  }
}
