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
import com.google.devtools.build.lib.bazel.rules.workspace.NewLocalRepositoryRule;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.RepositoryValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.nio.charset.Charset;

/**
 * Create a repository from a directory on the local filesystem.
 */
public class NewLocalRepositoryFunction extends RepositoryFunction {

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = RepositoryFunction.getRule(repositoryName, NewLocalRepositoryRule.NAME, env);
    if (rule == null) {
      return null;
    }

    // Given a rule that looks like this:
    // new_local_repository(
    //     name = 'x',
    //     path = '/some/path/to/y',
    //     build_file = 'x.BUILD'
    // )
    // This creates the following directory structure:
    // .external-repository/
    //   x/
    //     WORKSPACE
    //     BUILD -> <build_root>/x.BUILD
    //     y -> /some/path/to/y
    //
    Path repositoryDirectory = getExternalRepositoryDirectory().getRelative(rule.getName());
    try {
      FileSystemUtils.createDirectoryAndParents(repositoryDirectory);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
    FileValue directoryValue = getRepositoryDirectory(repositoryDirectory, env);
    if (directoryValue == null) {
      return null;
    }

    // Add x/WORKSPACE.
    try {
      Path workspaceFile = repositoryDirectory.getRelative("WORKSPACE");
      FileSystemUtils.writeContent(workspaceFile, Charset.forName("UTF-8"),
          "# DO NOT EDIT: automatically generated WORKSPACE file for " + rule + "\n");
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    // Link x/y to /some/path/to/y.
    String path = mapper.get("path", Type.STRING);
    PathFragment pathFragment = new PathFragment(path);
    if (!pathFragment.isAbsolute()) {
      throw new RepositoryFunctionException(
          new EvalException(
              rule.getLocation(),
              "In " + rule + " the 'path' attribute must specify an absolute path"),
          Transience.PERSISTENT);
    }
    Path pathTarget = getOutputBase().getFileSystem().getPath(pathFragment);
    Path symlinkPath = repositoryDirectory.getRelative(pathTarget.getBaseName());
    if (createSymbolicLink(symlinkPath, pathTarget, env) == null) {
      return null;
    }

    // Link x/BUILD to <build_root>/x.BUILD.
    PathFragment buildFile = new PathFragment(mapper.get("build_file", Type.STRING));
    Path buildFileTarget = getWorkspace().getRelative(buildFile);
    if (buildFile.equals(PathFragment.EMPTY_FRAGMENT) || buildFile.isAbsolute()
        || !buildFileTarget.exists()) {
      throw new RepositoryFunctionException(
          new EvalException(rule.getLocation(), "In " + rule
              + " the 'build_file' attribute must specify a relative path to an existing file"),
          Transience.PERSISTENT);
    }
    Path buildFilePath = repositoryDirectory.getRelative("BUILD");
    if (createSymbolicLink(buildFilePath, buildFileTarget, env) == null) {
      return null;
    }

    return new RepositoryValue(repositoryDirectory, directoryValue);
  }

  private FileValue createSymbolicLink(Path from, Path to, Environment env)
      throws RepositoryFunctionException {
    try {
      if (!from.exists()) {
        from.createSymbolicLink(to);
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
    return getRepositoryDirectory(from, env);
  }

  @Override
  public SkyFunctionName getSkyFunctionName() {
    return SkyFunctionName.computed(NewLocalRepositoryRule.NAME.toUpperCase());
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return NewLocalRepositoryRule.class;
  }
}
