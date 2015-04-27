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
import com.google.devtools.build.lib.skyframe.FileSymlinkCycleException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.skyframe.RepositoryValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
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
    //
    // Assume /some/path/to/y contains files z, w, and v. This creates the following directory
    // structure:
    // .external-repository/
    //   x/
    //     WORKSPACE
    //     BUILD -> <build_root>/x.BUILD
    //     z -> /some/path/to/y/z
    //     w -> /some/path/to/y/w
    //     v -> /some/path/to/y/v
    //
    // Create x/
    Path repositoryDirectory = getExternalRepositoryDirectory().getRelative(rule.getName());
    try {
      FileSystemUtils.deleteTree(repositoryDirectory);
      FileSystemUtils.createDirectoryAndParents(repositoryDirectory);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
    FileValue directoryValue = getRepositoryDirectory(repositoryDirectory, env);
    if (directoryValue == null) {
      return null;
    }

    // Add x/WORKSPACE.
    createWorkspaceFile(repositoryDirectory, rule);

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

    // Link x/y/z to /some/path/to/y/z.
    FileSystem fs = getOutputBase().getFileSystem();
    Path targetDirectory = fs.getPath(pathFragment);
    try {
      for (Path target : targetDirectory.getDirectoryEntries()) {
        Path symlinkPath = repositoryDirectory.getRelative(target.getBaseName());
        if (createSymbolicLink(symlinkPath, target, env) == null) {
          return null;
        }
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    // Link x/BUILD to <build_root>/x.BUILD.
    if (createBuildFile(rule, getWorkspace(), repositoryDirectory, env) == null) {
      return null;
    }

    return new RepositoryValue(repositoryDirectory, directoryValue);
  }

  public static void createWorkspaceFile(Path repositoryDirectory, Rule rule)
      throws RepositoryFunctionException {
    try {
      Path workspaceFile = repositoryDirectory.getRelative("WORKSPACE");
      FileSystemUtils.writeContent(workspaceFile, Charset.forName("UTF-8"),
          String.format("# DO NOT EDIT: automatically generated WORKSPACE file for %s\n", rule));
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  /**
   * Symlinks a BUILD file from the local filesystem into the external repository's root.
   * @param rule the rule that declares the build_file path.
   * @param workspaceDirectory the workspace root for the build.
   * @param repositoryDirectory the external repository's root directory.
   * @param env the Skyframe environment.
   * @return the file value of the symlink created.
   * @throws RepositoryFunctionException if the BUILD file specified does not exist or cannot be
   *         linked.
   */
  public static FileValue createBuildFile(
      Rule rule, Path workspaceDirectory, Path repositoryDirectory, Environment env)
      throws RepositoryFunctionException {
    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    PathFragment buildFile = new PathFragment(mapper.get("build_file", Type.STRING));
    Path buildFileTarget = workspaceDirectory.getRelative(buildFile);
    if (!buildFileTarget.exists()) {
      throw new RepositoryFunctionException(
          new EvalException(rule.getLocation(),
              String.format("In %s the 'build_file' attribute does not specify an existing file "
                  + "(%s does not exist)", rule, buildFileTarget)),
          Transience.PERSISTENT);
    }
    Path buildFilePath = repositoryDirectory.getRelative("BUILD");
    return createSymbolicLink(buildFilePath, buildFileTarget, env);
  }

  private static FileValue createSymbolicLink(Path from, Path to, Environment env)
      throws RepositoryFunctionException {
    try {
      if (!from.exists()) {
        from.createSymbolicLink(to);
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(String.format("Error creating symbolic link from %s to %s: %s",
              from, to, e.getMessage())), Transience.TRANSIENT);
    }

    SkyKey outputDirectoryKey = FileValue.key(RootedPath.toRootedPath(
        from, PathFragment.EMPTY_FRAGMENT));
    try {
      return (FileValue) env.getValueOrThrow(outputDirectoryKey, IOException.class,
          FileSymlinkCycleException.class, InconsistentFilesystemException.class);
    } catch (IOException | FileSymlinkCycleException | InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(
          new IOException(String.format("Could not access %s: %s", from, e.getMessage())),
          Transience.PERSISTENT);
    }
  }

  /**
   * @see RepositoryFunction#getRule(RepositoryName, String, Environment)
   */
  @Override
  public SkyFunctionName getSkyFunctionName() {
    return SkyFunctionName.computed(NewLocalRepositoryRule.NAME.toUpperCase());
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return NewLocalRepositoryRule.class;
  }
}
