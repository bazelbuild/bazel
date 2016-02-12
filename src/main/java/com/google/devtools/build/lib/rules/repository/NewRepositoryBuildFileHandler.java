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

package com.google.devtools.build.lib.rules.repository;

import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.skyframe.FileSymlinkException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;

import java.io.IOException;

/**
 * Encapsulates the 2-step behavior of creating build files for the new_*_repository rules.
 */
public class NewRepositoryBuildFileHandler {

  private final Path workspacePath;
  private FileValue buildFileValue;
  private String buildFileContent;

  public NewRepositoryBuildFileHandler(Path workspacePath) {
    this.workspacePath = workspacePath;
  }

  /**
   * Prepares for writing a build file by validating the build_file and build_file_content
   * attributes of the rule.
   *
   * @return true if the build file was successfully created, false if the environment is missing
   *     values (the calling fetch() function should return null in this case).
   * @throws RepositoryFunctionException if the rule does not define the build_file or
   *     build_file_content attributes, or if it defines both, or if the build file could not be
   *     retrieved, written, or symlinked.
   */
  public boolean prepareBuildFile(Rule rule, Environment env)
      throws RepositoryFunctionException {

    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    boolean hasBuildFile = mapper.isAttributeValueExplicitlySpecified("build_file");
    boolean hasBuildFileContent = mapper.isAttributeValueExplicitlySpecified("build_file_content");

    if (hasBuildFile && hasBuildFileContent) {

      String error = String.format(
          "Rule %s cannot have both a 'build_file' and 'build_file_content' attribute", rule);
      throw new RepositoryFunctionException(
          new EvalException(rule.getLocation(), error), Transience.PERSISTENT);

    } else if (hasBuildFile) {

      buildFileValue = getBuildFileValue(rule, env);
      if (env.valuesMissing()) {
        return false;
      }

    } else if (hasBuildFileContent) { 

      buildFileContent = mapper.get("build_file_content", Type.STRING);

    } else {

      String error = String.format(
          "Rule %s requires a 'build_file' or 'build_file_content' attribute", rule);
      throw new RepositoryFunctionException(
          new EvalException(rule.getLocation(), error), Transience.PERSISTENT);
    }

    return true;
  }

  /**
   * Writes the build file, based on the state set by prepareBuildFile().
   * 
   * @param outputDirectory the directory to write the build file.
   * @throws RepositoryFunctionException if the build file could not be written or symlinked
   * @throws IllegalStateException if prepareBuildFile() was not called before this, or if
   *     prepareBuildFile() failed and this was called.
   */
  public void finishBuildFile(Path outputDirectory) throws RepositoryFunctionException {
    if (buildFileValue != null) {
      // Link x/BUILD to <build_root>/x.BUILD.
      symlinkBuildFile(buildFileValue, outputDirectory);
    } else if (buildFileContent != null) {
      RepositoryFunction.writeBuildFile(outputDirectory, buildFileContent);
    } else {
      throw new IllegalStateException(
          "prepareBuildFile() must be called before finishBuildFile()");
    }
  }

  private FileValue getBuildFileValue(Rule rule, Environment env)
      throws RepositoryFunctionException {
    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    PathFragment buildFile = new PathFragment(mapper.get("build_file", Type.STRING));
    Path buildFileTarget = workspacePath.getRelative(buildFile);
    if (!buildFileTarget.exists()) {
      throw new RepositoryFunctionException(
          new EvalException(rule.getLocation(),
              String.format("In %s the 'build_file' attribute does not specify an existing file "
                  + "(%s does not exist)", rule, buildFileTarget)),
          Transience.PERSISTENT);
    }

    RootedPath rootedBuild;
    if (buildFile.isAbsolute()) {
      rootedBuild = RootedPath.toRootedPath(
          buildFileTarget.getParentDirectory(), new PathFragment(buildFileTarget.getBaseName()));
    } else {
      rootedBuild = RootedPath.toRootedPath(workspacePath, buildFile);
    }
    SkyKey buildFileKey = FileValue.key(rootedBuild);
    FileValue buildFileValue;
    try {
      // Note that this dependency is, strictly speaking, not necessary: the symlink could simply
      // point to this FileValue and the symlink chasing could be done while loading the package
      // but this results in a nicer error message and it's correct as long as RepositoryFunctions
      // don't write to things in the file system this FileValue depends on. In theory, the latter
      // is possible if the file referenced by build_file is a symlink to somewhere under the
      // external/ directory, but if you do that, you are really asking for trouble.
      buildFileValue = (FileValue) env.getValueOrThrow(buildFileKey, IOException.class,
          FileSymlinkException.class, InconsistentFilesystemException.class);
      if (buildFileValue == null) {
        return null;
      }
    } catch (IOException | FileSymlinkException | InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(
          new IOException("Cannot lookup " + buildFile + ": " + e.getMessage()),
          Transience.TRANSIENT);
    }

    return buildFileValue;
  }

  /**
   * Symlinks a BUILD file from the local filesystem into the external repository's root.
   * @param buildFileValue {@link FileValue} representing the BUILD file to be linked in
   * @param outputDirectory the directory of the remote repository
   * @throws RepositoryFunctionException if the BUILD file specified does not exist or cannot be
   *         linked.
   */
  private void symlinkBuildFile(
      FileValue buildFileValue, Path outputDirectory) throws RepositoryFunctionException {
    Path buildFilePath = outputDirectory.getRelative("BUILD");
    RepositoryFunction.createSymbolicLink(buildFilePath, buildFileValue.realRootedPath().asPath());
  }
}