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

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.FileSymlinkException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Map;

/**
 * Create a repository from a directory on the local filesystem.
 */
public class NewLocalRepositoryFunction extends RepositoryFunction {

  @Override
  public boolean isLocal(Rule rule) {
    return true;
  }

  @Override
  public RepositoryDirectoryValue.Builder fetch(Rule rule, Path outputDirectory,
      BlazeDirectories directories, Environment env, Map<String, String> markerData)
      throws SkyFunctionException, InterruptedException {

    NewRepositoryBuildFileHandler buildFileHandler =
        new NewRepositoryBuildFileHandler(directories.getWorkspace());
    if (!buildFileHandler.prepareBuildFile(rule, env)) {
      return null;
    }

    PathFragment pathFragment = getTargetPath(rule, directories.getWorkspace());

    FileSystem fs = directories.getOutputBase().getFileSystem();
    Path path = fs.getPath(pathFragment);

    RootedPath dirPath = RootedPath.toRootedPath(fs.getRootDirectory(), path);

    try {
      FileValue dirFileValue =
          (FileValue)
              env.getValueOrThrow(
                  FileValue.key(dirPath),
                  IOException.class,
                  FileSymlinkException.class,
                  InconsistentFilesystemException.class);
      if (dirFileValue == null) {
        return null;
      }

      if (!dirFileValue.exists()) {
        throw new RepositoryFunctionException(
            new IOException(
                "Expected directory at "
                    + dirPath.asPath().getPathString()
                    + " but it does not exist."),
            Transience.PERSISTENT);
      }
      if (!dirFileValue.isDirectory()) {
        // Someone tried to create a local repository from a file.
        throw new RepositoryFunctionException(
            new IOException(
                "Expected directory at "
                    + dirPath.asPath().getPathString()
                    + " but it is not a directory."),
            Transience.PERSISTENT);
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    } catch (FileSymlinkException e) {
      throw new RepositoryFunctionException(new IOException(e), Transience.PERSISTENT);
    } catch (InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(new IOException(e), Transience.PERSISTENT);
    }

    // fetch() creates symlinks to each child under 'path' and DiffAwareness handles checking all
    // of these files and directories for changes. However, if a new file/directory is added
    // directly under 'path', Bazel doesn't know that this has to be symlinked in. Thus, this
    // creates a dependency on the contents of the 'path' directory.
    SkyKey dirKey = DirectoryListingValue.key(dirPath);
    DirectoryListingValue directoryValue;
    try {
      directoryValue = (DirectoryListingValue) env.getValueOrThrow(
          dirKey, InconsistentFilesystemException.class);
    } catch (InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(new IOException(e), Transience.PERSISTENT);
    }
    if (directoryValue == null) {
      return null;
    }

    // Link x/y/z to /some/path/to/y/z.
    if (!symlinkLocalRepositoryContents(outputDirectory, path)) {
      return null;
    }

    buildFileHandler.finishBuildFile(outputDirectory);

    // If someone specified *new*_local_repository, we can assume they didn't want the existing
    // repository info.
    Path workspaceFile = outputDirectory.getRelative("WORKSPACE");
    if (workspaceFile.exists()) {
      try {
        workspaceFile.delete();
      } catch (IOException e) {
        throw new RepositoryFunctionException(e, Transience.TRANSIENT);
      }
    }
    createWorkspaceFile(outputDirectory, rule.getTargetKind(), rule.getName());

    return RepositoryDirectoryValue.builder().setPath(outputDirectory).setSourceDir(directoryValue);
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return NewLocalRepositoryRule.class;
  }
}
