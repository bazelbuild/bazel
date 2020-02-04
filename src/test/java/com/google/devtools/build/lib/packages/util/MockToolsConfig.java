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
package com.google.devtools.build.lib.packages.util;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import javax.annotation.Nullable;

/**
 * Configuration for the mock client setup that we use for testing.
 */
public final class MockToolsConfig {

  private final Path rootDirectory;
  private final boolean realFileSystem;

  // Allow the injection of the runfiles directory where actual tools are found.
  // TestUtil.getRunfilesDir() caches the value of the "TEST_SRCDIR" system property, which makes
  // it impossible to change if it doesn't get set early in test configuration setup.
  private final Path runfilesDirectory;

  public MockToolsConfig(Path rootDirectory) {
    this(rootDirectory, false, null);
  }

  public MockToolsConfig(Path rootDirectory, boolean realFileSystem) {
    this(rootDirectory, realFileSystem, null);
  }

  public MockToolsConfig(
      Path rootDirectory, boolean realFileSystem, @Nullable Path runfilesDirectoryOpt) {
    this.rootDirectory = rootDirectory;
    this.realFileSystem = realFileSystem;
    if (!realFileSystem) {
      this.runfilesDirectory = null;
    } else if (runfilesDirectoryOpt == null) {
      this.runfilesDirectory = rootDirectory.getRelative(BlazeTestUtils.runfilesDir());
    } else {
      this.runfilesDirectory = runfilesDirectoryOpt;
    }
  }

  public boolean isRealFileSystem() {
    return realFileSystem;
  }

  public Path getPath(String relativePath) {
    return rootDirectory.getRelative(relativePath);
  }

  public Path create(String relativePath, String... lines) throws IOException {
    Path path = rootDirectory.getRelative(relativePath);
    if (!path.exists()) {
      FileSystemUtils.writeIsoLatin1(path, lines);
    } else if (lines.length > 0) {
      String existingContent = new String(FileSystemUtils.readContentAsLatin1(path));

      StringBuilder newContent = new StringBuilder();
      for (String line : lines) {
        newContent.append(line);
        newContent.append(System.lineSeparator());
      }

      if (!newContent.toString().trim().equals(existingContent.trim())) {
        throw new IOException(
            "Conflict: '"
                + relativePath
                + "':\n'"
                + newContent
                + "'\n vs \n'"
                + existingContent
                + "'");
      }
    }
    return path;
  }

  public Path overwrite(String relativePath, String... lines) throws IOException {
    Path path = rootDirectory.getRelative(relativePath);
    if (path.exists()) {
      path.deleteTree();
    }
    return create(relativePath, lines);
  }

  public Path append(String relativePath, String... lines) throws IOException {
    Path path = rootDirectory.getRelative(relativePath);
    if (!path.exists()) {
      return create(relativePath, lines);
    }

    FileSystemUtils.appendIsoLatin1(path, lines);
    return path;
  }

  /**
   * Links a tool into the workspace by creating a symbolic link to a real file. The target location
   * in the workspace uses the same relative path as the given path to the tool in the runfiles
   * tree. Use this if you do not need to rename or relocate the file, i.e., if the location in the
   * workspace and the runfiles tree matches. Otherwise use {@link #linkTool(String, String)}.
   *
   * @param relativePath the relative path within the runfiles tree of the current test
   * @throws IOException
   */
  public void linkTool(String relativePath) throws IOException {
    Preconditions.checkState(realFileSystem);
    linkTool(relativePath, relativePath);
  }

  public void copyTool(String relativePath) throws IOException {
    Path runfiles = FileSystems.getNativeFileSystem().getPath(BlazeTestUtils.runfilesDir());
    Path source = runfiles.getRelative(TestConstants.WORKSPACE_NAME).getRelative(relativePath);
    create(relativePath, FileSystemUtils.readContent(source, StandardCharsets.ISO_8859_1));
  }

  /**
   * Links a tool into the workspace by creating a symbolic link to a real file.
   *
   * @param relativePath the relative path within the runfiles tree of the current test
   * @param dest the relative path in the mock client
   * @throws IOException
   */
  public void linkTool(String relativePath, String dest) throws IOException {
    Preconditions.checkState(realFileSystem);
    Path target = runfilesDirectory.getRelative(TestConstants.WORKSPACE_NAME + "/" + relativePath);
    if (!target.exists()) {
      // In some cases we run tests in a special client with a ../READONLY/ path where we may also
      // find the runfiles. Try that, too.
      Path readOnlyClientPath =
          rootDirectory.getRelative(
              "../READONLY/" + TestConstants.WORKSPACE_NAME + "/" + relativePath);
      if (!readOnlyClientPath.exists()) {
        throw new IOException("target does not exist " + target);
      } else {
        target = readOnlyClientPath;
      }
    }
    Path path = rootDirectory.getRelative(dest);
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    path.delete();
    path.createSymbolicLink(target);
  }

  /**
   * Convenience method to link multiple tools. Same as calling {@link #linkTool(String)} for each
   * parameter.
   */
  public void linkTools(String... tools) throws IOException {
    for (String tool : tools) {
      linkTool(tool);
    }
  }
}
