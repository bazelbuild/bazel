// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool.util;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;

/** Manages files and directories in the test workspace. */
public final class TestWorkspace {
  private final Path workspaceDir;
  private final FileSystem fileSystem;

  public TestWorkspace(Path workspaceDir) {
    this.workspaceDir = workspaceDir;
    this.fileSystem = workspaceDir.getFileSystem();
  }

  /** Returns the workspace root path. */
  public Path getRoot() {
    return workspaceDir;
  }

  /** Returns the underlying {@link FileSystem}. */
  public FileSystem getFileSystem() {
    return fileSystem;
  }

  /** Resolves a path relative to the workspace root. */
  public Path getPath(String workspaceRelativePath) {
    return workspaceDir.getRelative(PathFragment.create(workspaceRelativePath));
  }

  /**
   * Creates or overwrites a file with the given lines.
   *
   * @param workspaceRelativePath path relative to workspace root
   * @param lines content lines to write (empty array writes an empty file)
   * @return the written {@link Path}
   */
  @CanIgnoreReturnValue
  public Path write(String workspaceRelativePath, String... lines) throws IOException {
    Path path = getPath(workspaceRelativePath);
    path.getParentDirectory().createDirectoryAndParents();
    String content = lines.length == 0 ? "" : String.join("\n", lines) + "\n";
    FileSystemUtils.writeContent(path, UTF_8, content);
    return path;
  }

  /** Writes a WORKSPACE file. */
  @CanIgnoreReturnValue
  public Path setWorkspaceFile(String... lines) throws IOException {
    return write("WORKSPACE", lines);
  }

  /** Writes a MODULE.bazel file. */
  @CanIgnoreReturnValue
  public Path setModuleFile(String... lines) throws IOException {
    return write("MODULE.bazel", lines);
  }
}
