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
package com.google.devtools.build.lib.sandbox;

import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.io.IOException;
import org.junit.Before;

/**
 * Common parts of all {@link LinuxSandboxedStrategy} tests.
 */
public class LinuxSandboxedStrategyTestCase {
  protected FileSystem fileSystem;
  protected Path workspaceDir;

  @Before
  public final void createDirectoriesAndExecutor() throws Exception  {
    Path testRoot = createTestRoot();

    workspaceDir = testRoot.getRelative("workspace");
    workspaceDir.createDirectory();
  }

  private Path createTestRoot() throws IOException {
    fileSystem = FileSystems.getNativeFileSystem();
    Path testRoot = fileSystem.getPath(TestUtils.tmpDir());
    try {
      FileSystemUtils.deleteTreesBelow(testRoot);
    } catch (IOException e) {
      System.err.println("Failed to remove directory " + testRoot + ": " + e.getMessage());
      throw e;
    }
    return testRoot;
  }
}
