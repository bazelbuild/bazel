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
package com.google.devtools.build.lib.integration.util;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Performs setup for integration tests.
 */
public class IntegrationMock {
  public static IntegrationMock get() {
    return new IntegrationMock();
  }

  /**
   * Populates the _embedded_binaries/ directory with all files found in any of the directories in
   * {@link TestConstants#EMBEDDED_SCRIPTS_PATHS} by creating symlinks in
   * {@link BlazeDirectories#getEmbeddedBinariesRoot} that point to the runfiles tree
   * of the currently running test (as obtained from {@link BlazeTestUtils#runfilesDir}).
   */
  public BinTools getIntegrationBinTools(FileSystem fileSystem, BlazeDirectories directories)
      throws IOException {
    Path embeddedBinariesRoot = directories.getEmbeddedBinariesRoot();
    embeddedBinariesRoot.createDirectoryAndParents();

    Path runfiles = fileSystem.getPath(BlazeTestUtils.runfilesDir());
    // Copy over everything in embedded_scripts.
    Collection<Path> files = new ArrayList<>();
    for (String embeddedScriptPath : TestConstants.EMBEDDED_SCRIPTS_PATHS) {
      Path embeddedScripts = runfiles.getRelative(embeddedScriptPath);
      if (embeddedScripts.exists()) {
        files.addAll(embeddedScripts.getDirectoryEntries());
      } else {
        System.err.println("test does not have " + embeddedScripts);
      }
    }

    for (Path fromFile : files) {
      try {
        embeddedBinariesRoot.getChild(fromFile.getBaseName()).createSymbolicLink(fromFile);
      } catch (IOException e) {
        System.err.println("Could not symlink: " + e.getMessage());
      }
    }

    return BinTools.forIntegrationTesting(
        directories,
        TestConstants.EMBEDDED_TOOLS);
  }
}
