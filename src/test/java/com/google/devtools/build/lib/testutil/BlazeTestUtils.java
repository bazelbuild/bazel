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

package com.google.devtools.build.lib.testutil;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Some static utility functions for testing Blaze code. In contrast to {@link TestUtils}, these
 * functions are Blaze-specific.
 */
public class BlazeTestUtils {
  private BlazeTestUtils() {}

  /**
   * Populates the _embedded_binaries/ directory, containing all binaries/libraries, by symlinking
   * directories#getEmbeddedBinariesRoot() to the test's runfiles tree.
   */
  public static BinTools getIntegrationBinTools(BlazeDirectories directories) throws IOException {
    Path embeddedDir = directories.getEmbeddedBinariesRoot();
    FileSystemUtils.createDirectoryAndParents(embeddedDir);

    Path runfiles = directories.getFileSystem().getPath(BlazeTestUtils.runfilesDir());
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
        embeddedDir.getChild(fromFile.getBaseName()).createSymbolicLink(fromFile);
      } catch (IOException e) {
        System.err.println("Could not symlink: " + e.getMessage());
      }
    }

    return BinTools.forIntegrationTesting(
        directories, embeddedDir.toString(), TestConstants.EMBEDDED_TOOLS);
  }

  /**
   * Writes a FilesetRule to a String array.
   *
   * @param name the name of the rule.
   * @param out the output directory.
   * @param entries The FilesetEntry entries.
   * @return the String array of the rule.  One String for each line.
   */
  public static String[] createFilesetRule(String name, String out, String... entries) {
    return new String[] {
        String.format("Fileset(name = '%s', out = '%s',", name, out),
                      "        entries = [" +  Joiner.on(", ").join(entries) + "])"
    };
  }

  public static File undeclaredOutputDir() {
    String dir = System.getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (dir != null) {
      return new File(dir);
    }

    return TestUtils.tmpDirFile();
  }

  public static String runfilesDir() {
    String runfilesDirStr = TestUtils.getUserValue("TEST_SRCDIR");
    Preconditions.checkState(runfilesDirStr != null && runfilesDirStr.length() > 0,
        "TEST_SRCDIR unset or empty");
    return new File(runfilesDirStr).getAbsolutePath();
  }

  /** Creates an empty file, along with all its parent directories. */
  public static void makeEmptyFile(Path path) throws IOException {
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    FileSystemUtils.createEmptyFile(path);
  }

  /**
   * Changes the mtime of the file "path", which must exist.  No guarantee is
   * made about the new mtime except that it is different from the previous one.
   *
   * @throws IOException if the mtime could not be read or set.
   */
  public static void changeModtime(Path path)
    throws IOException {
    long prevMtime = path.getLastModifiedTime();
    long newMtime = prevMtime;
    do {
      newMtime += 1000;
      path.setLastModifiedTime(newMtime);
    } while (path.getLastModifiedTime() == prevMtime);
  }
}
