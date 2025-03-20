// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.windows.util;

import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.fail;

import com.google.common.base.Strings;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.windows.WindowsFileOperations;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/** Utilities for running Java tests on Windows. */
public final class WindowsTestUtil {

  /** A path where temp files can be created. It is NOT owned by this class. */
  private final String scratchRoot;

  public WindowsTestUtil(String scratchRoot) {
    this.scratchRoot = scratchRoot;
  }

  /**
   * Create directory junctions then assert their existence.
   *
   * <p>Each key in the map is a junction path, relative to {@link #scratchRoot}. These are the link
   * names.
   *
   * <p>Each value in the map is a directory or junction path, also relative to {@link
   * #scratchRoot}. These are the link targets.
   */
  public void createJunctions(Map<String, String> links) throws Exception {
    for (Map.Entry<String, String> e : links.entrySet()) {
      WindowsFileOperations.createJunction(
          scratchRoot + "/" + e.getKey(), scratchRoot + "/" + e.getValue());
    }

    for (Map.Entry<String, String> e : links.entrySet()) {
      assertWithMessage(
              String.format("Could not create junction '%s' -> '%s'", e.getKey(), e.getValue()))
          .that(new File(scratchRoot, e.getKey()).exists())
          .isTrue();
    }
  }

  /**
   * Create symbolic links.
   *
   * <p>Each key in the map is a symlink path relative to {@link #scratchRoot}. These are the link
   * names.
   *
   * <p>Each value in the map is a file path relative to {@link #scratchRoot}. These are the link
   * targets.
   */
  public void createSymlinks(Map<String, String> links) throws Exception {
    for (Map.Entry<String, String> entry : links.entrySet()) {
      WindowsFileOperations.createSymlink(
          scratchRoot + "/" + entry.getKey(), scratchRoot + "/" + entry.getValue());
    }
  }

  /** Delete everything under {@link #scratchRoot}/path. */
  public void deleteAllUnder(String path) throws IOException {
    if (Strings.isNullOrEmpty(path)) {
      path = scratchRoot;
    } else {
      path = scratchRoot + "\\" + path;
    }
    if (new File(path).exists()) {
      runCommand("cmd.exe /c rd /s /q \"" + path + "\"");
    }
  }

  /** Create a directory under `path`, relative to {@link #scratchRoot}. */
  public java.nio.file.Path scratchDir(String path) throws IOException {
    return Files.createDirectories(new File(scratchRoot, path).toPath());
  }

  /** Create a file with the given contents under `path`, relative to {@link #scratchRoot}. */
  public java.nio.file.Path scratchFile(String path, String... contents) throws IOException {
    File fd = new File(scratchRoot, path);
    Files.createDirectories(fd.toPath().getParent());
    try (FileWriter w = new FileWriter(fd)) {
      for (String line : contents) {
        w.write(line);
        w.write('\n');
      }
    }
    return fd.toPath();
  }

  /** Run a Command Prompt command. */
  public static void runCommand(String cmd) throws IOException {
    Process p = Runtime.getRuntime().exec(cmd);
    try {
      // Wait no more than 5 seconds to create all junctions.
      p.waitFor(5, TimeUnit.SECONDS);
    } catch (InterruptedException e) {
      fail("Failed to execute command; cmd: " + cmd);
    }
    assertWithMessage("Command failed: " + cmd).that(p.exitValue()).isEqualTo(0);
  }

  public Path createVfsPath(FileSystem fs, String path) throws IOException {
    return fs.getPath(scratchRoot + "/" + path);
  }
}
