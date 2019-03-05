// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.tests.workspace;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.blackbox.framework.BlackBoxTestContext;
import com.google.devtools.build.lib.blackbox.framework.BuilderRunner;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.util.OS;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Utility class for helping JUnit black box workspace tests.
 */
public class WorkspaceTestUtils {
  private static final String PATH = "PATH";

  /**
   * Create the BuilderRunner for workspace tests without MSYS/MINGW dependency on Windows.
   * @param context - BlackBoxTestContext instance
   * @return BuilderRunner for running Bazel
   */
  static BuilderRunner bazel(BlackBoxTestContext context) {
    if (!OS.WINDOWS.equals(OS.getCurrent())) {
      return context.bazel();
    }
    return context.bazel()
        .withEnv("BAZEL_SH", "C:/foo/bar/usr/bin/bash.exe")
        .withEnv(PATH, removeMsysFromPath(System.getenv(PATH)));
  }

  private static String removeMsysFromPath(String path) {
    if (!OS.WINDOWS.equals(OS.getCurrent())) {
      return path;
    }
    String[] parts = path.split(";");
    return Arrays.stream(parts)
        .filter(s -> !s.contains("msys")).collect(Collectors.joining(";"));
  }

  /**
   * Assert that the file exists and contains exactly one line with the certain text.
   * @param path - path to file
   * @param lines - lines of text expected in the file
   * @throws IOException if any file operation failed
   */
  static void assertLinesExactly(Path path, String... lines) throws IOException {
    Preconditions.checkState(lines.length > 0);

    assertThat(Files.exists(path)).isTrue();
    List<String> realLines = PathUtils.readFile(path);
    assertThat(realLines.size()).isEqualTo(lines.length);
    for (int i = 0; i < lines.length; i++) {
      assertThat(realLines.get(i)).isEqualTo(lines[i]);
    }
  }
}
