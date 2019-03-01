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

import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Utility class for keeping common assertion methods for workspace tests.
 */
public class AssertHelper {

  /**
   * Assert that the file exists and contains exactly one line with the certain text.
   * @param path - path to file
   * @param text - text expected in the file
   * @throws IOException if any file operation failed
   */
  static void assertOneLineFile(Path path, String text) throws IOException {
    assertThat(Files.exists(path)).isTrue();
    List<String> lines = PathUtils.readFile(path);
    assertThat(lines.size()).isEqualTo(1);
    assertThat(lines.get(0)).isEqualTo(text);
  }
}
