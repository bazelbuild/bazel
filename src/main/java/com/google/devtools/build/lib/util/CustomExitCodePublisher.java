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
package com.google.devtools.build.lib.util;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.annotation.Nullable;

/**
 * Provides an external way for the Bazel server to communicate its exit code to the client, when
 * the main gRPC channel is unavailable because the exit is too abrupt or originated in an async
 * thread.
 *
 * <p>Uses Java 8 {@link Path} objects rather than Bazel ones to avoid depending on the rest of
 * Bazel.
 */
public class CustomExitCodePublisher {
  private static final String EXIT_CODE_FILENAME = "exit_code_to_use_on_abrupt_exit";
  @Nullable private static Path abruptExitCodeFilePath = null;

  public static void setAbruptExitStatusFileDir(String path) {
    abruptExitCodeFilePath = Paths.get(path).resolve(EXIT_CODE_FILENAME);
  }

  public static void maybeWriteExitStatusFile(int exitCode) {
    if (abruptExitCodeFilePath != null) {
      try {
        Files.write(
            abruptExitCodeFilePath, String.valueOf(exitCode).getBytes(StandardCharsets.UTF_8));
      } catch (IOException ioe) {
        System.err.printf(
            "io error writing %d to abrupt exit status file %s: %s\n",
            exitCode, abruptExitCodeFilePath, ioe.getMessage());
      }
    }
  }
}
