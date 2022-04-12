// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.annotation.Nullable;

/**
 * Provides an external way for the Bazel server to communicate a failure_detail protobuf to its
 * user, when the main gRPC channel is unavailable because the server's exit is too abrupt, or the
 * failure occurred outside of a command.
 *
 * <p>Uses Java 8 {@link Path} objects rather than Bazel ones to avoid depending on the rest of
 * Bazel.
 */
public class CustomFailureDetailPublisher {
  @Nullable private static volatile Path failureDetailFilePath = null;

  private CustomFailureDetailPublisher() {}

  public static void setFailureDetailFilePath(String path) {
    failureDetailFilePath = Paths.get(path);
  }

  @VisibleForTesting
  public static void resetFailureDetailFilePath() {
    failureDetailFilePath = null;
  }

  public static boolean maybeWriteFailureDetailFile(FailureDetail failureDetail) {
    Path path = CustomFailureDetailPublisher.failureDetailFilePath;
    if (path != null) {
      try {
        Files.write(path, failureDetail.toByteArray());
        return true;
      } catch (IOException ioe) {
        System.err.printf(
            "io error writing failure detail to file %s.\nfailure_detail: %s\nIOException: %s",
            path, failureDetail, ioe.getMessage());
      }
    }
    return false;
  }
}
