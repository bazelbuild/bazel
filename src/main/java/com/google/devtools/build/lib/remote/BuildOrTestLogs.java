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

package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Recognizes build and test logs by exec path. These are the files that {@code
 * --remote_build_event_upload=minimal} still uploads.
 */
final class BuildOrTestLogs {
  private static final String OUTPUT_DIRECTORY_PREFIX = "bazel-out/";
  private static final String TEST_LOG_DIRECTORY = "/testlogs/";
  private static final String ACTION_STDOUT_PREFIX = "bazel-out/_tmp/actions/stdout-";
  private static final String ACTION_STDERR_PREFIX = "bazel-out/_tmp/actions/stderr-";

  private BuildOrTestLogs() {}

  /** Returns whether {@code execPath} is a test log or the stdout or stderr file of an action. */
  static boolean isBuildOrTestLog(PathFragment execPath) {
    String path = execPath.getPathString();
    if (!path.startsWith(OUTPUT_DIRECTORY_PREFIX)) {
      return false;
    }
    if (path.startsWith(ACTION_STDOUT_PREFIX) || path.startsWith(ACTION_STDERR_PREFIX)) {
      return true;
    }
    // Test logs are in bazel-out/<configuration>/testlogs.
    int configurationEnd = path.indexOf('/', OUTPUT_DIRECTORY_PREFIX.length());
    return configurationEnd != -1 && path.startsWith(TEST_LOG_DIRECTORY, configurationEnd);
  }
}
