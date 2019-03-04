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

import com.google.devtools.build.lib.blackbox.framework.BlackBoxTestContext;
import com.google.devtools.build.lib.blackbox.framework.BuilderRunner;
import com.google.devtools.build.lib.util.OS;
import java.util.Arrays;
import java.util.stream.Collectors;

public class WorkspaceTestUtils {
  public static final String PATH = "PATH";
  public static final String BAZEL_SH = "BAZEL_SH";

  public static BuilderRunner bazel(BlackBoxTestContext context) {
    return context.bazel()
        .withEnv(BAZEL_SH, "C:/foo/bar/usr/bin/bash.exe")
        .withEnv(PATH, WorkspaceTestUtils.removeMsysFromPath(System.getenv(PATH)));
  }

  private static String removeMsysFromPath(String path) {
    if (!OS.WINDOWS.equals(OS.getCurrent())) return path;
    if (path.indexOf(';') == -1) {
      return path;
    }
    String[] parts = path.split(";");
    return Arrays.stream(parts)
        .filter(s -> !s.contains("msys")).collect(Collectors.joining(";"));
  }
}
