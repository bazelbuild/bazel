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

package com.google.devtools.build.lib.windows.runfiles;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.Map;

/** Utility class for getting runfiles in tests on Windows. */
public final class WindowsRunfiles {

  private static Map<String, String> runfiles;

  /**
   * Returns the actual path on the system corresponding to a given runfiles path. On Unix-like with
   * the symlink tree, this function could just returns the runfiles path and let the symlinks do
   * the actual resolution.
   */
  public static String getRunfile(String runfilesPath) throws IOException {
    ensureRunfilesParsed();
    return runfiles.get(runfilesPath);
  }

  private static synchronized void ensureRunfilesParsed() throws IOException {
    if (runfiles != null) {
      return;
    }

    runfiles = new HashMap<>();
    InputStream fis = new FileInputStream(System.getenv("RUNFILES_MANIFEST_FILE"));
    InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
    try (BufferedReader br = new BufferedReader(isr)) {
      String line;
      while ((line = br.readLine()) != null) {
        line = line.trim();
        if (!line.isEmpty()) {
          // TODO(bazel-team): This is buggy when the path contains spaces, we should fix the
          // manifest format.
          String[] splitLine = line.split(" ");
          if (splitLine.length != 2) {
            throw new IllegalStateException("Runfiles manifest entry contains more than one space");
          }

          runfiles.put(splitLine[0], splitLine[1]);
        }
      }
    }
  }
}
