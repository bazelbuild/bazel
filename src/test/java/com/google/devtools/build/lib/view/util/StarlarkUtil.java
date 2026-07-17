// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.view.util;

import com.google.common.io.Files;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Paths;

/** Utility class to perform Starlark-related setup. */
public class StarlarkUtil {
  public static void setup(Scratch scratch) throws IOException {
    scratch.file("tools/build_rules/BUILD");
    scratch.file("rules/BUILD");
    copyExistingStarlarkFiles(scratch, "tools/build_rules", "rules");
    copyExistingStarlarkFiles(scratch, "third_party/bazel/tools/build_rules", "rules");
  }

  private static void copyExistingStarlarkFiles(Scratch scratch, String from, String to)
      throws IOException {
    File rulesDir = new File(from);
    if (rulesDir.exists() && rulesDir.isDirectory()) {
      for (String fileName : rulesDir.list()) {
        File file = new File(from + "/" + fileName);
        if (file.isFile() && (fileName.endsWith(".bzl") || fileName.endsWith(".scl"))) {
          String context = Files.asCharSource(file, Charset.defaultCharset()).read();
          Path path = scratch.resolve(to + "/" + fileName);
          if (path.exists()) {
            scratch.overwriteFile(path.getPathString(), context);
          } else {
            scratch.file(path.getPathString(), context);
          }
        }
      }
    }
  }

  public static void copyExistingStarlarkFile(MockToolsConfig mockToolsConfig, String bzlPath)
      throws IOException {
    String basename = bzlPath.substring(0, bzlPath.lastIndexOf('/'));
    mockToolsConfig.create(basename + "/BUILD");
    mockToolsConfig.create(
        bzlPath,
        new String(
            java.nio.file.Files.readString(
                Paths.get(BlazeTestUtils.runfilesDir(), "io_bazel", bzlPath))));
  }

  private StarlarkUtil() {}
}
