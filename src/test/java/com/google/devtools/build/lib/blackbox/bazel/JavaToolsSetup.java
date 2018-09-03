// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.blackbox.bazel;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.blackbox.framework.BlackBoxTestContext;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.framework.ToolsSetup;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

/** Setup for Bazel Java tools */
public class JavaToolsSetup implements ToolsSetup {
  @Override
  public void setup(BlackBoxTestContext context) throws IOException {
    Path jdkDirectory = context.getWorkDir().resolve("tools/jdk");
    List<Path> buildFiles =
        Files.list(jdkDirectory)
            .filter(path -> path.getFileName().toString().startsWith("BUILD."))
            .collect(Collectors.toList());
    assertThat(buildFiles.size()).isAtMost(1);
    if (!buildFiles.isEmpty()) {
      Path buildFile = jdkDirectory.resolve("BUILD");
      Files.copy(buildFiles.get(0), buildFile);
      assertThat(buildFile.toFile().setWritable(true)).isTrue();
    }

    String packageSubpath = JavaToolsSetup.class.getPackage().getName().replace('.', '/');
    Path langToolsJar =
        RunfilesUtil.find(String.format("io_bazel/src/test/java/%s/langtools.jar", packageSubpath));
    Files.copy(langToolsJar, jdkDirectory.resolve("langtools.jar"));
    PathUtils.writeBuild(
        jdkDirectory, "filegroup(name = \"test-langtools\", srcs = [\"langtools.jar\"])");

    PathUtils.copyTree(
        RunfilesUtil.find("io_bazel/third_party/java/jdk/langtools/BUILD").getParent(),
        context.getWorkDir().resolve("third_party/java/jdk/langtools"));
  }
}
