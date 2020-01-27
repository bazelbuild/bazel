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
import static com.google.common.truth.Truth8.assertThat;

import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link RepoWithRuleWritingTextGenerator}. */
@RunWith(JUnit4.class)
public class RepoWithRuleWritingTextGeneratorTest {
  private static final String BUILD_TEXT =
      "load(\"@bazel_tools//tools/build_defs/pkg:pkg.bzl\", \"pkg_tar\")\n"
          + "load('//:helper.bzl', 'write_to_file')\n"
          + "write_to_file(name = 'write_text', filename = 'out', text ='HELLO')\n"
          + "pkg_tar(name = \"pkg_tar_write_text\", srcs = glob([\"*\"]),)";
  private static final String BUILD_TEXT_PARAMS =
      "load(\"@bazel_tools//tools/build_defs/pkg:pkg.bzl\", \"pkg_tar\")\n"
          + "load('//:helper.bzl', 'write_to_file')\n"
          + "write_to_file(name = 'target', filename = 'file', text ='text')\n"
          + "pkg_tar(name = \"pkg_tar_target\", srcs = glob([\"*\"]),)";

  @Test
  public void testOutput() throws IOException {
    Path directory = Files.createTempDirectory("test_repo_output");
    try {
      RepoWithRuleWritingTextGenerator generator = new RepoWithRuleWritingTextGenerator(directory);

      Path repository = generator.setupRepository();
      assertThat(repository).isEqualTo(directory);
      assertThat(Files.exists(repository)).isTrue();

      String buildText = String.join("\n", PathUtils.readFile(repository.resolve("BUILD")));
      assertThat(buildText).isEqualTo(BUILD_TEXT);
      assertThat(generator.getPkgTarTarget()).isEqualTo("pkg_tar_write_text");
    } finally {
      PathUtils.deleteTree(directory);
    }
  }

  @Test
  public void testOutputWithParameters() throws IOException {
    Path directory = Files.createTempDirectory("test_repo_output_with_parameters");
    try {
      RepoWithRuleWritingTextGenerator generator =
          new RepoWithRuleWritingTextGenerator(directory)
              .withTarget("target")
              .withOutFile("file")
              .withOutputText("text");

      Path repository = generator.setupRepository();
      assertThat(repository).isEqualTo(directory);
      assertThat(Files.exists(repository)).isTrue();

      String buildText = String.join("\n", PathUtils.readFile(repository.resolve("BUILD")));
      assertThat(buildText).isEqualTo(BUILD_TEXT_PARAMS);
      assertThat(generator.getPkgTarTarget()).isEqualTo("pkg_tar_target");
    } finally {
      PathUtils.deleteTree(directory);
    }
  }

  @Test
  public void testStaticMethods() {
    String loadText = RepoWithRuleWritingTextGenerator.loadRule("@my_repo");
    assertThat(loadText).isEqualTo("load('@my_repo//:helper.bzl', 'write_to_file')");

    String callText =
        RepoWithRuleWritingTextGenerator.callRule("my_target", "filename", "out_text");
    assertThat(callText)
        .isEqualTo("write_to_file(name = 'my_target', filename = 'filename', text ='out_text')");
  }
}
