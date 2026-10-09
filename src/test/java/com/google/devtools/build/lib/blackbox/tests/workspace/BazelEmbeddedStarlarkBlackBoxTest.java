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

import com.google.devtools.build.lib.blackbox.framework.BuilderRunner;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.junit.Test;

/** Tests http_archive. */
public class BazelEmbeddedStarlarkBlackBoxTest extends AbstractBlackBoxTest {

  private static final String HELLO_FROM_EXTERNAL_REPOSITORY = "Hello from external repository!";
  private static final String HELLO_FROM_MAIN_REPOSITORY = "Hello from main repository!";

  @Test
  public void testHttpArchive() throws Exception {
    Path repo = context().getTmpDir().resolve("ext_repo");
    RepoWithRuleWritingTextGenerator generator = new RepoWithRuleWritingTextGenerator(repo);
    generator.withOutputText(HELLO_FROM_EXTERNAL_REPOSITORY).setupRepository();

    // pack the repository into an archive
    Path zipFile = context().getTmpDir().resolve("ext_repo.zip");
    assertThat(Files.exists(zipFile)).isFalse();
    try (ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(zipFile));
        Stream<Path> files = Files.list(repo)) {
      for (Path file : files.sorted().toList()) {
        zip.putNextEntry(new ZipEntry(file.getFileName().toString()));
        Files.copy(file, zip);
        zip.closeEntry();
      }
    }

    context()
        .write(
            "MODULE.bazel",
            "http_archive = use_repo_rule('@bazel_tools//tools/build_defs/repo:http.bzl',"
                + " 'http_archive')",
            String.format(
                "http_archive(name=\"ext\", urls=[\"%s\"],)", PathUtils.pathToFileURI(zipFile)));

    context()
        .write(
            "BUILD",
            RepoWithRuleWritingTextGenerator.loadRule("@ext"),
            RepoWithRuleWritingTextGenerator.callRule(
                "call_from_main", "main_out.txt", HELLO_FROM_MAIN_REPOSITORY));

    BuilderRunner bazel = bazel();
    // build the target from http_archive
    bazel.build("@ext//:" + RepoWithRuleWritingTextGenerator.TARGET);

    Path xPath = context().resolveBinPath(bazel, "external/+http_archive+ext/out");
    WorkspaceTestUtils.assertLinesExactly(xPath, HELLO_FROM_EXTERNAL_REPOSITORY);

    // and use the rule from http_archive in the main repository
    bazel.build("//:call_from_main");

    Path mainOutPath = context().resolveBinPath(bazel, "main_out.txt");
    WorkspaceTestUtils.assertLinesExactly(mainOutPath, HELLO_FROM_MAIN_REPOSITORY);
  }

  private BuilderRunner bazel() {
    return WorkspaceTestUtils.bazel(context());
  }
}
