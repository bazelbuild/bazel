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

import com.google.devtools.build.lib.bazel.repository.DecompressorDescriptor;
import com.google.devtools.build.lib.bazel.repository.TarFunction;
import com.google.devtools.build.lib.blackbox.framework.BlackBoxTestEnvironment;
import com.google.devtools.build.lib.blackbox.framework.BuilderRunner;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import org.junit.Test;

/** Tests pkg_tar and http_archive. */
public class BazelEmbeddedStarlarkBlackBoxTest extends AbstractBlackBoxTest {

  private static final String HELLO_FROM_EXTERNAL_REPOSITORY = "Hello from external repository!";
  private static final String HELLO_FROM_MAIN_REPOSITORY = "Hello from main repository!";

  @Test
  public void testPkgTar() throws Exception {
    context().write("main/WORKSPACE", BlackBoxTestEnvironment.getWorkspaceWithDefaultRepos());
    context().write("main/foo.txt", "Hello World");
    context().write("main/bar.txt", "Hello World, again");
    context()
        .write(
            "main/BUILD",
            """
            load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

            pkg_tar(
                name = "data",
                srcs = [
                    "bar.txt",
                    "foo.txt",
                ],
            )
            """);

    BuilderRunner bazel = bazel();
    bazel.build("...");

    Path dataTarPath = context().resolveBinPath(bazel, "main/data.tar");
    assertThat(Files.exists(dataTarPath)).isTrue();

    Path directory = decompress(dataTarPath);
    assertThat(directory.toFile().exists()).isTrue();

    Map<String, Path> map =
        Arrays.stream(Objects.requireNonNull(directory.toFile().listFiles()))
            .collect(Collectors.toMap(File::getName, file -> Paths.get(file.getAbsolutePath())));

    WorkspaceTestUtils.assertLinesExactly(map.get("foo.txt"), "Hello World");
    WorkspaceTestUtils.assertLinesExactly(map.get("bar.txt"), "Hello World, again");
  }

  @Test
  public void testHttpArchive() throws Exception {
    Path repo = context().getTmpDir().resolve("ext_repo");
    RepoWithRuleWritingTextGenerator generator = new RepoWithRuleWritingTextGenerator(repo);
    generator.withOutputText(HELLO_FROM_EXTERNAL_REPOSITORY).setupRepository();

    // file where we will manually copy the built archive
    Path zipFile = context().getTmpDir().resolve("ext_repo.tar");
    assertThat(Files.exists(zipFile)).isFalse();

    context()
        .write(
            "WORKSPACE",
            BlackBoxTestEnvironment.getWorkspaceWithDefaultRepos(),
            String.format(
                "local_repository(name=\"ext_local\", path=\"%s\",)",
                PathUtils.pathForStarlarkFile(repo)),
            String.format(
                "http_archive(name=\"ext\", urls=[\"%s\"],)", PathUtils.pathToFileURI(zipFile)));

    context()
        .write(
            "BUILD",
            RepoWithRuleWritingTextGenerator.loadRule("@ext"),
            RepoWithRuleWritingTextGenerator.callRule(
                "call_from_main", "main_out.txt", HELLO_FROM_MAIN_REPOSITORY));

    // first build the archive and copy it into zipFile
    BuilderRunner bazel = bazel();
    String tarTarget = generator.getPkgTarTarget();
    bazel.build("@ext_local//:" + tarTarget);
    Path packedFile =
        context().resolveBinPath(bazel, String.format("external/ext_local/%s.tar", tarTarget));
    Files.copy(packedFile, zipFile);

    // now build the target from http_archive
    bazel.build("@ext//:" + RepoWithRuleWritingTextGenerator.TARGET);

    Path xPath = context().resolveBinPath(bazel, "external/ext/out");
    WorkspaceTestUtils.assertLinesExactly(xPath, HELLO_FROM_EXTERNAL_REPOSITORY);

    // and use the rule from http_archive in the main repository
    bazel.build("//:call_from_main");

    Path mainOutPath = context().resolveBinPath(bazel, "main_out.txt");
    WorkspaceTestUtils.assertLinesExactly(mainOutPath, HELLO_FROM_MAIN_REPOSITORY);
  }

  private BuilderRunner bazel() {
    return WorkspaceTestUtils.bazel(context());
  }

  private Path decompress(Path dataTarPath) throws Exception {
    FileSystem fs = FileSystems.getNativeFileSystem();
    com.google.devtools.build.lib.vfs.Path dataTarPathForDecompress =
        fs.getPath(dataTarPath.toAbsolutePath().toString());

    com.google.devtools.build.lib.vfs.Path directory =
        TarFunction.INSTANCE.decompress(
            DecompressorDescriptor.builder()
                .setArchivePath(dataTarPathForDecompress)
                .setDestinationPath(dataTarPathForDecompress.getParentDirectory())
                .build());
    return Paths.get(directory.getPathString());
  }

  // TODO(ichern) test tar quoting
}
