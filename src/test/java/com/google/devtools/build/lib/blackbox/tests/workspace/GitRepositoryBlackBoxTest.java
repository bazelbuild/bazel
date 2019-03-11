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
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Test;

public class GitRepositoryBlackBoxTest extends AbstractBlackBoxTest {

  private static final String HELLO_FROM_EXTERNAL_REPOSITORY = "Hello from GIT repository!";

  @Test
  public void testNewGitRepository() throws Exception {
    Path repo = context().getTmpDir().resolve("ext_repo");
    setupGitRepository(context(), repo);

    String buildFileContent = String.format("%s\n%s", RepoWithRuleWritingTextGenerator.loadRule(""),
        RepoWithRuleWritingTextGenerator.callRule("call_write_text", "out.txt",
            HELLO_FROM_EXTERNAL_REPOSITORY));
    context().write("WORKSPACE",
        "load(\"@bazel_tools//tools/build_defs/repo:git.bzl\", \"new_git_repository\")",
        "new_git_repository(",
        "  name='ext',",
        String.format("  remote='%s',", PathUtils.pathToFileURI(repo.resolve(".git"))),
        "  tag='first',",
        String.format("  build_file_content=\"\"\"%s\"\"\",", buildFileContent),
        ")");

    BuilderRunner bazel = WorkspaceTestUtils.bazel(context());
    bazel.build("@ext//:call_write_text");
    Path outPath = context().resolveBinPath(bazel, "external/ext/out.txt");
    WorkspaceTestUtils.assertLinesExactly(outPath, HELLO_FROM_EXTERNAL_REPOSITORY);
  }

  @Test
  public void testCloneAtCommit() throws Exception {
    Path repo = context().getTmpDir().resolve("ext_repo");
    String commit = setupGitRepository(context(), repo);

    String buildFileContent = String.format("%s\n%s", RepoWithRuleWritingTextGenerator.loadRule(""),
        RepoWithRuleWritingTextGenerator.callRule("call_write_text", "out.txt",
            HELLO_FROM_EXTERNAL_REPOSITORY));
    context().write("WORKSPACE",
        "load(\"@bazel_tools//tools/build_defs/repo:git.bzl\", \"new_git_repository\")",
        "new_git_repository(",
        "  name='ext',",
        String.format("  remote='%s',", PathUtils.pathToFileURI(repo.resolve(".git"))),
        String.format("  commit='%s',", commit),
        String.format("  build_file_content=\"\"\"%s\"\"\",", buildFileContent),
        ")");

    BuilderRunner bazel = WorkspaceTestUtils.bazel(context());
    bazel.build("@ext//:call_write_text");
    Path outPath = context().resolveBinPath(bazel, "external/ext/out.txt");
    WorkspaceTestUtils.assertLinesExactly(outPath, HELLO_FROM_EXTERNAL_REPOSITORY);
  }

  @Test
  public void testCloneAtMaster() throws Exception {
    Path repo = context().getTmpDir().resolve("ext_repo");
    String commit = setupGitRepository(context(), repo);

    String buildFileContent = String.format("%s\n%s", RepoWithRuleWritingTextGenerator.loadRule(""),
        RepoWithRuleWritingTextGenerator.callRule("call_write_text", "out.txt",
            HELLO_FROM_EXTERNAL_REPOSITORY));
    context().write("WORKSPACE",
        "load(\"@bazel_tools//tools/build_defs/repo:git.bzl\", \"new_git_repository\")",
        "new_git_repository(",
        "  name='ext',",
        String.format("  remote='%s',", PathUtils.pathToFileURI(repo.resolve(".git"))),
        "  branch='master',",
        String.format("  build_file_content=\"\"\"%s\"\"\",", buildFileContent),
        ")");

    BuilderRunner bazel = WorkspaceTestUtils.bazel(context());
    bazel.build("@ext//:call_write_text");
    Path outPath = context().resolveBinPath(bazel, "external/ext/out.txt");
    WorkspaceTestUtils.assertLinesExactly(outPath, HELLO_FROM_EXTERNAL_REPOSITORY);
  }

  static String setupGitRepository(BlackBoxTestContext context, Path repo) throws Exception {
    PathUtils.deleteTree(repo);
    Files.createDirectories(repo);
    GitRepositoryHelper gitRepository = new GitRepositoryHelper(context, repo);
    gitRepository.init();

    RepoWithRuleWritingTextGenerator generator = new RepoWithRuleWritingTextGenerator(repo);
    generator.withOutputText(HELLO_FROM_EXTERNAL_REPOSITORY)
        .skipBuildFile()
        .setupRepository();

    gitRepository.addAll();
    gitRepository.commit("Initial commit");
    gitRepository.tag("first");
    return gitRepository.getHead();
  }
}
