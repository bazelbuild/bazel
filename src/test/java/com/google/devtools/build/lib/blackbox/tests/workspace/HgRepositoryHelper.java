// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.blackbox.tests.workspace;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.blackbox.framework.BlackBoxTestContext;
import com.google.devtools.build.lib.blackbox.framework.ProcessResult;
import java.nio.file.Path;

/**
 * Helper class for working with local mercurial repository in tests. Should not be used outside of tests.
 */
class HgRepositoryHelper {
  private final BlackBoxTestContext context;
  private final Path root;

  /**
   * Constructs the helper.
   *
   * @param context {@link BlackBoxTestContext} for running mercurial process
   * @param root working directory for running mercurial process, expected to be existing.
   */
  HgRepositoryHelper(BlackBoxTestContext context, Path root) {
    this.context = context;
    this.root = root;
  }

  /**
   * Calls 'hg init'.
   *
   * @throws Exception related to the invocation of the external hg process (like IOException or
   *     TimeoutException) or ProcessRunnerException if the process returned not expected return
   *     code.
   */
  void init() throws Exception {
    runHg("init");
    runHg("commit", "-m", "Initial commit", "--config", "ui.allowemptycommit=1");
  }

  /**
   * Add all files for the next commit.
   *
   * @throws Exception related to the invocation of the external hg process (like IOException or
   *     TimeoutException) or ProcessRunnerException if the process returned not expected return
   *     code.
   */
  void addAll() throws Exception {
    runHg("add");
  }

  /**
   * Commits all staged changed.
   *
   * @param commitMessage commit message
   * @throws Exception related to the invocation of the external hg process (like IOException or
   *     TimeoutException) or ProcessRunnerException if the process returned not expected return
   *     code.
   */
  void commit(String commitMessage) throws Exception {
    runHg("commit", "-m", commitMessage);
  }

  /**
   * Tags the HEAD commit.
   *
   * @param tagName tag name
   * @throws Exception related to the invocation of the external hg process (like IOException or
   *     TimeoutException) or ProcessRunnerException if the process returned not expected return
   *     code.
   */
  void tag(String tagName) throws Exception {
    runHg("tag", tagName);
  }

  /**
   * Creates the new branch with the specified name at HEAD.
   *
   * @param branchName branch name
   * @throws Exception related to the invocation of the external hg process (like IOException or
   *     TimeoutException) or ProcessRunnerException if the process returned not expected return
   *     code.
   */
  void createNewBranch(String branchName) throws Exception {
    runHg("branch", branchName);
  }

  /**
   * Closes the current branch.
   *
   * @throws Exception related to the invocation of the external hg process (like IOException or
   *     TimeoutException) or ProcessRunnerException if the process returned not expected return
   *     code.
   */
  void closeBranch() throws Exception {
    runHg("commit", "--close-branch", "-m", "Closing feature branch");
  }

  /**
   * Checks out specified revision or reference.
   *
   * @param ref reference to check out
   * @throws Exception related to the invocation of the external hg process (like IOException or
   *     TimeoutException) or ProcessRunnerException if the process returned not expected return
   *     code.
   */
  void checkout(String ref) throws Exception {
    runHg("update", "-r", ref);
  }

  /**
   * Returns the HEAD's commit hash.
   *
   * @throws Exception related to the invocation of the external hg process (like IOException or
   *     TimeoutException) or ProcessRunnerException if the process returned not expected return
   *     code.
   */
  String getHead() throws Exception {
    return runHg("id", "--id");
  }

  private String runHg(String... arguments) throws Exception {
    ProcessResult result =
        context.runBinary(
            root, "hg", false, ImmutableMap.of("HGUSER", "E X Ample <me@example.com>"), arguments);
    return result.outString();
  }
}
