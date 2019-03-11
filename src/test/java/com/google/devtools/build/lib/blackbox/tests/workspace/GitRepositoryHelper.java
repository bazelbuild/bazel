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

import com.google.devtools.build.lib.blackbox.framework.BlackBoxTestContext;
import com.google.devtools.build.lib.blackbox.framework.ProcessResult;
import java.nio.file.Path;

/**
 * Helper class for working with local git repository in tests.
 */
public class GitRepositoryHelper {
  private final BlackBoxTestContext context;
  private final Path root;

  public GitRepositoryHelper(BlackBoxTestContext context, Path root) {
    this.context = context;
    this.root = root;
  }

  Path init() throws Exception {
    runGit("init");
    runGit("config", "user.email", "'me@example.com'");
    runGit("config", "user.name", "'E X Ample'");
    return root;
  }

  String addAll() throws Exception {
    runGit("rm", "--cached");
    return runGit("add", ".");
  }

  String commit(String commitMessage) throws Exception {
    return runGit("commit", "-m", commitMessage);
  }

  String tag(String tagName) throws Exception {
    return runGit("tag", tagName);
  }

  String getHead() throws Exception {
    return runGit("rev-parse", "--short", "HEAD");
  }

  private String runGit(String... arguments) throws Exception {
    ProcessResult result = context.runBinary(root, "git", arguments);
    return result.outString();
  }
}
