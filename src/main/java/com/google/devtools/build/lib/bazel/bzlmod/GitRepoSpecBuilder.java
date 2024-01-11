// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import net.starlark.java.eval.StarlarkInt;

/**
 * Builder for a {@link RepoSpec} object that indicates how to materialize a repo corresponding to
 * a {@code git_repository} repo rule call.
 */
public class GitRepoSpecBuilder {

  public static final String GIT_REPO_PATH = "@bazel_tools//tools/build_defs/repo:git.bzl";

  private final ImmutableMap.Builder<String, Object> attrBuilder;

  public GitRepoSpecBuilder() {
    attrBuilder = new ImmutableMap.Builder<>();
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setRepoName(String repoName) {
    attrBuilder.put("name", repoName);
    return this;
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setRemote(String remoteRepoUrl) {
    attrBuilder.put("remote", remoteRepoUrl);
    return this;
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setCommit(String gitCommitHash) {
    attrBuilder.put("commit", gitCommitHash);
    return this;
  }

  public RepoSpec build() {
    attrBuilder.put("verbose", true);
    return RepoSpec.builder()
        .setBzlFile(GIT_REPO_PATH)
        .setRuleClassName("git_repository")
        .setAttributes(AttributeValues.create(attrBuilder.buildOrThrow()))
        .build();
  }
}
