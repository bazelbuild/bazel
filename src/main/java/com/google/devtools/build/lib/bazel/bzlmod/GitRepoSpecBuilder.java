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
import com.google.common.collect.ImmutableMap;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import net.starlark.java.eval.StarlarkInt;
import java.util.List;

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
    return setAttr("name", repoName);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setRemote(String remoteRepoUrl) {
    return setAttr("remote", remoteRepoUrl);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setCommit(String gitCommitHash) {
    return setAttr("commit", gitCommitHash);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setShallowSince(String shallowSince) {
    return setAttr("shallow_since", shallowSince);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setTag(String tag) {
    return setAttr("tag", tag);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setBranch(String branch) {
    return setAttr("branch", branch);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setInitSubmodules(boolean initSubmodules) {
    return setAttr("init_submodules", initSubmodules);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setRecursiveInitSubmodules(boolean recursiveInitSubmodules) {
    return setAttr("recursive_init_submodules", recursiveInitSubmodules);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setVerbose(boolean verbose) {
    return setAttr("verbose", verbose);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setStripPrefix(String stripPrefix) {
    return setAttr("strip_prefix", stripPrefix);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setPatches(List<String> patches) {
    return setAttr("patches", patches);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setPatchTool(String patchTool) {
    return setAttr("patch_tool", patchTool);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setPatchArgs(List<String> patchArgs) {
    return setAttr("patch_args", patchArgs);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setPatchCmds(List<String> patchCmds) {
    return setAttr("patch_cmds", patchCmds);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setPatchCmdsWin(List<String> patchCmdsWin) {
    return setAttr("patch_cmds_win", patchCmdsWin);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setBuildFile(String buildFile) {
    return setAttr("build_file", buildFile);
  }

  @CanIgnoreReturnValue
  public GitRepoSpecBuilder setBuildFileContent(String buildFileContent) {
    return setAttr("build_file_content", buildFileContent);
  }

  public RepoSpec build() {
    return RepoSpec.builder()
        .setBzlFile(GIT_REPO_PATH)
        .setRuleClassName("git_repository")
        .setAttributes(AttributeValues.create(attrBuilder.buildOrThrow()))
        .build();
  }

  private GitRepoSpecBuilder setAttr(String name, String value) {
    if (value != null && !value.isEmpty()) {
      attrBuilder.put(name, value);
    }
    return this;
  }

  private GitRepoSpecBuilder setAttr(String name, boolean value) {
    attrBuilder.put(name, value);
    return this;
  }

  private GitRepoSpecBuilder setAttr(String name, List<String> value) {
    if (value != null && !value.isEmpty()) {
      attrBuilder.put(name, value);
    }
    return this;
  }
}
