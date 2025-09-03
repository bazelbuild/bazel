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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.StarlarkList;

/**
 * Builder for a {@link RepoSpec} object that indicates how to materialize a repo corresponding to a
 * {@code git_repository} repo rule call.
 */
public class GitRepoSpecBuilder {

  public static final RepoRuleId GIT_REPOSITORY =
      new RepoRuleId(
          Label.parseCanonicalUnchecked("@@bazel_tools//tools/build_defs/repo:git.bzl"),
          "git_repository");

  private final Dict.Builder<String, Object> attrBuilder = Dict.builder();

  public GitRepoSpecBuilder() {}

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
  public GitRepoSpecBuilder setInitSubmodules(boolean initSubmodules) {
    setAttr("init_submodules", initSubmodules);
    setAttr("recursive_init_submodules", initSubmodules);
    return this;
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
  public GitRepoSpecBuilder setRemoteModuleFile(
      ArchiveRepoSpecBuilder.RemoteFile remoteModuleFile) {
    setAttr("remote_module_file_urls", remoteModuleFile.urls());
    setAttr("remote_module_file_integrity", remoteModuleFile.integrity());
    return this;
  }

  public RepoSpec build() {
    return new RepoSpec(GIT_REPOSITORY, AttributeValues.create(attrBuilder.buildImmutable()));
  }

  @CanIgnoreReturnValue
  private GitRepoSpecBuilder setAttr(String name, String value) {
    if (value != null && !value.isEmpty()) {
      attrBuilder.put(name, value);
    }
    return this;
  }

  @CanIgnoreReturnValue
  private GitRepoSpecBuilder setAttr(String name, boolean value) {
    attrBuilder.put(name, value);
    return this;
  }

  @CanIgnoreReturnValue
  private GitRepoSpecBuilder setAttr(String name, List<?> value) {
    if (value != null && !value.isEmpty()) {
      attrBuilder.put(name, StarlarkList.immutableCopyOf(value));
    }
    return this;
  }
}
