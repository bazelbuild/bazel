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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule.ResolutionReason;
import com.google.devtools.build.lib.cmdline.RepositoryName;

/** Specifies that a module should be retrieved from a Git repository. */
@AutoValue
public abstract class GitOverride implements NonRegistryOverride {
  public static GitOverride create(
      String remote,
      String commit,
      ImmutableList<String> patches,
      ImmutableList<String> patchCmds,
      int patchStrip) {
    return new AutoValue_GitOverride(remote, commit, patches, patchCmds, patchStrip);
  }

  /** The URL pointing to the git repository. */
  public abstract String getRemote();

  /** The commit hash to use. */
  public abstract String getCommit();

  /** The patches to apply after fetching from Git. Should be a list of labels. */
  public abstract ImmutableList<String> getPatches();

  /** The patch commands to execute after fetching from Git. Should be a list of commands. */
  public abstract ImmutableList<String> getPatchCmds();

  /** The number of path segments to strip from the paths in the supplied patches. */
  public abstract int getPatchStrip();

  /** Returns the {@link RepoSpec} that defines this repository. */
  @Override
  public RepoSpec getRepoSpec(RepositoryName repoName) {
    ImmutableMap.Builder<String, Object> attrBuilder = ImmutableMap.builder();
    attrBuilder
        .put("name", repoName.getName())
        .put("remote", getRemote())
        .put("commit", getCommit())
        .put("patches", getPatches())
        .put("patch_cmds", getPatchCmds())
        .put("patch_args", ImmutableList.of("-p" + getPatchStrip()));
    return RepoSpec.builder()
        .setBzlFile("@bazel_tools//tools/build_defs/repo:git.bzl")
        .setRuleClassName("git_repository")
        .setAttributes(AttributeValues.create(attrBuilder.buildOrThrow()))
        .build();
  }

  @Override
  public ResolutionReason getResolutionReason() {
    return ResolutionReason.GIT_OVERRIDE;
  }
}
