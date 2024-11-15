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

import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule.ResolutionReason;
import com.google.devtools.build.lib.cmdline.Label;

/**
 * Specifies that a module should be retrieved from a Git repository.
 *
 * @param remote The URL pointing to the git repository.
 * @param commit The commit hash to use.
 * @param patches The labels of patches to apply after fetching from Git.
 * @param patchCmds The patch commands to execute after fetching from Git. Should be a list of
 *     commands.
 * @param patchStrip The number of path segments to strip from the paths in the supplied patches.
 * @param initSubmodules Whether submodules in the fetched repo should be recursively initialized.
 * @param stripPrefix The directory prefix to strip from the extracted files.
 */
public record GitOverride(
    String remote,
    String commit,
    ImmutableList<Label> patches,
    ImmutableList<String> patchCmds,
    int patchStrip,
    boolean initSubmodules,
    String stripPrefix)
    implements NonRegistryOverride {
  public GitOverride {
    requireNonNull(remote, "remote");
    requireNonNull(commit, "commit");
    requireNonNull(patches, "patches");
    requireNonNull(patchCmds, "patchCmds");
    requireNonNull(stripPrefix, "stripPrefix");
  }

  public static GitOverride create(
      String remote,
      String commit,
      ImmutableList<Label> patches,
      ImmutableList<String> patchCmds,
      int patchStrip,
      boolean initSubmodules,
      String stripPrefix) {
    return new GitOverride(
        remote, commit, patches, patchCmds, patchStrip, initSubmodules, stripPrefix);
  }

  /** Returns the {@link RepoSpec} that defines this repository. */
  @Override
  public RepoSpec getRepoSpec() {
    return new GitRepoSpecBuilder()
        .setRemote(remote())
        .setCommit(commit())
        .setPatches(patches())
        .setPatchCmds(patchCmds())
        .setPatchArgs(ImmutableList.of("-p" + patchStrip()))
        .setInitSubmodules(initSubmodules())
        .setStripPrefix(stripPrefix())
        .build();
  }

  @Override
  public ResolutionReason getResolutionReason() {
    return ResolutionReason.GIT_OVERRIDE;
  }
}
