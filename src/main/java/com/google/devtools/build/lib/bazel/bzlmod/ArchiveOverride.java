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
import com.google.errorprone.annotations.InlineMe;

/**
 * Specifies that a module should be retrieved from an archive.
 *
 * @param urls The URLs pointing at the archives. Can be HTTP(S) or file URLs.
 * @param patches The labels of patches to apply after extracting the archive.
 * @param patchCmds The patch commands to execute after extracting the archive. Should be a list of
 *     commands.
 * @param integrity The subresource integirty metadata of the archive.
 * @param stripPrefix The prefix to strip from paths in the archive.
 * @param patchStrip The number of path segments to strip from the paths in the supplied patches.
 */
public record ArchiveOverride(
    ImmutableList<String> urls,
    ImmutableList<Label> patches,
    ImmutableList<String> patchCmds,
    String integrity,
    String stripPrefix,
    int patchStrip)
    implements NonRegistryOverride {
  public ArchiveOverride {
    requireNonNull(urls, "urls");
    requireNonNull(patches, "patches");
    requireNonNull(patchCmds, "patchCmds");
    requireNonNull(integrity, "integrity");
    requireNonNull(stripPrefix, "stripPrefix");
  }

  @InlineMe(replacement = "this.urls()")
  public ImmutableList<String> getUrls() {
    return urls();
  }

  @InlineMe(replacement = "this.patches()")
  public ImmutableList<Label> getPatches() {
    return patches();
  }

  @InlineMe(replacement = "this.patchCmds()")
  public ImmutableList<String> getPatchCmds() {
    return patchCmds();
  }

  @InlineMe(replacement = "this.integrity()")
  public String getIntegrity() {
    return integrity();
  }

  @InlineMe(replacement = "this.stripPrefix()")
  public String getStripPrefix() {
    return stripPrefix();
  }

  @InlineMe(replacement = "this.patchStrip()")
  public int getPatchStrip() {
    return patchStrip();
  }

  public static ArchiveOverride create(
      ImmutableList<String> urls,
      ImmutableList<Label> patches,
      ImmutableList<String> patchCmds,
      String integrity,
      String stripPrefix,
      int patchStrip) {
    return new ArchiveOverride(urls, patches, patchCmds, integrity, stripPrefix, patchStrip);
  }

  /** Returns the {@link RepoSpec} that defines this repository. */
  @Override
  public RepoSpec getRepoSpec() {
    return new ArchiveRepoSpecBuilder()
        .setUrls(getUrls())
        .setIntegrity(getIntegrity())
        .setStripPrefix(getStripPrefix())
        .setPatches(getPatches())
        .setPatchCmds(getPatchCmds())
        .setPatchStrip(getPatchStrip())
        .build();
  }

  @Override
  public ResolutionReason getResolutionReason() {
    return ResolutionReason.ARCHIVE_OVERRIDE;
  }
}
