// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * The result of downloading the top-level outputs of a single top-level target (or aspect) that
 * are only available as remote metadata, as requested by the download policy of the current
 * invocation.
 *
 * <p>The value records the output files that are present in the local filesystem even though the
 * metadata tracked for them in Skyframe is remote (which happens when they are downloaded without
 * their generating action being reexecuted). {@link FilesystemValueChecker} compares this record
 * against the local filesystem so that e.g. the deletion of such a file invalidates this node,
 * whose reevaluation restores the file - without invalidating the generating action.
 */
public final class ToplevelOutputsDownloadValue implements SkyValue {
  private final ImmutableMap<Artifact, FileContentsProxy> materializedOutputs;

  public ToplevelOutputsDownloadValue(
      ImmutableMap<Artifact, FileContentsProxy> materializedOutputs) {
    this.materializedOutputs = materializedOutputs;
  }

  /**
   * The output files that are present in the local filesystem while their metadata tracked in
   * Skyframe is remote, together with the contents proxy they were last observed with.
   */
  public ImmutableMap<Artifact, FileContentsProxy> getMaterializedOutputs() {
    return materializedOutputs;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof ToplevelOutputsDownloadValue that)) {
      return false;
    }
    return materializedOutputs.equals(that.materializedOutputs);
  }

  @Override
  public int hashCode() {
    return materializedOutputs.hashCode();
  }

  public static Key key(
      TopLevelActionLookupKeyWrapper completionKey, DownloadPolicy downloadPolicy) {
    return new AutoValue_ToplevelOutputsDownloadValue_Key(
        completionKey.topLevelArtifactContext(),
        completionKey.actionLookupKey(),
        downloadPolicy);
  }

  /**
   * An invocation's policy for downloading top-level outputs that are only available as remote
   * metadata. Since the policy determines which outputs to download, dependents of a {@link
   * ToplevelOutputsDownloadValue} must request a fresh node instead of reusing one that applied a
   * different policy (see {@link Key#downloadPolicy}).
   */
  public record DownloadPolicy(
      String outputsMode, String commandName, ImmutableList<String> downloadRegexes) {}

  /** The key of a {@link ToplevelOutputsDownloadValue}. */
  @AutoValue
  public abstract static class Key implements TopLevelActionLookupKeyWrapper {
    @Override
    public abstract ActionLookupKey actionLookupKey();

    abstract DownloadPolicy downloadPolicy();

    @Override
    public final SkyFunctionName functionName() {
      return SkyFunctions.TOPLEVEL_OUTPUTS_DOWNLOAD;
    }

    @Override
    public final boolean valueIsShareable() {
      return false;
    }
  }
}
