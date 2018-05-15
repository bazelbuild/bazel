// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import javax.annotation.Nullable;

/**
 * Cache provided by an {@link ActionExecutionFunction}, allowing Blaze to obtain artifact metadata
 * from the graph.
 *
 * <p>Data for the action's inputs is injected into this cache on construction, using the graph as
 * the source of truth.
 */
class PerActionFileCache implements ActionInputFileCache {
  private final InputArtifactData inputArtifactData;
  private final boolean missingArtifactsAllowed;

  /**
   * @param inputArtifactData Map from artifact to metadata, used to return metadata upon request.
   * @param missingArtifactsAllowed whether to tolerate missing artifacts: can happen during input
   *     discovery.
   */
  PerActionFileCache(InputArtifactData inputArtifactData, boolean missingArtifactsAllowed) {
    this.inputArtifactData = Preconditions.checkNotNull(inputArtifactData);
    this.missingArtifactsAllowed = missingArtifactsAllowed;
  }

  @Nullable
  @Override
  public Metadata getMetadata(ActionInput input) {
    if (!(input instanceof Artifact)) {
      return null;
    }
    Metadata result = inputArtifactData.get(input);
    Preconditions.checkState(missingArtifactsAllowed || result != null, "null for %s", input);
    return result;
  }

  @Override
  public Path getInputPath(ActionInput input) {
    return ((Artifact) input).getPath();
  }

  @Override
  public boolean contentsAvailableLocally(ByteString digest) {
    return inputArtifactData.contains(digest);
  }

  @Nullable
  @Override
  public Artifact getInputFromDigest(ByteString digest) {
    return inputArtifactData.get(digest);
  }
}
