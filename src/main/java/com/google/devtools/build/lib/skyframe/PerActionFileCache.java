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
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import javax.annotation.Nullable;

/**
 * Cache provided by an {@link ActionExecutionFunction}, allowing Blaze to obtain artifact metadata
 * from the graph.
 *
 * <p>Data for the action's inputs is injected into this cache on construction, using the graph as
 * the source of truth.
 */
class PerActionFileCache implements MetadataProvider {
  private final ActionInputMap inputArtifactData;
  private final boolean missingArtifactsAllowed;

  /**
   * @param inputArtifactData Map from artifact to metadata, used to return metadata upon request.
   * @param missingArtifactsAllowed whether to tolerate missing artifacts: can happen during input
   *     discovery.
   */
  PerActionFileCache(ActionInputMap inputArtifactData, boolean missingArtifactsAllowed) {
    this.inputArtifactData = Preconditions.checkNotNull(inputArtifactData);
    this.missingArtifactsAllowed = missingArtifactsAllowed;
  }

  @Nullable
  @Override
  public FileArtifactValue getMetadata(ActionInput input) {
    // TODO(shahan): is this bypass needed?
    if (!(input instanceof Artifact)) {
      return null;
    }
    FileArtifactValue result = inputArtifactData.getMetadata(input);
    Preconditions.checkState(missingArtifactsAllowed || result != null, "null for %s", input);
    return result;
  }

  @Override
  public ActionInput getInput(String execPath) {
    return inputArtifactData.getInput(execPath);
  }
}
