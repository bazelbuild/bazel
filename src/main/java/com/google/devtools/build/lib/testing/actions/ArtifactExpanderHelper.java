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

package com.google.devtools.build.lib.testing.actions;

import com.google.common.base.Functions;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.MiddlemanType;

/** Helper utility to create {@link ArtifactExpander} instances. */
public final class ArtifactExpanderHelper {
  private ArtifactExpanderHelper() {}

  public static ArtifactExpander actionGraphArtifactExpander(ActionGraph actionGraph) {
    return (mm, output) -> {
      // Skyframe is stricter in that it checks that "mm" is a input of the action, because
      // it cannot expand arbitrary middlemen without access to a global action graph.
      // We could check this constraint here too, but it seems unnecessary. This code is
      // going away anyway.
      Preconditions.checkArgument(mm.isMiddlemanArtifact(), "%s is not a middleman artifact", mm);
      ActionAnalysisMetadata middlemanAction = actionGraph.getGeneratingAction(mm);
      Preconditions.checkState(middlemanAction != null, mm);
      // TODO(bazel-team): Consider expanding recursively or throwing an exception here.
      // Most likely, this code will cause silent errors if we ever have a middleman that
      // contains a middleman.
      if (middlemanAction.getActionType() == MiddlemanType.AGGREGATING_MIDDLEMAN) {
        Artifact.addNonMiddlemanArtifacts(
            middlemanAction.getInputs().toList(), output, Functions.identity());
      }
    };
  }
}
