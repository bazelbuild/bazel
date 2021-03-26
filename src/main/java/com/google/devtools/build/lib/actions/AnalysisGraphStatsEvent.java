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
package com.google.devtools.build.lib.actions;

/**
 * Event transporting data about the size/shape of the analysis graph. Only emitted when Bazel is
 * forced to visit the entire analysis graph (for action/artifact conflict checking). See {@link
 * com.google.devtools.build.lib.skyframe.SkyframeBuildView#shouldCheckForConflicts}.
 */
public final class AnalysisGraphStatsEvent {
  private final TotalAndConfiguredTargetOnlyMetric actionLookupValueCount;
  private final TotalAndConfiguredTargetOnlyMetric actionCount;
  private final int outputArtifactCount;

  public AnalysisGraphStatsEvent(
      TotalAndConfiguredTargetOnlyMetric actionLookupValueCount,
      TotalAndConfiguredTargetOnlyMetric actionCount,
      int outputArtifactCount) {
    this.actionLookupValueCount = actionLookupValueCount;
    this.actionCount = actionCount;
    this.outputArtifactCount = outputArtifactCount;
  }

  public TotalAndConfiguredTargetOnlyMetric getActionLookupValueCount() {
    return actionLookupValueCount;
  }

  public TotalAndConfiguredTargetOnlyMetric getActionCount() {
    return actionCount;
  }

  public int getOutputArtifactCount() {
    return outputArtifactCount;
  }
}
