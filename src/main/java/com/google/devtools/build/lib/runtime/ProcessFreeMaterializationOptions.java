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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.common.options.OptionsProvider;

/** Enforces the option invariants required by process-free materialization. */
public final class ProcessFreeMaterializationOptions {
  private ProcessFreeMaterializationOptions() {}

  public static boolean isEnabled(OptionsProvider options) {
    BuildRequestOptions buildOptions = options.getOptions(BuildRequestOptions.class);
    return buildOptions != null && buildOptions.getMaterializeProcessFreeActions();
  }

  public static void normalize(BuildRequest request) {
    BuildRequestOptions buildOptions = request.getBuildOptions();
    if (buildOptions == null || !buildOptions.getMaterializeProcessFreeActions()) {
      return;
    }
    buildOptions.setPerformAnalysisPhase(true);
    buildOptions.setPerformExecutionPhase(true);

    CommonCommandOptions commonOptions = request.getOptions(CommonCommandOptions.class);
    request.getOptions(KeepStateAfterBuildOption.class).setKeepStateAfterBuild(true);
    commonOptions.setTrackIncrementalState(true);

    var analysisOptions = request.getViewOptions();
    if (analysisOptions != null) {
      analysisOptions.setDiscardAnalysisCache(false);
    }
    ExecutionOptions executionOptions = request.getOptions(ExecutionOptions.class);
    if (executionOptions != null) {
      executionOptions.setCheckUpToDate(false);
      executionOptions.setTestCheckUpToDate(false);
    }
  }

  public static boolean shouldKeepStateAfterBuild(
      boolean requestedValue, OptionsProvider options) {
    return requestedValue || isEnabled(options);
  }

  public static boolean shouldTrackIncrementalState(
      boolean requestedValue, OptionsProvider options) {
    return requestedValue || isEnabled(options);
  }

  public static boolean shouldDiscardAnalysisCache(
      boolean requestedValue, OptionsProvider options) {
    return requestedValue && !isEnabled(options);
  }
}
