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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Actions.GeneratingActions;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A Skyframe function to calculate the coverage report Action and Artifacts.
 */
public class CoverageReportFunction implements SkyFunction {

  static final Precomputed<ImmutableList<ActionAnalysisMetadata>> COVERAGE_REPORT_KEY =
      new Precomputed<>("coverage_report_actions");
  private final ActionKeyContext actionKeyContext;

  CoverageReportFunction(ActionKeyContext actionKeyContext) {
    this.actionKeyContext = actionKeyContext;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    Preconditions.checkState(
        CoverageReportValue.COVERAGE_REPORT_KEY.equals(skyKey),
        String.format(
            "Expected %s for SkyKey but got %s instead",
            CoverageReportValue.COVERAGE_REPORT_KEY, skyKey));

    ImmutableList<ActionAnalysisMetadata> actions = COVERAGE_REPORT_KEY.get(env);
    if (actions == null) {
      return null;
    }

    GeneratingActions generatingActions;
    try {
      generatingActions =
          Actions.assignOwnersAndFilterSharedActionsAndThrowActionConflict(
              actionKeyContext,
              actions,
              CoverageReportValue.COVERAGE_REPORT_KEY,
              /*outputFiles=*/ null);
    } catch (ActionConflictException e) {
      throw new IllegalStateException("Action conflicts not expected in coverage: " + skyKey, e);
    }
    return new CoverageReportValue(generatingActions);
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
