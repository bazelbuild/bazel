// Copyright 2022 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.skyframe.ArtifactConflictFinder.NUM_JOBS;

import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.concurrent.Sharder;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/** Represents the traversal of the ActionLookupValues in a build. */
public class ActionLookupValuesTraversal {
  // Some metrics indicate this is a rough average # of ALVs in a build.
  private final Sharder<ActionLookupValue> actionLookupValueShards =
      new Sharder<>(NUM_JOBS, /* expectedTotalSize= */ 200_000);

  // Metrics.
  private int configuredObjectCount = 0;
  private int configuredTargetCount = 0;
  private int actionCount = 0;
  private int actionCountNotIncludingAspects = 0;
  private int inputFileConfiguredTargetCount = 0;
  private int outputFileConfiguredTargetCount = 0;
  private int otherConfiguredTargetCount = 0;

  public ActionLookupValuesTraversal() {}

  void accumulate(ActionLookupKey key, SkyValue value) {
    boolean isConfiguredTarget = value instanceof ConfiguredTargetValue;
    boolean isActionLookupValue = value instanceof ActionLookupValue;
    if (!isConfiguredTarget && !isActionLookupValue) {
      BugReport.sendBugReport(
          new IllegalStateException(
              String.format(
                  "Should only be called with ConfiguredTargetValue or ActionLookupValue: %s %s"
                      + " %s",
                  value.getClass(), key, value)));
      return;
    }
    if (isConfiguredTarget
        && !Objects.equals(
            key.getConfigurationKey(),
            ((ConfiguredTargetValue) value).getConfiguredTarget().getConfigurationKey())) {
      // The configuration of the key doesn't match the configuration of the value. This means that
      // the ConfiguredTargetValue is delegated from a different key. This ConfiguredTargetValue
      // will show up again under its own key. Avoids double counting by skipping accumulation.
      return;
    }
    configuredObjectCount++;
    if (isConfiguredTarget) {
      configuredTargetCount++;
    }
    if (isActionLookupValue) {
      ActionLookupValue alv = (ActionLookupValue) value;
      int numActions = alv.getNumActions();
      actionCount += numActions;
      if (isConfiguredTarget) {
        actionCountNotIncludingAspects += numActions;
      }
      actionLookupValueShards.add(alv);
      return;
    }
    if (!(value instanceof NonRuleConfiguredTargetValue)) {
      BugReport.sendBugReport(
          new IllegalStateException(
              String.format("Unexpected value type: %s %s %s", value.getClass(), key, value)));
      return;
    }
    ConfiguredTarget configuredTarget =
        ((NonRuleConfiguredTargetValue) value).getConfiguredTarget();
    if (configuredTarget instanceof InputFileConfiguredTarget) {
      inputFileConfiguredTargetCount++;
    } else if (configuredTarget instanceof OutputFileConfiguredTarget) {
      outputFileConfiguredTargetCount++;
    } else {
      otherConfiguredTargetCount++;
    }
  }

  Sharder<ActionLookupValue> getActionLookupValueShards() {
    return actionLookupValueShards;
  }

  int getActionCount() {
    return actionCount;
  }

  BuildEventStreamProtos.BuildMetrics.BuildGraphMetrics.Builder getMetrics() {
    return BuildEventStreamProtos.BuildMetrics.BuildGraphMetrics.newBuilder()
        .setActionLookupValueCount(configuredObjectCount)
        .setActionLookupValueCountNotIncludingAspects(configuredTargetCount)
        .setActionCount(actionCount)
        .setActionCountNotIncludingAspects(actionCountNotIncludingAspects)
        .setInputFileConfiguredTargetCount(inputFileConfiguredTargetCount)
        .setOutputFileConfiguredTargetCount(outputFileConfiguredTargetCount)
        .setOtherConfiguredTargetCount(otherConfiguredTargetCount);
  }
}
