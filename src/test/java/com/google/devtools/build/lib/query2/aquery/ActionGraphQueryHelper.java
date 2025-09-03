// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.aquery;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment.TopLevelConfigurations;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.testutil.PostAnalysisQueryHelper;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator;
import com.google.devtools.build.skyframe.WalkableGraph;

/** Helper class for aquery test */
public class ActionGraphQueryHelper extends PostAnalysisQueryHelper<ConfiguredTargetValue> {

  @Override
  protected PostAnalysisQueryEnvironment<ConfiguredTargetValue> getPostAnalysisQueryEnvironment(
      WalkableGraph walkableGraph,
      TopLevelConfigurations topLevelConfigurations,
      ImmutableMap<String, BuildConfigurationValue> transitiveConfigurations,
      ImmutableMap<AspectKeyCreator.AspectKey, ConfiguredAspect> topLevelAspects) {
    ImmutableList<QueryFunction> extraFunctions =
        ImmutableList.copyOf(ActionGraphQueryEnvironment.AQUERY_FUNCTIONS);
    return new ActionGraphQueryEnvironment(
        keepGoing,
        getReporter(),
        extraFunctions,
        topLevelConfigurations,
        transitiveConfigurations,
        mainRepoTargetParser,
        analysisHelper.getPackageManager().getPackagePath(),
        () -> walkableGraph,
        settings,
        LabelPrinter.legacy());
  }

  @Override
  public String getLabel(ConfiguredTargetValue configuredTargetValue) {
    return configuredTargetValue.getConfiguredTarget().getOriginalLabel().toString();
  }
}
