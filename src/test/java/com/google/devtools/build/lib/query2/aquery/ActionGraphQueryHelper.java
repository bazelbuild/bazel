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
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment.TopLevelConfigurations;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.testutil.PostAnalysisQueryHelper;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.Collection;

/** Helper class for aquery test */
public class ActionGraphQueryHelper extends PostAnalysisQueryHelper<ConfiguredTargetValue> {

  @Override
  protected PostAnalysisQueryEnvironment<ConfiguredTargetValue> getPostAnalysisQueryEnvironment(
      WalkableGraph walkableGraph,
      TopLevelConfigurations topLevelConfigurations,
      Collection<SkyKey> transitiveConfigurationKeys) {
    ImmutableList<QueryFunction> extraFunctions =
        ImmutableList.copyOf(ActionGraphQueryEnvironment.AQUERY_FUNCTIONS);
    return new ActionGraphQueryEnvironment(
        keepGoing,
        getReporter(),
        extraFunctions,
        topLevelConfigurations,
        mainRepoTargetParser,
        analysisHelper.getPackageManager().getPackagePath(),
        () -> walkableGraph,
        settings);
  }

  @Override
  public String getLabel(ConfiguredTargetValue configuredTargetValue) {
    return configuredTargetValue.getConfiguredTarget().getOriginalLabel().toString();
  }
}
