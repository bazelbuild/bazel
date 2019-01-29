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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.query2.ConfiguredTargetQueryEnvironment;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment.TopLevelConfigurations;
import com.google.devtools.build.lib.query2.engine.AbstractQueryTest.QueryHelper;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.skyframe.WalkableGraph;

/**
 * {@link QueryHelper} for {@link ConfiguredTargetQueryTest}. Big warts: uses an {@link
 * AnalysisTestCase} to do analysis before query, but {@link AnalysisTestCase} is meant to be
 * inherited from, not composed. In particular, means that @Before and @After annotations of {@link
 * AnalysisTestCase} must be run manually. @BeforeClass and @AfterClass are completely ignored for
 * now.
 */
public class ConfiguredTargetQueryHelper extends PostAnalysisQueryHelper<ConfiguredTarget> {
  @Override
  protected ConfiguredTargetQueryEnvironment getPostAnalysisQueryEnvironment(
      WalkableGraph walkableGraph, TopLevelConfigurations topLevelConfigurations) {
    ImmutableList<QueryFunction> extraFunctions =
        ImmutableList.copyOf(ConfiguredTargetQueryEnvironment.CQUERY_FUNCTIONS);
    return new ConfiguredTargetQueryEnvironment(
        keepGoing,
        reporter,
        extraFunctions,
        topLevelConfigurations,
        analysisHelper.getHostConfiguration(),
        parserPrefix,
        analysisHelper.getPackageManager().getPackagePath(),
        () -> walkableGraph,
        this.settings);
  }

  @Override
  public String getLabel(ConfiguredTarget target) {
    return target.getLabel().toString();
  }
}
