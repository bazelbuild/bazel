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

package com.google.devtools.build.lib.skylarkbuildapi.test;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skylarkbuildapi.Bootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.test.AnalysisFailureInfoApi.AnalysisFailureInfoProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.test.AnalysisTestResultInfoApi.AnalysisTestResultInfoProviderApi;

/**
 * {@link Bootstrap} for skylark objects related to testing.
 */
public class TestingBootstrap implements Bootstrap {

  private final TestingModuleApi testingModule;
  private final CoverageCommonApi<?> coverageCommon;
  private final AnalysisFailureInfoProviderApi analysisFailureInfoProvider;
  private final AnalysisTestResultInfoProviderApi testResultInfoProvider;

  public TestingBootstrap(
      TestingModuleApi testingModule,
      CoverageCommonApi<?> coverageCommon,
      AnalysisFailureInfoProviderApi analysisFailureInfoProvider,
      AnalysisTestResultInfoProviderApi testResultInfoProvider) {
    this.testingModule = testingModule;
    this.coverageCommon = coverageCommon;
    this.analysisFailureInfoProvider = analysisFailureInfoProvider;
    this.testResultInfoProvider = testResultInfoProvider;
  }

  @Override
  public void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    builder.put("testing", testingModule);
    builder.put("coverage_common", coverageCommon);
    builder.put("AnalysisFailureInfo", analysisFailureInfoProvider);
    builder.put("AnalysisTestResultInfo", testResultInfoProvider);
  }
}
