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

package com.google.devtools.build.lib.analysis.test;

import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.skylarkbuildapi.test.AnalysisTestResultInfoApi;

/**
 * Encapsulates the result of analyis-phase testing. Build targets which return an instance of this
 * provider signal to the build system that it should generate 'stub' test executable.
 */
public class AnalysisTestResultInfo implements Info, AnalysisTestResultInfoApi {

  /**
   * Singleton provider instance for {@link AnalysisTestResultInfo}.
   */
  public static final TestResultInfoProvider SKYLARK_CONSTRUCTOR =
      new TestResultInfoProvider();

  private final Boolean success;
  private final String message;

  public AnalysisTestResultInfo(Boolean success, String message) {
    this.success = success;
    this.message = message;
  }

  @Override
  public TestResultInfoProvider getProvider() {
    return SKYLARK_CONSTRUCTOR;
  }

  @Override
  public Boolean getSuccess() {
    return success;
  }

  @Override
  public String getMessage() {
    return message;
  }

  /**
   * Provider implementation for {@link AnalysisTestResultInfo}.
   */
  public static class TestResultInfoProvider
      extends BuiltinProvider<AnalysisTestResultInfo> implements AnalysisTestResultInfoProviderApi {

    public TestResultInfoProvider() {
      super("AnalysisTestResultInfo", AnalysisTestResultInfo.class);
    }

    @Override
    public AnalysisTestResultInfoApi testResultInfo(Boolean success, String message) {
      return new AnalysisTestResultInfo(success, message);
    }
  }
}
