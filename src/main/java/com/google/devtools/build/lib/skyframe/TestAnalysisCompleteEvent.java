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

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import javax.annotation.concurrent.Immutable;

/**
 * The event that marks the completion of the analysis of a test target.
 *
 * <p>This is used to set up the {@code TestResultAggregator} for the test target before running the
 * test.
 */
@Immutable
public final class TestAnalysisCompleteEvent {

  private final ConfiguredTarget target;
  private final BuildConfigurationValue buildConfigurationValue;

  public TestAnalysisCompleteEvent(
      ConfiguredTarget target, BuildConfigurationValue buildConfigurationValue) {
    this.target = target;
    this.buildConfigurationValue = buildConfigurationValue;
  }

  public ConfiguredTarget getConfiguredTarget() {
    return target;
  }

  public BuildConfigurationValue getBuildConfigurationValue() {
    return buildConfigurationValue;
  }
}
