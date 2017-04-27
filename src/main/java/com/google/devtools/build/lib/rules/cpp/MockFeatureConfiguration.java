// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import java.util.List;

/**
 * A mock crosstool feature configuration. This allows for easier writing of unit tests isolated
 * from the inner workings of crosstool feature sets.
 *
 * <p>For example, instead of setting up a fully defined feature in crosstool configuration which
 * conditionally sets environment variables for specific actions, one can use this class to mock an
 * "always on" feature (which always adds a certain environment variable).
 */
public class MockFeatureConfiguration extends FeatureConfiguration {

  private final ImmutableMap<String, String> environmentVariables;
  private final ImmutableSet<String> requirements;

  public MockFeatureConfiguration(
      ImmutableMap<String, String> environmentVariables, ImmutableSet<String> requirements) {
    this.environmentVariables = environmentVariables;
    this.requirements = requirements;
  }

  @Override
  ImmutableMap<String, String> getEnvironmentVariables(String action, Variables variables) {
    return environmentVariables;
  }

  @Override
  CcToolchainFeatures.Tool getToolForAction(String actionName) {
    return new CcToolchainFeatures.Tool("tool_path", requirements);
  }

  @Override
  List<String> getCommandLine(String action, Variables variables) {
    return ImmutableList.of();
  }

  @Override
  boolean actionIsConfigured(String actionName) {
    return true;
  }
}
