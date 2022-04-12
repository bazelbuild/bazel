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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.test.TestEnvironmentInfoApi;
import java.util.List;
import java.util.Map;

/** Provider containing any additional environment variables for use in the test action. */
@Immutable
public final class TestEnvironmentInfo extends NativeInfo implements TestEnvironmentInfoApi {

  /** Starlark constructor and identifier for TestEnvironmentInfo. */
  public static final BuiltinProvider<TestEnvironmentInfo> PROVIDER =
      new BuiltinProvider<TestEnvironmentInfo>("TestEnvironment", TestEnvironmentInfo.class) {};

  private final ImmutableMap<String, String> environment;
  private final ImmutableList<String> inheritedEnvironment;

  /** Constructs a new provider with the given fixed and inherited environment variables. */
  public TestEnvironmentInfo(Map<String, String> environment, List<String> inheritedEnvironment) {
    this.environment = ImmutableMap.copyOf(Preconditions.checkNotNull(environment));
    this.inheritedEnvironment =
        ImmutableList.copyOf(Preconditions.checkNotNull(inheritedEnvironment));
  }

  @Override
  public BuiltinProvider<TestEnvironmentInfo> getProvider() {
    return PROVIDER;
  }

  /**
   * Returns environment variables which should be set on the test action.
   */
  @Override
  public Map<String, String> getEnvironment() {
    return environment;
  }

  /** Returns names of environment variables whose value should be obtained from the environment. */
  @Override
  public ImmutableList<String> getInheritedEnvironment() {
    return inheritedEnvironment;
  }
}
