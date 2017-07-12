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

package com.google.devtools.build.lib.rules.test;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import java.util.Map;

/** Provider containing any additional environment variables for use in the test action. */
@Immutable
public final class TestEnvironmentProvider extends SkylarkClassObject {

  /** Skylark constructor and identifier for TestEnvironmentProvider. */
  public static final NativeClassObjectConstructor<TestEnvironmentProvider> SKYLARK_CONSTRUCTOR =
      new NativeClassObjectConstructor<TestEnvironmentProvider>(
          TestEnvironmentProvider.class, "TestEnvironment") {};

  private final Map<String, String> environment;

  /** Constructs a new provider with the given variable name to variable value mapping. */
  public TestEnvironmentProvider(Map<String, String> environment) {
    super(SKYLARK_CONSTRUCTOR, ImmutableMap.<String, Object>of("environment", environment));
    this.environment = Preconditions.checkNotNull(environment);
  }

  /**
   * Returns environment variables which should be set on the test action.
   */
  public Map<String, String> getEnvironment() {
    return environment;
  }
}
