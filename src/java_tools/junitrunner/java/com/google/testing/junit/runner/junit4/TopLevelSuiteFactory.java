// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.junit4;

import com.google.testing.junit.runner.util.Factory;

/**
 * A factory that supplies a top level suite {@link Class<?>}.
 */
public final class TopLevelSuiteFactory implements Factory<Class<?>> {
  private final JUnit4InstanceModules.SuiteClass module;

  public TopLevelSuiteFactory(JUnit4InstanceModules.SuiteClass module) {
    assert module != null;
    this.module = module;
  }

  @Override
  public Class<?> get() {
    if (module.topLevelSuite() == null) {
      throw new NullPointerException();
    }
    return module.topLevelSuite();
  }

  public static Factory<Class<?>> create(JUnit4InstanceModules.SuiteClass module) {
    return new TopLevelSuiteFactory(module);
  }
}
