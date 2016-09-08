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
import com.google.testing.junit.runner.util.Supplier;

/**
 * A factory that supplies a top level suite {@link String}.
 */
public final class TopLevelSuiteNameFactory implements Factory<String> {
  private final Supplier<Class<?>> suiteSupplier;

  public TopLevelSuiteNameFactory(Supplier<Class<?>> suiteSupplier) {
    assert suiteSupplier != null;
    this.suiteSupplier = suiteSupplier;
  }

  @Override
  public String get() {
    String topLevelSuiteName =
        JUnit4InstanceModules.SuiteClass.topLevelSuiteName(suiteSupplier.get());
    if (topLevelSuiteName == null) {
      throw new NullPointerException();
    }
    return topLevelSuiteName;
  }

  public static Factory<String> create(Supplier<Class<?>> suiteSupplier) {
    return new TopLevelSuiteNameFactory(suiteSupplier);
  }
}
