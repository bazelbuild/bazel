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
 * A factory that supplies {@link JUnit4Config}.
 */
public final class JUnit4ConfigFactory implements Factory<JUnit4Config> {
  private final Supplier<JUnit4Options> optionsSupplier;

  public JUnit4ConfigFactory(Supplier<JUnit4Options> optionsSupplier) {
    assert optionsSupplier != null;
    this.optionsSupplier = optionsSupplier;
  }

  @Override
  public JUnit4Config get() {
    JUnit4Config config = JUnit4InstanceModules.Config.config(optionsSupplier.get());
    assert config != null;
    return config;
  }

  public static Factory<JUnit4Config> create(Supplier<JUnit4Options> optionsSupplier) {
    return new JUnit4ConfigFactory(optionsSupplier);
  }
}
