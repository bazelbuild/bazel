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

import com.google.testing.junit.runner.model.TestSuiteModel;
import com.google.testing.junit.runner.util.Factory;
import com.google.testing.junit.runner.util.Supplier;

/**
 * A factory that supplies {@link Supplier}<{@link TestSuiteModel}> from a
 * {@link Supplier}<{@link JUnit4TestModelBuilder}>.
 */
public final class TestSuiteModelSupplierFactory
    implements Factory<Supplier<TestSuiteModel>> {
  private final Supplier<JUnit4TestModelBuilder> builderSupplier;

  public TestSuiteModelSupplierFactory(Supplier<JUnit4TestModelBuilder> builderSupplier) {
    assert builderSupplier != null;
    this.builderSupplier = builderSupplier;
  }

  @Override
  public Supplier<TestSuiteModel> get() {
    Supplier<TestSuiteModel> testSuiteModelSupplier =
        JUnit4RunnerBaseModule.provideTestSuiteModelSupplier(builderSupplier.get());
    assert testSuiteModelSupplier != null;
    return testSuiteModelSupplier;
  }

  public static Factory<Supplier<TestSuiteModel>> create(
      Supplier<JUnit4TestModelBuilder> builderSupplier) {
    return new TestSuiteModelSupplierFactory(builderSupplier);
  }
}
