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
import org.junit.runner.Request;

/**
 * A factory that supplies {@link JUnit4TestModelBuilder}.
 */
public final class JUnit4TestModelBuilderFactory implements Factory<JUnit4TestModelBuilder> {
  private final Supplier<Request> requestSupplier;

  private final Supplier<String> suiteNameSupplier;

  private final Supplier<TestSuiteModel.Builder> builderSupplier;

  public JUnit4TestModelBuilderFactory(
      Supplier<Request> requestSupplier,
      Supplier<String> suiteNameSupplier,
      Supplier<TestSuiteModel.Builder> builderSupplier) {
    assert requestSupplier != null;
    this.requestSupplier = requestSupplier;
    assert suiteNameSupplier != null;
    this.suiteNameSupplier = suiteNameSupplier;
    assert builderSupplier != null;
    this.builderSupplier = builderSupplier;
  }

  @Override
  public JUnit4TestModelBuilder get() {
    return new JUnit4TestModelBuilder(
        requestSupplier.get(), suiteNameSupplier.get(), builderSupplier.get());
  }

  public static Factory<JUnit4TestModelBuilder> create(
      Supplier<Request> requestSupplier,
      Supplier<String> suiteNameSupplier,
      Supplier<TestSuiteModel.Builder> builderSupplier) {
    return new JUnit4TestModelBuilderFactory(requestSupplier, suiteNameSupplier, builderSupplier);
  }
}
