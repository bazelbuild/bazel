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

import com.google.testing.junit.runner.internal.junit4.CancellableRequestFactory;
import com.google.testing.junit.runner.model.TestSuiteModel;
import com.google.testing.junit.runner.util.Factory;
import com.google.testing.junit.runner.util.Supplier;
import java.io.PrintStream;
import java.util.Set;
import org.junit.runner.Request;
import org.junit.runner.notification.RunListener;

/**
 * A factory that supplies {@link JUnit4Runner}.
 */
public final class JUnit4RunnerFactory implements Factory<JUnit4Runner> {
  private final Supplier<Request> requestSupplier;

  private final Supplier<CancellableRequestFactory> requestFactorySupplier;

  private final Supplier<Supplier<TestSuiteModel>> modelSupplierSupplier;

  private final Supplier<PrintStream> testRunnerOutSupplier;

  private final Supplier<JUnit4Config> configSupplier;

  private final Supplier<Set<RunListener>> runListenersSupplier;

  private final Supplier<Set<JUnit4Runner.Initializer>> initializersSupplier;

  public JUnit4RunnerFactory(
      Supplier<Request> requestSupplier,
      Supplier<CancellableRequestFactory> requestFactorySupplier,
      Supplier<Supplier<TestSuiteModel>> modelSupplierSupplier,
      Supplier<PrintStream> testRunnerOutSupplier,
      Supplier<JUnit4Config> configSupplier,
      Supplier<Set<RunListener>> runListenersSupplier,
      Supplier<Set<JUnit4Runner.Initializer>> initializersSupplier) {
    assert requestSupplier != null;
    this.requestSupplier = requestSupplier;
    assert requestFactorySupplier != null;
    this.requestFactorySupplier = requestFactorySupplier;
    assert modelSupplierSupplier != null;
    this.modelSupplierSupplier = modelSupplierSupplier;
    assert testRunnerOutSupplier != null;
    this.testRunnerOutSupplier = testRunnerOutSupplier;
    assert configSupplier != null;
    this.configSupplier = configSupplier;
    assert runListenersSupplier != null;
    this.runListenersSupplier = runListenersSupplier;
    assert initializersSupplier != null;
    this.initializersSupplier = initializersSupplier;
  }

  @Override
  public JUnit4Runner get() {
    return new JUnit4Runner(
        requestSupplier.get(),
        requestFactorySupplier.get(),
        modelSupplierSupplier.get(),
        testRunnerOutSupplier.get(),
        configSupplier.get(),
        runListenersSupplier.get(),
        initializersSupplier.get());
  }

  public static Factory<JUnit4Runner> create(
      Supplier<Request> requestSupplier,
      Supplier<CancellableRequestFactory> requestFactorySupplier,
      Supplier<Supplier<TestSuiteModel>> modelSupplierSupplier,
      Supplier<PrintStream> testRunnerOutSupplier,
      Supplier<JUnit4Config> configSupplier,
      Supplier<Set<RunListener>> runListenersSupplier,
      Supplier<Set<JUnit4Runner.Initializer>> initializersSupplier) {
    return new JUnit4RunnerFactory(
        requestSupplier,
        requestFactorySupplier,
        modelSupplierSupplier,
        testRunnerOutSupplier,
        configSupplier,
        runListenersSupplier,
        initializersSupplier);
  }
}
