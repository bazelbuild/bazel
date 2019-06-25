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
import com.google.testing.junit.runner.model.AntXmlResultWriterFactory;
import com.google.testing.junit.runner.model.TestSuiteModel;
import com.google.testing.junit.runner.model.TestSuiteModelBuilderFactory;
import com.google.testing.junit.runner.model.XmlResultWriter;
import com.google.testing.junit.runner.sharding.ShardingEnvironment;
import com.google.testing.junit.runner.sharding.ShardingFilters;
import com.google.testing.junit.runner.util.MemoizingSupplier;
import com.google.testing.junit.runner.util.SetFactory;
import com.google.testing.junit.runner.util.Supplier;
import com.google.testing.junit.runner.util.TestClock;
import java.io.PrintStream;
import java.util.Set;
import org.junit.internal.TextListener;
import org.junit.runner.Request;
import org.junit.runner.notification.RunListener;

/**
 * Utility class to create a JUnit4Runner instance from a {@link Builder} for testing purposes. All
 * required dependencies are being injected automatically.
 */
public final class JUnit4BazelMock {
  private Supplier<Class<?>> topLevelSuiteSupplier;

  private Supplier<Request> requestMemoizingSupplier;

  private Supplier<CancellableRequestFactory> cancellableRequestFactorySupplier;

  private Supplier<String> topLevelSuiteNameSupplier;

  private Supplier<TestClock> tickerSupplier;

  private Supplier<ShardingEnvironment> shardingEnvironmentSupplier;

  private Supplier<ShardingFilters> shardingFiltersSupplier;

  private Supplier<XmlResultWriter> xmlResultWriterSupplier;

  private Supplier<TestSuiteModel.Builder> builderSupplier;

  private Supplier<JUnit4TestModelBuilder> jUnit4TestModelBuilderMemoizingSupplier;

  private Supplier<Supplier<TestSuiteModel>> provideTestSuiteModelSupplierMemoizingSupplier;

  private Supplier<PrintStream> stdoutStreamMemoizingSupplier;

  private Supplier<JUnit4Config> configSupplier;

  private Supplier<Set<RunListener>> mockRunListenerSupplier;

  private Supplier<TextListener> textListenerMemoizingSupplier;

  private Supplier<RunListener> textListenerSupplier;

  private Supplier<Set<RunListener>> setOfRunListenerSupplier;

  private Supplier<JUnit4Runner> jUnit4RunnerProvider;

  private JUnit4BazelMock(Builder builder) {
    assert builder != null;
    initialize(builder);
  }

  public static Builder builder() {
    return new Builder();
  }

  @SuppressWarnings("unchecked")
  private void initialize(final Builder builder) {

    this.topLevelSuiteSupplier = TopLevelSuiteFactory.create(builder.suiteClass);

    this.requestMemoizingSupplier =
        new MemoizingSupplier<>(RequestFactory.create(topLevelSuiteSupplier));

    this.cancellableRequestFactorySupplier =
        new MemoizingSupplier<>(CancellableRequestFactoryFactory.create());

    this.topLevelSuiteNameSupplier = TopLevelSuiteNameFactory.create(topLevelSuiteSupplier);

    this.tickerSupplier =
        new MemoizingSupplier<>(TestModuleTickerFactory.create(builder.testModule));

    this.shardingEnvironmentSupplier =
        TestModuleShardingEnvironmentFactory.create(builder.testModule);

    this.shardingFiltersSupplier =
        TestModuleShardingFiltersFactory.create(
            builder.testModule,
            shardingEnvironmentSupplier,
            ShardingFilterFactoryFactory.create());

    this.xmlResultWriterSupplier =
        TestModuleXmlResultWriterFactory.create(
            builder.testModule, AntXmlResultWriterFactory.create());

    this.builderSupplier =
        TestSuiteModelBuilderFactory.create(
            tickerSupplier,
            shardingFiltersSupplier,
            shardingEnvironmentSupplier,
            xmlResultWriterSupplier);

    this.jUnit4TestModelBuilderMemoizingSupplier =
        new MemoizingSupplier<>(JUnit4TestModelBuilderFactory.create(
                requestMemoizingSupplier, topLevelSuiteNameSupplier, builderSupplier));

    this.provideTestSuiteModelSupplierMemoizingSupplier =
        new MemoizingSupplier<>(TestSuiteModelSupplierFactory.create(
                jUnit4TestModelBuilderMemoizingSupplier));

    this.stdoutStreamMemoizingSupplier =
        new MemoizingSupplier<>(TestModuleProvideStdoutStreamFactory.create(builder.testModule));

    this.configSupplier = TestModuleConfigFactory.create(builder.testModule);

    this.mockRunListenerSupplier =
        TestModuleMockRunListenerFactory.create(builder.testModule);

    this.textListenerMemoizingSupplier =
        new MemoizingSupplier<>(ProvideTextListenerFactory.create(stdoutStreamMemoizingSupplier));

    this.textListenerSupplier = TextListenerFactory.create(textListenerMemoizingSupplier);

    this.setOfRunListenerSupplier =
        SetFactory.<RunListener>builder(1, 1)
            .addCollectionSupplier(mockRunListenerSupplier)
            .addSupplier(textListenerSupplier)
            .build();

    this.jUnit4RunnerProvider =
        JUnit4RunnerFactory.create(
            requestMemoizingSupplier,
            cancellableRequestFactorySupplier,
            provideTestSuiteModelSupplierMemoizingSupplier,
            stdoutStreamMemoizingSupplier,
            configSupplier,
            setOfRunListenerSupplier,
            SetFactory.<JUnit4Runner.Initializer>empty());
  }

  public JUnit4Runner runner() {
    return jUnit4RunnerProvider.get();
  }

  public CancellableRequestFactory cancellableRequestFactory() {
    return cancellableRequestFactorySupplier.get();
  }

  public static final class Builder {
    private JUnit4InstanceModules.SuiteClass suiteClass;

    private JUnit4RunnerTest.TestModule testModule;

    private Builder() {}

    public JUnit4BazelMock build() {
      if (suiteClass == null) {
        throw new IllegalStateException(
            JUnit4InstanceModules.SuiteClass.class.getCanonicalName() + " must be set");
      }
      if (testModule == null) {
        throw new IllegalStateException(
            JUnit4RunnerTest.TestModule.class.getCanonicalName() + " must be set");
      }
      return new JUnit4BazelMock(this);
    }

    public Builder testModule(JUnit4RunnerTest.TestModule testModule) {
      if (testModule == null) {
        throw new NullPointerException();
      }
      this.testModule = testModule;
      return this;
    }

    public Builder suiteClass(JUnit4InstanceModules.SuiteClass suiteClass) {
      if (suiteClass == null) {
        throw new NullPointerException();
      }
      this.suiteClass = suiteClass;
      return this;
    }
  }
}
