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

package com.google.testing.junit.runner;

import com.google.testing.junit.runner.internal.SignalHandlers;
import com.google.testing.junit.runner.internal.SignalHandlersFactory;
import com.google.testing.junit.runner.junit4.CancellableRequestFactoryFactory;
import com.google.testing.junit.runner.junit4.CurrentRunningTestFactory;
import com.google.testing.junit.runner.junit4.JUnit4ConfigFactory;
import com.google.testing.junit.runner.junit4.JUnit4InstanceModules;
import com.google.testing.junit.runner.junit4.JUnit4OptionsFactory;
import com.google.testing.junit.runner.junit4.JUnit4Runner;
import com.google.testing.junit.runner.junit4.JUnit4RunnerFactory;
import com.google.testing.junit.runner.junit4.JUnit4RunnerModule;
import com.google.testing.junit.runner.junit4.JUnit4TestModelBuilderFactory;
import com.google.testing.junit.runner.junit4.JUnit4TestNameListenerFactory;
import com.google.testing.junit.runner.junit4.JUnit4TestStackTraceListenerFactory;
import com.google.testing.junit.runner.junit4.JUnit4TestXmlListenerFactory;
import com.google.testing.junit.runner.junit4.NameListenerFactory;
import com.google.testing.junit.runner.junit4.ProvideTextListenerFactory;
import com.google.testing.junit.runner.junit4.ProvideXmlStreamFactory;
import com.google.testing.junit.runner.junit4.RequestFactory;
import com.google.testing.junit.runner.junit4.ShardingFilterFactoryFactory;
import com.google.testing.junit.runner.junit4.SignalHandlerInstallerFactory;
import com.google.testing.junit.runner.junit4.StackTraceListenerFactory;
import com.google.testing.junit.runner.junit4.TestSuiteModelSupplierFactory;
import com.google.testing.junit.runner.junit4.TextListenerFactory;
import com.google.testing.junit.runner.junit4.TickerFactory;
import com.google.testing.junit.runner.junit4.TopLevelSuiteFactory;
import com.google.testing.junit.runner.junit4.TopLevelSuiteNameFactory;
import com.google.testing.junit.runner.junit4.XmlListenerFactory;
import com.google.testing.junit.runner.model.AntXmlResultWriterFactory;
import com.google.testing.junit.runner.model.TestSuiteModel;
import com.google.testing.junit.runner.model.TestSuiteModelBuilderFactory;
import com.google.testing.junit.runner.model.XmlResultWriter;
import com.google.testing.junit.runner.sharding.ShardingEnvironmentFactory;
import com.google.testing.junit.runner.sharding.ShardingFilters;
import com.google.testing.junit.runner.sharding.ShardingFiltersFactory;
import com.google.testing.junit.runner.util.MemoizingSupplier;
import com.google.testing.junit.runner.util.SetFactory;
import com.google.testing.junit.runner.util.Supplier;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Set;
import org.junit.internal.TextListener;
import org.junit.runner.Request;
import org.junit.runner.notification.RunListener;

/**
 * Utility class to create a JUnit4Runner instance from a {@link Builder}. All required 
 * dependencies are being injected automatically.
 */
public final class JUnit4Bazel {
  private Supplier<Class<?>> topLevelSuiteSupplier;

  private Supplier<Request> requestMemoizingSupplier;

  @SuppressWarnings("rawtypes")
  private Supplier cancellableRequestFactorySupplier;

  private Supplier<String> topLevelSuiteNameSupplier;

  private Supplier<ShardingFilters> shardingFiltersSupplier;

  private Supplier<XmlResultWriter> resultWriterSupplier;

  private Supplier<TestSuiteModel.Builder> builderSupplier;

  @SuppressWarnings("rawtypes")
  private Supplier jUnit4TestModelBuilderMemoizingSupplier;

  private Supplier<Supplier<TestSuiteModel>> provideTestSuiteModelSupplierMemoizingSupplier;

  private Supplier<PrintStream> stdoutStreamMemoizingSupplier;

  @SuppressWarnings("rawtypes")
  private Supplier optionsMemoizingSupplier;

  @SuppressWarnings("rawtypes")
  private Supplier configMemoizingSupplier;

  private Supplier<SignalHandlers> signalHandlersSupplier;

  private Supplier<PrintStream> stderrStreamMemoizingSupplier;

  @SuppressWarnings("rawtypes")
  private Supplier jUnit4TestStackTraceListenerMemoizingSupplier;

  private Supplier<RunListener> stackTraceListenerSupplier;

  private Supplier<OutputStream> provideXmlStreamMemoizingSupplier;

  @SuppressWarnings("rawtypes")
  private Supplier jUnit4TestXmlListenerMemoizingSupplier;

  private Supplier<RunListener> xmlListenerSupplier;

  @SuppressWarnings("rawtypes")
  private Supplier provideCurrentRunningTestMemoizingSupplier;

  @SuppressWarnings("rawtypes")
  private Supplier jUnit4TestNameListenerMemoizingSupplier;

  private Supplier<RunListener> nameListenerSupplier;

  private Supplier<TextListener> provideTextListenerMemoizingSupplier;

  private Supplier<RunListener> textListenerSupplier;

  private Supplier<Set<RunListener>> setOfRunListenerProvider;

  private Supplier<JUnit4Runner> jUnit4RunnerProvider;

  private JUnit4Bazel(Builder builder) {
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

    this.shardingFiltersSupplier = ShardingFiltersFactory.create(
            ShardingEnvironmentFactory.create(),
            ShardingFilterFactoryFactory.create());

    this.resultWriterSupplier = ResultWriterFactory.create(AntXmlResultWriterFactory.create());

    this.builderSupplier =
        TestSuiteModelBuilderFactory.create(
            TickerFactory.create(),
            shardingFiltersSupplier,
            ShardingEnvironmentFactory.create(),
            resultWriterSupplier);

    this.jUnit4TestModelBuilderMemoizingSupplier =
        new MemoizingSupplier<>(JUnit4TestModelBuilderFactory.create(
                requestMemoizingSupplier, topLevelSuiteNameSupplier, builderSupplier));

    this.provideTestSuiteModelSupplierMemoizingSupplier =
        new MemoizingSupplier<Supplier<TestSuiteModel>>(
            TestSuiteModelSupplierFactory.create(jUnit4TestModelBuilderMemoizingSupplier));

    this.stdoutStreamMemoizingSupplier = new MemoizingSupplier<>(StdoutStreamFactory.create());

    this.optionsMemoizingSupplier =
        new MemoizingSupplier<>(JUnit4OptionsFactory.create(builder.config));

    this.configMemoizingSupplier =
        new MemoizingSupplier<Object>(JUnit4ConfigFactory.create(optionsMemoizingSupplier));

    this.signalHandlersSupplier =
        SignalHandlersFactory.create(SignalHandlerInstallerFactory.create());

    this.stderrStreamMemoizingSupplier = new MemoizingSupplier<>(StderrStreamFactory.create());

    this.jUnit4TestStackTraceListenerMemoizingSupplier =
        new MemoizingSupplier<>(JUnit4TestStackTraceListenerFactory.create(
            signalHandlersSupplier, stderrStreamMemoizingSupplier));

    this.stackTraceListenerSupplier =
        StackTraceListenerFactory.create(jUnit4TestStackTraceListenerMemoizingSupplier);

    this.provideXmlStreamMemoizingSupplier =
        new MemoizingSupplier<OutputStream>(
            ProvideXmlStreamFactory.create(configMemoizingSupplier));

    this.jUnit4TestXmlListenerMemoizingSupplier =
        new MemoizingSupplier<Object>(JUnit4TestXmlListenerFactory.create(
            provideTestSuiteModelSupplierMemoizingSupplier,
            cancellableRequestFactorySupplier,
            signalHandlersSupplier,
            provideXmlStreamMemoizingSupplier,
            stderrStreamMemoizingSupplier));

    this.xmlListenerSupplier = XmlListenerFactory.create(jUnit4TestXmlListenerMemoizingSupplier);

    this.provideCurrentRunningTestMemoizingSupplier =
        new MemoizingSupplier<>(CurrentRunningTestFactory.create(builder.jUnit4RunnerModule));

    this.jUnit4TestNameListenerMemoizingSupplier =
        new MemoizingSupplier<Object>(
            JUnit4TestNameListenerFactory.create(provideCurrentRunningTestMemoizingSupplier));

    this.nameListenerSupplier = NameListenerFactory.create(jUnit4TestNameListenerMemoizingSupplier);

    this.provideTextListenerMemoizingSupplier =
        new MemoizingSupplier<>(ProvideTextListenerFactory.create(stdoutStreamMemoizingSupplier));

    this.textListenerSupplier =
        TextListenerFactory.create(provideTextListenerMemoizingSupplier);

    this.setOfRunListenerProvider =
        SetFactory.<RunListener>builder(4, 0)
            .addSupplier(stackTraceListenerSupplier)
            .addSupplier(xmlListenerSupplier)
            .addSupplier(nameListenerSupplier)
            .addSupplier(textListenerSupplier)
            .build();

    this.jUnit4RunnerProvider =
        JUnit4RunnerFactory.create(
            requestMemoizingSupplier,
            cancellableRequestFactorySupplier,
            provideTestSuiteModelSupplierMemoizingSupplier,
            stdoutStreamMemoizingSupplier,
            configMemoizingSupplier,
            setOfRunListenerProvider,
            SetFactory.<JUnit4Runner.Initializer>empty());
  }

  public JUnit4Runner runner() {
    return jUnit4RunnerProvider.get();
  }

  /**
   * A builder for instantiating {@link JUnit4Bazel}.
   */
  public static final class Builder {
    private JUnit4InstanceModules.SuiteClass suiteClass;

    private JUnit4InstanceModules.Config config;

    private JUnit4RunnerModule jUnit4RunnerModule;

    private Builder() {}

    public JUnit4Bazel build() {
      if (suiteClass == null) {
        throw new IllegalStateException(
            JUnit4InstanceModules.SuiteClass.class.getCanonicalName() + " must be set");
      }
      if (config == null) {
        throw new IllegalStateException(
            JUnit4InstanceModules.Config.class.getCanonicalName() + " must be set");
      }
      if (jUnit4RunnerModule == null) {
        this.jUnit4RunnerModule = new JUnit4RunnerModule();
      }
      return new JUnit4Bazel(this);
    }

    public Builder jUnit4RunnerModule(JUnit4RunnerModule jUnit4RunnerModule) {
      this.jUnit4RunnerModule = checkNotNull(jUnit4RunnerModule);
      return this;
    }

    public Builder suiteClass(JUnit4InstanceModules.SuiteClass suiteClass) {
      this.suiteClass = checkNotNull(suiteClass);
      return this;
    }

    public Builder config(JUnit4InstanceModules.Config config) {
      this.config = checkNotNull(config);
      return this;
    }
  }

  private static <T> T checkNotNull(T reference) {
    if (reference == null) {
      throw new NullPointerException();
    }
    return reference;
  }
}
