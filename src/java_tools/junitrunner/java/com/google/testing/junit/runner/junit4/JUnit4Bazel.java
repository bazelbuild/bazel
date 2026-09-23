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

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.testing.junit.runner.internal.junit4.CancellableRequestFactory;
import com.google.testing.junit.runner.internal.junit4.MemoizingRequest;
import com.google.testing.junit.runner.model.AntXmlResultWriter;
import com.google.testing.junit.runner.model.TestSuiteModel;
import com.google.testing.junit.runner.model.XmlResultWriter;
import com.google.testing.junit.runner.sharding.ShardingEnvironment;
import com.google.testing.junit.runner.sharding.ShardingFilters;
import com.google.testing.junit.runner.util.MemoizingSupplier;
import com.google.testing.junit.runner.util.SuiteUtil;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;
import org.junit.runner.Request;
import org.junit.runner.notification.RunListener;

/**
 * Utility class to create a JUnit4Runner instance from a {@link Builder}. All required dependencies
 * are being injected automatically.
 */
public final class JUnit4Bazel {

  private Request request;

  private CancellableRequestFactory cancellableRequestFactory;

  private JUnit4TestModelBuilder jUnit4TestModelBuilder;

  private Supplier<TestSuiteModel> testSuiteModelSupplier;

  private PrintStream stdoutStream;

  private JUnit4Config config;

  private Set<RunListener> setOfRunListeners;

  JUnit4Bazel(Builder<?> builder) {
    initialize(checkNotNull(builder));
  }

  public static Builder<?> builder() {
    return new Builder<>();
  }

  private void initialize(final Builder<?> builder) {
    List<Class<?>> topLevelSuites = builder.suiteClasses;
    // Request.classes is self-memoizing because it creates the runner eagerly and delegates to
    // Request.runner(...). Request.aClass only needs MemoizingRequest for compatibility with
    // pre-4.11 JUnit versions where getRunner() reconstructed the runner on each call.
    this.request =
        (topLevelSuites.size() == 1)
            ? new MemoizingRequest(Request.aClass(topLevelSuites.get(0)))
            : Request.classes(topLevelSuites.toArray(new Class<?>[0]));
    this.cancellableRequestFactory = builder.module.cancellableRequestFactory();
    String topLevelSuiteName = SuiteUtil.getTopLevelSuiteName(topLevelSuites);
    ShardingEnvironment shardingEnvironment = builder.module.shardingEnvironment();
    ShardingFilters shardingFilters = builder.module.shardingFilters(shardingEnvironment);
    XmlResultWriter resultWriter = new AntXmlResultWriter();
    TestSuiteModel.Builder builder1 =
        new TestSuiteModel.Builder(
            builder.module.clock(), shardingFilters, shardingEnvironment, resultWriter);
    this.jUnit4TestModelBuilder = new JUnit4TestModelBuilder(request, topLevelSuiteName, builder1);
    this.testSuiteModelSupplier = new MemoizingSupplier<>(() -> jUnit4TestModelBuilder.get());
    this.stdoutStream = builder.module.stdout();
    this.config = builder.module.config();
    this.setOfRunListeners =
        builder.module.setOfRunListeners(config, testSuiteModelSupplier, cancellableRequestFactory);
  }

  public JUnit4Runner runner() {
    return new JUnit4Runner(
        request,
        cancellableRequestFactory,
        testSuiteModelSupplier,
        stdoutStream,
        config,
        setOfRunListeners,
        Collections.emptySet());
  }

  /** A builder for instantiating {@link JUnit4Bazel}. */
  public static class Builder<B extends Builder<B>> {
    private List<Class<?>> suiteClasses = new ArrayList<>();
    private JUnit4InstanceModules.Config config;
    protected JUnit4RunnerModule module;

    public JUnit4Bazel build() {
      if (suiteClasses.isEmpty()) {
        throw new IllegalStateException("suiteClass or suiteClasses must be set");
      }
      if (module == null) {
        this.module = createModule();
      }
      return new JUnit4Bazel(this);
    }

    private JUnit4RunnerModule createModule() {
      if (config == null) {
        throw new IllegalStateException(
            JUnit4InstanceModules.Config.class.getCanonicalName() + " must be set");
      }
      return new JUnit4RunnerModule(config.options());
    }

    @CanIgnoreReturnValue
    @SuppressWarnings("unchecked") // unchecked: safe cast for fluent builder pattern
    public B suiteClass(Class<?> suiteClass) {
      this.suiteClasses = Collections.singletonList(checkNotNull(suiteClass));
      return (B) this;
    }

    @CanIgnoreReturnValue
    @SuppressWarnings("unchecked") // Safe cast for fluent builder pattern
    public B suiteClasses(List<Class<?>> suiteClasses) {
      this.suiteClasses = new ArrayList<>(checkNotNull(suiteClasses));
      return (B) this;
    }

    @CanIgnoreReturnValue
    @SuppressWarnings("unchecked") // Safe cast for fluent builder pattern
    public B config(JUnit4InstanceModules.Config config) {
      this.config = checkNotNull(config);
      return (B) this;
    }
  }

  private static <T> T checkNotNull(T reference) {
    if (reference == null) {
      throw new NullPointerException();
    }
    return reference;
  }
}
