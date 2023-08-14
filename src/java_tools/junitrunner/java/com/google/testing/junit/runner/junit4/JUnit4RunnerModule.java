// Copyright 2015 The Bazel Authors. All Rights Reserved.
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

import com.google.testing.junit.runner.internal.SignalHandlers;
import com.google.testing.junit.runner.internal.Xml;
import com.google.testing.junit.runner.internal.junit4.CancellableRequestFactory;
import com.google.testing.junit.runner.internal.junit4.JUnit4TestNameListener;
import com.google.testing.junit.runner.internal.junit4.JUnit4TestStackTraceListener;
import com.google.testing.junit.runner.internal.junit4.JUnit4TestXmlListener;
import com.google.testing.junit.runner.internal.junit4.SettableCurrentRunningTest;
import com.google.testing.junit.runner.model.TestSuiteModel;
import com.google.testing.junit.runner.sharding.ShardingEnvironment;
import com.google.testing.junit.runner.sharding.ShardingFilters;
import com.google.testing.junit.runner.util.TestClock;
import com.google.testing.junit.runner.util.TestNameProvider;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import javax.inject.Singleton;
import org.junit.runner.notification.RunListener;

/** Utility class for real test runs. This is a legacy Dagger module. */
class JUnit4RunnerModule {

  private final JUnit4Options options;

  public JUnit4RunnerModule(JUnit4Options options) {
    this.options = options;
  }

  @Singleton
  @Xml
  static OutputStream provideXmlStream(JUnit4Config config) {
    @Nullable Path path = config.getXmlOutputPath();

    if (path != null) {
      try {
        // TODO(bazel-team): Change the provider method to return ByteSink or CharSink
        return new FileOutputStream(path.toFile());
      } catch (FileNotFoundException e) {
        /*
         * We try to avoid throwing exceptions in the runner code. There is no
         * way to induce a test failure here, so the only thing we can do is
         * print a message and move on.
         */
        e.printStackTrace();
      }
    }

    // Returns an OutputStream that discards everything written into it.
    return new OutputStream() {
      @Override
      public void write(int b) {}

      @Override
      public void write(byte[] b) {
        if (b == null) {
          throw new NullPointerException();
        }
      }

      @Override
      public void write(byte[] b, int off, int len) {
        if (b == null) {
          throw new NullPointerException();
        }
      }

      @Override
      public String toString() {
        return "null OutputStream";
      }
    };
  }

  private static SettableCurrentRunningTest provideCurrentRunningTest() {
    return new SettableCurrentRunningTest() {
      @Override
      protected void setGlobalTestNameProvider(TestNameProvider provider) {
        testNameProvider = provider;
      }
    };
  }

  ShardingEnvironment shardingEnvironment() {
    return new ShardingEnvironment();
  }

  ShardingFilters shardingFilters(ShardingEnvironment shardingEnvironment) {
    return new ShardingFilters(shardingEnvironment, ShardingFilters.DEFAULT_SHARDING_STRATEGY);
  }

  PrintStream stdout() {
    return System.out;
  }

  JUnit4Config config() {
    return new JUnit4Config(
        options.getTestRunnerFailFast(),
        options.getTestIncludeFilter(),
        options.getTestExcludeFilter());
  }

  TestClock clock() {
    return TestClock.systemClock();
  }

  Set<RunListener> setOfRunListeners(
      JUnit4Config config,
      Supplier<TestSuiteModel> testSuiteModelSupplier,
      CancellableRequestFactory cancellableRequestFactory) {
    Set<RunListener> listeners = new HashSet<>();
    listeners.add(
        new JUnit4TestStackTraceListener(
            new SignalHandlers(SignalHandlers.createRealHandlerInstaller()), System.err));
    listeners.add(
        new JUnit4TestXmlListener(
            testSuiteModelSupplier,
            cancellableRequestFactory,
            new SignalHandlers(SignalHandlers.createRealHandlerInstaller()),
            new ProvideXmlStreamFactory(() -> config).get(),
            System.err));
    listeners.add(new JUnit4TestNameListener(provideCurrentRunningTest()));
    listeners.add(JUnit4RunnerBaseModule.provideTextListener(stdout()));
    return Collections.unmodifiableSet(listeners);
  }

  CancellableRequestFactory cancellableRequestFactory() {
    return new CancellableRequestFactory();
  }
}
