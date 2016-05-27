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

import static com.google.inject.multibindings.Multibinder.newSetBinder;

import com.google.common.base.Optional;
import com.google.common.base.Ticker;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.inject.AbstractModule;
import com.google.inject.Provides;
import com.google.inject.Singleton;
import com.google.inject.multibindings.Multibinder;
import com.google.testing.junit.runner.internal.SignalHandlers;
import com.google.testing.junit.runner.util.TestNameProvider;

import org.junit.runner.notification.RunListener;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.List;

/**
 * Guice module for real test runs.
 */
public class JUnit4RunnerModule extends AbstractModule {
  private final Class<?> suite;
  private final JUnit4Config config;
  private final ImmutableList<String> unparsedArgs;

  public static JUnit4RunnerModule create(Class<?> suite, List<String> args) {
    JUnit4Options options = JUnit4Options.parse(System.getenv(), ImmutableList.copyOf(args));
    JUnit4Config config = new JUnit4Config(
        options.getTestIncludeFilter(),
        options.getTestExcludeFilter(),
        Optional.<Path>absent());
    return new JUnit4RunnerModule(suite, config, ImmutableList.copyOf(options.getUnparsedArgs()));
  }

  private JUnit4RunnerModule(
      Class<?> suite, JUnit4Config config, ImmutableList<String> unparsedArgs) {
    this.suite = suite;
    this.config = config;
    this.unparsedArgs = unparsedArgs;
  }

  @Override
  protected void configure() {
    install(new JUnit4RunnerBaseModule(suite));

    // We require explicit bindings so we don't use an unexpected just-in-time binding
    bind(SignalHandlers.class);

    // Normal bindings
    bind(JUnit4Config.class).toInstance(config);
    bind(Ticker.class).toInstance(Ticker.systemTicker());
    bind(SignalHandlers.HandlerInstaller.class).toInstance(
        SignalHandlers.createRealHandlerInstaller());

    // Bind listeners
    Multibinder<RunListener> listenerBinder = newSetBinder(binder(), RunListener.class);
    listenerBinder.addBinding().to(JUnit4TestNameListener.class);
    listenerBinder.addBinding().to(JUnit4TestXmlListener.class);
    listenerBinder.addBinding().to(JUnit4TestStackTraceListener.class);
  }

  @Provides @Singleton @Xml
  OutputStream provideXmlStream() {
    Optional<Path> path = config.getXmlOutputPath();

    if (path.isPresent()) {
      try {
        // TODO(bazel-team): Change the provider method to return ByteSink or CharSink
        return new FileOutputStream(path.get().toFile());
      } catch (FileNotFoundException e) {
        /*
         * We try to avoid throwing exceptions in the runner code. There is no
         * way to induce a test failure here, so the only thing we can do is
         * print a message and move on.
         */
        e.printStackTrace();
      }
    }

    return ByteStreams.nullOutputStream();
  }

  @Provides @Singleton
  SettableCurrentRunningTest provideCurrentRunningTest() {
    return new SettableCurrentRunningTest() {
      void setGlobalTestNameProvider(TestNameProvider provider) {
        testNameProvider = provider;
      }
    };
  }

  /**
   * Gets the list of unparsed command line arguments.
   */
  public ImmutableList<String> getUnparsedArgs() {
    return unparsedArgs;
  }
}
