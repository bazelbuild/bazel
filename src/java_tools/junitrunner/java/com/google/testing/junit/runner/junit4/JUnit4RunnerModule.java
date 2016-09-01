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
import com.google.testing.junit.runner.util.TestNameProvider;
import com.google.testing.junit.runner.util.Ticker;

import dagger.Module;
import dagger.Provides;
import dagger.multibindings.IntoSet;

import org.junit.runner.notification.RunListener;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import javax.annotation.Nullable;
import javax.inject.Singleton;

/**
 * Dagger module for real test runs.
 */
@Module(includes = {JUnit4RunnerBaseModule.class, JUnit4InstanceModules.Config.class})
public final class JUnit4RunnerModule {
  @Provides
  static Ticker ticker() {
    return Ticker.systemTicker();
  }

  @Provides
  static SignalHandlers.HandlerInstaller signalHandlerInstaller() {
    return SignalHandlers.createRealHandlerInstaller();
  }

  @Provides
  @IntoSet
  static RunListener nameListener(JUnit4TestNameListener impl) {
    return impl;
  }

  @Provides
  @IntoSet
  static RunListener xmlListener(JUnit4TestXmlListener impl) {
    return impl;
  }

  @Provides
  @IntoSet
  static RunListener stackTraceListener(JUnit4TestStackTraceListener impl) {
    return impl;
  }


  @Provides
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

  @Provides @Singleton
  SettableCurrentRunningTest provideCurrentRunningTest() {
    return new SettableCurrentRunningTest() {
      @Override
      void setGlobalTestNameProvider(TestNameProvider provider) {
        testNameProvider = provider;
      }
    };
  }
}
