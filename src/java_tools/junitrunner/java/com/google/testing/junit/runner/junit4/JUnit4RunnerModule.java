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
import com.google.testing.junit.runner.internal.junit4.JUnit4TestNameListener;
import com.google.testing.junit.runner.internal.junit4.JUnit4TestStackTraceListener;
import com.google.testing.junit.runner.internal.junit4.JUnit4TestXmlListener;
import com.google.testing.junit.runner.internal.junit4.SettableCurrentRunningTest;
import com.google.testing.junit.runner.util.TestNameProvider;
import com.google.testing.junit.runner.util.Ticker;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.annotation.Nullable;
import javax.inject.Singleton;
import org.junit.runner.notification.RunListener;

/**
 * Utility class for real test runs. This is a legacy Dagger module.
 */
public final class JUnit4RunnerModule {
  static Ticker ticker() {
    return Ticker.systemTicker();
  }

  static SignalHandlers.HandlerInstaller signalHandlerInstaller() {
    return SignalHandlers.createRealHandlerInstaller();
  }

  static RunListener nameListener(JUnit4TestNameListener impl) {
    return impl;
  }

  static RunListener xmlListener(JUnit4TestXmlListener impl) {
    return impl;
  }

  static RunListener stackTraceListener(JUnit4TestStackTraceListener impl) {
    return impl;
  }

  @Singleton
  @Xml
  static OutputStream provideXmlStream(JUnit4Config config) {
    @Nullable Path path = config.getXmlOutputPath();

    if (path != null) {
      try {
        // TODO(bazel-team): Change the provider method to return ByteSink or CharSink
        return Files.newOutputStream(path);
      } catch (IOException e) {
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

  @Singleton
  SettableCurrentRunningTest provideCurrentRunningTest() {
    return new SettableCurrentRunningTest() {
      @Override
      protected void setGlobalTestNameProvider(TestNameProvider provider) {
        testNameProvider = provider;
      }
    };
  }
}
