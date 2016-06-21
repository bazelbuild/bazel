// Copyright 2012 The Bazel Authors. All Rights Reserved.
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

import static com.google.testing.junit.runner.sharding.ShardingFilters.DEFAULT_SHARDING_STRATEGY;

import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.testing.junit.junit4.runner.MemoizingRequest;
import com.google.testing.junit.runner.internal.Stdout;
import com.google.testing.junit.runner.junit4.JUnit4InstanceModules.SuiteClass;
import com.google.testing.junit.runner.model.TestSuiteModel;
import com.google.testing.junit.runner.sharding.api.ShardingFilterFactory;

import dagger.Module;
import dagger.Multibindings;
import dagger.Provides;
import dagger.multibindings.IntoSet;

import org.junit.internal.TextListener;
import org.junit.runner.Request;
import org.junit.runner.notification.RunListener;

import java.io.OutputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.Set;

import javax.inject.Singleton;

/**
 * Dagger module for creating a {@link JUnit4Runner}. This contains the common
 * bindings used when either the runner runs actual tests or when we do
 * integration tests of the runner itself.
 *
 */
@Module(includes = SuiteClass.class)
public final class JUnit4RunnerBaseModule {

  @Multibindings
  interface MultiBindings {
    Set<JUnit4Runner.Initializer> initializers();
  }

  @Provides
  static ShardingFilterFactory shardingFilterFactory() {
    return DEFAULT_SHARDING_STRATEGY;
  }

  @Provides
  @IntoSet
  static RunListener textListener(TextListener impl) {
    return impl;
  }


  @Provides
  @Singleton
  static Supplier<TestSuiteModel> provideTestSuiteModelSupplier(JUnit4TestModelBuilder builder) {
    return Suppliers.memoize(builder);
  }

  @Provides
  @Singleton
  static TextListener provideTextListener(@Stdout PrintStream testRunnerOut) {
    return new TextListener(asUtf8PrintStream(testRunnerOut));
  }

  private static PrintStream asUtf8PrintStream(OutputStream stream) {
    try {
      return new PrintStream(stream, false /* autoFlush */, StandardCharsets.UTF_8.toString());
    } catch (UnsupportedEncodingException e) {
      throw new IllegalStateException("UTF-8 must be supported as per the java language spec", e);
    }
  }

  @Provides
  @Singleton
  static Request provideRequest(@TopLevelSuite Class<?> suiteClass) {
    /*
     * JUnit4Runner requests the Runner twice, once to build the model (before
     * filtering) and once to run the tests (after filtering). Constructing the
     * Runner can be expensive, so Memoize the Runner.
     */
    Request request = Request.aClass(suiteClass);
    return new MemoizingRequest(request);
  }
}
