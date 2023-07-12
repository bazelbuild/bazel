// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar.runtime;

import static com.google.common.truth.Truth.assertThat;

import java.lang.reflect.Method;

/**
 * A utility class for testing ThrowableExtension. It uses reflection to get the strategy name, so
 * as to avoid dependency on the runtime library. This is beneficial, because we can test whether
 * the runtime library is on the classpath.
 */
public class ThrowableExtensionTestUtility {

  private static final String SYSTEM_PROPERTY_EXPECTED_STRATEGY = "expected.strategy";

  public static String getTwrStrategyClassNameSpecifiedInSystemProperty() {
    String className = unquote(System.getProperty(SYSTEM_PROPERTY_EXPECTED_STRATEGY));
    assertThat(className).isNotEmpty();
    return className;
  }

  private static final String THROWABLE_EXTENSION_CLASS_NAME =
      "com.google.devtools.build.android.desugar.runtime.ThrowableExtension";

  private static boolean isStrategyOfClass(String className) {
    return getStrategyClassName().equals(className);
  }

  public static String getStrategyClassName() {
    try {
      Class<?> klass = Class.forName(THROWABLE_EXTENSION_CLASS_NAME);
      Method method = klass.getMethod("getStrategy");
      Object strategy = method.invoke(null);
      return strategy.getClass().getName();
    } catch (Throwable e) {
      throw new AssertionError(e);
    }
  }

  public static boolean isMimicStrategy() {
    return isStrategyOfClass(THROWABLE_EXTENSION_CLASS_NAME + "$MimicDesugaringStrategy");
  }

  public static boolean isNullStrategy() {
    return isStrategyOfClass(THROWABLE_EXTENSION_CLASS_NAME + "$NullDesugaringStrategy");
  }

  public static boolean isReuseStrategy() {
    return isStrategyOfClass(THROWABLE_EXTENSION_CLASS_NAME + "$ReuseDesugaringStrategy");
  }

  private static String unquote(String s) {
    if (s.startsWith("'") || s.startsWith("\"")) {
      assertThat(s).endsWith(s.substring(0, 1));
      return s.substring(1, s.length() - 1);
    } else {
      return s;
    }
  }
}
