// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.testutil;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.util.OS;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A base class for constructing test suites by searching the classpath for
 * tests, possibly restricted to a predicate.
 */
public class BazelTestSuiteBuilder {

  static {
    // Avoid verbose INFO logging in tests.
    Logger.getLogger(BazelTestSuiteBuilder.class.getName()).getParent().setLevel(Level.WARNING);
  }

  /**
   * @return a TestSuiteBuilder configured for Bazel.
   */
  protected TestSuiteBuilder getBuilder() {
    return new TestSuiteBuilder()
        .addPackageRecursive("com.google.devtools.build.lib");
  }

  /** A predicate that succeeds only if the test supports the current operating system. */
  public static final Predicate<Class<?>> TEST_SUPPORTS_CURRENT_OS =
      new Predicate<Class<?>>() {
        @Override
        public boolean apply(Class<?> testClass) {
          ImmutableSet<OS> supportedOs = ImmutableSet.copyOf(getSupportedOs(testClass));
          return supportedOs.isEmpty() || supportedOs.contains(OS.getCurrent());
        }
      };

  /** Given a class, determine the list of operating systems its tests can run under. */
  private static OS[] getSupportedOs(Class<?> clazz) {
    return getAnnotationElementOrDefault(clazz, "supportedOs");
  }

  /**
   * Returns the value of the given element in the {@link TestSpec} annotation of the given class,
   * or the default value of that element if the class doesn't have a {@link TestSpec} annotation.
   */
  @SuppressWarnings("unchecked")
  private static <T> T getAnnotationElementOrDefault(Class<?> clazz, String elementName) {
    TestSpec spec = clazz.getAnnotation(TestSpec.class);
    try {
      Method method = TestSpec.class.getMethod(elementName);
      return spec != null ? (T) method.invoke(spec) : (T) method.getDefaultValue();
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException("no such element " + elementName, e);
    } catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
      throw new IllegalStateException("can't invoke accessor for element " + elementName, e);
    }
  }
}
