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
          ImmutableSet<OS> supportedOs = ImmutableSet.copyOf(Suite.getSupportedOs(testClass));
          return supportedOs.isEmpty() || supportedOs.contains(OS.getCurrent());
        }
      };
}
