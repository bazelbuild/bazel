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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
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

  /** A predicate that succeeds only for LARGE tests. */
  public static final Predicate<Class<?>> TEST_IS_LARGE =
      hasSize(Suite.LARGE_TESTS);

  /** A predicate that succeeds only for MEDIUM tests. */
  public static final Predicate<Class<?>> TEST_IS_MEDIUM =
      hasSize(Suite.MEDIUM_TESTS);

  /** A predicate that succeeds only for SMALL tests. */
  public static final Predicate<Class<?>> TEST_IS_SMALL =
      hasSize(Suite.SMALL_TESTS);

  /** A predicate that succeeds only for non-flaky tests. */
  public static final Predicate<Class<?>> TEST_IS_FLAKY = new Predicate<Class<?>>() {
    @Override
    public boolean apply(Class<?> testClass) {
      return Suite.isFlaky(testClass);
    }
  };

  /** A predicate that succeeds only for non-local-only tests. */
  public static final Predicate<Class<?>> TEST_IS_LOCAL_ONLY =
      new Predicate<Class<?>>() {
        @Override
        public boolean apply(Class<?> testClass) {
          return Suite.isLocalOnly(testClass);
        }
      };

  /** A predicate that succeeds only if the test supports the current operating system. */
  public static final Predicate<Class<?>> TEST_SUPPORTS_CURRENT_OS =
      new Predicate<Class<?>>() {
        @Override
        public boolean apply(Class<?> testClass) {
          ImmutableSet<OS> supportedOs = ImmutableSet.copyOf(Suite.getSupportedOs(testClass));
          return supportedOs.isEmpty() || supportedOs.contains(OS.getCurrent());
        }
      };


  private static Predicate<Class<?>> hasSize(final Suite size) {
    return new Predicate<Class<?>>() {
      @Override
      public boolean apply(Class<?> testClass) {
        return Suite.getSize(testClass) == size;
      }
    };
  }

  protected static Predicate<Class<?>> inSuite(final String suiteName) {
    return new Predicate<Class<?>>() {
      @Override
      public boolean apply(Class<?> testClass) {
        return Suite.getSuiteName(testClass).equalsIgnoreCase(suiteName);
      }
    };
  }

  /**
   * Given a TestCase subclass, returns its designated suite annotation, if
   * any, or the empty string otherwise.
   */
  public static String getSuite(Class<?> clazz) {
    TestSpec spec = clazz.getAnnotation(TestSpec.class);
    return spec == null ? "" : spec.suite();
  }

  /**
   * Returns a predicate over TestCases that is true iff the TestCase has a
   * TestSpec annotation whose suite="..." value (a comma-separated list of
   * tags) matches all of the query operators specified in the system property
   * {@code blaze.suite}.  The latter is also a comma-separated list, but of
   * query operators, each of which is either the name of a tag which must be
   * present (e.g. "foo"), or the !-prefixed name of a tag that must be absent
   * (e.g. "!foo").
   */
  public static Predicate<Class<?>> matchesSuiteQuery() {
    final String suiteProperty = System.getProperty("blaze.suite");
    if (suiteProperty == null) {
      throw new IllegalArgumentException("blaze.suite property not found");
    }
    final Set<String> queryTokens = splitCommas(suiteProperty);
    return new Predicate<Class<?>>() {
      @Override
      public boolean apply(Class<?> testClass) {
        // Return true iff every queryToken is satisfied by suiteTags.
        Set<String> suiteTags = splitCommas(getSuite(testClass));
        for (String queryToken : queryTokens) {
          if (queryToken.startsWith("!")) { // forbidden tag
            if (suiteTags.contains(queryToken.substring(1))) {
              return false;
            }
          } else { // mandatory tag
            if (!suiteTags.contains(queryToken)) {
              return false;
            }
          }
        }
        return true;
      }
    };
  }

  private static Set<String> splitCommas(String s) {
    return new HashSet<>(Arrays.asList(s.split(",")));
  }

}
