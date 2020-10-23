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
import com.google.common.base.Predicates;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.util.Classpath;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import java.lang.reflect.Modifier;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.Set;
import junit.framework.TestCase;
import org.junit.runner.RunWith;

/**
 * A collector for test classes, for both JUnit 3 and 4. To be used in combination with {@link
 * CustomSuite}.
 */
public final class TestSuiteBuilder {

  private Set<Class<?>> testClasses = Sets.newTreeSet(new TestClassNameComparator());
  private Predicate<Class<?>> matchClassPredicate = Predicates.alwaysTrue();
  private final boolean tolerateEmptyTestSuites;

  public TestSuiteBuilder() {
    tolerateEmptyTestSuites = false;
  }

  /**
   * @param tolerateEmptyTestSuites set this to true to add an empty test which passes to the suite.
   *     Its better for Test Suites to fail when they create an empty set of classes to test, so new
   *     suites should avoid setting this to true.
   */
  public TestSuiteBuilder(boolean tolerateEmptyTestSuites) {
    this.tolerateEmptyTestSuites = tolerateEmptyTestSuites;
  }

  /**
   * Adds the tests found (directly) in class {@code c} to the set of tests
   * this builder will search.
   */
  public TestSuiteBuilder addTestClass(Class<?> c) {
    testClasses.add(c);
    return this;
  }

  /**
   * Adds all the test classes (top-level or nested) found in package
   * {@code pkgName} or its subpackages to the set of tests this builder will
   * search.
   */
  public TestSuiteBuilder addPackageRecursive(String pkgName) {
    for (Class<?> c : getClassesRecursive(pkgName)) {
      addTestClass(c);
    }
    return this;
  }

  private Set<Class<?>> getClassesRecursive(String pkgName) {
    Set<Class<?>> result = new LinkedHashSet<>();
    try {
      for (Class<?> clazz : Classpath.findClasses(pkgName)) {
        if (isTestClass(clazz)) {
          result.add(clazz);
        }
      }
    } catch (ClassPathException e) {
      throw new AssertionError("Cannot retrive classes: " + e.getMessage());
    }
    return result;
  }

  /**
   * Specifies a predicate returns false for classes we want to exclude.
   */
  public TestSuiteBuilder matchClasses(Predicate<Class<?>> predicate) {
    matchClassPredicate = predicate;
    return this;
  }

  /**
   * Creates and returns a TestSuite containing the tests from the given
   * classes and/or packages which matched the given tags.
   */
  public Set<Class<?>> create() {
    Set<Class<?>> result = new LinkedHashSet<>();
    for (Class<?> testClass : Iterables.filter(testClasses, matchClassPredicate)) {
      result.add(testClass);
    }
    if (tolerateEmptyTestSuites && result.isEmpty()) {
      // We have some cases where the resulting test suite is empty, which some of our test
      // infrastructure treats as an error.
      result.add(TautologyTest.class);
    }
    return result;
  }

  /**
   * Determines if a given class is a test class.
   *
   * @param container class to test
   * @return <code>true</code> if the test is a test class.
   */
  private static boolean isTestClass(Class<?> container) {
    return (isJunit4Test(container) || isJunit3Test(container))
        && !isSuite(container)
        && !Modifier.isAbstract(container.getModifiers());
  }

  private static boolean isJunit4Test(Class<?> container) {
    return container.isAnnotationPresent(RunWith.class);
  }

  private static boolean isJunit3Test(Class<?> container) {
    return TestCase.class.isAssignableFrom(container);
  }

  /**
   * Classes that have a {@code RunWith} annotation for {@link ClasspathSuite} or {@link
   * CustomSuite} are automatically excluded to avoid picking up the suite class itself.
   */
  private static boolean isSuite(Class<?> container) {
    RunWith runWith = container.getAnnotation(RunWith.class);
    return (runWith != null)
        && ((runWith.value() == ClasspathSuite.class) || (runWith.value() == CustomSuite.class));
  }

  private static class TestClassNameComparator implements Comparator<Class<?>> {
    @Override
    public int compare(Class<?> o1, Class<?> o2) {
      return o1.getName().compareTo(o2.getName());
    }
  }

  /**
   * A test that does nothing and always passes. We have some cases where an empty test suite is
   * treated as an error, so we use this test to make sure that the test suite is always non-empty.
   */
  public static class TautologyTest extends TestCase {
    public void testThatNothingHappens() {
    }
  }
}
