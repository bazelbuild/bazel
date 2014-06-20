// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * A JUnit3 test-suite builder. This will go away when we migrate to JUnit4.
 */
public final class TestSuiteBuilder {

  private boolean withClassPath;
  private Set<TestClass> testClasses = Sets.newTreeSet(
      new TestClassNameComparator());
  private String suiteName;
  private Predicate<Class<?>> matchClassPredicate = Predicates.alwaysTrue();

  /**
   * Finds tests on the application class path.
   */
  public TestSuiteBuilder withClassPath() {
    this.withClassPath = true;
    return this;
  }

  /**
   * Adds the tests found (directly) in class {@code c} to the set of tests
   * this builder will search.
   */
  public TestSuiteBuilder addTestClass(Class<? extends TestCase> c) {
    testClasses.add(new TestClass(c));
    return this;
  }

  /**
   * Adds all the test classes (top-level or nested) found in package
   * {@code pkgName} or its subpackages to the set of tests this builder will
   * search.
   */
  public TestSuiteBuilder addPackageRecursive(String pkgName) {
    for (Class<?> c : getClassesRecursive(pkgName)) {
      addTestClass(c.asSubclass(TestCase.class));
    }
    return this;
  }

  private Set<Class<?>> getClassesRecursive(String pkgName) {
    if (!withClassPath) {
      throw new UnsupportedOperationException("Only classpath is supported test class collection");
    }
    Set<Class<?>> result = new LinkedHashSet<>();
    for (Class<?> clazz : Classpath.findClasses(pkgName)) {
      if (isTestClass(clazz)) {
        result.add(clazz);
      }
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
  public TestSuite create() {
    // Create a filter with which to exclude some of the objects from
    // this.testClasses.
    Predicate<TestClass> nonExcluded = new Predicate<TestClass>() {
      @Override
      public boolean apply(TestClass t) {
        Class<?> cachedClass = t.testClass;
        if (!matchClassPredicate.apply(cachedClass)) {
          return false;
        }
        return true;
      }
    };

    TestSuite ts = createEmptyTestSuite(suiteName);
    for (TestClass testClass : Iterables.filter(testClasses, nonExcluded)) {
      TestSuite suiteForThisClass = testClass.createSuite();
      if (suiteForThisClass != null) {
        ts.addTest(suiteForThisClass);
      }
    }

    return ts;
  }

  private TestSuite createEmptyTestSuite(String suiteName) {
    return (suiteName == null) ? new TestSuite() : new TestSuite(suiteName);
  }

  /**
   * Set the title of the suite (fun for IDEs).
   */
  public TestSuiteBuilder withName(String suiteName) {
    this.suiteName = suiteName;
    return this;
  }

  /**
   * Determines if a given class is a test class.
   *
   * @param container class to test
   * @return <code>true</code> if the test is a test class.
   */
  private boolean isTestClass(Class<?> container) {
    return TestCase.class.isAssignableFrom(container)
           && Modifier.isPublic(container.getModifiers())
           && !Modifier.isAbstract(container.getModifiers())
           && hasValidConstructor(container);
  }

  private static boolean hasValidConstructor(Class<?> container) {
    // TODO(kevinb): this is an unsafe typecast.
    @SuppressWarnings("unchecked") // generic type cast
    Class<TestCase> containerClass = (Class<TestCase>) container;

    return (findConstructor(containerClass) != null);
  }

  private static Constructor<? extends TestCase> findConstructor(
      Class<? extends TestCase> container) {
    Constructor<? extends TestCase> result = null;

    Constructor<? extends TestCase>[] constructors =
        getAllTestConstructors(container);
    for (Constructor<? extends TestCase> constructor : constructors) {
      if (Modifier.isPublic(constructor.getModifiers())) {
        Class<?>[] parameterTypes = constructor.getParameterTypes();
        if (parameterTypes.length == 0) {
          result = constructor;
          // JUnit prefers the String constructor, so keep looking
        } else if (parameterTypes.length == 1
            && parameterTypes[0] == String.class) {
          return constructor;
        }
      }
    }
    return result;
  }

  private static Constructor<? extends TestCase>[] getAllTestConstructors(
      Class<? extends TestCase> container) {
    // The cast below is not necessary in Java 5, but necessary in Java 6,
    // where the return type of Class.getDeclaredConstructors() was changed
    // from Constructor<T>[] to Constructor<?>[]
    @SuppressWarnings("unchecked")
    Constructor<? extends TestCase>[] constructors =
        (Constructor<? extends TestCase>[]) container.getConstructors();
    return constructors;
  }

  private static class TestClass {
    private static final Pattern TEST_NAME_PATTERN = Pattern.compile("test\\w*");

    private final Class<? extends TestCase> testClass;
    private final Constructor<? extends TestCase> constructor;
    private final List<Method> testMethods = Lists.newArrayList();

    TestClass(Class<? extends TestCase> c) {
      if (c == null) {
        throw new NullPointerException();
      }

      if (Modifier.isAbstract(c.getModifiers())) {
        throw new IllegalArgumentException("abstract");
      }

      this.testClass = c;

      /*
       * If there is a static "suite" method in the class, and we were not
       * told to ignore them, then first check for a suite method. If a suite
       * method is found, the test's constructor may not be public.
       */
      constructor = findConstructor(testClass);

      if (constructor == null) {
        throw new IllegalArgumentException(
          "Class " + testClass + " has no appropriate constructor");
      } else {
        Method[] methods = testClass.getMethods();
        Arrays.sort(methods, new Comparator<Method>() {
          @Override
          public int compare(Method m1, Method m2) {
            int i1 = m1.getName().hashCode();
            int i2 = m2.getName().hashCode();
            return (i1 != i2)
                ? ((i1 < i2) ? -1 : 1)
                : m1.toString().compareTo(m2.toString());
          }
        });

        /*
         * Note that the logic JUnit actually uses is much more complicated,
         * but I have absolutely no idea why.  This works.
         */
        for (Method m : methods) {
          if (isTestMethod(m)) {
            testMethods.add(m);
          }
        }
      }
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof TestClass) {
        TestClass that = (TestClass) o;
        return (this.testClass).equals(that.testClass);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return testClass.hashCode();
    }

    private static Test erroringTest(String name, final Throwable error) {
      return new TestCase(name) {
        @Override
        protected void runTest() throws Throwable {
          throw error;
        }
      };
    }

    /**
     * Creates a {@link TestSuite} for the given query.
     *
     * @return new suite, or null if no tests match the query
     */
    TestSuite createSuite() {
      List<Test> testCases = instantiateTestCases();
      if (testCases.isEmpty()) {
        return null;
      }

      TestSuite suite = new TestSuite(testClass.getName());
      for (Test tc : testCases) {
        suite.addTest(tc);
      }
      return suite;
    }

    private List<Test> instantiateTestCases() {
      List<Test> tests = Lists.newArrayList();
      for (Method m : testMethods) {
        try {
          TestCase tc = instantiateTestCase(m.getName());
          tests.add(tc);
        } catch (Exception e) {
          tests.add(erroringTest(testClass.getName() + "." + m.getName(), e));
        }
      }
      return tests;
    }

    private TestCase instantiateTestCase(String methodName) throws Exception {
      TestCase tc;
      if (constructor.getParameterTypes().length == 1) {
        tc = constructor.newInstance(methodName);
      } else {
        tc = constructor.newInstance();
        tc.setName(methodName);
      }
      return tc;
    }

    private boolean isTestMethod(Method m) {
      return m.getReturnType() == void.class
          && Modifier.isPublic(m.getModifiers())
          && !Modifier.isAbstract(m.getModifiers())
          && m.getParameterTypes().length == 0
          && TEST_NAME_PATTERN.matcher(m.getName()).matches();
    }
  }

  private static class TestClassNameComparator implements Comparator<TestClass> {
    @Override
    public int compare(TestClass o1, TestClass o2) {
      return o1.testClass.getName().compareTo(o2.testClass.getName());
    }
  }
}
