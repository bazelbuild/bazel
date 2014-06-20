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

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.model.RunnerBuilder;

import java.lang.reflect.Modifier;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * A JUnit4 suite implementation that finds all classes on the current classpath in or below the
 * package of the annotated class, except classes that are annotated with ClasspathSuite.
 */
public final class ClasspathSuite extends Suite {

  /**
   * Only called reflectively. Do not use programmatically.
   */
  public ClasspathSuite(Class<?> klass, RunnerBuilder builder) throws Throwable {
    super(builder, klass, getClasses(klass));
  }

  private static Class<?>[] getClasses(Class<?> klass) {
    Set<Class<?>> result = new LinkedHashSet<>();
    for (Class<?> clazz : Classpath.findClasses(klass.getPackage().getName())) {
      if (isTestClass(clazz)) {
        result.add(clazz);
      }
    }
    return result.toArray(new Class<?>[result.size()]);
  }

  /**
   * Determines if a given class is a test class.
   *
   * @param container class to test
   * @return <code>true</code> if the test is a test class.
   */
  private static boolean isTestClass(Class<?> container) {
    return (container.getAnnotation(RunWith.class) != null)
        && (container.getAnnotation(RunWith.class).value() != ClasspathSuite.class)
        && Modifier.isPublic(container.getModifiers())
        && !Modifier.isAbstract(container.getModifiers());
  }
}
