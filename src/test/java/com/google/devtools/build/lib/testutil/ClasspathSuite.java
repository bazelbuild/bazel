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

import org.junit.runners.Suite;
import org.junit.runners.model.RunnerBuilder;

import java.util.Set;

/**
 * A suite implementation that finds all JUnit 3 and 4 classes on the current classpath in or below
 * the package of the annotated class, except classes that are annotated with {@code ClasspathSuite}
 * or {@link CustomSuite}.
 *
 * <p>If you need to specify a custom test class filter or a different package prefix, then use
 * {@link CustomSuite} instead.
 */
public final class ClasspathSuite extends Suite {

  /**
   * Only called reflectively. Do not use programmatically.
   */
  public ClasspathSuite(Class<?> klass, RunnerBuilder builder) throws Throwable {
    super(builder, klass, getClasses(klass));
  }

  private static Class<?>[] getClasses(Class<?> klass) {
    Set<Class<?>> result = new TestSuiteBuilder().addPackageRecursive(klass.getPackage().getName())
        .create();
    return result.toArray(new Class<?>[result.size()]);
  }
}
