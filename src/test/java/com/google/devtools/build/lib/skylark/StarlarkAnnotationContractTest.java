// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylark;

import com.google.devtools.build.lib.util.Classpath;
import java.lang.reflect.Method;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkInterfaceUtils;
import net.starlark.java.annot.StarlarkMethod;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests that bazel usages of {@link StarlarkMethod} and {@link StarlarkBuiltin} abide by the
 * contracts specified in their documentation.
 *
 * <p>Tests in this class use the java reflection API.
 *
 * <p>This verification *would* be done via annotation processor, but annotation processors in java
 * don't have access to the full set of information that the java reflection API has.
 */
@RunWith(JUnit4.class)
public class StarlarkAnnotationContractTest {

  // Common prefix of packages in bazel that may have classes that implement or extend a
  // Starlark type.
  private static final String MODULES_PACKAGE_PREFIX = "com/google/devtools/build";

  /**
   * Verifies that every class in bazel that implements or extends a Starlark type has a clearly
   * resolvable type.
   *
   * <p>If this test fails, it indicates the following error scenario:
   *
   * <p>Suppose class A is a subclass of both B and C, where B and C are annotated with {@link
   * StarlarkBuiltin} annotations (and are thus considered "Starlark types"). If B is not a subclass
   * of C (nor visa versa), then it's impossible to resolve whether A is of type B or if A is of
   * type C. It's both! The way to resolve this is usually to have A be its own type (annotated with
   * {@link StarlarkBuiltin}), and thus have the explicit type of A be semantically "B and C".
   */
  @Test
  public void testResolvableStarlarkBuiltins() throws Exception {
    for (Class<?> candidateClass : Classpath.findClasses(MODULES_PACKAGE_PREFIX)) {
      StarlarkInterfaceUtils.getStarlarkBuiltin(candidateClass);
    }
  }

  /**
   * Verifies that no class or interface has a method annotated with {@link StarlarkMethod} unless
   * that class or interface is annotated with either {@link StarlarkGlobalLibrary} or with {@link
   * StarlarkBuiltin}.
   */
  @Test
  public void testStarlarkCallableScope() throws Exception {
    for (Class<?> candidateClass : Classpath.findClasses(MODULES_PACKAGE_PREFIX)) {
      if (StarlarkInterfaceUtils.getStarlarkBuiltin(candidateClass) == null
          && !StarlarkInterfaceUtils.hasStarlarkGlobalLibrary(candidateClass)) {
        for (Method method : candidateClass.getMethods()) {
          StarlarkMethod callable = StarlarkInterfaceUtils.getStarlarkMethod(method);
          if (callable != null && method.getDeclaringClass() == candidateClass) {
            throw new AssertionError(
                String.format(
                    "Class %s has a StarlarkMethod method %s but is neither a @StarlarkBuiltin"
                        + " nor a @StarlarkGlobalLibrary",
                    candidateClass, method.getName()));
          }
        }
      }
    }
  }
}
