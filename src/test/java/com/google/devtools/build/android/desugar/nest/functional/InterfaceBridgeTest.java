// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.nest.functional;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DynamicClassLiteral;
import com.google.testing.testsize.MediumTest;
import com.google.testing.testsize.MediumTestAttribute;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import javax.inject.Inject;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for accessing private interface methods from another class within a nest. */
@RunWith(JUnit4.class)
@MediumTest(MediumTestAttribute.FILE)
public class InterfaceBridgeTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addInputs(Paths.get(System.getProperty("input_jar")))
          .setWorkingJavaPackage(
              "com.google.devtools.build.android.desugar.nest.functional.testsrc.simpleunit.interfacemethod")
          .addCommandOptions("desugar_nest_based_private_access", "true")
          .build();

  @Inject
  @DynamicClassLiteral(value = "InterfaceNest$InterfaceMate")
  private Class<?> mate;

  @Inject
  @DynamicClassLiteral(value = "InterfaceNest")
  private Class<?> invoker;

  private Object mateInstance;

  @Inject
  @DynamicClassLiteral(value = "InterfaceNest$ConcreteMate")
  private Class<?> concreteMate;

  @Before
  public void loadClassesInNest() throws Exception {
    mateInstance = concreteMate.getDeclaredConstructor().newInstance();
  }

  @Test
  public void interfaceDeclaredMethods() {
    List<String> interfaceMethodNames =
        Arrays.stream(mate.getDeclaredMethods()).map(Method::getName).collect(Collectors.toList());
    assertThat(interfaceMethodNames)
        .containsExactly("invokeInstanceMethodInClassBoundary", "mateValues");
  }

  @Test
  public void invokePrivateStaticMethod() throws Exception {
    assertThat(
            invoker
                .getDeclaredMethod("invokePrivateStaticMethod", long.class, int.class)
                .invoke(null, 10L, 20))
        .isEqualTo(30L);
  }

  @Test
  public void invokePrivateInstanceMethod() throws Exception {
    assertThat(
            invoker
                .getDeclaredMethod("invokePrivateInstanceMethod", mate, long.class, int.class)
                .invoke(null, mateInstance, 20L, 30))
        .isEqualTo(50L);
  }

  @Test
  public void invokeInClassBoundaryStaticMethod() throws Exception {
    assertThat(
            invoker
                .getDeclaredMethod("invokeInClassBoundaryStaticMethod", long.class)
                .invoke(null, 2L))
        .isEqualTo(3L); // 0 + 1 + 2
  }

  @Test
  public void invokeInClassBoundaryInstanceMethod() throws Exception {
    assertThat(
            invoker
                .getDeclaredMethod("invokeInClassBoundaryInstanceMethod", mate, long.class)
                .invoke(null, mateInstance, 3L))
        .isEqualTo(6L); // 0 + 1 + 2 + 3
  }

  @Test
  public void invokePrivateStaticMethodWithLambda() throws Exception {
    assertThat(
            invoker
                .getDeclaredMethod(
                    "invokePrivateStaticMethodWithLambda",
                    long.class,
                    long.class,
                    long.class,
                    long.class)
                .invoke(null, 2L, 3L, 4L, 5L))
        .isEqualTo(17L);
  }

  @Test
  public void invokePrivateInstanceMethodWithLambda() throws Exception {
    assertThat(
            invoker
                .getDeclaredMethod(
                    "invokePrivateInstanceMethodWithLambda",
                    mate,
                    Collection.class,
                    long.class,
                    long.class)
                .invoke(null, mateInstance, ImmutableList.of(2L, 3L), 4L, 5L))
        .isEqualTo(64850L);
  }
}
