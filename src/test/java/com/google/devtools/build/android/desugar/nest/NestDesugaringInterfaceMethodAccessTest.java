/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.nest;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DynamicClassLiteral;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle;
import com.google.testing.testsize.MediumTest;
import com.google.testing.testsize.MediumTestAttribute;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.nio.file.Paths;
import java.util.Arrays;
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
public class NestDesugaringInterfaceMethodAccessTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addInputs(Paths.get(System.getProperty("input_jar")))
          .setWorkingJavaPackage(
              "com.google.devtools.build.android.desugar.nest.testsrc.simpleunit.interfacemethod")
          .addCommandOptions("desugar_nest_based_private_access", "true")
          .build();

  @Inject
  @DynamicClassLiteral(value = "InterfaceNest$InterfaceMate")
  private Class<?> mate;

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

  @Inject
  @RuntimeMethodHandle(className = "InterfaceNest", memberName = "invokePrivateStaticMethod")
  private MethodHandle invokePrivateStaticMethod;

  @Test
  public void invokePrivateStaticMethod() throws Throwable {
    long result = (long) invokePrivateStaticMethod.invoke(10L, 20);
    assertThat(result).isEqualTo(30L);
  }

  @Inject
  @RuntimeMethodHandle(className = "InterfaceNest", memberName = "invokePrivateInstanceMethod")
  private MethodHandle invokePrivateInstanceMethod;

  @Test
  public void invokePrivateInstanceMethod() throws Throwable {
    long result = (long) invokePrivateInstanceMethod.invoke(mateInstance, 20L, 30);
    assertThat(result).isEqualTo(50L);
  }

  @Inject
  @RuntimeMethodHandle(
      className = "InterfaceNest",
      memberName = "invokeInClassBoundaryStaticMethod")
  private MethodHandle invokeInClassBoundaryStaticMethod;

  @Test
  public void invokeInClassBoundaryStaticMethod() throws Throwable {
    long result = (long) invokeInClassBoundaryStaticMethod.invoke(2L);
    assertThat(result).isEqualTo(3L); // 0 + 1 + 2
  }

  @Inject
  @RuntimeMethodHandle(
      className = "InterfaceNest",
      memberName = "invokeInClassBoundaryInstanceMethod")
  private MethodHandle invokeInClassBoundaryInstanceMethod;

  @Test
  public void invokeInClassBoundaryInstanceMethod() throws Throwable {
    long result = (long) invokeInClassBoundaryInstanceMethod.invoke(mateInstance, 3L);
    assertThat(result).isEqualTo(6L); // 0 + 1 + 2 + 3
  }

  @Inject
  @RuntimeMethodHandle(
      className = "InterfaceNest",
      memberName = "invokePrivateStaticMethodWithLambda")
  private MethodHandle invokePrivateStaticMethodWithLambda;

  @Test
  public void invokePrivateStaticMethodWithLambda() throws Throwable {
    long result = (long) invokePrivateStaticMethodWithLambda.invoke(2L, 3L, 4L, 5L);
    assertThat(result).isEqualTo(17L);
  }

  @Inject
  @RuntimeMethodHandle(
      className = "InterfaceNest",
      memberName = "invokePrivateInstanceMethodWithLambda")
  private MethodHandle invokePrivateInstanceMethodWithLambda;

  @Test
  public void invokePrivateInstanceMethodWithLambda() throws Throwable {
    long result =
        (long)
            invokePrivateInstanceMethodWithLambda.invoke(
                mateInstance, ImmutableList.of(2L, 3L), 4L, 5L);
    assertThat(result).isEqualTo(64850L);
  }
}
