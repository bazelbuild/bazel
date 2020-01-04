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

import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DynamicClassLiteral;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle.MemberUseContext;
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

/** Tests for accessing private methods from another class within a nest. */
@RunWith(JUnit4.class)
@MediumTest(MediumTestAttribute.FILE)
public final class NestDesugaringMethodAccessTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addInputs(Paths.get(System.getProperty("input_jar")))
          .setWorkingJavaPackage(
              "com.google.devtools.build.android.desugar.nest.testsrc.simpleunit.method")
          .addCommandOptions("desugar_nest_based_private_access", "true")
          .build();

  @Inject
  @DynamicClassLiteral(value = "MethodNest$MethodOwnerMate")
  private Class<?> mate;

  @Inject
  @DynamicClassLiteral("MethodNest$SubMate")
  private Class<?> subClassMate;

  @Inject
  @DynamicClassLiteral("MethodNest")
  private Class<?> invoker;

  private Object mateInstance;
  private Object invokerInstance;

  @Before
  public void loadClassesInNest() throws Exception {
    mateInstance = mate.getConstructor().newInstance();
    invokerInstance = invoker.getDeclaredConstructor().newInstance();
  }

  @Test
  public void methodBridgeGeneration() throws Exception {
    List<String> bridgeMethodNames =
        Arrays.stream(mate.getDeclaredMethods())
            .map(Method::getName)
            .filter(name -> !name.startsWith("$jacoco"))
            .collect(Collectors.toList());
    assertThat(bridgeMethodNames)
        .containsExactly(
            "staticMethod",
            "instanceMethod",
            "privateStaticMethod",
            "privateInstanceMethod",
            "inClassBoundStaticMethod",
            "inClassBoundInstanceMethod",
            "privateStaticMethod$bridge",
            "privateInstanceMethod$bridge");
  }

  @Inject
  @RuntimeMethodHandle(
      className = "MethodNest",
      memberName = "populatedFromInvokePrivateStaticMethod",
      usage = MemberUseContext.FIELD_GETTER)
  private MethodHandle populatedFromInvokePrivateStaticMethod;

  @Test
  public void invokePrivateStaticMethod_staticInitializer() throws Throwable {
    long result = (long) populatedFromInvokePrivateStaticMethod.invoke();
    assertThat(result).isEqualTo(385L); // 128L + 256 + 1
  }

  @Inject
  @RuntimeMethodHandle(
      className = "MethodNest",
      memberName = "populatedFromInvokePrivateInstanceMethod",
      usage = MemberUseContext.FIELD_GETTER)
  private MethodHandle populatedFromInvokePrivateInstanceMethod;

  @Test
  public void invokePrivateInstanceMethod_instanceInitializer() throws Throwable {
    long result = (long) populatedFromInvokePrivateInstanceMethod.invoke(invokerInstance);
    assertThat(result).isEqualTo(768L); // 128L + 256 + 1
  }

  @Inject
  @RuntimeMethodHandle(className = "MethodNest", memberName = "invokePrivateStaticMethod")
  private MethodHandle invokePrivateStaticMethod;

  @Test
  public void invokePrivateStaticMethod() throws Throwable {
    long result = (long) invokePrivateStaticMethod.invokeExact((long) 1L, (int) 2);
    assertThat(result).isEqualTo(1L + 2);
  }

  @Inject
  @RuntimeMethodHandle(className = "MethodNest", memberName = "invokePrivateInstanceMethod")
  private MethodHandle invokePrivateInstanceMethod;

  @Test
  public void invokePrivateInstanceMethod() throws Throwable {
    long result = (long) invokePrivateInstanceMethod.invoke(mateInstance, 2L, 3);
    assertThat(result).isEqualTo(2L + 3);
  }

  @Inject
  @RuntimeMethodHandle(className = "MethodNest", memberName = "invokeStaticMethod")
  private MethodHandle invokeStaticMethod;

  @Test
  public void invokeStaticMethod() throws Throwable {
    long x = 3L;
    int y = 4;

    long result = (long) invokeStaticMethod.invoke(x, y);
    assertThat(result).isEqualTo(x + y);
  }

  @Inject
  @RuntimeMethodHandle(className = "MethodNest", memberName = "invokeInstanceMethod")
  private MethodHandle invokeInstanceMethod;

  @Test
  public void invokeInstanceMethod() throws Throwable {
    long x = 4L;
    int y = 5;

    long result = (long) invokeInstanceMethod.invoke(mateInstance, x, y);
    assertThat(result).isEqualTo(x + y);
  }

  @Inject
  @RuntimeMethodHandle(
      className = "MethodNest",
      memberName = "invokeSuperAccessPrivateInstanceMethod")
  private MethodHandle invokeSuperAccessPrivateInstanceMethod;

  @Test
  public void invokeSuperAccessPrivateInstanceMethod() throws Throwable {
    assertThat(
            invokeSuperAccessPrivateInstanceMethod.invoke(
                subClassMate.getConstructor().newInstance(), 7L, 8))
        .isEqualTo(16L); // 15 + 1
  }

  @Inject
  @RuntimeMethodHandle(
      className = "MethodNest",
      memberName = "invokeCastAccessPrivateInstanceMethod")
  private MethodHandle invokeCastAccessPrivateInstanceMethod;

  @Test
  public void invokeCastAccessPrivateInstanceMethod() throws Throwable {
    long result =
        (long)
            invokeCastAccessPrivateInstanceMethod.invoke(
                subClassMate.getConstructor().newInstance(), 9L, 10);
    assertThat(result).isEqualTo(21L); // 19 + 2
  }
}
