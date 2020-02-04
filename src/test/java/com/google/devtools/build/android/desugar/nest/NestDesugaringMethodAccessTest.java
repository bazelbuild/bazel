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

import com.google.devtools.build.android.desugar.testing.junit.AsmNode;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRunner;
import com.google.devtools.build.android.desugar.testing.junit.DynamicClassLiteral;
import com.google.devtools.build.android.desugar.testing.junit.JdkSuppress;
import com.google.devtools.build.android.desugar.testing.junit.JdkVersion;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle.MemberUseContext;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import javax.inject.Inject;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.objectweb.asm.tree.ClassNode;

/** Tests for accessing private methods from another class within a nest. */
@RunWith(DesugarRunner.class)
@JdkSuppress(minJdkVersion = JdkVersion.V11)
public final class NestDesugaringMethodAccessTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addSourceInputsFromJvmFlag("input_srcs")
          .addJavacOptions("-source 11", "-target 11")
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
  public void inputClassFileMajorVersions(
      @AsmNode(className = "MethodNest", round = 0) ClassNode beforeDesugarClassNode,
      @AsmNode(className = "MethodNest", round = 1) ClassNode afterDesugarClassNode) {
    assertThat(beforeDesugarClassNode.version).isEqualTo(JdkVersion.V11);
    assertThat(afterDesugarClassNode.version).isEqualTo(JdkVersion.V1_7);
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

  @Test
  public void invokePrivateStaticMethod_staticInitializer(
      @RuntimeMethodHandle(
              className = "MethodNest",
              memberName = "populatedFromInvokePrivateStaticMethod",
              usage = MemberUseContext.FIELD_GETTER)
          MethodHandle populatedFromInvokePrivateStaticMethod)
      throws Throwable {
    long result = (long) populatedFromInvokePrivateStaticMethod.invoke();
    assertThat(result).isEqualTo(385L); // 128L + 256 + 1
  }

  @Test
  public void invokePrivateInstanceMethod_instanceInitializer(
      @RuntimeMethodHandle(
              className = "MethodNest",
              memberName = "populatedFromInvokePrivateInstanceMethod",
              usage = MemberUseContext.FIELD_GETTER)
          MethodHandle populatedFromInvokePrivateInstanceMethod)
      throws Throwable {
    long result = (long) populatedFromInvokePrivateInstanceMethod.invoke(invokerInstance);
    assertThat(result).isEqualTo(768L); // 128L + 256 + 1
  }

  @Test
  public void invokePrivateStaticMethod(
      @RuntimeMethodHandle(className = "MethodNest", memberName = "invokePrivateStaticMethod")
          MethodHandle invokePrivateStaticMethod)
      throws Throwable {
    long result = (long) invokePrivateStaticMethod.invokeExact((long) 1L, (int) 2);
    assertThat(result).isEqualTo(1L + 2);
  }

  @Test
  public void invokePrivateInstanceMethod(
      @RuntimeMethodHandle(className = "MethodNest", memberName = "invokePrivateInstanceMethod")
          MethodHandle invokePrivateInstanceMethod)
      throws Throwable {
    long result = (long) invokePrivateInstanceMethod.invoke(mateInstance, 2L, 3);
    assertThat(result).isEqualTo(2L + 3);
  }

  @Test
  public void invokeStaticMethod(
      @RuntimeMethodHandle(className = "MethodNest", memberName = "invokeStaticMethod")
          MethodHandle invokeStaticMethod)
      throws Throwable {
    long x = 3L;
    int y = 4;

    long result = (long) invokeStaticMethod.invoke(x, y);
    assertThat(result).isEqualTo(x + y);
  }

  @Test
  public void invokeInstanceMethod(
      @RuntimeMethodHandle(className = "MethodNest", memberName = "invokeInstanceMethod")
          MethodHandle invokeInstanceMethod)
      throws Throwable {
    long x = 4L;
    int y = 5;

    long result = (long) invokeInstanceMethod.invoke(mateInstance, x, y);
    assertThat(result).isEqualTo(x + y);
  }

  @Test
  public void invokeSuperAccessPrivateInstanceMethod(
      @RuntimeMethodHandle(
              className = "MethodNest",
              memberName = "invokeSuperAccessPrivateInstanceMethod")
          MethodHandle invokeSuperAccessPrivateInstanceMethod)
      throws Throwable {
    assertThat(
            invokeSuperAccessPrivateInstanceMethod.invoke(
                subClassMate.getConstructor().newInstance(), 7L, 8))
        .isEqualTo(16L); // 15 + 1
  }

  @Test
  public void invokeCastAccessPrivateInstanceMethod(
      @RuntimeMethodHandle(
              className = "MethodNest",
              memberName = "invokeCastAccessPrivateInstanceMethod")
          MethodHandle invokeCastAccessPrivateInstanceMethod)
      throws Throwable {
    long result =
        (long)
            invokeCastAccessPrivateInstanceMethod.invoke(
                subClassMate.getConstructor().newInstance(), 9L, 10);
    assertThat(result).isEqualTo(21L); // 19 + 2
  }
}
