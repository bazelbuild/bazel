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
import com.google.devtools.build.android.desugar.testing.junit.AsmNode;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRunner;
import com.google.devtools.build.android.desugar.testing.junit.DynamicClassLiteral;
import com.google.devtools.build.android.desugar.testing.junit.JdkSuppress;
import com.google.devtools.build.android.desugar.testing.junit.JdkVersion;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle;
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

/** Tests for accessing private interface methods from another class within a nest. */
@RunWith(DesugarRunner.class)
@JdkSuppress(minJdkVersion = JdkVersion.V11)
public final class NestDesugaringInterfaceMethodAccessTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addSourceInputsFromJvmFlag("input_srcs")
          .addJavacOptions("-source 11", "-target 11")
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
  public void inputClassFileMajorVersions(
      @AsmNode(className = "InterfaceNest", round = 0) ClassNode beforeDesugarClassNode,
      @AsmNode(className = "InterfaceNest", round = 1) ClassNode afterDesugarClassNode) {
    assertThat(beforeDesugarClassNode.version).isEqualTo(JdkVersion.V11);
    assertThat(afterDesugarClassNode.version).isEqualTo(JdkVersion.V1_7);
  }

  @Test
  public void interfaceDeclaredMethods() {
    List<String> interfaceMethodNames =
        Arrays.stream(mate.getDeclaredMethods()).map(Method::getName).collect(Collectors.toList());
    assertThat(interfaceMethodNames)
        .containsExactly("invokeInstanceMethodInClassBoundary", "mateValues");
  }

  @Test
  public void invokePrivateStaticMethod(
      @RuntimeMethodHandle(className = "InterfaceNest", memberName = "invokePrivateStaticMethod")
          MethodHandle invokePrivateStaticMethod)
      throws Throwable {
    long result = (long) invokePrivateStaticMethod.invoke(10L, 20);
    assertThat(result).isEqualTo(30L);
  }

  @Test
  public void invokePrivateInstanceMethod(
      @RuntimeMethodHandle(className = "InterfaceNest", memberName = "invokePrivateInstanceMethod")
          MethodHandle invokePrivateInstanceMethod)
      throws Throwable {
    long result = (long) invokePrivateInstanceMethod.invoke(mateInstance, 20L, 30);
    assertThat(result).isEqualTo(50L);
  }

  @Test
  public void invokeInClassBoundaryStaticMethod(
      @RuntimeMethodHandle(
              className = "InterfaceNest",
              memberName = "invokeInClassBoundaryStaticMethod")
          MethodHandle invokeInClassBoundaryStaticMethod)
      throws Throwable {
    long result = (long) invokeInClassBoundaryStaticMethod.invoke(2L);
    assertThat(result).isEqualTo(3L); // 0 + 1 + 2
  }

  @Test
  public void invokeInClassBoundaryInstanceMethod(
      @RuntimeMethodHandle(
              className = "InterfaceNest",
              memberName = "invokeInClassBoundaryInstanceMethod")
          MethodHandle invokeInClassBoundaryInstanceMethod)
      throws Throwable {
    long result = (long) invokeInClassBoundaryInstanceMethod.invoke(mateInstance, 3L);
    assertThat(result).isEqualTo(6L); // 0 + 1 + 2 + 3
  }

  @Test
  public void invokePrivateStaticMethodWithLambda(
      @RuntimeMethodHandle(
              className = "InterfaceNest",
              memberName = "invokePrivateStaticMethodWithLambda")
          MethodHandle invokePrivateStaticMethodWithLambda)
      throws Throwable {
    long result = (long) invokePrivateStaticMethodWithLambda.invoke(2L, 3L, 4L, 5L);
    assertThat(result).isEqualTo(17L);
  }

  @Test
  public void invokePrivateInstanceMethodWithLambda(
      @RuntimeMethodHandle(
              className = "InterfaceNest",
              memberName = "invokePrivateInstanceMethodWithLambda")
          MethodHandle invokePrivateInstanceMethodWithLambda)
      throws Throwable {
    long result =
        (long)
            invokePrivateInstanceMethodWithLambda.invoke(
                mateInstance, ImmutableList.of(2L, 3L), 4L, 5L);
    assertThat(result).isEqualTo(64850L);
  }
}
