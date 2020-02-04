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
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import javax.inject.Inject;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.MethodNode;

/** Tests for accessing private constructors from another class within a nest. */
@RunWith(DesugarRunner.class)
@JdkSuppress(minJdkVersion = JdkVersion.V11)
public final class NestDesugaringConstructorAccessTest {

  private static final Lookup lookup = MethodHandles.lookup();

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, lookup)
          .addSourceInputsFromJvmFlag("input_srcs")
          .addJavacOptions("-source 11", "-target 11")
          .setWorkingJavaPackage(
              "com.google.devtools.build.android.desugar.nest.testsrc.simpleunit.constructor")
          .addCommandOptions("desugar_nest_based_private_access", "true")
          .build();

  @Inject
  @DynamicClassLiteral("ConstructorNest$ConstructorServiceMate")
  private Class<?> mate;

  @Inject
  @DynamicClassLiteral("ConstructorNest")
  private Class<?> invoker;

  @Test
  public void inputClassFileMajorVersions(
      @AsmNode(className = "ConstructorNest", round = 0) ClassNode beforeDesugarClassNode,
      @AsmNode(className = "ConstructorNest", round = 1) ClassNode afterDesugarClassNode) {
    assertThat(beforeDesugarClassNode.version).isEqualTo(JdkVersion.V11);
    assertThat(afterDesugarClassNode.version).isEqualTo(JdkVersion.V1_7);
  }

  @Test
  public void companionClassIsPresent(
      @DynamicClassLiteral("ConstructorNest$NestCC") Class<?> companion) {
    assertThat(companion).isNotNull();
  }

  @Test
  public void companionClassHierarchy(
      @DynamicClassLiteral("ConstructorNest$NestCC") Class<?> companion) {
    assertThat(companion.getEnclosingClass()).isEqualTo(invoker);
    assertThat(companion.getEnclosingConstructor()).isNull();
    assertThat(companion.getEnclosingMethod()).isNull();
  }

  @Test
  public void companionClassModifiers(
      @DynamicClassLiteral("ConstructorNest$NestCC") Class<?> companion) {
    assertThat(companion.isSynthetic()).isTrue();
    assertThat(companion.isMemberClass()).isTrue();
    assertThat(Modifier.isAbstract(companion.getModifiers())).isTrue();
    assertThat(Modifier.isStatic(companion.getModifiers())).isTrue();
  }

  @Test
  public void constructorBridgeGeneration() {
    assertThat(mate.getDeclaredConstructors()).hasLength(4);
  }

  @Test
  public void zeroArgConstructorBridge(
      @DynamicClassLiteral("ConstructorNest$NestCC") Class<?> companion) throws Exception {
    Constructor<?> constructor = mate.getDeclaredConstructor(companion);
    assertThat(constructor.getModifiers() & 0x7).isEqualTo(0);
  }

  @Test
  public void multiArgConstructorBridge(
      @DynamicClassLiteral("ConstructorNest$NestCC") Class<?> companion) throws Exception {
    Constructor<?> constructor = mate.getDeclaredConstructor(long.class, int.class, companion);

    assertThat(Modifier.isPublic(constructor.getModifiers())).isFalse();
    assertThat(Modifier.isPrivate(constructor.getModifiers())).isFalse();
    assertThat(Modifier.isProtected(constructor.getModifiers())).isFalse();
  }

  @Test
  public void createFromEmptyArgConstructor(
      @RuntimeMethodHandle(
              className = "ConstructorNest",
              memberName = "createFromZeroArgConstructor")
          MethodHandle createFromZeroArgConstructor)
      throws Throwable {
    long result = (long) createFromZeroArgConstructor.invoke();
    assertThat(result).isEqualTo(30L);
  }

  @Test
  public void createFromMultiArgConstructor(
      @RuntimeMethodHandle(
              className = "ConstructorNest",
              memberName = "createFromMultiArgConstructor")
          MethodHandle createFromMultiArgConstructor)
      throws Throwable {
    long result = (long) createFromMultiArgConstructor.invoke((long) 20L, (int) 30);
    assertThat(result).isEqualTo(50L);
  }

  @Test
  public void nestWithDollarSignNamedClasses_nestHostSyntheticConstructor(
      @AsmNode(
              className = "$Dollar$Sign$Named$Nest$",
              memberName = "<init>",
              memberDescriptor =
                  "(JLcom/google/devtools/build/android/desugar/nest/testsrc/simpleunit/constructor/$Dollar$Sign$Named$Nest$$NestCC;)V")
          MethodNode constructor)
      throws Throwable {
    assertThat(constructor).isNotNull();
  }

  @Test
  public void nestWithDollarSignNamedClasses_nestMemberSyntheticConstructor(
      @AsmNode(
              className = "$Dollar$Sign$Named$Nest$$$Dollar$Sign$Named$Member$",
              memberName = "<init>",
              memberDescriptor =
                  "(Lcom/google/devtools/build/android/desugar/nest/testsrc/simpleunit/constructor/$Dollar$Sign$Named$Nest$;Lcom/google/devtools/build/android/desugar/nest/testsrc/simpleunit/constructor/$Dollar$Sign$Named$Nest$$NestCC;)V")
          MethodNode constructor)
      throws Throwable {
    assertThat(constructor).isNotNull();
  }

  @Test
  public void nestWithDollarSignNamedClasses_execute(
      @RuntimeMethodHandle(className = "DollarSignNamedNest", memberName = "execute")
          MethodHandle execute)
      throws Throwable {

    long result = (long) execute.invoke((long) 10L);
    assertThat(result).isEqualTo(14L);
  }
}
