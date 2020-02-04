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
import static com.google.common.truth.Truth8.assertThat;

import com.google.devtools.build.android.desugar.testing.junit.AsmNode;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRunner;
import com.google.devtools.build.android.desugar.testing.junit.DynamicClassLiteral;
import com.google.devtools.build.android.desugar.testing.junit.JdkSuppress;
import com.google.devtools.build.android.desugar.testing.junit.JdkVersion;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import javax.inject.Inject;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.MethodNode;

/** Tests for accessing private fields from another class within a nest. */
@RunWith(DesugarRunner.class)
@JdkSuppress(minJdkVersion = JdkVersion.V11)
public final class NestDesugaringFieldAccessTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addSourceInputsFromJvmFlag("input_srcs")
          .setWorkingJavaPackage(
              "com.google.devtools.build.android.desugar.nest.testsrc.simpleunit.field")
          .addJavacOptions("-source 11", "-target 11")
          .addCommandOptions("desugar_nest_based_private_access", "true")
          .build();

  @Inject
  @DynamicClassLiteral("FieldNest$FieldOwnerMate")
  private Class<?> mate;

  private Object mateInstance;

  @Before
  public void loadClasses() throws Exception {
    mateInstance = mate.getConstructor().newInstance();
  }

  @Test
  public void inputClassFileMajorVersions(
      @AsmNode(className = "FieldNest", round = 0) ClassNode beforeDesugarClassNode,
      @AsmNode(className = "FieldNest", round = 1) ClassNode afterDesugarClassNode) {
    assertThat(beforeDesugarClassNode.version).isEqualTo(JdkVersion.V11);
    assertThat(afterDesugarClassNode.version).isEqualTo(JdkVersion.V1_7);
  }

  @Test
  public void bridgeMethodGeneration() {
    List<String> bridgeMethodNames =
        Arrays.stream(mate.getDeclaredMethods())
            .map(Method::getName)
            .filter(name -> !name.startsWith("$jacoco"))
            .collect(Collectors.toList());
    assertThat(bridgeMethodNames)
        .containsExactly(
            "privateStaticField$bridge_getter",
            "privateStaticField$bridge_setter",
            "privateInstanceField$bridge_getter",
            "privateInstanceField$bridge_setter",
            "privateInstanceWideField$bridge_setter",
            "getPrivateStaticFieldInBoundary",
            "getPrivateInstanceFieldInBoundary",
            "privateStaticFieldReadOnly$bridge_getter",
            "privateInstanceFieldReadOnly$bridge_getter",
            "privateStaticArrayField$bridge_getter",
            "privateInstanceArrayField$bridge_getter");
  }

  @Test
  public void getStaticField(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "getStaticField")
          MethodHandle getStaticField)
      throws Throwable {
    long result = (long) getStaticField.invoke();
    assertThat(result).isEqualTo(10L);
  }

  @Test
  public void getInstanceField(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "getInstanceField")
          MethodHandle getInstanceField)
      throws Throwable {
    long result = (long) getInstanceField.invoke(mateInstance);
    assertThat(result).isEqualTo(20);
  }

  @Test
  public void getPrivateStaticField(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateStaticField")
          MethodHandle getPrivateStaticField)
      throws Throwable {
    long result = (long) getPrivateStaticField.invoke();
    assertThat(result).isEqualTo(30L);
  }

  @Test
  public void setPrivateStaticField(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "setPrivateStaticField")
          MethodHandle setPrivateStaticField)
      throws Throwable {
    long result = (long) setPrivateStaticField.invoke((long) 35L);
    assertThat(result).isEqualTo(35L);

    Field privateStaticField = mate.getDeclaredField("privateStaticField");
    privateStaticField.setAccessible(true);
    assertThat(privateStaticField.get(null)).isEqualTo(35L);
  }

  @Test
  public void getPrivateInstanceField(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateInstanceField")
          MethodHandle getPrivateInstanceField)
      throws Throwable {
    int result = (int) getPrivateInstanceField.invoke(mateInstance);
    assertThat(result).isEqualTo(40);
  }

  @Test
  public void setPrivateInstanceField(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "setPrivateInstanceField")
          MethodHandle setPrivateInstanceField)
      throws Throwable {
    int result = (int) setPrivateInstanceField.invoke(mateInstance, (int) 45);
    assertThat(result).isEqualTo(45);

    Field privateInstanceField = mate.getDeclaredField("privateInstanceField");
    privateInstanceField.setAccessible(true);
    assertThat(privateInstanceField.get(mateInstance)).isEqualTo(45);
  }

  @Test
  public void setPrivateInstanceWideField(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "setPrivateInstanceWideField")
          MethodHandle setPrivateInstanceWideField)
      throws Throwable {
    long result = (long) setPrivateInstanceWideField.invoke(mateInstance, 47L);
    assertThat(result).isEqualTo(47L);

    Field privateInstanceField = mate.getDeclaredField("privateInstanceWideField");
    privateInstanceField.setAccessible(true);
    assertThat(privateInstanceField.get(mateInstance)).isEqualTo(47L);
  }

  @Test
  public void setPrivateInstanceWideField_opcodes(
      @AsmNode(className = "FieldNest", memberName = "setPrivateInstanceWideField")
          MethodNode setPrivateInstanceWideField)
      throws Throwable {
    AbstractInsnNode[] instructions = setPrivateInstanceWideField.instructions.toArray();
    assertThat(Arrays.stream(instructions).map(AbstractInsnNode::getOpcode))
        .containsAtLeast(
            Opcodes.ALOAD,
            Opcodes.LLOAD,
            Opcodes.DUP2_X1,
            Opcodes.INVOKESTATIC,
            Opcodes.POP2,
            Opcodes.LRETURN)
        .inOrder();
  }

  @Test
  public void getPrivateStaticFieldReadOnly(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateStaticFieldReadOnly")
          MethodHandle getPrivateStaticFieldReadOnly)
      throws Throwable {
    long result = (long) getPrivateStaticFieldReadOnly.invoke();
    assertThat(result).isEqualTo(50L);
  }

  @Test
  public void getPrivateInstanceFieldReadOnly(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateInstanceFieldReadOnly")
          MethodHandle getPrivateInstanceFieldReadOnly)
      throws Throwable {
    long result = (long) getPrivateInstanceFieldReadOnly.invoke(mateInstance);
    assertThat(result).isEqualTo(60L);
  }

  @Test
  public void getPrivateInstanceFieldReadOnly_trace(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateInstanceFieldReadOnly")
          MethodHandle getPrivateInstanceFieldReadOnly)
      throws Throwable {
    long result = (long) getPrivateInstanceFieldReadOnly.invoke(mateInstance);
    assertThat(result).isEqualTo(60L);
  }

  @Test
  public void getPrivateStaticFieldInBoundary(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateStaticFieldInBoundary")
          MethodHandle getPrivateStaticFieldInBoundary)
      throws Throwable {
    long result = (long) getPrivateStaticFieldInBoundary.invoke();
    assertThat(result).isEqualTo(70L);
  }

  @Test
  public void getPrivateInstanceFieldInBoundary(
      @RuntimeMethodHandle(
              className = "FieldNest",
              memberName = "getPrivateInstanceFieldInBoundary")
          MethodHandle getPrivateInstanceFieldInBoundary)
      throws Throwable {
    int result = (int) getPrivateInstanceFieldInBoundary.invoke(mateInstance);
    assertThat(result).isEqualTo(80);
  }

  @Test
  public void getPrivateStaticArrayFieldElement(
      @RuntimeMethodHandle(
              className = "FieldNest",
              memberName = "getPrivateStaticArrayFieldElement")
          MethodHandle getPrivateStaticArrayFieldElement)
      throws Throwable {
    long[] arrayFieldInitialValue = {100L, 200L, 300L};

    Field field = mate.getDeclaredField("privateStaticArrayField");
    field.setAccessible(true);
    field.set(null, arrayFieldInitialValue);

    long result = (long) getPrivateStaticArrayFieldElement.invoke(1); // 1 is for index.
    assertThat(result).isEqualTo(200L);
  }

  @Test
  public void setPrivateStaticArrayFieldElement(
      @RuntimeMethodHandle(
              className = "FieldNest",
              memberName = "setPrivateStaticArrayFieldElement")
          MethodHandle setPrivateStaticArrayFieldElement)
      throws Throwable {
    long[] arrayFieldInitialValue = {200L, 300L, 400L};
    long overriddenValue = 3000L;

    Field field = mate.getDeclaredField("privateStaticArrayField");
    field.setAccessible(true);
    field.set(null, arrayFieldInitialValue);

    long result = (long) setPrivateStaticArrayFieldElement.invoke(1, overriddenValue);
    assertThat(result).isEqualTo(overriddenValue);

    long[] actual = (long[]) field.get(null);
    assertThat(actual).asList().containsExactly(200L, 3000L, 400L).inOrder();
  }

  @Test
  public void getPrivateInstanceArrayFieldElement(
      @RuntimeMethodHandle(
              className = "FieldNest",
              memberName = "getPrivateInstanceArrayFieldElement")
          MethodHandle getPrivateInstanceArrayFieldElement)
      throws Throwable {
    int[] arrayFieldInitialValue = {300, 400, 500};

    Field field = mate.getDeclaredField("privateInstanceArrayField");
    field.setAccessible(true);
    field.set(mateInstance, arrayFieldInitialValue);

    int result = (int) getPrivateInstanceArrayFieldElement.invoke(mateInstance, 1);
    assertThat(result).isEqualTo(400);
  }

  @Test
  public void setPrivateInstanceArrayFieldElement(
      @RuntimeMethodHandle(
              className = "FieldNest",
              memberName = "setPrivateInstanceArrayFieldElement")
          MethodHandle setPrivateInstanceArrayFieldElement)
      throws Throwable {
    int[] arrayFieldInitialValue = {400, 500, 600};
    int overriddenValue = 5000;

    Field field = mate.getDeclaredField("privateInstanceArrayField");
    field.setAccessible(true);
    field.set(mateInstance, arrayFieldInitialValue);

    long result =
        (long) setPrivateInstanceArrayFieldElement.invoke(mateInstance, 1, overriddenValue);

    assertThat(result).isEqualTo(overriddenValue);

    int[] actual = (int[]) field.get(mateInstance);
    assertThat(actual).asList().containsExactly(400, 5000, 600).inOrder();
  }

  @Test
  public void compoundSetPrivateStaticField(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "compoundSetPrivateStaticField")
          MethodHandle compoundSetPrivateStaticField)
      throws Throwable {
    long fieldInitialValue = 100;
    long fieldValueDelta = 20;

    Field field = mate.getDeclaredField("privateStaticField");
    field.setAccessible(true);
    field.set(null, fieldInitialValue);

    long result = (long) compoundSetPrivateStaticField.invoke(fieldValueDelta);
    assertThat(result).isEqualTo(fieldInitialValue + fieldValueDelta);

    assertThat(field.get(null)).isEqualTo(fieldInitialValue + fieldValueDelta);
  }

  @Test
  public void preIncrementPrivateStaticField(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "preIncrementPrivateStaticField")
          MethodHandle preIncrementPrivateStaticField)
      throws Throwable {
    long fieldInitialValue = 200;

    Field field = mate.getDeclaredField("privateStaticField");
    field.setAccessible(true);
    field.set(null, fieldInitialValue);

    long result = (long) preIncrementPrivateStaticField.invoke();
    assertThat(result).isEqualTo(fieldInitialValue + 1);

    assertThat(field.get(null)).isEqualTo(fieldInitialValue + 1);
  }

  @Test
  public void postIncrementPrivateStaticField(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "postIncrementPrivateStaticField")
          MethodHandle postIncrementPrivateStaticField)
      throws Throwable {
    long fieldInitialValue = 300;

    Field field = mate.getDeclaredField("privateStaticField");
    field.setAccessible(true);
    field.set(null, fieldInitialValue);

    long result = (long) postIncrementPrivateStaticField.invoke();
    assertThat(result).isEqualTo(fieldInitialValue);

    assertThat(field.get(null)).isEqualTo(fieldInitialValue + 1);
  }

  @Test
  public void compoundSetPrivateInstanceField(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "compoundSetPrivateInstanceField")
          MethodHandle compoundSetPrivateInstanceField)
      throws Throwable {
    int fieldInitialValue = 400;
    int fieldDelta = 10;

    Field field = mate.getDeclaredField("privateInstanceField");
    field.setAccessible(true);
    field.set(mateInstance, fieldInitialValue);

    int result = (int) compoundSetPrivateInstanceField.invoke(mateInstance, fieldDelta);
    assertThat(result).isEqualTo(fieldInitialValue + fieldDelta);

    assertThat(field.get(mateInstance)).isEqualTo(fieldInitialValue + fieldDelta);
  }

  @Test
  public void preIncrementPrivateInstanceField(
      @RuntimeMethodHandle(className = "FieldNest", memberName = "preIncrementPrivateInstanceField")
          MethodHandle preIncrementPrivateInstanceField)
      throws Throwable {
    int fieldInitialValue = 500;

    Field field = mate.getDeclaredField("privateInstanceField");
    field.setAccessible(true);
    field.set(mateInstance, fieldInitialValue);

    int result = (int) preIncrementPrivateInstanceField.invoke(mateInstance);
    assertThat(result).isEqualTo(fieldInitialValue + 1);

    assertThat(field.get(mateInstance)).isEqualTo(fieldInitialValue + 1);
  }

  @Test
  public void postIncrementPrivateInstanceField(
      @RuntimeMethodHandle(
              className = "FieldNest",
              memberName = "postIncrementPrivateInstanceField")
          MethodHandle postIncrementPrivateInstanceField)
      throws Throwable {
    int fieldInitialValue = 600;

    Field field = mate.getDeclaredField("privateInstanceField");
    field.setAccessible(true);
    field.set(mateInstance, fieldInitialValue);

    int result = (int) postIncrementPrivateInstanceField.invoke(mateInstance);
    assertThat(result).isEqualTo(fieldInitialValue);

    assertThat(field.get(mateInstance)).isEqualTo(fieldInitialValue + 1);
  }
}
