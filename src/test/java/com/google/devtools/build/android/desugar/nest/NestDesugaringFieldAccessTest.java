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
import com.google.testing.testsize.MediumTest;
import com.google.testing.testsize.MediumTestAttribute;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
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

/** Tests for accessing private fields from another class within a nest. */
@RunWith(JUnit4.class)
@MediumTest(MediumTestAttribute.FILE)
public final class NestDesugaringFieldAccessTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addInputs(Paths.get(System.getProperty("input_jar")))
          .setWorkingJavaPackage(
              "com.google.devtools.build.android.desugar.nest.testsrc.simpleunit.field")
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
  public void bridgeMethodGeneration() throws Exception {
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
            "getPrivateStaticFieldInBoundary",
            "getPrivateInstanceFieldInBoundary",
            "privateStaticFieldReadOnly$bridge_getter",
            "privateInstanceFieldReadOnly$bridge_getter",
            "privateStaticArrayField$bridge_getter",
            "privateInstanceArrayField$bridge_getter");
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "getStaticField")
  private MethodHandle getStaticField;

  @Test
  public void getStaticField() throws Throwable {
    long result = (long) getStaticField.invoke();
    assertThat(result).isEqualTo(10L);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "getInstanceField")
  private MethodHandle getInstanceField;

  @Test
  public void getInstanceField() throws Throwable {
    long result = (long) getInstanceField.invoke(mateInstance);
    assertThat(result).isEqualTo(20);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateStaticField")
  private MethodHandle getPrivateStaticField;

  @Test
  public void getPrivateStaticField() throws Throwable {
    long result = (long) getPrivateStaticField.invoke();
    assertThat(result).isEqualTo(30L);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "setPrivateStaticField")
  private MethodHandle setPrivateStaticField;

  @Test
  public void setPrivateStaticField() throws Throwable {
    long result = (long) setPrivateStaticField.invoke((long) 35L);
    assertThat(result).isEqualTo(35L);

    Field privateStaticField = mate.getDeclaredField("privateStaticField");
    privateStaticField.setAccessible(true);
    assertThat(privateStaticField.get(null)).isEqualTo(35L);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateInstanceField")
  private MethodHandle getPrivateInstanceField;

  @Test
  public void getPrivateInstanceField() throws Throwable {
    int result = (int) getPrivateInstanceField.invoke(mateInstance);
    assertThat(result).isEqualTo(40);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "setPrivateInstanceField")
  private MethodHandle setPrivateInstanceField;

  @Test
  public void setPrivateInstanceField() throws Throwable {
    int result = (int) setPrivateInstanceField.invoke(mateInstance, (int) 45);
    assertThat(result).isEqualTo(45);

    Field privateInstanceField = mate.getDeclaredField("privateInstanceField");
    privateInstanceField.setAccessible(true);
    assertThat(privateInstanceField.get(mateInstance)).isEqualTo(45);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateStaticFieldReadOnly")
  private MethodHandle getPrivateStaticFieldReadOnly;

  @Test
  public void getPrivateStaticFieldReadOnly() throws Throwable {
    long result = (long) getPrivateStaticFieldReadOnly.invoke();
    assertThat(result).isEqualTo(50L);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateInstanceFieldReadOnly")
  private MethodHandle getPrivateInstanceFieldReadOnly;

  @Test
  public void getPrivateInstanceFieldReadOnly() throws Throwable {
    long result = (long) getPrivateInstanceFieldReadOnly.invoke(mateInstance);
    assertThat(result).isEqualTo(60L);
  }

  @Test
  public void getPrivateInstanceFieldReadOnly_trace() throws Throwable {
    long result = (long) getPrivateInstanceFieldReadOnly.invoke(mateInstance);
    assertThat(result).isEqualTo(60L);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateStaticFieldInBoundary")
  private MethodHandle getPrivateStaticFieldInBoundary;

  @Test
  public void getPrivateStaticFieldInBoundary() throws Throwable {
    long result = (long) getPrivateStaticFieldInBoundary.invoke();
    assertThat(result).isEqualTo(70L);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateInstanceFieldInBoundary")
  private MethodHandle getPrivateInstanceFieldInBoundary;

  @Test
  public void getPrivateInstanceFieldInBoundary() throws Throwable {
    int result = (int) getPrivateInstanceFieldInBoundary.invoke(mateInstance);
    assertThat(result).isEqualTo(80);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateStaticArrayFieldElement")
  private MethodHandle getPrivateStaticArrayFieldElement;

  @Test
  public void getPrivateStaticArrayFieldElement() throws Throwable {
    long[] arrayFieldInitialValue = {100L, 200L, 300L};

    Field field = mate.getDeclaredField("privateStaticArrayField");
    field.setAccessible(true);
    field.set(null, arrayFieldInitialValue);

    long result = (long) getPrivateStaticArrayFieldElement.invoke(1); // 1 is for index.
    assertThat(result).isEqualTo(200L);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "setPrivateStaticArrayFieldElement")
  private MethodHandle setPrivateStaticArrayFieldElement;

  @Test
  public void setPrivateStaticArrayFieldElement() throws Throwable {
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

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "getPrivateInstanceArrayFieldElement")
  private MethodHandle getPrivateInstanceArrayFieldElement;

  @Test
  public void getPrivateInstanceArrayFieldElement() throws Throwable {
    int[] arrayFieldInitialValue = {300, 400, 500};

    Field field = mate.getDeclaredField("privateInstanceArrayField");
    field.setAccessible(true);
    field.set(mateInstance, arrayFieldInitialValue);

    int result = (int) getPrivateInstanceArrayFieldElement.invoke(mateInstance, 1);
    assertThat(result).isEqualTo(400);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "setPrivateInstanceArrayFieldElement")
  private MethodHandle setPrivateInstanceArrayFieldElement;

  @Test
  public void setPrivateInstanceArrayFieldElement() throws Throwable {
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

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "compoundSetPrivateStaticField")
  private MethodHandle compoundSetPrivateStaticField;

  @Test
  public void compoundSetPrivateStaticField() throws Throwable {
    long fieldInitialValue = 100;
    long fieldValueDelta = 20;

    Field field = mate.getDeclaredField("privateStaticField");
    field.setAccessible(true);
    field.set(null, fieldInitialValue);

    long result = (long) compoundSetPrivateStaticField.invoke(fieldValueDelta);
    assertThat(result).isEqualTo(fieldInitialValue + fieldValueDelta);

    assertThat(field.get(null)).isEqualTo(fieldInitialValue + fieldValueDelta);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "preIncrementPrivateStaticField")
  private MethodHandle preIncrementPrivateStaticField;

  @Test
  public void preIncrementPrivateStaticField() throws Throwable {
    long fieldInitialValue = 200;

    Field field = mate.getDeclaredField("privateStaticField");
    field.setAccessible(true);
    field.set(null, fieldInitialValue);

    long result = (long) preIncrementPrivateStaticField.invoke();
    assertThat(result).isEqualTo(fieldInitialValue + 1);

    assertThat(field.get(null)).isEqualTo(fieldInitialValue + 1);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "postIncrementPrivateStaticField")
  private MethodHandle postIncrementPrivateStaticField;

  @Test
  public void postIncrementPrivateStaticField() throws Throwable {
    long fieldInitialValue = 300;

    Field field = mate.getDeclaredField("privateStaticField");
    field.setAccessible(true);
    field.set(null, fieldInitialValue);

    long result = (long) postIncrementPrivateStaticField.invoke();
    assertThat(result).isEqualTo(fieldInitialValue);

    assertThat(field.get(null)).isEqualTo(fieldInitialValue + 1);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "compoundSetPrivateInstanceField")
  private MethodHandle compoundSetPrivateInstanceField;

  @Test
  public void compoundSetPrivateInstanceField() throws Throwable {
    int fieldInitialValue = 400;
    int fieldDelta = 10;

    Field field = mate.getDeclaredField("privateInstanceField");
    field.setAccessible(true);
    field.set(mateInstance, fieldInitialValue);

    int result = (int) compoundSetPrivateInstanceField.invoke(mateInstance, fieldDelta);
    assertThat(result).isEqualTo(fieldInitialValue + fieldDelta);

    assertThat(field.get(mateInstance)).isEqualTo(fieldInitialValue + fieldDelta);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "preIncrementPrivateInstanceField")
  private MethodHandle preIncrementPrivateInstanceField;

  @Test
  public void preIncrementPrivateInstanceField() throws Throwable {
    int fieldInitialValue = 500;

    Field field = mate.getDeclaredField("privateInstanceField");
    field.setAccessible(true);
    field.set(mateInstance, fieldInitialValue);

    int result = (int) preIncrementPrivateInstanceField.invoke(mateInstance);
    assertThat(result).isEqualTo(fieldInitialValue + 1);

    assertThat(field.get(mateInstance)).isEqualTo(fieldInitialValue + 1);
  }

  @Inject
  @RuntimeMethodHandle(className = "FieldNest", memberName = "postIncrementPrivateInstanceField")
  private MethodHandle postIncrementPrivateInstanceField;

  @Test
  public void postIncrementPrivateInstanceField() throws Throwable {
    int fieldInitialValue = 600;

    Field field = mate.getDeclaredField("privateInstanceField");
    field.setAccessible(true);
    field.set(mateInstance, fieldInitialValue);

    int result = (int) postIncrementPrivateInstanceField.invoke(mateInstance);
    assertThat(result).isEqualTo(fieldInitialValue);

    assertThat(field.get(mateInstance)).isEqualTo(fieldInitialValue + 1);
  }
}
