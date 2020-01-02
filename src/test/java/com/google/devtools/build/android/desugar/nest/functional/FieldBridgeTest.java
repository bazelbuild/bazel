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

import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DynamicClassLiteral;
import com.google.testing.testsize.MediumTest;
import com.google.testing.testsize.MediumTestAttribute;
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
public final class FieldBridgeTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addInputs(Paths.get(System.getProperty("input_jar")))
          .setWorkingJavaPackage(
              "com.google.devtools.build.android.desugar.nest.functional.testsrc.simpleunit.field")
          .addCommandOptions("desugar_nest_based_private_access", "true")
          .build();

  @Inject
  @DynamicClassLiteral("FieldNest$FieldOwnerMate")
  private Class<?> mate;

  @Inject
  @DynamicClassLiteral("FieldNest")
  private Class<?> invoker;

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

  @Test
  public void getStaticField() throws Exception {
    assertThat(invoker.getDeclaredMethod("getStaticField").invoke(null)).isEqualTo(10L);
  }

  @Test
  public void getInstanceField() throws Exception {
    assertThat(invoker.getDeclaredMethod("getInstanceField", mate).invoke(null, mateInstance))
        .isEqualTo(20);
  }

  @Test
  public void getPrivateStaticField() throws Exception {
    assertThat(invoker.getDeclaredMethod("getPrivateStaticField").invoke(null)).isEqualTo(30L);
  }

  @Test
  public void setPrivateStaticField() throws Exception {
    assertThat(invoker.getDeclaredMethod("setPrivateStaticField", long.class).invoke(null, 35L))
        .isEqualTo(35L);

    Field privateStaticField = mate.getDeclaredField("privateStaticField");
    privateStaticField.setAccessible(true);
    assertThat(privateStaticField.get(null)).isEqualTo(35L);
  }

  @Test
  public void getPrivateInstanceField() throws Exception {
    invoker.getDeclaredMethod("getPrivateInstanceField", mate).invoke(null, mateInstance);
  }

  @Test
  public void setPrivateInstanceField() throws Exception {
    assertThat(
            invoker
                .getDeclaredMethod("setPrivateInstanceField", mate, int.class)
                .invoke(null, mateInstance, 45))
        .isEqualTo(45);

    Field privateInstanceField = mate.getDeclaredField("privateInstanceField");
    privateInstanceField.setAccessible(true);
    assertThat(privateInstanceField.get(mateInstance)).isEqualTo(45);
  }

  @Test
  public void getPrivateStaticFieldReadOnly() throws Exception {
    assertThat(invoker.getDeclaredMethod("getPrivateStaticFieldReadOnly").invoke(null))
        .isEqualTo(50L);
  }

  @Test
  public void getPrivateInstanceFieldReadOnly() throws Exception {
    assertThat(
            invoker
                .getDeclaredMethod("getPrivateInstanceFieldReadOnly", mate)
                .invoke(null, mateInstance))
        .isEqualTo(60L);
  }

  @Test
  public void getPrivateInstanceFieldReadOnly_trace() throws Exception {
    assertThat(
            invoker
                .getDeclaredMethod("getPrivateInstanceFieldReadOnly", mate)
                .invoke(null, mateInstance))
        .isEqualTo(60L);
  }

  @Test
  public void getPrivateStaticFieldInBoundary() throws Exception {
    assertThat(invoker.getDeclaredMethod("getPrivateStaticFieldInBoundary").invoke(null))
        .isEqualTo(70L);
  }

  @Test
  public void getPrivateInstanceFieldInBoundary() throws Exception {
    assertThat(
            invoker
                .getDeclaredMethod("getPrivateInstanceFieldInBoundary", mate)
                .invoke(null, mateInstance))
        .isEqualTo(80);
  }

  @Test
  public void getPrivateStaticArrayFieldElement() throws Exception {
    long[] arrayFieldInitialValue = {100L, 200L, 300L};

    Field field = mate.getDeclaredField("privateStaticArrayField");
    field.setAccessible(true);
    field.set(null, arrayFieldInitialValue);

    assertThat(
            invoker
                .getDeclaredMethod("getPrivateStaticArrayFieldElement", int.class)
                .invoke(null, 1)) // index
        .isEqualTo(200);
  }

  @Test
  public void setPrivateStaticArrayFieldElement() throws Exception {
    long[] arrayFieldInitialValue = {200L, 300L, 400L};
    long overriddenValue = 3000L;

    Field field = mate.getDeclaredField("privateStaticArrayField");
    field.setAccessible(true);
    field.set(null, arrayFieldInitialValue);

    assertThat(
            invoker
                .getDeclaredMethod("setPrivateStaticArrayFieldElement", int.class, long.class)
                .invoke(null, 1, overriddenValue)) // index
        .isEqualTo(overriddenValue);

    long[] actual = (long[]) field.get(null);
    assertThat(actual).asList().containsExactly(200L, 3000L, 400L).inOrder();
  }

  @Test
  public void getPrivateInstanceArrayFieldElement() throws Exception {
    int[] arrayFieldInitialValue = {300, 400, 500};

    Field field = mate.getDeclaredField("privateInstanceArrayField");
    field.setAccessible(true);
    field.set(mateInstance, arrayFieldInitialValue);

    assertThat(
            invoker
                .getDeclaredMethod("getPrivateInstanceArrayFieldElement", mate, int.class)
                .invoke(null, mateInstance, 1)) // index
        .isEqualTo(400);
  }

  @Test
  public void setPrivateInstanceArrayFieldElement() throws Exception {
    int[] arrayFieldInitialValue = {400, 500, 600};
    int overriddenValue = 5000;

    Field field = mate.getDeclaredField("privateInstanceArrayField");
    field.setAccessible(true);
    field.set(mateInstance, arrayFieldInitialValue);

    assertThat(
            invoker
                .getDeclaredMethod(
                    "setPrivateInstanceArrayFieldElement", mate, int.class, int.class)
                .invoke(null, mateInstance, 1, overriddenValue)) // index
        .isEqualTo(overriddenValue);

    int[] actual = (int[]) field.get(mateInstance);
    assertThat(actual).asList().containsExactly(400, 5000, 600).inOrder();
  }

  @Test
  public void compoundSetPrivateStaticField() throws Exception {
    long fieldInitialValue = 100;
    long fieldValueDelta = 20;

    Field field = mate.getDeclaredField("privateStaticField");
    field.setAccessible(true);
    field.set(null, fieldInitialValue);

    assertThat(
            invoker
                .getDeclaredMethod("compoundSetPrivateStaticField", long.class)
                .invoke(null, fieldValueDelta))
        .isEqualTo(fieldInitialValue + fieldValueDelta);

    assertThat(field.get(null)).isEqualTo(fieldInitialValue + fieldValueDelta);
  }

  @Test
  public void preIncrementPrivateStaticField() throws Exception {
    long fieldInitialValue = 200;

    Field field = mate.getDeclaredField("privateStaticField");
    field.setAccessible(true);
    field.set(null, fieldInitialValue);

    assertThat(invoker.getDeclaredMethod("preIncrementPrivateStaticField").invoke(null))
        .isEqualTo(fieldInitialValue + 1);

    assertThat(field.get(null)).isEqualTo(fieldInitialValue + 1);
  }

  @Test
  public void postIncrementPrivateStaticField() throws Exception {
    long fieldInitialValue = 300;

    Field field = mate.getDeclaredField("privateStaticField");
    field.setAccessible(true);
    field.set(null, fieldInitialValue);

    assertThat(invoker.getDeclaredMethod("postIncrementPrivateStaticField").invoke(null))
        .isEqualTo(fieldInitialValue);

    assertThat(field.get(null)).isEqualTo(fieldInitialValue + 1);
  }

  @Test
  public void compoundSetPrivateInstanceField() throws Exception {
    int fieldInitialValue = 400;
    int fieldDelta = 10;

    Field field = mate.getDeclaredField("privateInstanceField");
    field.setAccessible(true);
    field.set(mateInstance, fieldInitialValue);

    assertThat(
            invoker
                .getDeclaredMethod("compoundSetPrivateInstanceField", mate, int.class)
                .invoke(null, mateInstance, fieldDelta))
        .isEqualTo(fieldInitialValue + fieldDelta);

    assertThat(field.get(mateInstance)).isEqualTo(fieldInitialValue + fieldDelta);
  }

  @Test
  public void preIncrementPrivateInstanceField() throws Exception {
    int fieldInitialValue = 500;

    Field field = mate.getDeclaredField("privateInstanceField");
    field.setAccessible(true);
    field.set(mateInstance, fieldInitialValue);
    assertThat(
            invoker
                .getDeclaredMethod("preIncrementPrivateInstanceField", mate)
                .invoke(null, mateInstance))
        .isEqualTo(fieldInitialValue + 1);

    assertThat(field.get(mateInstance)).isEqualTo(fieldInitialValue + 1);
  }

  @Test
  public void postIncrementPrivateInstanceField() throws Exception {
    int fieldInitialValue = 600;

    Field field = mate.getDeclaredField("privateInstanceField");
    field.setAccessible(true);
    field.set(mateInstance, fieldInitialValue);
    assertThat(
            invoker
                .getDeclaredMethod("postIncrementPrivateInstanceField", mate)
                .invoke(null, mateInstance))
        .isEqualTo(fieldInitialValue);

    assertThat(field.get(mateInstance)).isEqualTo(fieldInitialValue + 1);
  }
}
