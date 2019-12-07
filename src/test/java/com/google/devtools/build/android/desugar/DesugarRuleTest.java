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

package com.google.devtools.build.android.desugar;

import static com.google.common.truth.Truth8.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.android.desugar.DesugarRule.LoadClass;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Arrays;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** The test for {@link DesugarRule}. */
@RunWith(JUnit4.class)
public class DesugarRuleTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .enableIterativeTransformation(3)
          .addRuntimeInputs(
              "third_party/bazel/src/test/java/com/google/devtools/build/android/desugar/libdesugar_rule_test_target.jar")
          .build();

  @LoadClass(
      value =
          "com.google.devtools.build.android.desugar.DesugarRuleTestTarget$InterfaceSubjectToDesugar",
      round = 1)
  private Class<?> interfaceSubjectToDesugarClass;

  @LoadClass(
      value =
          "com.google.devtools.build.android.desugar.DesugarRuleTestTarget$InterfaceSubjectToDesugar",
      round = 2)
  private Class<?> interfaceSubjectToDesugarClassTwice;

  @LoadClass(
      "com.google.devtools.build.android.desugar.DesugarRuleTestTarget$InterfaceSubjectToDesugar$$CC")
  private Class<?> interfaceSubjectToDesugarCompanionClass;

  @Test
  public void staticMethodsAreMovedFromOriginatingClass() {
    assertThrows(
        NoSuchMethodException.class,
        () -> interfaceSubjectToDesugarClass.getDeclaredMethod("staticMethod"));
  }

  @Test
  public void staticMethodsAreMovedFromOriginatingClass_desugarTwice() {
    assertThrows(
        NoSuchMethodException.class,
        () -> interfaceSubjectToDesugarClassTwice.getDeclaredMethod("staticMethod"));
  }

  @Test
  public void staticMethodsAreMovedToCompanionClass() {
    assertThat(
            Arrays.stream(interfaceSubjectToDesugarCompanionClass.getDeclaredMethods())
                .map(Method::getName))
        .contains("staticMethod$$STATIC$$");
  }
}
