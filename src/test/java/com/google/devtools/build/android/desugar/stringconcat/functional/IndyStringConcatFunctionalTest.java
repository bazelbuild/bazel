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

package com.google.devtools.build.android.desugar.stringconcat.functional;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;

import com.google.common.collect.Streams;
import com.google.devtools.build.android.desugar.testing.junit.AsmNode;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DynamicClassLiteral;
import com.google.testing.testsize.MediumTest;
import com.google.testing.testsize.MediumTestAttribute;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Paths;
import javax.inject.Inject;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.MethodNode;

/**
 * Tests for accessing a series of private fields, constructors and methods from another class
 * within a nest.
 */
@RunWith(JUnit4.class)
@MediumTest(MediumTestAttribute.FILE)
public final class IndyStringConcatFunctionalTest {

  private static final MethodHandles.Lookup lookup = MethodHandles.lookup();

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, lookup)
          .addInputs(Paths.get(System.getProperty("input_jar")))
          .setWorkingJavaPackage(
              "com.google.devtools.build.android.desugar.stringconcat.functional")
          .addCommandOptions("desugar_indy_string_concat", "true")
          .build();

  @Inject
  @DynamicClassLiteral("StringConcatCases")
  private Class<?> stringConcatCases;

  @Inject
  @AsmNode(className = "StringConcatCases", memberName = "simplePrefix", round = 0)
  private MethodNode simpleStrContatBeforeDesugar;

  @Inject
  @AsmNode(className = "StringConcatCases", memberName = "simplePrefix", round = 1)
  private MethodNode simpleStrConcatAfterDesugar;

  @Test
  public void invokeDynamicInstr_presentBeforeDesugaring() {
    assertThat(
            Streams.stream(simpleStrContatBeforeDesugar.instructions)
                .map(AbstractInsnNode::getOpcode))
        .contains(Opcodes.INVOKEDYNAMIC);
  }

  @Test
  public void invokeDynamicInstr_absentAfterDesugaring() {
    assertThat(
            Streams.stream(simpleStrConcatAfterDesugar.instructions)
                .map(AbstractInsnNode::getOpcode))
        .doesNotContain(Opcodes.INVOKEDYNAMIC);
  }

  @Test
  public void twoConcat() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "twoConcat",
            MethodType.methodType(String.class, String.class, String.class));
    String result = (String) testHandle.invoke("ab", "cd");

    assertThat(result).isEqualTo("abcd");
  }

  @Test
  public void twoConcat_StringAndObject() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "twoConcat",
            MethodType.methodType(String.class, String.class, Object.class));
    String result = (String) testHandle.invoke("ab", (Object) "cd");
    assertThat(result).isEqualTo("T:abcd");
  }

  @Test
  public void twoConcatWithConstants() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "twoConcatWithConstants",
            MethodType.methodType(String.class, String.class, String.class));
    String result = (String) testHandle.invoke("ab", "cd");

    assertThat(result).isEqualTo("ab<constant>cd");
  }

  @Test
  public void threeConcat() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "threeConcat",
            MethodType.methodType(String.class, String.class, String.class, String.class));
    String result = (String) testHandle.invoke("ab", "cd", "ef");

    assertThat(result).isEqualTo("abcdef");
  }

  @Test
  public void twoConcatWithRecipe() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "twoConcatWithRecipe",
            MethodType.methodType(String.class, String.class, String.class));
    String result = (String) testHandle.invoke("ab", "cd");

    assertThat(result).isEqualTo("<p>ab<br>cd</p>");
  }

  @Test
  public void threeConcatWithRecipe() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "threeConcatWithRecipe",
            MethodType.methodType(String.class, String.class, String.class, String.class));
    String result = (String) testHandle.invoke("ab", "cd", "ef");
    assertThat(result).isEqualTo("<p>ab<br>cd<br>ef</p>");
  }

  @Test
  public void twoConcatWithPrimitives_StringAndInt() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "twoConcatWithPrimitives",
            MethodType.methodType(String.class, String.class, int.class));
    String result = (String) testHandle.invoke("ab", 123);
    assertThat(result).isEqualTo("ab123");
  }

  @Test
  public void twoConcatWithPrimitives_StringAndLong() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "twoConcatWithPrimitives",
            MethodType.methodType(String.class, String.class, long.class));
    String result = (String) testHandle.invoke("ab", 123L);
    assertThat(result).isEqualTo("ab123");
  }

  @Test
  public void twoConcatWithPrimitives_StringAndDouble() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "twoConcatWithPrimitives",
            MethodType.methodType(String.class, String.class, double.class));
    String result = (String) testHandle.invoke("ab", 123.125);
    assertThat(result).isEqualTo("ab123.125");
  }

  @Test
  public void twoConcatWithPrimitives_intAndString() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "twoConcatWithPrimitives",
            MethodType.methodType(String.class, int.class, String.class));
    String result = (String) testHandle.invoke(123, "ABC");
    assertThat(result).isEqualTo("123ABC");
  }

  @Test
  public void twoConcatWithPrimitives_longAndString() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "twoConcatWithPrimitives",
            MethodType.methodType(String.class, long.class, String.class));
    String result = (String) testHandle.invoke(123L, "ABC");
    assertThat(result).isEqualTo("123ABC");
  }

  @Test
  public void twoConcatWithPrimitives_doubleAndString() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "twoConcatWithPrimitives",
            MethodType.methodType(String.class, double.class, String.class));
    String result = (String) testHandle.invoke(123.125, "ABC");
    assertThat(result).isEqualTo("123.125ABC");
  }

  @Test
  public void concatWithAllPrimitiveTypes() throws Throwable {
    MethodHandle testHandle =
        lookup.findStatic(
            stringConcatCases,
            "concatWithAllPrimitiveTypes",
            MethodType.methodType(
                String.class,
                String.class,
                int.class,
                boolean.class,
                byte.class,
                char.class,
                short.class,
                double.class,
                float.class,
                long.class));
    String result =
        (String)
            testHandle.invoke(
                "head", // string value
                1, // int value
                true, // boolean value
                (byte) 2, // byte value
                'c', // char value
                (short) 4, // short value
                5.25D, // double value
                6.5F, // float value
                7L); // long value
    assertThat(result).isEqualTo("head-1-true-2-c-4-5.25-6.5-7");
  }
}
