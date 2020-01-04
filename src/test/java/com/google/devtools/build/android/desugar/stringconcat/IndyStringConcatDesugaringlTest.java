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

package com.google.devtools.build.android.desugar.stringconcat;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;

import com.google.common.collect.Streams;
import com.google.devtools.build.android.desugar.testing.junit.AsmNode;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle;
import com.google.testing.testsize.MediumTest;
import com.google.testing.testsize.MediumTestAttribute;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
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
public final class IndyStringConcatDesugaringlTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addInputs(Paths.get(System.getProperty("input_jar")))
          .setWorkingJavaPackage("com.google.devtools.build.android.desugar.stringconcat")
          .addCommandOptions("desugar_indy_string_concat", "true")
          .build();

  @Inject
  @AsmNode(className = "StringConcatTestCases", memberName = "simplePrefix", round = 0)
  private MethodNode simpleStrContatBeforeDesugar;

  @Inject
  @AsmNode(className = "StringConcatTestCases", memberName = "simplePrefix", round = 1)
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

  @Inject
  @RuntimeMethodHandle(
      className = "StringConcatTestCases",
      memberName = "twoConcat",
      memberDescriptor = "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;")
  private MethodHandle twoConcat;

  @Test
  public void twoConcat() throws Throwable {
    String result = (String) twoConcat.invoke("ab", "cd");
    assertThat(result).isEqualTo("abcd");
  }

  @Inject
  @RuntimeMethodHandle(
      className = "StringConcatTestCases",
      memberName = "twoConcat",
      memberDescriptor = "(Ljava/lang/String;Ljava/lang/Object;)Ljava/lang/String;")
  private MethodHandle twoConcatStringAndObject;

  @Test
  public void twoConcat_StringAndObject() throws Throwable {
    String result = (String) twoConcatStringAndObject.invoke("ab", (Object) "cd");
    assertThat(result).isEqualTo("T:abcd");
  }

  @Inject
  @RuntimeMethodHandle(
      className = "StringConcatTestCases",
      memberName = "twoConcatWithConstants",
      memberDescriptor = "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;")
  private MethodHandle twoConcatWithConstants;

  @Test
  public void twoConcatWithConstants() throws Throwable {
    String result = (String) twoConcatWithConstants.invoke("ab", "cd");
    assertThat(result).isEqualTo("ab<constant>cd");
  }

  @Inject
  @RuntimeMethodHandle(className = "StringConcatTestCases", memberName = "threeConcat")
  private MethodHandle threeConcat;

  @Test
  public void threeConcat() throws Throwable {
    String result = (String) threeConcat.invoke("ab", "cd", "ef");
    assertThat(result).isEqualTo("abcdef");
  }

  @Inject
  @RuntimeMethodHandle(className = "StringConcatTestCases", memberName = "twoConcatWithRecipe")
  private MethodHandle twoConcatWithRecipe;

  @Test
  public void twoConcatWithRecipe() throws Throwable {
    String result = (String) twoConcatWithRecipe.invoke("ab", "cd");
    assertThat(result).isEqualTo("<p>ab<br>cd</p>");
  }

  @Inject
  @RuntimeMethodHandle(className = "StringConcatTestCases", memberName = "threeConcatWithRecipe")
  private MethodHandle threeConcatWithRecipe;

  @Test
  public void threeConcatWithRecipe() throws Throwable {
    String result = (String) threeConcatWithRecipe.invoke("ab", "cd", "ef");
    assertThat(result).isEqualTo("<p>ab<br>cd<br>ef</p>");
  }

  @Inject
  @RuntimeMethodHandle(
      className = "StringConcatTestCases",
      memberName = "twoConcatWithPrimitives",
      memberDescriptor = "(Ljava/lang/String;I)Ljava/lang/String;")
  private MethodHandle twoConcatWithPrimitives;

  @Test
  public void twoConcatWithPrimitives_StringAndInt() throws Throwable {
    String result = (String) twoConcatWithPrimitives.invoke("ab", 123);
    assertThat(result).isEqualTo("ab123");
  }

  @Inject
  @RuntimeMethodHandle(
      className = "StringConcatTestCases",
      memberName = "twoConcatWithPrimitives",
      memberDescriptor = "(Ljava/lang/String;J)Ljava/lang/String;")
  private MethodHandle twoConcatOfStringAndPrimitiveLong;

  @Test
  public void twoConcatWithPrimitives_StringAndLong() throws Throwable {
    String result = (String) twoConcatOfStringAndPrimitiveLong.invoke("ab", 123L);
    assertThat(result).isEqualTo("ab123");
  }

  @Inject
  @RuntimeMethodHandle(
      className = "StringConcatTestCases",
      memberName = "twoConcatWithPrimitives",
      memberDescriptor = "(Ljava/lang/String;D)Ljava/lang/String;")
  private MethodHandle twoConcatOfStringAndPrimitiveDouble;

  @Test
  public void twoConcatWithPrimitives_StringAndDouble() throws Throwable {
    String result = (String) twoConcatOfStringAndPrimitiveDouble.invoke("ab", 123.125);
    assertThat(result).isEqualTo("ab123.125");
  }

  @Inject
  @RuntimeMethodHandle(
      className = "StringConcatTestCases",
      memberName = "twoConcatWithPrimitives",
      memberDescriptor = "(ILjava/lang/String;)Ljava/lang/String;")
  private MethodHandle twoConcatOfPrimitiveIntAndString;

  @Test
  public void twoConcatWithPrimitives_intAndString() throws Throwable {
    String result = (String) twoConcatOfPrimitiveIntAndString.invoke(123, "ABC");
    assertThat(result).isEqualTo("123ABC");
  }

  @Inject
  @RuntimeMethodHandle(
      className = "StringConcatTestCases",
      memberName = "twoConcatWithPrimitives",
      memberDescriptor = "(JLjava/lang/String;)Ljava/lang/String;")
  private MethodHandle twoConcatOfPrimitiveLongAndString;

  @Test
  public void twoConcatWithPrimitives_longAndString() throws Throwable {
    String result = (String) twoConcatOfPrimitiveLongAndString.invoke(123L, "ABC");
    assertThat(result).isEqualTo("123ABC");
  }

  @Inject
  @RuntimeMethodHandle(
      className = "StringConcatTestCases",
      memberName = "twoConcatWithPrimitives",
      memberDescriptor = "(DLjava/lang/String;)Ljava/lang/String;")
  private MethodHandle twoConcatOfPrimitiveDoubleAndString;

  @Test
  public void twoConcatWithPrimitives_doubleAndString() throws Throwable {
    String result = (String) twoConcatOfPrimitiveDoubleAndString.invoke(123.125, "ABC");
    assertThat(result).isEqualTo("123.125ABC");
  }

  @Inject
  @RuntimeMethodHandle(
      className = "StringConcatTestCases",
      memberName = "concatWithAllPrimitiveTypes")
  private MethodHandle concatWithAllPrimitiveTypes;

  @Test
  public void concatWithAllPrimitiveTypes() throws Throwable {
    String result =
        (String)
            concatWithAllPrimitiveTypes.invoke(
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
