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

import com.google.devtools.build.android.desugar.testing.junit.AsmNode;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRunner;
import com.google.devtools.build.android.desugar.testing.junit.JdkSuppress;
import com.google.devtools.build.android.desugar.testing.junit.JdkVersion;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.nio.file.Paths;
import java.util.Arrays;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.MethodNode;

/**
 * Tests for accessing a series of private fields, constructors and methods from another class
 * within a nest.
 */
@RunWith(DesugarRunner.class)
public final class IndyStringConcatDesugaringlTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addInputs(Paths.get(System.getProperty("input_jar")))
          .setWorkingJavaPackage("com.google.devtools.build.android.desugar.stringconcat")
          .addCommandOptions("desugar_indy_string_concat", "true")
          .build();

  @Test
  @JdkSuppress(minJdkVersion = JdkVersion.V11)
  public void invokeDynamicInstr_presentBeforeDesugaring(
      @AsmNode(className = "StringConcatTestCases", memberName = "simplePrefix", round = 0)
          MethodNode simpleStrConcatBeforeDesugar) {
    AbstractInsnNode[] instructions = simpleStrConcatBeforeDesugar.instructions.toArray();
    assertThat(Arrays.stream(instructions).map(AbstractInsnNode::getOpcode))
        .contains(Opcodes.INVOKEDYNAMIC);
  }

  @Test
  @JdkSuppress(minJdkVersion = JdkVersion.V11)
  public void invokeDynamicInstr_absentAfterDesugaring(
      @AsmNode(className = "StringConcatTestCases", memberName = "simplePrefix", round = 1)
          MethodNode simpleStrConcatAfterDesugar) {
    AbstractInsnNode[] instructions = simpleStrConcatAfterDesugar.instructions.toArray();
    assertThat(Arrays.stream(instructions).map(AbstractInsnNode::getOpcode))
        .doesNotContain(Opcodes.INVOKEDYNAMIC);
  }

  @Test
  public void twoConcat(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcat",
              memberDescriptor = "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;")
          MethodHandle twoConcat)
      throws Throwable {
    String result = (String) twoConcat.invoke("ab", "cd");
    assertThat(result).isEqualTo("abcd");
  }

  @Test
  public void twoConcat_StringAndObject(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcat",
              memberDescriptor = "(Ljava/lang/String;Ljava/lang/Object;)Ljava/lang/String;")
          MethodHandle twoConcat)
      throws Throwable {
    String result = (String) twoConcat.invoke("ab", (Object) "cd");
    assertThat(result).isEqualTo("T:abcd");
  }

  @Test
  public void twoConcatWithConstants(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcatWithConstants",
              memberDescriptor = "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;")
          MethodHandle twoConcat)
      throws Throwable {
    String result = (String) twoConcat.invoke("ab", "cd");
    assertThat(result).isEqualTo("ab<constant>cd");
  }

  @Test
  public void threeConcat(
      @RuntimeMethodHandle(className = "StringConcatTestCases", memberName = "threeConcat")
          MethodHandle threeConcat)
      throws Throwable {
    String result = (String) threeConcat.invoke("ab", "cd", "ef");
    assertThat(result).isEqualTo("abcdef");
  }

  @Test
  public void twoConcatWithRecipe(
      @RuntimeMethodHandle(className = "StringConcatTestCases", memberName = "twoConcatWithRecipe")
          MethodHandle twoConcat)
      throws Throwable {
    String result = (String) twoConcat.invoke("ab", "cd");
    assertThat(result).isEqualTo("<p>ab<br>cd</p>");
  }

  @Test
  public void threeConcatWithRecipe(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "threeConcatWithRecipe")
          MethodHandle threeConcat)
      throws Throwable {
    String result = (String) threeConcat.invoke("ab", "cd", "ef");
    assertThat(result).isEqualTo("<p>ab<br>cd<br>ef</p>");
  }

  @Test
  public void twoConcatWithPrimitives_StringAndInt(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcatWithPrimitives",
              memberDescriptor = "(Ljava/lang/String;I)Ljava/lang/String;")
          MethodHandle twoConcat)
      throws Throwable {
    String result = (String) twoConcat.invoke("ab", 123);
    assertThat(result).isEqualTo("ab123");
  }

  @Test
  public void twoConcatWithPrimitives_StringAndLong(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcatWithPrimitives",
              memberDescriptor = "(Ljava/lang/String;J)Ljava/lang/String;")
          MethodHandle twoConcat)
      throws Throwable {
    String result = (String) twoConcat.invoke("ab", 123L);
    assertThat(result).isEqualTo("ab123");
  }

  @Test
  public void twoConcatWithPrimitives_StringAndDouble(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcatWithPrimitives",
              memberDescriptor = "(Ljava/lang/String;D)Ljava/lang/String;")
          MethodHandle twoConcat)
      throws Throwable {
    String result = (String) twoConcat.invoke("ab", 123.125);
    assertThat(result).isEqualTo("ab123.125");
  }

  @Test
  public void twoConcatWithPrimitives_intAndString(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcatWithPrimitives",
              memberDescriptor = "(ILjava/lang/String;)Ljava/lang/String;")
          MethodHandle twoConcat)
      throws Throwable {
    String result = (String) twoConcat.invoke(123, "ABC");
    assertThat(result).isEqualTo("123ABC");
  }

  @Test
  public void twoConcatWithPrimitives_longAndString(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcatWithPrimitives",
              memberDescriptor = "(JLjava/lang/String;)Ljava/lang/String;")
          MethodHandle twoConcat)
      throws Throwable {
    String result = (String) twoConcat.invoke(123L, "ABC");
    assertThat(result).isEqualTo("123ABC");
  }

  @Test
  public void twoConcatWithPrimitives_doubleAndString(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcatWithPrimitives",
              memberDescriptor = "(DLjava/lang/String;)Ljava/lang/String;")
          MethodHandle twoConcat)
      throws Throwable {
    String result = (String) twoConcat.invoke(123.125, "ABC");
    assertThat(result).isEqualTo("123.125ABC");
  }

  @Test
  public void concatWithAllPrimitiveTypes(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "concatWithAllPrimitiveTypes")
          MethodHandle concatWithAllPrimitiveTypes)
      throws Throwable {
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
