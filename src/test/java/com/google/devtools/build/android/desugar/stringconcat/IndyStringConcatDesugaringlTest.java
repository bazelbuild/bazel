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
import com.google.devtools.build.android.desugar.testing.junit.FromParameterValueSource;
import com.google.devtools.build.android.desugar.testing.junit.JdkSuppress;
import com.google.devtools.build.android.desugar.testing.junit.JdkVersion;
import com.google.devtools.build.android.desugar.testing.junit.ParameterValueSource;
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
  @ParameterValueSource({"", "", ""})
  @ParameterValueSource({"", "b", "b"})
  @ParameterValueSource({"a", "", "a"})
  @ParameterValueSource({"a", "b", "ab"})
  @ParameterValueSource({"ab", "cd", "abcd"})
  public void twoConcat(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcat",
              memberDescriptor = "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;")
          MethodHandle twoConcat,
      @FromParameterValueSource String x,
      @FromParameterValueSource String y,
      @FromParameterValueSource String expectedResult)
      throws Throwable {
    String result = (String) twoConcat.invoke(x, y);
    assertThat(result).isEqualTo(expectedResult);
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
  @ParameterValueSource({"a", "0", "a0"})
  @ParameterValueSource({"a", "1", "a1"})
  @ParameterValueSource({"b", "3", "b3"})
  @ParameterValueSource({"c", "7", "c7"})
  @ParameterValueSource({"a", "15", "a15"})
  @ParameterValueSource({"d", "2147483647", "d2147483647"}) // Integer.MAX_VALUE
  @ParameterValueSource({"d", "-2147483648", "d-2147483648"}) // Integer.MIN_VALUE
  public void twoConcatWithPrimitives_StringAndInt(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcatWithPrimitives",
              memberDescriptor = "(Ljava/lang/String;I)Ljava/lang/String;")
          MethodHandle twoConcat,
      @FromParameterValueSource String x,
      @FromParameterValueSource int y,
      @FromParameterValueSource String expectedResult)
      throws Throwable {
    String result = (String) twoConcat.invoke(x, y);
    assertThat(result).isEqualTo(expectedResult);
  }

  @Test
  @ParameterValueSource({"a", "1", "a1"})
  @ParameterValueSource({"b", "3", "b3"})
  @ParameterValueSource({"c", "7", "c7"})
  @ParameterValueSource({"a", "15", "a15"})
  @ParameterValueSource({"d", "9223372036854775807", "d9223372036854775807"}) // Long.MAX_VALUE
  @ParameterValueSource({"e", "-9223372036854775808", "e-9223372036854775808"}) // Long.MIN_VALUE
  public void twoConcatWithPrimitives_StringAndLong(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcatWithPrimitives",
              memberDescriptor = "(Ljava/lang/String;J)Ljava/lang/String;")
          MethodHandle twoConcat,
      @FromParameterValueSource String x,
      @FromParameterValueSource long y,
      @FromParameterValueSource String expectedResult)
      throws Throwable {
    String result = (String) twoConcat.invoke(x, y);
    assertThat(result).isEqualTo(expectedResult);
  }

  @Test
  @ParameterValueSource({"a", "123.125", "a123.125"})
  @ParameterValueSource({
    "max-double-",
    "1.7976931348623157E+308",
    "max-double-1.7976931348623157E308"
  })
  @ParameterValueSource({
    "min-double-",
    "2.2250738585072014E-308",
    "min-double-2.2250738585072014E-308"
  })
  public void twoConcatWithPrimitives_StringAndDouble(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "twoConcatWithPrimitives",
              memberDescriptor = "(Ljava/lang/String;D)Ljava/lang/String;")
          MethodHandle twoConcat,
      @FromParameterValueSource String x,
      @FromParameterValueSource double y,
      @FromParameterValueSource String expectedResult)
      throws Throwable {
    String result = (String) twoConcat.invoke(x, y);
    assertThat(result).isEqualTo(expectedResult);
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
  @ParameterValueSource({
    "head", // string value
    "1", // int value
    "true", // boolean value
    "2", // byte value
    "c", // char value
    "4", // short value
    "5.25", // double value
    "6.5F", // float value
    "7", // long value
    "head/1/true/2/c/4/5.25/6.5/7" // expectedResult
  })
  @ParameterValueSource({
    "min:", // string value
    "-2147483648", // int value
    "false", // boolean value
    "-128", // byte value
    "~", // char value
    "-32768", // short value
    "1.7976931348623157E+308", // double value
    "1.4E-45F", // float value
    "-9223372036854775808", // long value
    "min:/-2147483648/false/-128/~/-32768/1.7976931348623157E308/1.4E-45/-9223372036854775808"
  })
  @ParameterValueSource({
    "max:", // string value
    "2147483647", // int value
    "true", // boolean value
    "127", // byte value
    "~", // char value
    "32767", // short value
    "2.2250738585072014E-308", // double value
    "3.4028235E+38F", // float value
    "9223372036854775807", // long value
    "max:/2147483647/true/127/~/32767/2.2250738585072014E-308/3.4028235E38/9223372036854775807"
  })
  public void concatWithAllPrimitiveTypes(
      @RuntimeMethodHandle(
              className = "StringConcatTestCases",
              memberName = "concatWithAllPrimitiveTypes")
          MethodHandle concatWithAllPrimitiveTypes,
      @FromParameterValueSource String stringValue,
      @FromParameterValueSource int intValue,
      @FromParameterValueSource boolean booleanValue,
      @FromParameterValueSource byte byteValue,
      @FromParameterValueSource char charValue,
      @FromParameterValueSource short shortValue,
      @FromParameterValueSource double doubleValue,
      @FromParameterValueSource float floatValue,
      @FromParameterValueSource long longValue,
      @FromParameterValueSource String expectedResult)
      throws Throwable {
    String result =
        (String)
            concatWithAllPrimitiveTypes.invoke(
                stringValue,
                intValue,
                booleanValue,
                byteValue,
                charValue,
                shortValue,
                doubleValue,
                floatValue,
                longValue);
    assertThat(result).isEqualTo(expectedResult);
  }
}
