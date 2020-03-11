/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.covariantreturn;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static org.objectweb.asm.tree.AbstractInsnNode.METHOD_INSN;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.android.desugar.testing.junit.AsmNode;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRunner;
import com.google.devtools.build.android.desugar.testing.junit.JdkSuppress;
import com.google.devtools.build.android.desugar.testing.junit.JdkVersion;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.nio.charset.Charset;
import java.util.Arrays;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.TypeInsnNode;

/** Functional Tests for {@link NioBufferRefConverter}. */
@RunWith(DesugarRunner.class)
@JdkSuppress(minJdkVersion = JdkVersion.V11)
public class NioBufferRefConverterTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addSourceInputsFromJvmFlag("input_srcs")
          .addJavacOptions("-source 11", "-target 11")
          .addCommandOptions("desugar_nest_based_private_access", "true")
          .addCommandOptions("allow_empty_bootclasspath", "true")
          .addCommandOptions("core_library", "true")
          .setWorkingJavaPackage("com.google.devtools.build.android.desugar.covariantreturn")
          .build();

  @Test
  public void methodOfNioBufferWithCovariantTypes_beforeDesugar(
      @AsmNode(className = "NioBufferInvocations", memberName = "getByteBufferPosition", round = 0)
          MethodNode before) {
    ImmutableList<AbstractInsnNode> methodInvocations =
        Arrays.stream(before.instructions.toArray())
            .filter(insnNode -> insnNode.getType() == METHOD_INSN)
            .collect(toImmutableList());

    assertThat(methodInvocations).hasSize(1);
    MethodInsnNode methodInsnNode = (MethodInsnNode) Iterables.getOnlyElement(methodInvocations);

    assertThat(methodInsnNode.owner).isEqualTo("java/nio/ByteBuffer");
    assertThat(methodInsnNode.name).isEqualTo("position");
    assertThat(methodInsnNode.desc).isEqualTo("(I)Ljava/nio/ByteBuffer;");

    assertThat(methodInsnNode.getNext().getOpcode()).isEqualTo(Opcodes.ARETURN);
  }

  @Test
  public void methodOfNioBufferWithCovariantTypes_afterDesugar(
      @AsmNode(className = "NioBufferInvocations", memberName = "getByteBufferPosition", round = 1)
          MethodNode after) {
    ImmutableList<AbstractInsnNode> methodInvocations =
        Arrays.stream(after.instructions.toArray())
            .filter(insnNode -> insnNode.getType() == METHOD_INSN)
            .collect(toImmutableList());

    assertThat(methodInvocations).hasSize(1);
    MethodInsnNode methodInsnNode = (MethodInsnNode) Iterables.getOnlyElement(methodInvocations);

    assertThat(methodInsnNode.owner).isEqualTo("java/nio/ByteBuffer");
    assertThat(methodInsnNode.name).isEqualTo("position");
    assertThat(methodInsnNode.desc).isEqualTo("(I)Ljava/nio/Buffer;");

    TypeInsnNode typeInsnNode = (TypeInsnNode) methodInsnNode.getNext();
    assertThat(typeInsnNode.getOpcode()).isEqualTo(Opcodes.CHECKCAST);
    assertThat(typeInsnNode.desc).isEqualTo("java/nio/ByteBuffer");

    assertThat(typeInsnNode.getNext().getOpcode()).isEqualTo(Opcodes.ARETURN);
  }

  @Test
  public void methodOfNioBufferWithCovariantTypes_beforeDesugarInvocation(
      @RuntimeMethodHandle(
              className = "NioBufferInvocations",
              memberName = "getByteBufferPosition",
              round = 0)
          MethodHandle before)
      throws Throwable {
    ByteBuffer buffer = ByteBuffer.wrap("random text".getBytes(Charset.defaultCharset()));
    int expectedPos = 2;

    ByteBuffer result = (ByteBuffer) before.invoke(buffer, expectedPos);
    assertThat(result.position()).isEqualTo(expectedPos);
  }

  @Test
  public void methodOfNioBufferWithCovariantTypes_afterDesugarInvocationOfByteBufferMethod(
      @RuntimeMethodHandle(className = "NioBufferInvocations", memberName = "getByteBufferPosition")
          MethodHandle after)
      throws Throwable {
    ByteBuffer buffer = ByteBuffer.wrap("random text".getBytes(Charset.defaultCharset()));
    int expectedPos = 2;

    ByteBuffer result = (ByteBuffer) after.invoke(buffer, expectedPos);
    assertThat(result.position()).isEqualTo(expectedPos);
  }

  @Test
  public void methodOfNioBufferWithCovariantTypes_afterDesugarInvocationOfCharBufferMethod(
      @RuntimeMethodHandle(className = "NioBufferInvocations", memberName = "getCharBufferPosition")
          MethodHandle after)
      throws Throwable {
    CharBuffer buffer = CharBuffer.wrap("random text".toCharArray());
    int expectedPos = 2;

    CharBuffer result = (CharBuffer) after.invoke(buffer, expectedPos);
    assertThat(result.position()).isEqualTo(expectedPos);
  }

  @Test
  public void methodOfNioBufferWithCovariantTypes_afterDesugarInvocationOfIntBufferMethod(
      @RuntimeMethodHandle(className = "NioBufferInvocations", memberName = "getIntBufferPosition")
          MethodHandle after)
      throws Throwable {
    IntBuffer buffer = IntBuffer.wrap(new int[] {10, 20, 30});
    int expectedPos = 2;

    IntBuffer result = (IntBuffer) after.invoke(buffer, expectedPos);
    assertThat(result.position()).isEqualTo(expectedPos);
  }

  @Test
  public void methodOfNioBufferWithCovariantTypes_afterDesugarInvocationOfFloatBufferMethod(
      @RuntimeMethodHandle(
              className = "NioBufferInvocations",
              memberName = "getFloatBufferPosition")
          MethodHandle after)
      throws Throwable {
    FloatBuffer buffer = FloatBuffer.wrap(new float[] {10f, 20f, 30f});
    int expectedPos = 2;

    FloatBuffer result = (FloatBuffer) after.invoke(buffer, expectedPos);
    assertThat(result.position()).isEqualTo(expectedPos);
  }

  @Test
  public void methodOfNioBufferWithCovariantTypes_afterDesugarInvocationOfDoubleBufferMethod(
      @RuntimeMethodHandle(
              className = "NioBufferInvocations",
              memberName = "getDoubleBufferPosition")
          MethodHandle after)
      throws Throwable {
    DoubleBuffer buffer = DoubleBuffer.wrap(new double[] {10.0, 20.0, 30.0});
    int expectedPos = 2;

    DoubleBuffer result = (DoubleBuffer) after.invoke(buffer, expectedPos);
    assertThat(result.position()).isEqualTo(expectedPos);
  }

  @Test
  public void methodOfNioBufferWithCovariantTypes_afterDesugarInvocationOfShortBufferMethod(
      @RuntimeMethodHandle(
              className = "NioBufferInvocations",
              memberName = "getShortBufferPosition")
          MethodHandle after)
      throws Throwable {
    ShortBuffer buffer = ShortBuffer.wrap(new short[] {10, 20, 30});
    int expectedPos = 2;

    ShortBuffer result = (ShortBuffer) after.invoke(buffer, expectedPos);
    assertThat(result.position()).isEqualTo(expectedPos);
  }

  @Test
  public void methodOfNioBufferWithCovariantTypes_afterDesugarInvocationOfLongBufferMethod(
      @RuntimeMethodHandle(className = "NioBufferInvocations", memberName = "getLongBufferPosition")
          MethodHandle after)
      throws Throwable {
    LongBuffer buffer = LongBuffer.wrap(new long[] {10L, 20L, 30L});
    int expectedPos = 2;

    LongBuffer result = (LongBuffer) after.invoke(buffer, expectedPos);
    assertThat(result.position()).isEqualTo(expectedPos);
  }
}
