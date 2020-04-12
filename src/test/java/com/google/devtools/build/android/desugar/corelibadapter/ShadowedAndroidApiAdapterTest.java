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

package com.google.devtools.build.android.desugar.corelibadapter;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.android.desugar.testing.junit.DesugarTestHelpers.findMethodInvocationSites;
import static com.google.devtools.build.android.desugar.testing.junit.DesugarTestHelpers.getRuntimePathsFromJvmFlag;

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
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.MethodNode;

/** Tests for accessing private constructors from another class within a nest. */
@RunWith(DesugarRunner.class)
@JdkSuppress(minJdkVersion = JdkVersion.V11)
public class ShadowedAndroidApiAdapterTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addSourceInputsFromJvmFlag("input_srcs")
          .addClasspathEntries(getRuntimePathsFromJvmFlag("desugar_jdk_jar"))
          .addClasspathEntries(getRuntimePathsFromJvmFlag("desugar_runtime_libs"))
          .addClasspathEntries(getRuntimePathsFromJvmFlag("jdk_jar"))
          .addJavacOptions("-source 11", "-target 11")
          .addCommandOptions("core_library", "true")
          .addCommandOptions("desugar_supported_core_libs", "true")
          .enableIterativeTransformation(2)
          .addCommandOptions("rewrite_core_library_prefix", "javadesugar/testing/")
          .build();

  @Test
  public void inputClassFileMajorVersions_preDesugar(
      @AsmNode(className = "com.app.testing.CuboidCalculator", round = 0) ClassNode before) {
    assertThat(before.version).isEqualTo(JdkVersion.V11);
  }

  @Test
  public void inputClassFileMajorVersions_postDesugar(
      @AsmNode(className = "com.app.testing.CuboidCalculator", round = 1) ClassNode before) {
    assertThat(before.version).isEqualTo(JdkVersion.V1_7);
  }

  @Test
  public void executeUserApp_invokeConstructorPreDesugar(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeConstructor",
              round = 0)
          MethodHandle method)
      throws Throwable {
    long result = (long) method.invoke(2L, 3L, 4L);
    assertThat(result).isEqualTo(24L);
  }

  @Test
  public void executeUserApp_invokeConstructorDesugarOnce(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeConstructor",
              round = 1)
          MethodHandle method)
      throws Throwable {
    long result = (long) method.invoke(2L, 3L, 4L);
    assertThat(result).isEqualTo(24L);
  }

  @Test
  public void executeUserApp_invokeConstructorDesugarTwice(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeConstructor",
              round = 2)
          MethodHandle method)
      throws Throwable {
    long result = (long) method.invoke(2L, 3L, 4L);
    assertThat(result).isEqualTo(24L);
  }

  @Test
  public void executeUserApp_invokeDerivedClassConstructorWithEmbeddedInstancePreDesugar(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeDerivedClassConstructorWithEmbeddedInstance",
              round = 0)
          MethodHandle method)
      throws Throwable {
    long result = (long) method.invoke(2L, 3L, 4L);
    assertThat(result).isEqualTo(24L);
  }

  @Test
  public void executeUserApp_invokeDerivedClassConstructorWithEmbeddedInstanceDesugarOnce(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeDerivedClassConstructorWithEmbeddedInstance",
              round = 1)
          MethodHandle method)
      throws Throwable {
    long result = (long) method.invoke(2L, 3L, 4L);
    assertThat(result).isEqualTo(24L);
  }

  @Test
  public void executeUserApp_invokeDerivedClassConstructorWithEmbeddedInstanceDesugarTwice(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeDerivedClassConstructorWithEmbeddedInstance",
              round = 2)
          MethodHandle method)
      throws Throwable {
    long result = (long) method.invoke(2L, 3L, 4L);
    assertThat(result).isEqualTo(24L);
  }

  @Test
  @ParameterValueSource({"2", "3", "4", "64"})
  @ParameterValueSource({"20", "30", "40", "24000"})
  public void executeUserApp_invokeDerivedClassConstructorWithDimensionsPreDesugar(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeDerivedClassConstructorWithDimensions",
              round = 0)
          MethodHandle method,
      @FromParameterValueSource long width,
      @FromParameterValueSource long length,
      @FromParameterValueSource long height,
      @FromParameterValueSource long expectedVolume)
      throws Throwable {
    long result = (long) method.invoke(width, length, height);
    assertThat(result).isEqualTo(expectedVolume);
  }

  @Test
  @ParameterValueSource({"2", "3", "4", "64"})
  @ParameterValueSource({"20", "30", "40", "24000"})
  public void executeUserApp_invokeDerivedClassConstructorWithDimensionsDesugarOnce(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeDerivedClassConstructorWithDimensions",
              round = 1)
          MethodHandle method,
      @FromParameterValueSource long width,
      @FromParameterValueSource long length,
      @FromParameterValueSource long height,
      @FromParameterValueSource long expectedVolume)
      throws Throwable {
    long result = (long) method.invoke(width, length, height);
    assertThat(result).isEqualTo(expectedVolume);
  }

  @Test
  @ParameterValueSource({"2", "3", "4", "64"})
  @ParameterValueSource({"20", "30", "40", "24000"})
  public void executeUserApp_invokeDerivedClassConstructorWithDimensionsDesugarTwice(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeDerivedClassConstructorWithDimensions",
              round = 2)
          MethodHandle method,
      @FromParameterValueSource long width,
      @FromParameterValueSource long length,
      @FromParameterValueSource long height,
      @FromParameterValueSource long expectedVolume)
      throws Throwable {
    long result = (long) method.invoke(width, length, height);
    assertThat(result).isEqualTo(expectedVolume);
  }

  @Test
  public void executeUserApp_invokeStaticMethodPreDesugar(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeStaticMethod",
              round = 0)
          MethodHandle method)
      throws Throwable {
    long result = (long) method.invoke(2L, 3L, 4L, 10L, 10L, 10L);
    assertThat(result).isEqualTo(24000L);
  }

  @Test
  public void executeUserApp_invokeStaticMethodDesugarOnce(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeStaticMethod",
              round = 1)
          MethodHandle method)
      throws Throwable {
    long result = (long) method.invoke(2L, 3L, 4L, 10L, 10L, 10L);
    assertThat(result).isEqualTo(24000L);
  }

  @Test
  public void executeUserApp_invokeStaticMethodDesugarTwice(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeStaticMethod",
              round = 2)
          MethodHandle method)
      throws Throwable {
    long result = (long) method.invoke(2L, 3L, 4L, 10L, 10L, 10L);
    assertThat(result).isEqualTo(24000L);
  }

  @Test
  public void executeUserApp_invokeInstanceMethodPreDesugar(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeInstanceMethod",
              round = 0)
          MethodHandle method)
      throws Throwable {
    long result = (long) method.invoke(2L, 3L, 4L, 10L, 10L, 10L);
    assertThat(result).isEqualTo(24000L);
  }

  @Test
  public void executeUserApp_invokeInstanceMethodDesugarOnce(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeInstanceMethod",
              round = 1)
          MethodHandle method)
      throws Throwable {
    long result = (long) method.invoke(2L, 3L, 4L, 10L, 10L, 10L);
    assertThat(result).isEqualTo(24000L);
  }

  @Test
  public void executeUserApp_invokeInstanceMethodDesugarTwice(
      @RuntimeMethodHandle(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeInstanceMethod",
              round = 2)
          MethodHandle method)
      throws Throwable {
    long result = (long) method.invoke(2L, 3L, 4L, 10L, 10L, 10L);
    assertThat(result).isEqualTo(24000L);
  }

  @Test
  public void executeUserApp_invokeConstructorDesugarOnceInlineConversion(
      @AsmNode(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeConstructor",
              round = 1)
          MethodNode method) {
    assertThat(
            findMethodInvocationSites(
                method,
                "com/google/devtools/build/android/desugar/typeadapter/javadesugar/testing/CuboidConverter",
                "from",
                ".*"))
        .isEmpty();
    assertThat(
            findMethodInvocationSites(
                method,
                "com/google/devtools/build/android/desugar/typeadapter/javadesugar/testing/CuboidConverter",
                "to",
                ".*"))
        .isNotEmpty();
  }

  @Test
  public void executeUserApp_invokeStaticMethodDesugarConversionVerification(
      @AsmNode(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeStaticMethod",
              round = 1)
          MethodNode method) {
    assertThat(
            findMethodInvocationSites(
                method,
                "com/google/devtools/build/android/desugar/typeadapter/android/testing/CuboidInflater\\$.*\\$Adapter",
                "inflateStatic",
                ".*"))
        .isNotEmpty();
    assertThat(
            findMethodInvocationSites(
                method,
                "com/google/devtools/build/android/desugar/typeadapter/javadesugar/testing/CuboidConverter",
                "from",
                ".*"))
        .isEmpty();
    assertThat(
            findMethodInvocationSites(
                method,
                "com/google/devtools/build/android/desugar/typeadapter/javadesugar/testing/CuboidConverter",
                "to",
                ".*"))
        .isEmpty();
  }

  @Test
  public void executeUserApp_invokeInstanceMethodDesugarConversionVerification(
      @AsmNode(
              className = "com.app.testing.CuboidCalculator",
              memberName = "invokeInstanceMethod",
              round = 1)
          MethodNode method) {
    assertThat(
            findMethodInvocationSites(
                method,
                "com/google/devtools/build/android/desugar/typeadapter/android/testing/CuboidInflater\\$.*\\$Adapter",
                "inflateInstance",
                ".*"))
        .isNotEmpty();
    assertThat(
            findMethodInvocationSites(
                method,
                "com/google/devtools/build/android/desugar/typeadapter/javadesugar/testing/CuboidConverter",
                "from",
                ".*"))
        .isEmpty();
    assertThat(
            findMethodInvocationSites(
                method,
                "com/google/devtools/build/android/desugar/typeadapter/javadesugar/testing/CuboidConverter",
                "to",
                ".*"))
        .isEmpty();
  }
}
