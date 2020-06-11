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

package com.google.devtools.build.android.desugar.nest;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.android.desugar.testing.junit.DesugarTestHelpers.getRuntimePathsFromJvmFlag;

import com.google.devtools.build.android.desugar.testing.junit.AsmNode;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRunner;
import com.google.devtools.build.android.desugar.testing.junit.JdkSuppress;
import com.google.devtools.build.android.desugar.testing.junit.JdkVersion;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.objectweb.asm.tree.ClassNode;

/** Tests for accessing private constructors from another class within a nest. */
@RunWith(DesugarRunner.class)
@JdkSuppress(minJdkVersion = JdkVersion.V11)
public class NestDesugaringCoreLibTest {

  private static final Lookup lookup = MethodHandles.lookup();

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, lookup)
          .addSourceInputs(getRuntimePathsFromJvmFlag("input_srcs"))
          .addJavacOptions("-source 11", "-target 11")
          .addCommandOptions("desugar_nest_based_private_access", "true")
          .addCommandOptions("allow_empty_bootclasspath", "true")
          .addCommandOptions("core_library", "true")
          .addCommandOptions("desugar_supported_core_libs", "true")
          .addCommandOptions("rewrite_core_library_prefix", "javadesugar/testing/")
          .build();

  @Test
  public void inputClassFileMajorVersions(
      @AsmNode(className = "javadesugar.testing.TestCoreType$MateA", round = 0) ClassNode before,
      @AsmNode(className = "jd$.testing.TestCoreType$MateA", round = 1) ClassNode after) {
    assertThat(before.version).isEqualTo(JdkVersion.V11);
    assertThat(after.version).isEqualTo(JdkVersion.V1_7);
  }

  @Test
  public void invokeInterMatePrivateStaticMethodOfCoreLibType(
      @RuntimeMethodHandle(className = "jd$.testing.TestCoreType", memberName = "twoSum")
          MethodHandle twoSum)
      throws Throwable {
    long result = (long) twoSum.invoke(1L, 2L);
    assertThat(result).isEqualTo(3L);
  }

  @Test
  public void invokeInterMatePrivateInstanceMethodOfCoreLibType(
      @RuntimeMethodHandle(className = "jd$.testing.TestCoreType", memberName = "twoSumWithBase")
          MethodHandle twoSum)
      throws Throwable {
    long result = (long) twoSum.invoke(1000L, 1L, 2L);
    assertThat(result).isEqualTo(1003L);
  }
}
