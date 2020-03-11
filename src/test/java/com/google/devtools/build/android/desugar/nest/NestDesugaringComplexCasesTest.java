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

import com.google.devtools.build.android.desugar.testing.junit.AsmNode;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRunner;
import com.google.devtools.build.android.desugar.testing.junit.JdkSuppress;
import com.google.devtools.build.android.desugar.testing.junit.JdkVersion;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.objectweb.asm.tree.ClassNode;

/**
 * Tests for accessing a series of private fields, constructors and methods from another class
 * within a nest.
 */
@RunWith(DesugarRunner.class)
@JdkSuppress(minJdkVersion = JdkVersion.V11)
public final class NestDesugaringComplexCasesTest {

  private static final MethodHandles.Lookup lookup = MethodHandles.lookup();

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, lookup)
          .addSourceInputsFromJvmFlag("input_srcs")
          .addJavacOptions("-source 11", "-target 11")
          .setWorkingJavaPackage(
              "com.google.devtools.build.android.desugar.nest.testsrc.complexcase")
          .addCommandOptions("desugar_nest_based_private_access", "true")
          .build();

  @Test
  public void inputClassFileMajorVersions(
      @AsmNode(className = "Xylem", round = 0) ClassNode beforeDesugarClassNode,
      @AsmNode(className = "Xylem", round = 1) ClassNode afterDesugarClassNode) {
    assertThat(beforeDesugarClassNode.version).isEqualTo(JdkVersion.V11);
    assertThat(afterDesugarClassNode.version).isEqualTo(JdkVersion.V1_7);
  }

  @Test
  public void comprehensiveTest(
      @RuntimeMethodHandle(className = "Xylem", memberName = "execute") MethodHandle xylemExecute)
      throws Throwable {
    long result = (long) xylemExecute.invoke((long) 2L, (int) 3);
    assertThat(result).isEqualTo(14004004171L);
  }
}
