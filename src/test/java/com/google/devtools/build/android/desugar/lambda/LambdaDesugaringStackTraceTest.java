/*
 * Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.lambda;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRunner;
import com.google.devtools.build.android.desugar.testing.junit.RuntimeMethodHandle;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.List;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Test class for testing the stack trace related behaviors from lambda desugaring. */
@RunWith(DesugarRunner.class)
public final class LambdaDesugaringStackTraceTest {
  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addSourceInputsFromJvmFlag("input_srcs")
          .addJavacOptions("-source 11", "-target 11")
          .setWorkingJavaPackage(
              "com.google.devtools.build.android.desugar.lambda.testsrc.stacktrace")
          .enableIterativeTransformation(2)
          .build();

  @Test
  public void stackTraceFileNamesThroughLambda_beforeDesugaring(
      @RuntimeMethodHandle(
              className = "StackTraceTestTarget",
              memberName = "getStackTraceFileNamesThroughLambda",
              round = 0)
          MethodHandle stackTraceTestTarget)
      throws Throwable {
    List<String> result = (List<String>) stackTraceTestTarget.invoke();
    assertThat(result.subList(0, 4))
        .containsExactly(
            "StackTraceTestTarget.java",
            "StackTraceTestTarget.java",
            "StackTraceTestTarget.java",
            "LambdaDesugaringStackTraceTest.java")
        .inOrder();
  }

  @Test
  public void stackTraceFileNamesThroughLambda_afterDesugaring(
      @RuntimeMethodHandle(
              className = "StackTraceTestTarget",
              memberName = "getStackTraceFileNamesThroughLambda",
              round = 1)
          MethodHandle stackTraceTestTarget)
      throws Throwable {
    List<String> result = (List<String>) stackTraceTestTarget.invoke();
    assertThat(result.subList(0, 5))
        .containsExactly(
            "StackTraceTestTarget.java",
            "StackTraceTestTarget.java",
            "StackTraceTestTarget.java",
            "StackTraceTestTarget.java",
            "LambdaDesugaringStackTraceTest.java")
        .inOrder();
  }

  @Test
  public void stackTraceFileNamesThroughLambda_afterDesugaringTwice(
      @RuntimeMethodHandle(
              className = "StackTraceTestTarget",
              memberName = "getStackTraceFileNamesThroughLambda",
              round = 2)
          MethodHandle stackTraceTestTarget)
      throws Throwable {
    List<String> result = (List<String>) stackTraceTestTarget.invoke();
    assertThat(result.subList(0, 5))
        .containsExactly(
            "StackTraceTestTarget.java",
            "StackTraceTestTarget.java",
            "StackTraceTestTarget.java",
            "StackTraceTestTarget.java",
            "LambdaDesugaringStackTraceTest.java")
        .inOrder();
  }

  @Test
  public void stackTraceFileNamesThroughNestedLambda_afterDesugaring(
      @RuntimeMethodHandle(
              className = "StackTraceTestTarget",
              memberName = "getStackTraceFileNamesThroughNestedLambda",
              round = 1)
          MethodHandle stackTraceTestTarget)
      throws Throwable {
    List<String> result = (List<String>) stackTraceTestTarget.invoke();
    assertThat(result.subList(0, 5))
        .containsExactly(
            "StackTraceTestTarget.java",
            "StackTraceTestTarget.java",
            "StackTraceTestTarget.java",
            "StackTraceTestTarget.java",
            "LambdaDesugaringStackTraceTest.java")
        .inOrder();
  }
}
